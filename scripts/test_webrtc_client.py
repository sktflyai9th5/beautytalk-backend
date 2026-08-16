"""WebRTC 전체 E2E 테스트 클라이언트.

합성 프레임을 담은 video track을 WebRTC로 송신하고,
data channel과 WebSocket 두 경로 모두로 analyze 트리거를 보낸 뒤
WebSocket으로 analysis_result가 돌아오는 전체 흐름을 검증한다.

사용법:
    python scripts/test_webrtc_client.py --http-url http://127.0.0.1:8000 --ws-url ws://127.0.0.1:8000
"""

import argparse
import asyncio
import json
import sys
import uuid

import httpx
import numpy as np
import websockets
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class SyntheticFaceTrack(VideoStreamTrack):
    """움직이는 그라데이션 + 얼굴 모양 타원을 그린 합성 640x480 프레임을 송신."""

    def __init__(self):
        super().__init__()
        self._count = 0
        h, w = 480, 640
        yy, xx = np.ogrid[:h, :w]
        self._face_mask = ((xx - w / 2) ** 2 / 120**2 + (yy - h / 2) ** 2 / 160**2) <= 1
        self._gradient = np.linspace(0, 255, w, dtype=np.uint8)[None, :]

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        self._count += 1
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 0] = self._gradient
        img[:, :, 1] = (self._count * 3) % 256
        img[self._face_mask] = (170, 190, 220)
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


async def wait_result(queue: asyncio.Queue, request_id: str, timeout: float) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f"request_id={request_id} 결과가 오지 않음")
        msg = await asyncio.wait_for(queue.get(), timeout=remaining)
        if msg.get("type") == "analysis_result" and msg.get("request_id") == request_id:
            return msg


async def run(args) -> None:
    session_id = args.session_id or f"rtctest-{uuid.uuid4().hex[:8]}"
    results: asyncio.Queue = asyncio.Queue()

    # 1) WebSocket 결과 채널 연결
    ws_url = f"{args.ws_url}/ws/{session_id}"
    print(f"[*] connecting {ws_url}")
    ws = await websockets.connect(ws_url)
    connected = json.loads(await asyncio.wait_for(ws.recv(), 10))
    assert connected["type"] == "connected", connected
    print(f"[+] websocket connected: session_id={session_id}")

    async def ws_reader():
        async for raw in ws:
            msg = json.loads(raw)
            print(f"[ws<] {json.dumps(msg, ensure_ascii=False)}")
            await results.put(msg)

    reader_task = asyncio.create_task(ws_reader())

    # 2) WebRTC 피어 구성 (Tailscale 내부망 전제: ICE 서버 없이 host candidate만)
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    channel = pc.createDataChannel("control")
    channel_open = asyncio.Event()
    pc_connected = asyncio.Event()

    @channel.on("open")
    def on_channel_open():
        channel_open.set()

    @channel.on("message")
    def on_channel_message(message):
        print(f"[dc<] {message}")

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[pc] connectionState={pc.connectionState}")
        if pc.connectionState == "connected":
            pc_connected.set()

    pc.addTrack(SyntheticFaceTrack())

    # 3) 시그널링: POST /webrtc/offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.post(
            f"{args.http_url}/webrtc/offer",
            json={
                "session_id": session_id,
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            },
        )
        response.raise_for_status()
        answer = response.json()
    print("[+] answer received")
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )

    await asyncio.wait_for(pc_connected.wait(), 20)
    print("[+] webrtc connected")
    await asyncio.wait_for(channel_open.wait(), 20)
    print("[+] data channel open")

    # 4) 프레임이 서버 버퍼에 쌓일 시간을 준다
    await asyncio.sleep(args.stream_seconds)
    async with httpx.AsyncClient(timeout=10.0) as http:
        health = (await http.get(f"{args.http_url}/health")).json()
    print(f"[+] health during stream: {health}")
    assert health["webrtc_sessions"] >= 1, health
    assert health["ws_connections"] >= 1, health

    # mock 모드는 결정적 결과를 검증하고, 실모델 모드(--real-model)는
    # 유효한 분석 응답(ok/no_face)이 돌아오는지만 검증한다 (합성 영상엔 얼굴이 없으므로).
    ok_statuses = ("ok", "no_face") if args.real_model else ("ok",)

    # 5) 트리거 경로 1: WebRTC data channel
    channel.send(
        json.dumps(
            {"type": "analyze", "question": "립 봐줘잉", "request_id": "trigger-dc"},
            ensure_ascii=False,
        )
    )
    print("[*] analyze trigger sent via data channel")
    result = await wait_result(results, "trigger-dc", args.timeout)
    assert result["status"] in ok_statuses, result
    if not args.real_model:
        assert result["region"] == "lips", result
    assert result["message"], result
    print(f"[+] datachannel-trigger result: status={result['status']} message={result['message']}")

    # 6) 트리거 경로 2: WebSocket
    await ws.send(
        json.dumps(
            {"type": "analyze", "question": "볼 어때?", "request_id": "trigger-ws"},
            ensure_ascii=False,
        )
    )
    print("[*] analyze trigger sent via websocket")
    result = await wait_result(results, "trigger-ws", args.timeout)
    assert result["status"] in ok_statuses, result
    if not args.real_model:
        assert result["region"] == "cheeks", result
    assert result["message"], result
    print(f"[+] websocket-trigger result: status={result['status']} message={result['message']}")

    # 7) 정리
    reader_task.cancel()
    await pc.close()
    await ws.close()
    print("PASS: WebRTC full E2E ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--stream-seconds", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="실제 모델 서버 대상: no_face도 정상으로 간주하고 mock 전용 검증은 건너뜀",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
