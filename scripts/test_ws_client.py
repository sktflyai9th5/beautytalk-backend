"""WebSocket E2E 테스트 클라이언트.

/ws/{session_id} 접속 → ping/pong → analyze 트리거 전송 →
analysis_result 수신까지 확인한다 (mock 모드에서 서버만 떠 있으면 통과).

사용법:
    python scripts/test_ws_client.py --url ws://127.0.0.1:8000 --question "립 봐줘잉"
"""

import argparse
import asyncio
import json
import sys
import uuid

import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_KEYS = ("session_id", "request_id", "status", "region", "message", "detail", "timestamp")


async def run(url: str, session_id: str, question: str, timeout: float) -> None:
    ws_url = f"{url}/ws/{session_id}"
    print(f"[*] connecting {ws_url}")
    async with websockets.connect(ws_url) as ws:
        connected = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        assert connected["type"] == "connected", connected
        assert connected["session_id"] == session_id, connected
        print(f"[+] connected: session_id={connected['session_id']}")

        await ws.send(json.dumps({"type": "ping"}))
        pong = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        assert pong["type"] == "pong", pong
        print("[+] ping/pong ok")

        request_id = uuid.uuid4().hex
        await ws.send(
            json.dumps(
                {"type": "analyze", "question": question, "request_id": request_id},
                ensure_ascii=False,
            )
        )
        print(f"[*] analyze trigger sent: request_id={request_id} question={question!r}")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("analysis_result가 제한 시간 내에 도착하지 않음")
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            print(f"[<] {json.dumps(msg, ensure_ascii=False)}")
            if msg["type"] == "analysis_result":
                assert msg["request_id"] == request_id, msg
                assert msg["status"] in ("ok", "no_face", "error"), msg
                missing = [k for k in RESULT_KEYS if k not in msg]
                assert not missing, f"결과에 누락된 필드: {missing}"
                print(
                    f"[+] analysis_result: status={msg['status']} "
                    f"region={msg['region']} verdict={msg.get('verdict')}"
                )
                break

    print("PASS: WebSocket E2E ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000", help="ws://host:port")
    parser.add_argument("--session-id", default=f"wstest-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--question", default="립 봐줘잉")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.session_id, args.question, args.timeout))


if __name__ == "__main__":
    main()
