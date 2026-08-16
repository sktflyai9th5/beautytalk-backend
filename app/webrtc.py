"""WebRTC 시그널링 및 영상 수신.

POST /webrtc/offer 로 SDP offer를 받아 answer를 돌려주고,
앱이 보내는 video track의 최신 프레임을 세션 버퍼에 유지한다.
Tailscale 내부망 전제이므로 ICE 서버 없이 host candidate만 사용한다.
"""

import asyncio
import contextlib
import logging

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from fastapi import APIRouter, HTTPException, Request

from .analysis import handle_client_message
from .logging_utils import log_event
from .schemas import OfferRequest, OfferResponse
from .sessions import Session, SessionLimitError, spawn_tracked

logger = logging.getLogger("beautytalk.webrtc")

router = APIRouter()

FRAME_LOG_INTERVAL = 300  # 프레임 수신 로그 주기 (약 10초 @30fps)


async def _consume_video(session: Session, track) -> None:
    """video track에서 프레임을 계속 읽어 최신 1장만 버퍼에 유지."""
    try:
        while True:
            frame = await track.recv()
            session.frames.put(frame)
            count = session.frames.frames_received
            if count == 1 or count % FRAME_LOG_INTERVAL == 0:
                log_event(
                    logger,
                    "video_frames_buffered",
                    session_id=session.session_id,
                    frames_received=count,
                    width=frame.width,
                    height=frame.height,
                )
    except MediaStreamError:
        log_event(logger, "video_track_ended", session_id=session.session_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        log_event(
            logger,
            "video_consume_error",
            level=logging.ERROR,
            session_id=session.session_id,
            exc_info=True,
        )


async def _discard_track(session: Session, track) -> None:
    """사용하지 않는 track(예: audio)을 소비해서 버려 수신 큐 무한 증가를 막는다."""
    try:
        while True:
            await track.recv()
    except MediaStreamError:
        pass
    except asyncio.CancelledError:
        raise
    except Exception:
        log_event(
            logger,
            "discard_track_error",
            level=logging.WARNING,
            session_id=session.session_id,
            kind=track.kind,
            exc_info=True,
        )


def _cancel_media_tasks(session: Session) -> None:
    for task in list(session.media_tasks):
        task.cancel()
    session.media_tasks.clear()


class SignalingAborted(Exception):
    """시그널링 도중 세션이 정리되어 협상을 중단함."""


async def negotiate_offer(state, session: Session, sdp: str, sdp_type: str) -> OfferResponse:
    """offer SDP를 받아 피어를 구성하고 answer를 돌려준다 (HTTP/WS 시그널링 공용)."""
    manager = state.manager
    session_id = session.session_id
    session.touch()

    # 같은 session_id의 동시 offer는 세션별 락으로 직렬화한다.
    async with session.signaling_lock:
        # 같은 session_id로 재시그널링하면 이전 피어를 정리하고 교체한다.
        # 진행 중인 분석 task는 살아있는 WebSocket으로 결과를 보내야 하므로 취소하지 않는다.
        old_pc = session.pc
        if old_pc is not None:
            session.pc = None
            session.data_channel = None
            _cancel_media_tasks(session)
            with contextlib.suppress(Exception):
                await old_pc.close()
            log_event(logger, "webrtc_peer_replaced", session_id=session_id)

        # old_pc.close()를 기다리는 사이 세션이 정리됐을 수 있다 (WS 끊김, 유휴 회수 등)
        if session.closing or manager.get(session_id) is not session:
            raise SignalingAborted("시그널링 중 세션이 종료되었습니다. 다시 연결해 주세요.")

        # Tailscale 내부망: STUN/TURN 없이 host candidate만으로 연결
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        session.pc = pc

        @pc.on("track")
        def on_track(track):
            log_event(logger, "webrtc_track_received", session_id=session_id, kind=track.kind)
            consumer = _consume_video if track.kind == "video" else _discard_track
            spawn_tracked(
                session.media_tasks,
                consumer(session, track),
                task_logger=logger,
                event="track_consumer_crashed",
                session_id=session_id,
            )

        @pc.on("datachannel")
        def on_datachannel(channel):
            session.data_channel = channel
            log_event(
                logger,
                "webrtc_datachannel_open",
                session_id=session_id,
                label=channel.label,
            )

            @channel.on("message")
            def on_message(message):
                spawn_tracked(
                    session.analysis_tasks,
                    handle_client_message(state, session, message, source="datachannel"),
                    task_logger=logger,
                    event="datachannel_message_crashed",
                    session_id=session_id,
                )

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_event(
                logger,
                "webrtc_connection_state",
                session_id=session_id,
                state=pc.connectionState,
            )
            if pc.connectionState in ("failed", "closed"):
                current = manager.get(session_id)
                if current is not None and current.pc is pc:
                    await manager.close_session(
                        session_id, reason=f"webrtc_{pc.connectionState}"
                    )
                else:
                    # 세션이 더 이상 이 피어를 소유하지 않으면 (교체/정리 경합) 직접 닫아 누수를 막는다
                    with contextlib.suppress(Exception):
                        await pc.close()

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)  # aiortc는 ICE gathering 완료까지 대기한다

        # ICE gathering을 기다리는 사이에도 세션이 정리됐을 수 있다
        if session.closing or manager.get(session_id) is not session:
            with contextlib.suppress(Exception):
                await pc.close()
            raise SignalingAborted("시그널링 중 세션이 종료되었습니다. 다시 연결해 주세요.")

        log_event(logger, "webrtc_answer_sent", session_id=session_id)
        return OfferResponse(
            session_id=session_id,
            sdp=pc.localDescription.sdp,
            type=pc.localDescription.type,
        )


@router.post("/webrtc/offer", response_model=OfferResponse)
async def webrtc_offer(body: OfferRequest, request: Request) -> OfferResponse:
    state = request.app.state
    log_event(logger, "webrtc_offer_received", session_id=body.session_id, via="http")
    try:
        session = state.manager.get_or_create(body.session_id)
    except SessionLimitError:
        raise HTTPException(status_code=503, detail="동시 세션 수가 가득 찼습니다. 잠시 후 다시 시도해 주세요.")
    try:
        return await negotiate_offer(state, session, body.sdp, body.type)
    except SignalingAborted as e:
        raise HTTPException(status_code=409, detail=str(e))
