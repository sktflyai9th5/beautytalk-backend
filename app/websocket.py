"""WebSocket 결과 push 채널.

WS /ws/{session_id} 로 연결을 유지하며 분석 결과를 push한다.
{"type": "ping"} → {"type": "pong"} 응답으로 연결 상태를 확인할 수 있고,
{"type": "analyze", ...} 트리거도 이 채널로 받을 수 있다.
"""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .analysis import handle_client_message
from .logging_utils import log_event
from .schemas import ConnectedMessage, ErrorMessage, WebRTCAnswerMessage
from .sessions import SessionLimitError
from .webrtc import SignalingAborted, negotiate_offer

logger = logging.getLogger("beautytalk.websocket")

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    state = websocket.app.state
    manager = state.manager
    await websocket.accept()
    try:
        session = await manager.attach_ws(session_id, websocket)
    except SessionLimitError:
        await websocket.send_text(
            ErrorMessage(message="동시 세션 수가 가득 찼습니다. 잠시 후 다시 시도해 주세요.").model_dump_json()
        )
        await websocket.close(code=1013, reason="session limit reached")
        return
    log_event(logger, "ws_connected", session_id=session_id)
    await websocket.send_text(ConnectedMessage(session_id=session_id).model_dump_json())
    try:
        while True:
            raw = await websocket.receive_text()
            if await _try_ws_signaling(state, session, websocket, raw):
                continue
            await handle_client_message(state, session, raw, source="websocket")
    except WebSocketDisconnect as e:
        log_event(logger, "ws_disconnected", session_id=session_id, code=e.code)
    finally:
        await manager.detach_ws(session_id, websocket)


async def _try_ws_signaling(state, session, websocket: WebSocket, raw: str) -> bool:
    """{"type": "webrtc_offer", "sdp": ..., "sdp_type": "offer"} 메시지면 WS로 시그널링.

    앱 WebView(file:// 페이지)는 fetch가 막혀 있어 HTTP POST 대신 이 경로를 쓴다.
    처리했으면 True를 반환한다.
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(msg, dict) or msg.get("type") != "webrtc_offer":
        return False

    session.touch()
    log_event(logger, "webrtc_offer_received", session_id=session.session_id, via="websocket")
    sdp = msg.get("sdp")
    sdp_type = msg.get("sdp_type") or "offer"
    if not isinstance(sdp, str) or not sdp or sdp_type != "offer":
        await websocket.send_text(ErrorMessage(message="invalid webrtc_offer").model_dump_json())
        return True
    try:
        answer = await negotiate_offer(state, session, sdp, sdp_type)
    except SignalingAborted as e:
        await websocket.send_text(ErrorMessage(message=str(e)).model_dump_json())
        return True
    except Exception:
        log_event(
            logger,
            "ws_signaling_failed",
            level=logging.ERROR,
            session_id=session.session_id,
            exc_info=True,
        )
        await websocket.send_text(
            ErrorMessage(message="시그널링에 실패했어요. 다시 연결해 주세요.").model_dump_json()
        )
        return True
    await websocket.send_text(
        WebRTCAnswerMessage(
            session_id=session.session_id, sdp=answer.sdp, sdp_type=answer.type
        ).model_dump_json()
    )
    return True
