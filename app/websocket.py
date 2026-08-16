"""WebSocket 결과 push 채널.

WS /ws/{session_id} 로 연결을 유지하며 분석 결과를 push한다.
{"type": "ping"} → {"type": "pong"} 응답으로 연결 상태를 확인할 수 있고,
{"type": "analyze", ...} 트리거도 이 채널로 받을 수 있다.
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .analysis import handle_client_message
from .logging_utils import log_event
from .schemas import ConnectedMessage, ErrorMessage
from .sessions import SessionLimitError

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
            await handle_client_message(state, session, raw, source="websocket")
    except WebSocketDisconnect as e:
        log_event(logger, "ws_disconnected", session_id=session_id, code=e.code)
    finally:
        await manager.detach_ws(session_id, websocket)
