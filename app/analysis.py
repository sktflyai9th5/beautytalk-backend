"""분석 파이프라인: 트리거 수신 → 최신 프레임 캡처 → 분석 → 결과 push.

트리거는 WebSocket 메시지와 WebRTC data channel 메시지 두 경로로 들어오며,
결과는 WebSocket 우선으로 push하고 WebSocket이 없으면 data channel로 보낸다.
"""

import asyncio
import io
import json
import logging
import time
import uuid

from pydantic import BaseModel

from .analyzer import AnalyzerError, NoFrameError
from .logging_utils import log_event
from .schemas import (
    AnalysisResult,
    AnalysisStartedMessage,
    ErrorMessage,
    PongMessage,
)
from .sessions import Session, spawn_tracked

logger = logging.getLogger("beautytalk.analysis")

DEFAULT_QUESTION = "지금 메이크업 상태 어때?"

NO_FRAME_MESSAGE = (
    "카메라 영상이 아직 도착하지 않았어요. 휴대폰을 얼굴 앞에 두고 잠시 후 다시 말씀해 주세요."
)
TIMEOUT_MESSAGE = "분석이 너무 오래 걸리고 있어요. 잠시 후 다시 한 번 물어봐 주세요."
GENERIC_ERROR_MESSAGE = "분석 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요."
BUSY_MESSAGE = "아직 이전 분석이 진행 중이에요. 잠시 후 다시 물어봐 주세요."


async def _frame_to_jpeg(frame) -> bytes:
    """av.VideoFrame → JPEG bytes. 인코딩은 CPU 작업이라 스레드로 넘긴다."""

    def _encode() -> bytes:
        image = frame.to_image()
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    return await asyncio.to_thread(_encode)


async def send_to_session(session: Session, message: BaseModel) -> str:
    """WebSocket 우선, 없으면 data channel로 전송. 전송 경로 문자열을 반환."""
    text = message.model_dump_json()
    ws = session.ws
    if ws is not None:
        try:
            await ws.send_text(text)
            return "websocket"
        except Exception:
            log_event(
                logger,
                "ws_send_failed",
                level=logging.WARNING,
                session_id=session.session_id,
                exc_info=True,
            )
    channel = session.data_channel
    if channel is not None and getattr(channel, "readyState", None) == "open":
        try:
            channel.send(text)
            return "datachannel"
        except Exception:
            log_event(
                logger,
                "datachannel_send_failed",
                level=logging.WARNING,
                session_id=session.session_id,
                exc_info=True,
            )
    log_event(
        logger,
        "message_dropped",
        level=logging.WARNING,
        session_id=session.session_id,
        message_type=getattr(message, "type", type(message).__name__),
    )
    return "dropped"


async def run_analysis(state, session: Session, *, question: str, request_id: str) -> None:
    """분석 1회 실행 후 결과를 세션으로 push. 어떤 예외도 error 결과로 변환한다."""
    settings = state.settings
    started = time.perf_counter()
    try:
        async with session.analysis_lock:
            frame = session.frames.latest()
            jpeg = await _frame_to_jpeg(frame) if frame is not None else None
            payload = await asyncio.wait_for(
                state.analyzer.analyze(jpeg, question),
                timeout=settings.analysis_timeout,
            )
        result = AnalysisResult.from_payload(
            payload, session_id=session.session_id, request_id=request_id
        )
    except NoFrameError:
        result = AnalysisResult(
            session_id=session.session_id,
            request_id=request_id,
            status="error",
            message=NO_FRAME_MESSAGE,
        )
    except asyncio.TimeoutError:
        result = AnalysisResult(
            session_id=session.session_id,
            request_id=request_id,
            status="error",
            message=TIMEOUT_MESSAGE,
        )
    except AnalyzerError:
        log_event(
            logger,
            "analysis_failed",
            level=logging.ERROR,
            session_id=session.session_id,
            request_id=request_id,
            exc_info=True,
        )
        result = AnalysisResult(
            session_id=session.session_id,
            request_id=request_id,
            status="error",
            message=GENERIC_ERROR_MESSAGE,
        )
    except Exception:
        log_event(
            logger,
            "analysis_unexpected_error",
            level=logging.ERROR,
            session_id=session.session_id,
            request_id=request_id,
            exc_info=True,
        )
        result = AnalysisResult(
            session_id=session.session_id,
            request_id=request_id,
            status="error",
            message=GENERIC_ERROR_MESSAGE,
        )

    duration_ms = round((time.perf_counter() - started) * 1000)
    channel = await send_to_session(session, result)
    log_event(
        logger,
        "analysis_done",
        session_id=session.session_id,
        request_id=request_id,
        status=result.status,
        region=result.region,
        verdict=result.verdict,
        duration_ms=duration_ms,
        sent_via=channel,
    )


async def handle_client_message(state, session: Session, raw, *, source: str) -> None:
    """앱 → 서버 JSON 메시지(WebSocket/data channel 공통) 처리."""
    if session.closing:
        return
    session.touch()
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            await send_to_session(session, ErrorMessage(message="binary message not supported"))
            return
    try:
        msg = json.loads(raw)
        if not isinstance(msg, dict):
            raise ValueError("message must be a JSON object")
    except (json.JSONDecodeError, ValueError):
        log_event(
            logger,
            "invalid_client_message",
            level=logging.WARNING,
            session_id=session.session_id,
            source=source,
        )
        await send_to_session(session, ErrorMessage(message="invalid JSON message"))
        return

    msg_type = msg.get("type")
    if msg_type == "ping":
        await send_to_session(session, PongMessage())
        return

    if msg_type == "analyze":
        request_id = str(msg.get("request_id") or uuid.uuid4().hex)
        question = str(msg.get("question") or DEFAULT_QUESTION).strip() or DEFAULT_QUESTION
        log_event(
            logger,
            "analyze_trigger",
            session_id=session.session_id,
            request_id=request_id,
            question=question,
            source=source,
            has_frame=session.frames.latest() is not None,
        )
        # 분석이 이미 진행 중이면 무한 큐잉하지 않고 거절한다 (음성 트리거 중복 발화 대비)
        if session.analysis_lock.locked():
            log_event(
                logger,
                "analyze_rejected_busy",
                level=logging.WARNING,
                session_id=session.session_id,
                request_id=request_id,
            )
            await send_to_session(
                session,
                AnalysisResult(
                    session_id=session.session_id,
                    request_id=request_id,
                    status="error",
                    message=BUSY_MESSAGE,
                ),
            )
            return
        await send_to_session(
            session,
            AnalysisStartedMessage(session_id=session.session_id, request_id=request_id),
        )
        # send를 기다리는 사이 세션이 정리됐을 수 있다
        if session.closing:
            return
        spawn_tracked(
            session.analysis_tasks,
            run_analysis(state, session, question=question, request_id=request_id),
            task_logger=logger,
            event="analysis_task_crashed",
            session_id=session.session_id,
        )
        return

    log_event(
        logger,
        "unknown_client_message",
        level=logging.WARNING,
        session_id=session.session_id,
        source=source,
        message_type=msg_type,
    )
    await send_to_session(session, ErrorMessage(message=f"unknown message type: {msg_type}"))
