"""세션 관리.

session_id 하나가 (WebRTC 피어 + WebSocket 연결 + 최신 프레임 버퍼)를 묶는다.
어느 한쪽 연결이 끊기면 세션 전체를 정리하고, 유휴 세션은 주기적으로 회수한다.
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import WebSocket

from .logging_utils import log_event

logger = logging.getLogger("beautytalk.sessions")


class SessionLimitError(Exception):
    """동시 세션 수 상한 초과."""


def spawn_tracked(task_set: set, coro, *, task_logger, event: str, session_id: str):
    """task_set에 등록되는 asyncio task 생성.

    참조를 유지해 GC로 사라지는 것을 막고, 세션 정리 시 취소 대상이 되며,
    잡히지 않은 예외를 구조화 로그로 남긴다.
    """
    task = asyncio.create_task(coro)
    task_set.add(task)

    def _done(t: asyncio.Task) -> None:
        task_set.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log_event(
                task_logger,
                event,
                level=logging.ERROR,
                session_id=session_id,
                exc_info=t.exception(),
            )

    task.add_done_callback(_done)
    return task


class LatestFrameBuffer:
    """세션당 최신 프레임 1장만 유지해 메모리 누수를 방지한다."""

    def __init__(self):
        self._frame = None  # av.VideoFrame
        self.frames_received = 0
        self.last_frame_at: Optional[float] = None

    def put(self, frame) -> None:
        self._frame = frame
        self.frames_received += 1
        self.last_frame_at = time.monotonic()

    def latest(self):
        return self._frame

    def clear(self) -> None:
        self._frame = None


@dataclass
class Session:
    session_id: str
    pc: Optional[object] = None  # aiortc.RTCPeerConnection
    ws: Optional[WebSocket] = None
    data_channel: Optional[object] = None  # aiortc RTCDataChannel
    frames: LatestFrameBuffer = field(default_factory=LatestFrameBuffer)
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    analysis_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    signaling_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    media_tasks: set = field(default_factory=set)  # 프레임/트랙 소비 task
    analysis_tasks: set = field(default_factory=set)  # 트리거 처리/분석 task
    closing: bool = False

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        last = max(self.last_activity, self.frames.last_frame_at or 0.0)
        return time.monotonic() - last


class ConnectionManager:
    """session_id 기준으로 WebRTC 피어와 WebSocket 연결을 매핑/정리한다."""

    def __init__(
        self,
        *,
        idle_timeout: float = 300.0,
        reap_interval: float = 30.0,
        max_sessions: int = 50,
    ):
        self._sessions: dict[str, Session] = {}
        self._idle_timeout = idle_timeout
        self._reap_interval = reap_interval
        self._max_sessions = max_sessions
        self._reaper_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------ 조회

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            if len(self._sessions) >= self._max_sessions:
                log_event(
                    logger,
                    "session_limit_reached",
                    level=logging.WARNING,
                    session_id=session_id,
                    max_sessions=self._max_sessions,
                )
                raise SessionLimitError(
                    f"동시 세션 수 상한({self._max_sessions}) 초과"
                )
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            log_event(logger, "session_created", session_id=session_id)
        return session

    def counts(self) -> dict:
        sessions = list(self._sessions.values())
        return {
            "active_sessions": len(sessions),
            "webrtc_sessions": sum(1 for s in sessions if s.pc is not None),
            "ws_connections": sum(1 for s in sessions if s.ws is not None),
        }

    # ------------------------------------------------------------ 연결 등록

    async def attach_ws(self, session_id: str, ws: WebSocket) -> Session:
        session = self.get_or_create(session_id)
        old = session.ws
        session.ws = ws
        session.touch()
        if old is not None and old is not ws:
            with contextlib.suppress(Exception):
                await old.close(code=4000, reason="replaced by new connection")
            log_event(logger, "ws_replaced", session_id=session_id)
        return session

    async def detach_ws(self, session_id: str, ws: WebSocket) -> None:
        """WebSocket이 끊기면 세션 전체(WebRTC 포함)를 정리한다."""
        session = self._sessions.get(session_id)
        if session is not None and session.ws is ws:
            session.ws = None
            await self.close_session(session_id, reason="websocket_disconnected")

    # ------------------------------------------------------------ 정리

    async def close_session(self, session_id: str, *, reason: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None or session.closing:
            return
        session.closing = True

        for task in list(session.media_tasks) + list(session.analysis_tasks):
            task.cancel()
        session.media_tasks.clear()
        session.analysis_tasks.clear()

        pc, session.pc = session.pc, None
        if pc is not None:
            with contextlib.suppress(Exception):
                await pc.close()

        ws, session.ws = session.ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close(code=1000, reason=reason)

        session.data_channel = None
        session.frames.clear()
        log_event(
            logger,
            "session_closed",
            session_id=session_id,
            reason=reason,
            frames_received=session.frames.frames_received,
            lifetime_s=round(time.monotonic() - session.created_at, 1),
        )

    async def close_all(self, *, reason: str = "server_shutdown") -> None:
        for session_id in list(self._sessions):
            await self.close_session(session_id, reason=reason)

    # ------------------------------------------------------------ 유휴 세션 회수

    async def start(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reap_loop())

    async def stop(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None
        await self.close_all()

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reap_interval)
            try:
                await self._reap_idle_once()
            except Exception:
                log_event(logger, "reaper_error", level=logging.ERROR, exc_info=True)

    async def _reap_idle_once(self) -> None:
        idle_ids = [
            sid
            for sid, session in list(self._sessions.items())
            if session.idle_seconds > self._idle_timeout
        ]
        for sid in idle_ids:
            log_event(logger, "session_idle_timeout", session_id=sid)
            await self.close_session(sid, reason="idle_timeout")
