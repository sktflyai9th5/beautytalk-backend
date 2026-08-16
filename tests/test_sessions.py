import asyncio
import logging

import pytest

from app.sessions import (
    ConnectionManager,
    LatestFrameBuffer,
    SessionLimitError,
    spawn_tracked,
)


class FakeWebSocket:
    def __init__(self):
        self.closed = False
        self.close_code = None
        self.sent = []

    async def close(self, code=1000, reason=""):
        self.closed = True
        self.close_code = code

    async def send_text(self, text):
        self.sent.append(text)


class FakePeerConnection:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def test_latest_frame_buffer_keeps_only_one_frame():
    buffer = LatestFrameBuffer()
    assert buffer.latest() is None
    buffer.put("frame1")
    buffer.put("frame2")
    assert buffer.latest() == "frame2"
    assert buffer.frames_received == 2
    buffer.clear()
    assert buffer.latest() is None
    assert buffer.frames_received == 2


async def test_get_or_create_is_idempotent():
    manager = ConnectionManager()
    first = manager.get_or_create("s1")
    second = manager.get_or_create("s1")
    assert first is second
    assert manager.counts()["active_sessions"] == 1


async def test_attach_ws_replaces_old_connection():
    manager = ConnectionManager()
    old_ws = FakeWebSocket()
    new_ws = FakeWebSocket()
    await manager.attach_ws("s1", old_ws)
    session = await manager.attach_ws("s1", new_ws)
    assert session.ws is new_ws
    assert old_ws.closed
    assert old_ws.close_code == 4000
    # 교체된 옛 소켓의 detach는 세션을 죽이면 안 된다
    await manager.detach_ws("s1", old_ws)
    assert manager.get("s1") is not None


async def test_detach_ws_tears_down_whole_session():
    manager = ConnectionManager()
    ws = FakeWebSocket()
    pc = FakePeerConnection()
    session = await manager.attach_ws("s1", ws)
    session.pc = pc
    await manager.detach_ws("s1", ws)
    assert manager.get("s1") is None
    assert pc.closed


async def test_close_session_closes_ws_pc_and_tasks():
    manager = ConnectionManager()
    ws = FakeWebSocket()
    pc = FakePeerConnection()
    session = await manager.attach_ws("s1", ws)
    session.pc = pc
    session.frames.put("frame")

    async def forever():
        await asyncio.sleep(3600)

    media_task = asyncio.create_task(forever())
    analysis_task = asyncio.create_task(forever())
    session.media_tasks.add(media_task)
    session.analysis_tasks.add(analysis_task)

    await manager.close_session("s1", reason="test")
    await asyncio.sleep(0)
    assert manager.get("s1") is None
    assert session.closing
    assert ws.closed
    assert pc.closed
    assert media_task.cancelled()
    assert analysis_task.cancelled()
    assert session.frames.latest() is None
    # 중복 호출해도 안전
    await manager.close_session("s1", reason="test")


async def test_session_limit_rejects_new_but_allows_existing():
    manager = ConnectionManager(max_sessions=1)
    first = manager.get_or_create("s1")
    with pytest.raises(SessionLimitError):
        manager.get_or_create("s2")
    assert manager.get_or_create("s1") is first
    await manager.close_session("s1", reason="test")
    manager.get_or_create("s2")  # 자리가 나면 다시 허용


async def test_spawn_tracked_keeps_reference_and_logs_crash(caplog):
    task_set: set = set()
    started = asyncio.Event()

    async def boom():
        started.set()
        raise RuntimeError("boom")

    logger = logging.getLogger("beautytalk.test")
    task = spawn_tracked(
        task_set, boom(), task_logger=logger, event="task_crashed", session_id="s1"
    )
    assert task in task_set
    await started.wait()
    with caplog.at_level(logging.ERROR):
        await asyncio.sleep(0.05)
    assert task_set == set()
    assert any(
        getattr(r, "event", "") == "task_crashed" for r in caplog.records
    )


async def test_idle_reaper_closes_stale_sessions():
    manager = ConnectionManager(idle_timeout=0.05, reap_interval=0.02)
    ws = FakeWebSocket()
    await manager.attach_ws("stale", ws)
    await manager.start()
    try:
        await asyncio.sleep(0.3)
        assert manager.get("stale") is None
        assert ws.closed
    finally:
        await manager.stop()


async def test_active_session_survives_reaper():
    manager = ConnectionManager(idle_timeout=0.2, reap_interval=0.02)
    session = manager.get_or_create("busy")
    await manager.start()
    try:
        for _ in range(6):
            session.touch()
            await asyncio.sleep(0.05)
        assert manager.get("busy") is not None
    finally:
        await manager.stop()


async def test_stop_closes_all_sessions():
    manager = ConnectionManager()
    manager.get_or_create("a")
    manager.get_or_create("b")
    await manager.start()
    await manager.stop()
    assert manager.counts()["active_sessions"] == 0
