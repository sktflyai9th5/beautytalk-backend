"""mock 분석 파이프라인(run_analysis / handle_client_message) 테스트."""

import asyncio
import json
from types import SimpleNamespace

from app.analysis import handle_client_message, run_analysis
from app.analyzer import AnalyzerError, MockAnalyzer, NoFrameError
from app.config import Settings
from app.sessions import Session


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def close(self, code=1000, reason=""):
        pass


def make_settings(**overrides):
    values = dict(
        qwen_api_url="",
        qwen_api_key="EMPTY",
        qwen_model="test",
        analyzer_mock=True,
        analyzer_mock_auto=False,
        session_idle_timeout=300,
        session_reap_interval=30,
        analysis_timeout=5,
        max_sessions=50,
        debug_save_frames=False,
        qwen_no_think=True,
    )
    values.update(overrides)
    return Settings(**values)


def make_state(analyzer=None):
    return SimpleNamespace(
        settings=make_settings(),
        analyzer=analyzer or MockAnalyzer(),
    )


class FailingAnalyzer:
    name = "failing"

    async def analyze(self, image_jpeg, question):
        raise AnalyzerError("boom")


class NoFrameAnalyzer:
    name = "noframe"

    async def analyze(self, image_jpeg, question):
        raise NoFrameError("no frame")


class SlowAnalyzer:
    name = "slow"

    async def analyze(self, image_jpeg, question):
        await asyncio.sleep(60)


async def test_run_analysis_sends_ok_result_over_ws():
    state = make_state()
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    await run_analysis(state, session, question="립 봐줘잉", request_id="r1")
    [result] = session.ws.sent
    assert result["type"] == "analysis_result"
    assert result["session_id"] == "s1"
    assert result["request_id"] == "r1"
    assert result["status"] == "ok"
    assert result["region"] == "lips"
    assert result["message"]
    assert result["detail"]["issues"]
    assert result["timestamp"]


async def test_run_analysis_analyzer_error_sends_error_result():
    state = make_state(FailingAnalyzer())
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    await run_analysis(state, session, question="립", request_id="r1")
    [result] = session.ws.sent
    assert result["status"] == "error"
    assert result["message"]


async def test_run_analysis_no_frame_error_result():
    state = make_state(NoFrameAnalyzer())
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    await run_analysis(state, session, question="립", request_id="r1")
    [result] = session.ws.sent
    assert result["status"] == "error"
    assert "카메라" in result["message"]


async def test_run_analysis_timeout_sends_error_result():
    state = make_state(SlowAnalyzer())
    state.settings = make_settings(analysis_timeout=0.05)
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    await run_analysis(state, session, question="립", request_id="r1")
    [result] = session.ws.sent
    assert result["status"] == "error"


async def test_handle_client_message_ping_pong():
    state = make_state()
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    await handle_client_message(state, session, '{"type": "ping"}', source="websocket")
    [pong] = session.ws.sent
    assert pong["type"] == "pong"


async def test_handle_client_message_analyze_full_flow():
    state = make_state()
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    await handle_client_message(
        state,
        session,
        '{"type": "analyze", "question": "볼 어때?", "request_id": "req-7"}',
        source="datachannel",
    )
    for _ in range(100):
        if len(session.ws.sent) >= 2:
            break
        await asyncio.sleep(0.02)
    started, result = session.ws.sent
    assert started["type"] == "analysis_started"
    assert started["request_id"] == "req-7"
    assert result["type"] == "analysis_result"
    assert result["request_id"] == "req-7"
    assert result["region"] == "cheeks"


async def test_handle_client_message_invalid_json():
    state = make_state()
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    await handle_client_message(state, session, "not json {", source="websocket")
    [error] = session.ws.sent
    assert error["type"] == "error"


async def test_handle_client_message_unknown_type():
    state = make_state()
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    await handle_client_message(state, session, '{"type": "dance"}', source="websocket")
    [error] = session.ws.sent
    assert error["type"] == "error"
    assert "dance" in error["message"]


async def test_analyze_rejected_while_previous_analysis_running():
    gate = asyncio.Event()

    class BlockingAnalyzer:
        name = "blocking"

        async def analyze(self, image_jpeg, question):
            await gate.wait()
            return await MockAnalyzer().analyze(image_jpeg, question)

    state = make_state(BlockingAnalyzer())
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()

    await handle_client_message(
        state, session, '{"type": "analyze", "request_id": "first"}', source="websocket"
    )
    await asyncio.sleep(0.05)  # 첫 분석이 락을 잡을 시간
    await handle_client_message(
        state, session, '{"type": "analyze", "request_id": "second"}', source="websocket"
    )
    gate.set()
    for _ in range(100):
        if len(session.ws.sent) >= 3:
            break
        await asyncio.sleep(0.02)

    by_request = {m.get("request_id"): m for m in session.ws.sent if m.get("type") == "analysis_result"}
    assert by_request["second"]["status"] == "error"
    assert "진행 중" in by_request["second"]["message"]
    assert by_request["first"]["status"] == "ok"


async def test_handle_client_message_ignored_when_session_closing():
    state = make_state()
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    session.closing = True
    await handle_client_message(state, session, '{"type": "ping"}', source="websocket")
    assert session.ws.sent == []
    assert session.analysis_tasks == set()


async def test_debug_save_frames_writes_jpeg(tmp_path, monkeypatch):
    import numpy as np
    from av import VideoFrame

    from app import analysis as analysis_module

    monkeypatch.setattr(analysis_module, "DEBUG_FRAMES_DIR", tmp_path / "debug_frames")
    state = make_state()
    state.settings = make_settings(debug_save_frames=True)
    session = Session(session_id="s1")
    session.ws = FakeWebSocket()
    frame = VideoFrame.from_ndarray(
        np.zeros((48, 64, 3), dtype=np.uint8), format="bgr24"
    )
    session.frames.put(frame)
    await run_analysis(state, session, question="립", request_id="rframe")
    saved = list((tmp_path / "debug_frames").glob("*_s1_rframe.jpg"))
    assert len(saved) == 1
    assert saved[0].stat().st_size > 100
    [result] = session.ws.sent
    assert result["status"] == "ok"


async def test_analyze_without_ws_falls_back_to_datachannel():
    state = make_state()
    session = Session(session_id="s1")

    class FakeChannel:
        readyState = "open"

        def __init__(self):
            self.sent = []

        def send(self, text):
            self.sent.append(json.loads(text))

    channel = FakeChannel()
    session.data_channel = channel
    await run_analysis(state, session, question="립", request_id="r1")
    [result] = channel.sent
    assert result["type"] == "analysis_result"
    assert result["status"] == "ok"
