"""WS /ws/{session_id} 엔드포인트 통합 테스트 (mock 분석기)."""


def test_ws_connect_and_ping_pong(client):
    with client.websocket_connect("/ws/ws-test-1") as ws:
        connected = ws.receive_json()
        assert connected["type"] == "connected"
        assert connected["session_id"] == "ws-test-1"

        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_ws_analyze_trigger_returns_result(client):
    with client.websocket_connect("/ws/ws-test-2") as ws:
        assert ws.receive_json()["type"] == "connected"

        ws.send_json({"type": "analyze", "question": "립 봐줘잉", "request_id": "req-1"})
        started = ws.receive_json()
        assert started["type"] == "analysis_started"
        assert started["request_id"] == "req-1"

        result = ws.receive_json()
        assert result["type"] == "analysis_result"
        assert result["session_id"] == "ws-test-2"
        assert result["request_id"] == "req-1"
        assert result["status"] == "ok"
        assert result["region"] == "lips"
        assert result["verdict"] == "needs_fix"
        assert result["message"]
        assert result["detail"]["issues"]
        assert result["timestamp"]


def test_ws_analyze_without_request_id_generates_one(client):
    with client.websocket_connect("/ws/ws-test-3") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"type": "analyze", "question": "지금 어때?"})
        started = ws.receive_json()
        result = ws.receive_json()
        assert started["request_id"] == result["request_id"]
        assert result["region"] == "overall"
        assert result["verdict"] == "perfect"


def test_ws_invalid_json_gets_error_not_disconnect(client):
    with client.websocket_connect("/ws/ws-test-4") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_text("this is not json")
        error = ws.receive_json()
        assert error["type"] == "error"

        # 연결은 유지되어야 한다
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_ws_disconnect_cleans_up_session(client):
    with client.websocket_connect("/ws/ws-test-5") as ws:
        assert ws.receive_json()["type"] == "connected"
        assert client.get("/health").json()["active_sessions"] == 1
    assert client.get("/health").json()["active_sessions"] == 0
