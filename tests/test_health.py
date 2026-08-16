"""기존 엔드포인트가 깨지지 않았는지 확인하는 기준선 테스트."""


def test_root_keeps_original_contract(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "beautytalk-backend"
    assert body["status"] == "ok"


def test_health_reports_status_and_sessions(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["analyzer"] == "mock"
    assert body["active_sessions"] == 0
    assert body["webrtc_sessions"] == 0
    assert body["ws_connections"] == 0


def test_health_counts_ws_sessions(client):
    with client.websocket_connect("/ws/health-count-test"):
        body = client.get("/health").json()
        assert body["active_sessions"] == 1
        assert body["ws_connections"] == 1
        assert body["webrtc_sessions"] == 0
