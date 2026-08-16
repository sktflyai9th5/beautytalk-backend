import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ANALYZER_MOCK", "true")
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT", "60")
    monkeypatch.setenv("SESSION_REAP_INTERVAL", "5")
    with TestClient(main.app) as test_client:
        yield test_client
