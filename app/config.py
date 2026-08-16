"""환경변수 기반 설정.

- QWEN_API_URL      : OpenAI 호환 API base URL (예: http://127.0.0.1:8001/v1)
- QWEN_API_KEY      : API 키 (vLLM 등 로컬 서버는 보통 불필요, 기본 "EMPTY")
- QWEN_MODEL        : 모델 이름 (기본 Qwen/Qwen3-VL-8B-Instruct)
- ANALYZER_MOCK     : "true"면 mock 분석기 사용. 미설정 시 QWEN_API_URL이 없으면
                      자동으로 mock 모드 (모델 없이도 서버가 뜨도록)
- SESSION_IDLE_TIMEOUT : 유휴 세션 정리 기준 초 (기본 300)
- SESSION_REAP_INTERVAL: 유휴 세션 검사 주기 초 (기본 30)
- ANALYSIS_TIMEOUT  : 분석 1회 제한 시간 초 (기본 60)
- MAX_SESSIONS      : 동시 세션 수 상한 (기본 50)
"""

import os
from dataclasses import dataclass

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    qwen_api_url: str
    qwen_api_key: str
    qwen_model: str
    analyzer_mock: bool
    analyzer_mock_auto: bool  # ANALYZER_MOCK 미설정 + URL 없음으로 자동 활성화된 경우
    session_idle_timeout: float
    session_reap_interval: float
    analysis_timeout: float
    max_sessions: int


def load_settings() -> Settings:
    qwen_api_url = os.environ.get("QWEN_API_URL", "").strip().rstrip("/")
    mock_raw = os.environ.get("ANALYZER_MOCK")
    analyzer_mock_auto = False
    if mock_raw is None:
        analyzer_mock = not qwen_api_url
        analyzer_mock_auto = analyzer_mock
    else:
        analyzer_mock = mock_raw.strip().lower() in _TRUTHY
    return Settings(
        qwen_api_url=qwen_api_url,
        qwen_api_key=os.environ.get("QWEN_API_KEY", "EMPTY"),
        qwen_model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        analyzer_mock=analyzer_mock,
        analyzer_mock_auto=analyzer_mock_auto,
        session_idle_timeout=float(os.environ.get("SESSION_IDLE_TIMEOUT", "300")),
        session_reap_interval=float(os.environ.get("SESSION_REAP_INTERVAL", "30")),
        analysis_timeout=float(os.environ.get("ANALYSIS_TIMEOUT", "60")),
        max_sessions=int(os.environ.get("MAX_SESSIONS", "50")),
    )
