import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.analyzer import get_analyzer
from app.config import load_settings
from app.logging_utils import log_event, setup_logging
from app.sessions import ConnectionManager
from app.webrtc import router as webrtc_router
from app.websocket import router as websocket_router

setup_logging()
logger = logging.getLogger("beautytalk.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.manager = ConnectionManager(
        idle_timeout=settings.session_idle_timeout,
        reap_interval=settings.session_reap_interval,
        max_sessions=settings.max_sessions,
    )
    app.state.analyzer = get_analyzer(settings)
    await app.state.manager.start()
    if settings.analyzer_mock_auto:
        log_event(
            logger,
            "analyzer_mock_auto_enabled",
            level=logging.WARNING,
            hint="QWEN_API_URL이 없어 mock 분석기로 기동. 실서비스는 QWEN_API_URL 설정 필요",
        )
    log_event(
        logger,
        "startup",
        analyzer=app.state.analyzer.name,
        qwen_api_url=settings.qwen_api_url or None,
        session_idle_timeout=settings.session_idle_timeout,
        max_sessions=settings.max_sessions,
    )
    yield
    await app.state.manager.stop()
    await app.state.analyzer.aclose()
    log_event(logger, "shutdown")


app = FastAPI(title="beautytalk-backend", lifespan=lifespan)
app.include_router(webrtc_router)
app.include_router(websocket_router)


@app.get("/")
def root():
    return {"service": "beautytalk-backend", "status": "ok"}


@app.get("/health")
def health():
    manager = getattr(app.state, "manager", None)
    analyzer = getattr(app.state, "analyzer", None)
    counts = manager.counts() if manager else {
        "active_sessions": 0,
        "webrtc_sessions": 0,
        "ws_connections": 0,
    }
    return {
        "status": "healthy",
        "analyzer": analyzer.name if analyzer else None,
        **counts,
    }
