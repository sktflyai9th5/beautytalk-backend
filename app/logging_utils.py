"""JSON 구조화 로깅.

사용 예:
    logger = logging.getLogger("beautytalk.webrtc")
    log_event(logger, "webrtc_offer_received", session_id=sid)
→ {"ts": "...", "level": "INFO", "logger": "beautytalk.webrtc",
   "event": "webrtc_offer_received", "session_id": "..."}
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    exc_info=None,
    **fields,
) -> None:
    logger.log(level, event, extra={"event": event, "fields": fields}, exc_info=exc_info)
