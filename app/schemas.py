"""API / WebSocket 메시지 스키마.

서버 → 앱 (WebSocket 또는 data channel):
  connected / pong / analysis_started / analysis_result / error
앱 → 서버 (WebSocket 또는 data channel, JSON 텍스트):
  {"type": "ping"}
  {"type": "analyze", "question": "립 봐줘잉", "request_id": "선택"}
"""

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

Region = Literal["lips", "eyes", "cheeks", "overall"]
Verdict = Literal["perfect", "needs_fix", "bad_color"]
AnalysisStatus = Literal["ok", "no_face", "error"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------- WebRTC 시그널링


class OfferRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    sdp: str
    type: Literal["offer"]


class OfferResponse(BaseModel):
    session_id: str
    sdp: str
    type: str


# ---------------------------------------------------------------- 분석 결과


class Issue(BaseModel):
    area: str
    problem: str
    suggestion: str


class AnalysisDetail(BaseModel):
    issues: list[Issue] = Field(default_factory=list)


class AnalysisPayload(BaseModel):
    """분석기(Qwen/mock)가 돌려주는 순수 분석 결과."""

    status: AnalysisStatus = "ok"
    region: Region = "overall"
    verdict: Optional[Verdict] = None
    message: str
    detail: AnalysisDetail = Field(default_factory=AnalysisDetail)


class AnalysisResult(BaseModel):
    """WebSocket으로 앱에 push되는 최종 결과. message는 앱이 그대로 TTS로 읽는다."""

    type: Literal["analysis_result"] = "analysis_result"
    session_id: str
    request_id: str
    status: AnalysisStatus
    region: Region = "overall"
    verdict: Optional[Verdict] = None
    message: str
    detail: AnalysisDetail = Field(default_factory=AnalysisDetail)
    timestamp: str = Field(default_factory=utc_now_iso)

    @classmethod
    def from_payload(
        cls, payload: AnalysisPayload, *, session_id: str, request_id: str
    ) -> "AnalysisResult":
        return cls(
            session_id=session_id,
            request_id=request_id,
            status=payload.status,
            region=payload.region,
            verdict=payload.verdict,
            message=payload.message,
            detail=payload.detail,
        )


# ---------------------------------------------------------------- 제어 메시지


class ConnectedMessage(BaseModel):
    type: Literal["connected"] = "connected"
    session_id: str
    timestamp: str = Field(default_factory=utc_now_iso)


class PongMessage(BaseModel):
    type: Literal["pong"] = "pong"
    timestamp: str = Field(default_factory=utc_now_iso)


class AnalysisStartedMessage(BaseModel):
    type: Literal["analysis_started"] = "analysis_started"
    session_id: str
    request_id: str
    timestamp: str = Field(default_factory=utc_now_iso)


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
    timestamp: str = Field(default_factory=utc_now_iso)
