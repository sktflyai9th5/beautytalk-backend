import pytest
from pydantic import ValidationError

from app.schemas import (
    AnalysisDetail,
    AnalysisPayload,
    AnalysisResult,
    Issue,
    OfferRequest,
)


def test_offer_request_requires_offer_type():
    OfferRequest(session_id="s1", sdp="v=0", type="offer")
    with pytest.raises(ValidationError):
        OfferRequest(session_id="s1", sdp="v=0", type="answer")
    with pytest.raises(ValidationError):
        OfferRequest(session_id="", sdp="v=0", type="offer")


def test_analysis_result_matches_spec_shape():
    result = AnalysisResult(
        session_id="s1",
        request_id="r1",
        status="ok",
        region="lips",
        verdict="needs_fix",
        message="입술 라인이 번졌어요.",
        detail=AnalysisDetail(
            issues=[Issue(area="입술", problem="번짐", suggestion="닦아내기")]
        ),
    )
    data = result.model_dump()
    assert set(data.keys()) == {
        "type",
        "session_id",
        "request_id",
        "status",
        "region",
        "verdict",
        "message",
        "detail",
        "timestamp",
    }
    assert data["type"] == "analysis_result"
    assert data["detail"]["issues"][0] == {
        "area": "입술",
        "problem": "번짐",
        "suggestion": "닦아내기",
    }
    assert "T" in data["timestamp"]  # ISO8601


def test_analysis_result_rejects_bad_enum_values():
    with pytest.raises(ValidationError):
        AnalysisResult(
            session_id="s1",
            request_id="r1",
            status="weird",
            message="x",
        )
    with pytest.raises(ValidationError):
        AnalysisResult(
            session_id="s1",
            request_id="r1",
            status="ok",
            region="nose",
            message="x",
        )


def test_analysis_result_from_payload():
    payload = AnalysisPayload(
        status="ok", region="cheeks", verdict="bad_color", message="볼 색이 진해요."
    )
    result = AnalysisResult.from_payload(payload, session_id="s1", request_id="r9")
    assert result.session_id == "s1"
    assert result.request_id == "r9"
    assert result.region == "cheeks"
    assert result.verdict == "bad_color"
    assert result.message == "볼 색이 진해요."


def test_analysis_detail_defaults_are_independent():
    a = AnalysisResult(session_id="a", request_id="1", status="ok", message="x")
    b = AnalysisResult(session_id="b", request_id="2", status="ok", message="y")
    a.detail.issues.append(Issue(area="1", problem="2", suggestion="3"))
    assert b.detail.issues == []
