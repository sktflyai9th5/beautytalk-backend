import pytest

from app.analyzer import (
    AnalyzerError,
    MockAnalyzer,
    guess_region,
    parse_model_output,
)


@pytest.mark.parametrize(
    ("question", "region"),
    [
        ("립 봐줘잉", "lips"),
        ("입술 어때?", "lips"),
        ("눈 화장 봐줘", "eyes"),
        ("아이라인 어때", "eyes"),
        ("볼 어때?", "cheeks"),
        ("블러셔 봐줘", "cheeks"),
        ("지금 어때?", "overall"),
    ],
)
def test_guess_region(question, region):
    assert guess_region(question) == region


async def test_mock_analyzer_returns_region_specific_payload():
    analyzer = MockAnalyzer()
    payload = await analyzer.analyze(None, "립 봐줘잉")
    assert payload.status == "ok"
    assert payload.region == "lips"
    assert payload.verdict == "needs_fix"
    assert payload.message
    assert payload.detail.issues


async def test_mock_analyzer_overall_is_perfect():
    analyzer = MockAnalyzer()
    payload = await analyzer.analyze(b"fake-jpeg", "지금 어때?")
    assert payload.region == "overall"
    assert payload.verdict == "perfect"


async def test_mock_analyzer_payloads_are_isolated():
    analyzer = MockAnalyzer()
    first = await analyzer.analyze(None, "립 봐줘잉")
    first.detail.issues.clear()
    second = await analyzer.analyze(None, "립 봐줘잉")
    assert second.detail.issues


def test_parse_model_output_plain_json():
    text = (
        '{"status": "ok", "region": "lips", "verdict": "needs_fix",'
        ' "message": "입술 라인이 번졌어요.",'
        ' "issues": [{"area": "입술", "problem": "번짐", "suggestion": "닦기"}]}'
    )
    payload = parse_model_output(text, "립 봐줘잉")
    assert payload.region == "lips"
    assert payload.verdict == "needs_fix"
    assert payload.detail.issues[0].area == "입술"


def test_parse_model_output_with_code_fence_and_think():
    text = (
        "<think>사용자의 입술을 보자...</think>\n"
        "```json\n"
        '{"status": "ok", "region": "lips", "verdict": "perfect", "message": "완벽해요."}\n'
        "```"
    )
    payload = parse_model_output(text, "립 봐줘잉")
    assert payload.verdict == "perfect"
    assert payload.message == "완벽해요."


def test_parse_model_output_invalid_region_falls_back_to_question():
    text = '{"status": "ok", "region": "nose", "verdict": "needs_fix", "message": "m"}'
    payload = parse_model_output(text, "볼 어때?")
    assert payload.region == "cheeks"


def test_parse_model_output_invalid_status_defaults_ok():
    text = '{"status": "great", "region": "lips", "verdict": "needs_fix", "message": "m"}'
    payload = parse_model_output(text, "립")
    assert payload.status == "ok"


def test_parse_model_output_no_face_without_verdict():
    text = '{"status": "no_face", "message": "얼굴이 보이지 않아요."}'
    payload = parse_model_output(text, "지금 어때?")
    assert payload.status == "no_face"
    assert payload.verdict is None


def test_parse_model_output_missing_message_raises():
    with pytest.raises(AnalyzerError):
        parse_model_output('{"status": "ok", "region": "lips"}', "립")


def test_parse_model_output_garbage_raises():
    with pytest.raises(AnalyzerError):
        parse_model_output("죄송하지만 분석할 수 없어요.", "립")


def test_parse_model_output_json_with_trailing_text():
    text = '{"status": "ok", "region": "eyes", "verdict": "needs_fix", "message": "m"} 이상입니다.'
    payload = parse_model_output(text, "눈")
    assert payload.region == "eyes"
