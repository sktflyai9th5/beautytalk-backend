# -*- coding: utf-8 -*-
"""경로 선택 규칙.

여기서 잘못 고르면 크롭도 프롬프트도 어댑터도 전부 어긋난다. 게다가 사용자는
"왜 엉뚱한 답이 나왔는지" 화면으로 확인할 수 없으므로 조용히 틀린다.
"""
import pytest

from beautytalk.router import LIP, MAKEUP, normalize, pick


@pytest.mark.parametrize("q", [
    "립 어때?",
    "립 봐줘",
    "입술 괜찮아?",
    "입술 화장 봐주라",
    "내 립스틱 잘 발렸어?",
    "틴트 번졌어?",
    "립밤 상태 어때",
    "입꼬리 쪽 지저분해?",
])
def test_립_단서는_립_경로(q):
    route, _ = pick(q)
    assert route == LIP


@pytest.mark.parametrize("q", [
    "잎술 어때",     # STT 오인식
    "입쑬 봐줘",
    "맆 봐주라",
])
def test_흔한_오인식도_립_경로(q):
    """온디바이스 STT 는 '립'을 자주 흘린다. 정확 일치만 보면 놓친다."""
    route, _ = pick(q)
    assert route == LIP


@pytest.mark.parametrize("q", [
    "지금 어때?",
    "눈썹 괜찮아?",
    "아이라인 번졌어?",
    "피부 톤 어때",
    "화장 전체적으로 봐줘",
    "볼 터치 진해?",
])
def test_그_외에는_메이크업_경로(q):
    route, _ = pick(q)
    assert route == MAKEUP


@pytest.mark.parametrize("q", ["립 말고 눈 봐줘", "입술말고 눈썹"])
def test_립_제외_표현은_메이크업_경로(q):
    """'립'이 들어 있어도 배제 표현이면 립이 아니다."""
    route, _ = pick(q)
    assert route == MAKEUP


def test_빈_문장은_기본값():
    route, reason = pick("")
    assert route == MAKEUP
    assert "기본값" in reason


def test_공백과_기호를_무시한다():
    assert normalize("립,  어때?") == "립어때"
    assert pick("립   어    때")[0] == LIP


def test_사유가_항상_붙는다():
    """지표로 쓰므로 비어 있으면 안 된다."""
    for q in ["립 어때", "지금 어때", "", "립 말고 눈"]:
        assert pick(q)[1]
