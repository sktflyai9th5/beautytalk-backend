# -*- coding: utf-8 -*-
"""TTS 로 나갈 문장 다듬기."""
from beautytalk.sanitize import FALLBACK, action_of, sanitize


def test_마크다운_기호를_없앤다():
    """TTS 는 별표를 '별표'라고 읽는다."""
    assert sanitize("**오른쪽** 입꼬리가 번졌어요") == "오른쪽 입꼬리가 번졌어요"
    assert "[" not in sanitize("[확인] 괜찮아요")


def test_빈_응답은_폴백():
    """조용히 끊기면 사용자는 아무 정보도 못 받는다."""
    assert sanitize("") == FALLBACK
    assert sanitize(None) == FALLBACK
    assert sanitize("   ") == FALLBACK
    assert sanitize("***") == FALLBACK


def test_정상_문장은_그대로():
    t = "전체적으로 고르게 발렸습니다. 더 손보지 않아도 됩니다."
    assert sanitize(t) == t


def test_줄바꿈은_공백으로():
    assert sanitize("첫 문장.\n둘째 문장.") == "첫 문장. 둘째 문장."


def test_동작_분류():
    assert action_of("입술 바깥으로 번졌어요. 면봉으로 닦아 주세요.") == "wipe"
    assert action_of("색이 옅어요. 조금 더 덧발라 펴 주세요.") == "blend"
    assert action_of("전체적으로 고르게 발렸습니다. 더 손보지 않아도 됩니다.") == "ok"
