# -*- coding: utf-8 -*-
"""TTS 안전 후처리.

모델 응답은 원문 그대로 읽어 주는 게 원칙이다(요약·재작성 금지). 다만 마크다운
기호나 괄호가 섞여 들어오면 TTS 가 그걸 그대로 읽어 버리므로 값싼 필터 하나만 둔다.
"""
from __future__ import annotations

import re

FALLBACK = "지금은 확인이 어려워요. 잠시 뒤에 다시 찍어 주세요."

_STRIP = re.compile(r"[*_#`\[\](){}<>|~]")
_SPACES = re.compile(r"\s+")


def sanitize(text: str | None) -> str:
    t = _STRIP.sub("", text or "")
    t = _SPACES.sub(" ", t).strip()
    return t or FALLBACK


# ── 응답에서 동작 뽑기 (로깅·지표 전용) ────────────────────────────
# 사용자에게 나가는 문구를 만드는 데는 절대 쓰지 않는다.
_IMPER = r"주세요|세요|십시오|하시면|해 보시|바르시|눌러서"
_WIPE = r"닦|지워|지우"
_BLEND = r"펴|덧발|더 발라|한 겹 더|밀어|문질|두드려|마주 눌러|얹|채워"


def action_of(text: str) -> str:
    """ok | wipe | blend. 검증 때와 분포가 크게 다르면 뭔가 어긋난 신호다."""
    sents = re.split(r"(?<=[.!?])\s*", text or "")
    ins = " ".join(s for s in sents if re.search(_IMPER, s))
    if re.search(_WIPE, ins):
        return "wipe"
    if re.search(_BLEND, ins):
        return "blend"
    if not ins:
        if re.search(r"닦아내면|닦으면 됩|지우면 됩", text or ""):
            return "wipe"
        if re.search(r"덧발|더 발라|한 겹 더", text or ""):
            return "blend"
    return "ok"
