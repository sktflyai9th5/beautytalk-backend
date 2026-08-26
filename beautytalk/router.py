# -*- coding: utf-8 -*-
"""STT 문장 → 처리 경로 선택.

경로가 갈리면 전처리(크롭)도 모델(어댑터)도 프롬프트도 전부 달라지므로,
여기서 한 번 잘못 고르면 그 뒤가 통째로 어긋난다. 그래서 규칙을 한곳에 모았다.

  "립 어때?"      → lip     (입술 크롭 + 립 프롬프트 + few-shot 3장)
  "눈썹 괜찮아?"  → makeup  (얼굴 크롭 + 메이크업 어댑터)
  "지금 어때?"    → makeup  (기본값)

STT 오인식을 견뎌야 한다. 온디바이스 인식은 "립"을 "입", "잎", "맆" 으로 흘리고
"입술"을 "입쑬"로 적기도 한다. 그래서 정확 일치가 아니라 부분 문자열 + 흔한
오인식 변형을 함께 본다.
"""
from __future__ import annotations

import re
import unicodedata

LIP = "lip"
MAKEUP = "makeup"

# 립 경로로 보낼 단서. 공백 제거한 문장에서 부분 문자열로 찾는다.
LIP_KEYWORDS = (
    "립", "입술", "입슬", "입쑬", "잎술", "리브", "맆",
    "틴트", "립스틱", "립글", "립밤", "립라인", "입꼬리", "인중",
)

# 립처럼 보이지만 립이 아닌 것 — 먼저 걸러낸다.
# "립 말고 눈", "입술 말고" 처럼 사용자가 명시적으로 배제하는 경우.
LIP_NEGATIONS = ("립말고", "입술말고", "립빼고", "입술빼고", "립말구", "입술말구")

# 메이크업 경로임을 분명히 하는 단서 (지표·로그용. 기본값이 makeup 이라 필수는 아니다)
MAKEUP_KEYWORDS = (
    "눈썹", "눈", "아이", "섀도", "쉐도", "마스카라", "아이라인", "라이너",
    "피부", "베이스", "파운데", "쿠션", "볼", "블러", "치크", "코", "턱",
    "전체", "화장", "메이크업",
)


def normalize(text: str) -> str:
    """공백·기호 제거 + 자모 결합. 매칭 전에 항상 통과시킨다."""
    t = unicodedata.normalize("NFC", text or "")
    return re.sub(r"[\s\W_]+", "", t)


def pick(text: str) -> tuple[str, str]:
    """(경로, 사유) 를 돌려준다. 사유는 로그·지표용."""
    t = normalize(text)
    if not t:
        return MAKEUP, "빈 문장 → 기본값"

    for neg in LIP_NEGATIONS:
        if neg in t:
            return MAKEUP, f"립 제외 표현({neg})"

    for kw in LIP_KEYWORDS:
        if kw in t:
            return LIP, f"립 단서({kw})"

    for kw in MAKEUP_KEYWORDS:
        if kw in t:
            return MAKEUP, f"메이크업 단서({kw})"

    return MAKEUP, "단서 없음 → 기본값"
