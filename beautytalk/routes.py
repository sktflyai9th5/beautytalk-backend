# -*- coding: utf-8 -*-
"""경로 정의 — 전처리 · 프롬프트 · 어댑터를 한 묶음으로.

경로를 추가하려면 assets/<이름>/ 에 system_prompt.txt 를 두고 여기에 Route 를
하나 더 만들면 된다. few-shot 은 있으면 실리고 없으면 안 실린다.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from . import config, makeup_regions, router, vision

log = logging.getLogger("bt.routes")


@dataclass
class Shot:
    image: Image.Image
    user: str
    assistant: str


@dataclass
class Route:
    name: str
    # 사용자 사진 → 크롭. (Prepared, None) 또는 (None, PrepError)
    prepare: Callable[[Image.Image], tuple]
    system: str
    #  모델 출력이 그대로 읽을 문장이 아닐 때 (예: 좌표 JSON) 변환한다.
    #  (원문, Prepared) → (읽어 줄 문장, 지표, 화면용 항목). 문장이 비면 실패.
    postprocess: Callable[[str, object], tuple[str, str]] | None = None
    #  True 면 사용자 발화를 모델에 넣지 않고 default_question 을 쓴다.
    fixed_question: bool = False
    #  경로별 생성 길이 상한. 0 이면 전역값. 좌표 JSON 은 짧아서 길게 잡을 이유가 없다.
    max_new_tokens: int = 0
    #  JSON 배열이 닫히면 생성을 끊는다 (좌표 경로).
    stop_on_json: bool = False
    #  LoRA 어댑터 이름. 빈 문자열이면 베이스 그대로 쓴다.
    adapter: str = ""
    # 사용자가 아무 말 없이 촬영만 했을 때 넣을 질문
    default_question: str = "지금 어때?"
    shots: list[Shot] = field(default_factory=list)

    def build_messages(self, user_image: Image.Image, user_text: str):
        """(messages, images). images 순서 = messages 안 <image> 등장 순서.

        순서가 어긋나면 모델이 엉뚱한 이미지를 보고 답한다.
        """
        messages = [{"role": "system", "content": [{"type": "text", "text": self.system}]}]
        images: list[Image.Image] = []
        for s in self.shots:
            messages += [
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": s.user}]},
                {"role": "assistant", "content": [{"type": "text", "text": s.assistant}]},
            ]
            images.append(s.image)
        messages.append({"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": user_text}]})
        images.append(user_image)
        return messages, images


def _load_shots(base: Path) -> list[Shot]:
    """assets/<경로>/fewshot/fewshot.json 이 있으면 읽는다. 순서 유지."""
    meta = base / "fewshot" / "fewshot.json"
    if not meta.exists():
        return []
    shots = []
    for s in json.loads(meta.read_text(encoding="utf-8")):
        img = vision.fit(Image.open(base / s["image"]), config.FEWSHOT_MAX_SIDE)
        shots.append(Shot(img, s["user"], s["assistant"]))
    return shots


def _system(base: Path) -> str:
    return (base / "system_prompt.txt").read_text(encoding="utf-8")


def _makeup_answer(raw: str, prep) -> tuple:
    """좌표 JSON → 부위 이름 → 음성 문장. 학습 정답과 같은 규칙으로 이름을 붙인다."""
    pred = makeup_regions.parse_coord_answer(raw)
    if prep.landmarks is None:
        # 얼굴 박스는 잡혔는데 468점 랜드마크가 안 나온 경우(옆얼굴·가림·저해상도).
        # 부위 이름을 붙일 수 없으니 좌표를 읽어 줄 수는 없다. 폴백 문구보다
        # 다시 찍어 달라는 안내가 사용자에게 실제로 도움이 된다.
        return ("얼굴이 또렷하게 잡히지 않았어요. 정면을 보고 조금 더 밝은 곳에서 다시 찍어 주세요.",
                "no_landmarks", [])
    return makeup_regions.answer_from_boxes(
        pred, prep.landmarks, prep.face_box, prep.rect,
        on_face=lambda b: vision.on_face(b, prep.hull),
        img_size=(prep.img_w, prep.img_h),
    )


_routes: dict[str, Route] = {}


def load() -> dict[str, Route]:
    """부팅 시 1회. 프롬프트와 few-shot 이미지를 메모리에 올린다."""
    if _routes:
        return _routes

    lip_dir = config.ASSETS / "lip"
    _routes[router.LIP] = Route(
        name=router.LIP,
        prepare=vision.prepare_lip,
        system=_system(lip_dir),
        adapter=config.LIP_ADAPTER,  # 기본은 없음 — 프롬프트 + few-shot 만으로 검증됐다
        default_question="립 봐주라.",
        shots=_load_shots(lip_dir),
    )

    makeup_dir = config.ASSETS / "makeup"
    _routes[router.MAKEUP] = Route(
        name=router.MAKEUP,
        prepare=vision.prepare_face,
        system=_system(makeup_dir),
        adapter=config.MAKEUP_ADAPTER,
        # 학습 때 쓴 질문 문구 그대로 (COORD_QUESTION). 사용자가 뭐라고 묻든
        # 모델에는 이걸 넣는다 — 좌표 검출기라 질문 표현이 달라질 이유가 없다.
        default_question="이 얼굴에서 베이스가 고르지 않은 부분을 찾아줘.",
        fixed_question=True,
        max_new_tokens=128,   # 박스 최대 6개짜리 JSON 배열이면 충분하다
        stop_on_json=True,
        shots=_load_shots(makeup_dir),
        # 이 모델은 문장이 아니라 좌표를 낸다. 숫자를 읽어 줄 수는 없다.
        postprocess=_makeup_answer,
    )

    for r in _routes.values():
        log.info(
            "경로 %-6s | 어댑터=%-45s | 시스템 %d자 | few-shot %d장 | 후처리 %s",
            r.name, r.adapter or "(없음, 베이스)", len(r.system), len(r.shots),
            "좌표→한국어" if r.postprocess else "없음(원문)",
        )
    return _routes


def get(name: str) -> Route:
    return load()[name]
