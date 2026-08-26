# -*- coding: utf-8 -*-
"""환경변수 설정. 전부 기본값이 있어서 아무것도 안 줘도 뜬다."""
from __future__ import annotations

import os
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# ── 모델 ──────────────────────────────────────────────────────────
# 두 경로가 **같은 베이스**를 쓴다. 메이크업 경로의 LoRA 어댑터가 이 체크포인트
# 위에서 학습됐고(adapter_config.json 의 base_model_name_or_path), 립 경로도
# 인계 문서가 이 4bit 체크포인트로 검증했다. 그래서 한 번만 올리고 공유한다.
#
# ★ 4B 로 내리지 말 것.
#   인계 문서 config.json 에 "4B 도 few-shot 시 동급 성능" 이라고 적혀 있고
#   추론도 대략 절반으로 줄지만, 립 프롬프트와 few-shot 3장은 **8B 기준으로
#   작성·검증된 세트**다. 모델을 바꾸면 그 검증이 통째로 무효가 되고, 앞이
#   보이지 않는 사용자는 답이 나빠진 걸 스스로 확인할 방법이 없다.
#   (메이크업 어댑터는 애초에 8B 위에서 학습돼 바꿀 수도 없다)
BASE_MODEL = os.environ.get(
    "BT_BASE_MODEL", "unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit"
)
MAKEUP_ADAPTER = os.environ.get(
    "BT_MAKEUP_ADAPTER", "iouty/qwen3vl-makeup-qwen3vl_augmented"
)
# 립 경로는 파인튜닝 없음(프롬프트 + few-shot 만). 어댑터를 붙이고 싶으면 여기에.
LIP_ADAPTER = os.environ.get("BT_LIP_ADAPTER", "").strip()

# unsloth 로더 사용 여부. 이 체크포인트는 unsloth 사전 양자화본이라 순정
# transformers 로는 생성이 안 된다 (비전 타워 4bit 양자화 상태가 안 붙는다).
USE_UNSLOTH = _b("BT_USE_UNSLOTH", True)

MAX_NEW_TOKENS = _i("BT_MAX_NEW_TOKENS", 200)
TOKENIZER_MAX_LEN = _i("BT_TOKENIZER_MAX_LEN", 16384)  # few-shot 3턴이 들어간다
# 정상 추론은 20~35초다. 60초를 넘기면 GPU 가 저전력 상태이거나 뭔가 잘못된 것이라
# 계속 붙잡느니 재촬영을 안내하는 편이 빠르다. 앱은 이보다 10초 더 기다린다.
GENERATE_TIMEOUT = _f("BT_GENERATE_TIMEOUT", 60.0)

# ── 이미지 ────────────────────────────────────────────────────────
QUERY_MAX_SIDE = _i("BT_QUERY_MAX_SIDE", 640)
FEWSHOT_MAX_SIDE = _i("BT_FEWSHOT_MAX_SIDE", 448)

# 전면 카메라 미러링 되돌리기.
# EXIF 로는 잡히지 않는다(프리뷰 반전이 픽셀에 구워져 저장되는 경우가 많다).
# 틀리면 좌우 안내가 통째로 뒤집히는데 사용자는 스스로 확인할 수 없다.
# 앱이 요청에 mirrored 를 실어 보내면 그 값이 이 기본값을 덮어쓴다.
MIRRORED_DEFAULT = _b("BT_MIRRORED_DEFAULT", False)

# ── 전처리 ────────────────────────────────────────────────────────
LANDMARKER_TASK = os.environ.get("BT_LANDMARKER", str(ASSETS / "face_landmarker.task"))
PREPROCESS_WORKERS = _i("BT_PREPROCESS_WORKERS", 3)
TOO_DARK_MEAN = _f("BT_TOO_DARK_MEAN", 40.0)

# 립 크롭 — 인계 문서가 검증한 값. 바꾸면 재측정 대상이다.
LIP_PAD_X = _f("BT_LIP_PAD_X", 0.50)
LIP_PAD_Y = _f("BT_LIP_PAD_Y", 0.70)
LIP_MIN_W = _i("BT_LIP_MIN_W", 60)

# 얼굴 크롭 — 메이크업 경로용. 검증된 값이 아니라 합리적 기본값이다.
FACE_PAD = _f("BT_FACE_PAD", 0.25)
FACE_MIN_W = _i("BT_FACE_MIN_W", 120)

# ── 서버 ──────────────────────────────────────────────────────────
HOST = os.environ.get("BT_HOST", "0.0.0.0")
PORT = _i("BT_PORT", 8100)
LOG_LEVEL = os.environ.get("BT_LOG_LEVEL", "INFO").upper()

# 모델 없이 라우팅·전처리만 확인할 때 (CI·개발용)
MOCK = _b("BT_MOCK", False)
