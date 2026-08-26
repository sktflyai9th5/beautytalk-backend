# -*- coding: utf-8 -*-
"""메이크업 경로 후처리 — 학습 코드(makeup_inference.py)에서 그대로 옮긴 것.

이 모듈의 함수는 **학습 정답을 만들 때 쓴 것과 같아야 한다.** 학습 코드의 설계가

    모델 = 결함 위치(박스) + 종류만 출력
    코드 = 그 좌표를 describe_region_band() 로 부위 이름으로 변환 -> 음성 문장

이기 때문이다. 이름 붙이는 규칙이 학습 때와 다르면 좌표가 맞아도 엉뚱한 부위를
말하게 되고, 사용자는 화면을 못 보므로 그게 틀렸다는 걸 알 방법이 없다.

그래서 여기서는 **고쳐 쓰지 않는다.** 임계값·랜드마크 번호·문장 표현 전부 원본 그대로다.
원본이 바뀌면 이 파일도 같이 바꿔야 한다.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("bt.makeup")

# 학습 데이터의 부위별 등장 횟수. 이 값 미만인 세밀 라벨은 상위 표현으로 낮춘다.
# 예시가 적은 라벨을 세밀하게 부르면 구역 경계에 걸린 박스에서 이름이 자주 틀어진다.
FINE_LABEL_MIN_COUNT = 15

def _load_hist() -> dict:
    f = Path(__file__).parent / "assets" / "makeup" / "fine_hist.json"
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
        h = {k: v for k, v in raw.items() if not k.startswith("_")}
        log.info("부위 히스토그램 %d종 (기준 %d회 미만은 상위 표현)",
                 len(h), FINE_LABEL_MIN_COUNT)
        return h
    except Exception:
        # 없으면 전부 상위 표현으로 — 안전한 쪽이다
        log.warning("fine_hist.json 없음 — 세밀 라벨을 전부 상위 표현으로 낮춘다")
        return {}


FINE_HIST = _load_hist()

COORD_SCALE = 1000
MAX_SPEAK_BOXES = 2   # 한 번에 여러 곳 말해봤자 기억 못 함
NO_DEFECT = "전체적으로 베이스가 고르게 잘 발려 있어요. 지금 상태 그대로 두셔도 좋아요."
CLOSING = "손끝으로 쓸어봤을 때 주변과 두께 차이가 안 느껴지면 잘 된 거예요."

TYPE_TO_KO = {"missing": "덜발림", "boundary": "경계", "uneven": "불균형"}
KO_TO_TYPE = {v: k for k, v in TYPE_TO_KO.items()}


# ---------------------------------------------------------------- 좌표 변환
def crop_norm_to_bbox(norm, rect):
    """크롭 기준 0~1000 정규화 좌표 -> 원본 이미지 좌표."""
    ox, oy, cw, ch = rect
    x1, y1, x2, y2 = [v / COORD_SCALE for v in norm]
    return [ox + x1 * cw, oy + y1 * ch, ox + x2 * cw, oy + y2 * ch]


def parse_coord_answer(text):
    """모델 출력에서 bbox_2d 목록을 관대하게 파싱.

    출력이 토큰 제한에 걸려 중간에 잘려도 완성된 객체까지는 건져낸다.
    """
    data = None
    m = re.search(r"\[.*\]", text or "", re.S)
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = None
    if data is None:
        # 잘린 출력 대비: 온전한 중괄호 덩어리만 개별로 긁어모음
        data = []
        for chunk in re.findall(r"\{[^{}]*\}", text or "", re.S):
            try:
                data.append(json.loads(chunk))
            except json.JSONDecodeError:
                continue
    out = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        box = d.get("bbox_2d") or d.get("bbox")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        try:
            box = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:  # 뒤집힌 박스 버림
            continue
        out.append((box, KO_TO_TYPE.get(str(d.get("label", "")).strip(), "uneven")))
    return out[:2]


# ---------------------------------------------------------------- 부위 판정
def _lm_xy(pts, idxs, axis):
    return sum(pts[i][axis] for i in idxs) / len(idxs)


def describe_region_band(bbox, landmarks):
    """랜드마크 기준선으로 얼굴을 세로 5구간 x 가로 3구간으로 나눠 부위 이름을 정한다."""
    if not landmarks:
        return None
    p = landmarks
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    try:
        forehead_y = _lm_xy(p, [10], 1)                    # 이마 맨 위
        brow_y = _lm_xy(p, [105, 334, 107, 336], 1)        # 눈썹 선
        eye_bot_y = _lm_xy(p, [145, 374], 1)               # 아래 눈꺼풀
        nose_bot_y = _lm_xy(p, [2], 1)                     # 코 밑
        mouth_bot_y = _lm_xy(p, [17], 1)                   # 아랫입술 아래
        nose_x = _lm_xy(p, [6, 168, 4, 1], 0)              # 코 중심선
        alar_half = abs(_lm_xy(p, [331], 0) - _lm_xy(p, [102], 0)) / 2   # 콧방울 반폭
        mouth_half = abs(_lm_xy(p, [291], 0) - _lm_xy(p, [61], 0)) / 2   # 입 반폭
    except (IndexError, ZeroDivisionError):
        return None

    # 코 중심선 기준 좌우 (이미지 왼쪽 = 사용자 오른쪽으로 반전)
    side = "오른쪽" if cx < nose_x else "왼쪽"
    dx = abs(cx - nose_x)
    is_center = dx < max(alar_half, 1)
    is_nose_side = dx < max(alar_half * 1.8, 1)
    is_mouth_side = dx < max(mouth_half * 1.3, 1)

    if cy < brow_y:                                        # 1. 이마
        if is_center:
            mid = (forehead_y + brow_y) / 2
            return "이마 중앙" if cy < mid else "미간"
        return f"이마 {side}"

    if cy < eye_bot_y:                                     # 2. 눈썹~눈 아래
        return "콧대" if is_center else f"눈두덩이 {side}"

    if cy < nose_bot_y:                                    # 3. 눈 아래~코 밑
        if is_center:
            mid = (eye_bot_y + nose_bot_y) / 2
            return "콧대" if cy < mid else "코끝"
        if is_nose_side:
            return f"콧볼 {side}"
        mid = (eye_bot_y + nose_bot_y) / 2
        return f"눈 아래 {side}" if cy < mid else f"볼 위쪽 {side}"

    if cy < mouth_bot_y:                                   # 4. 코 밑~입 아래
        if is_center:
            mid = (nose_bot_y + mouth_bot_y) / 2
            return "인중" if cy < mid else "입"
        return f"입 옆 {side}" if is_mouth_side else f"볼 아래쪽 {side}"

    return "턱 중앙" if is_center else f"턱 {side}"          # 5. 입 아래


# 세밀 라벨 -> 상위 대표 표현. 학습 데이터에 예시가 적은 라벨은 이쪽으로 낮춘다.
FINE_TO_HEAD = {
    "이마 중앙": "이마", "이마": "이마",
    "미간": "미간",
    "콧대": "코", "코끝": "코", "콧볼": "코",
    "눈썹 위": "눈", "눈썹 바로 밑": "눈", "눈두덩이": "눈", "눈 아래": "눈",
    "애교살": "눈", "눈꼬리": "눈",
    "볼 위쪽": "볼", "볼 중앙": "볼", "볼 아래쪽": "볼",
    "인중": "입", "입 옆": "입", "입": "입",
    "턱 중앙": "턱", "턱": "턱",
}


def coarsen_fine(fine):
    """세밀 라벨을 같은 계열 상위 표현으로 한 단계만 낮춘다. 눈두덩이 왼쪽 -> 눈 왼쪽."""
    side, base = "", fine
    for s in (" 왼쪽", " 오른쪽"):
        if fine.endswith(s):
            side, base = s, fine[: -len(s)]
            break
    head = FINE_TO_HEAD.get(base)
    if head is None:
        return None
    return f"{head}{side}" if side else head


def region_label(bbox, landmarks):
    """부위 이름. 학습 데이터에 충분히 나온 라벨만 세밀하게 부른다.

    학습 코드 get_region_label() 과 같은 규칙이다. 예시가 15개 미만인 라벨은
    같은 계열 상위 표현으로 낮춘다 ("눈두덩이 왼쪽" -> "눈 왼쪽").
    """
    fine = describe_region_band(bbox, landmarks)
    if fine is None:
        return None
    if FINE_HIST.get(fine, 0) >= FINE_LABEL_MIN_COUNT:
        return fine
    return coarsen_fine(fine) or fine


def is_wide_area(bbox, face_box):
    """얼굴 너비/높이의 28% 이상이면 국소 부위가 아니라 넓은 범위로 본다."""
    if not face_box:
        return False
    x1, y1, x2, y2 = bbox
    fx1, fy1, fx2, fy2 = face_box
    fw, fh = max(fx2 - fx1, 1), max(fy2 - fy1, 1)
    return (x2 - x1) / fw > 0.28 or (y2 - y1) / fh > 0.28


# ---------------------------------------------------------------- 문장 만들기
def _josa(word, with_batchim, without_batchim):
    """받침 유무에 따라 조사 선택. 턱+이 / 볼 아래쪽 중앙+가."""
    ch = word[-1]
    if "가" <= ch <= "힣":
        return with_batchim if (ord(ch) - 0xAC00) % 28 else without_batchim
    return without_batchim


def state_and_action(region, wide, dtype):
    """결함 종류에 따라 필요한 행동이 정반대다 (덜 발린 곳을 두드려봤자 안 채워진다)."""
    subj = (f"{region} 전체{_josa('전체', '이', '가')}" if wide
            else f"{region}{_josa(region, '이', '가')}")
    loc = f"{region} 전체" if wide else region
    if dtype == "missing":
        state = f"{subj} 덜 발렸어요."
        action = ("그 부위를 손끝으로 넓게 펴 발라주세요." if wide
                  else "그 부분에 손끝으로 조금 더 펴 발라주세요.")
    elif dtype == "boundary":
        state = f"{loc}에 발린 곳과 안 발린 곳 경계가 티나요."
        action = "그 선을 손끝으로 문질러서 흐려주세요."
    else:
        state = f"{loc}에 베이스가 고르지 않게 발려 있어요."
        action = ("좁게 콕 집기보다 그 부위 전체를 손이나 퍼프로 넓게 펴 발라서 정리해 주세요."
                  if wide else "퍼프나 손끝으로 그 부분을 가볍게 두드려서 정리해 주세요.")
    return state, action


def answer_from_boxes(pred, landmarks, face_box, rect, on_face=None, img_size=None):
    """모델 좌표 -> 부위 이름 -> 음성 안내 문장.

    (읽어 줄 문장, 지표, 부위별 항목) 을 돌려준다. 항목은 화면 표시용이고,
    음성은 합쳐진 문장을 그대로 쓴다 — 둘을 따로 만들면 서로 어긋난다.

    pred       parse_coord_answer 결과 (크롭 0~1000 좌표, 종류)
    landmarks  원본 이미지 픽셀 좌표계의 FaceMesh 468점
    face_box   원본 좌표계의 얼굴 박스 (넓은 범위 판정용)
    rect       크롭 사각형 (ox, oy, cw, ch)
    on_face    bbox -> bool. 얼굴 밖(도구/손/머리카락) 예측을 걸러낸다.
    """
    if not pred:
        return NO_DEFECT, "none", []

    boxes = [(crop_norm_to_bbox(b, rect), t) for b, t in pred]
    if on_face is not None:
        boxes = [(b, t) for b, t in boxes if on_face(b)]
    if not boxes:
        return NO_DEFECT, "off_face", []

    # 예측 박스 둘이 같은 부위로 판정되면 같은 문장을 두 번 말하게 된다.
    # 같은 (부위, 종류)는 한 번만 말하고 다음 박스를 끌어올린다.
    said, items = set(), []
    for bbox, dtype in boxes:
        region = region_label(bbox, landmarks)
        if region is None:
            continue
        key = (region, dtype)
        if key in said:
            continue
        said.add(key)
        items.append((bbox, region, dtype))
        if len(items) >= MAX_SPEAK_BOXES:
            break

    if not items:
        return NO_DEFECT, "no_region", []

    parts = []
    structured = []
    for i, (bbox, region, dtype) in enumerate(items):
        state, action = state_and_action(region, is_wide_area(bbox, face_box), dtype)
        parts.append(state if i == 0 else "그리고 " + state)
        parts.append(action)
        # 화면은 부위/상태/동작을 따로 보여 준다. 문장을 다시 쪼개면 어긋나므로
        # 만들 때 같이 내보낸다. 음성은 여전히 아래 합친 문장을 그대로 읽는다.
        item = {
            "region": region,
            "state": state,
            "action": action,
            "type": TYPE_TO_KO[dtype],
        }
        # 어디가 문제인지 — 원본 기준 0..1 박스. 앱이 사진 위에 그린다.
        if img_size and img_size[0] > 0 and img_size[1] > 0:
            iw, ih = img_size
            item["box"] = [round(bbox[0] / iw, 4), round(bbox[1] / ih, 4),
                           round(bbox[2] / iw, 4), round(bbox[3] / ih, 4)]
        structured.append(item)
    parts.append(CLOSING)
    detail = ",".join(f"{r}:{TYPE_TO_KO[t]}" for _, r, t in items)
    return " ".join(parts), f"{len(items)}곳 {detail}", structured
