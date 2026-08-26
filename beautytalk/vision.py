# -*- coding: utf-8 -*-
"""MediaPipe FaceLandmarker 전처리.

인계 문서(assets/lip/HANDOFF.md)의 preprocess.py 를 서버용으로 옮긴 것.
립 크롭 상수(PAD_X/PAD_Y/OUTER/처리순서)는 **검증에 쓴 값 그대로**다.

두 가지를 고쳤다.
  1) 검출기를 스레드로컬로. 원본은 락 없는 전역 싱글턴인데 FaceLandmarker(IMAGE
     모드)는 스레드 안전이 아니라서 워커 스레드에 그대로 붙이면 안 된다.
  2) 얼굴 전체 크롭(makeup 경로) 추가. 립 크롭과 같은 검출 결과를 재사용한다.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from . import config

log = logging.getLogger("bt.vision")

# iOS 기본 포맷(HEIC). 없으면 아이폰 업로드가 통째로 예외가 난다.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - 선택 의존성
    log.warning("pillow-heif 없음 — HEIC 업로드는 실패한다")

# MediaPipe FaceMesh 입술 바깥 윤곽 (인계 문서 그대로. 바꾸면 크롭이 달라진다)
LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
             375, 321, 405, 314, 17, 84, 181, 91, 146]

# 프레이밍 판정에 쓰는 기준점 (널리 쓰이는 인덱스만)
CHIN = 152                # 턱 끝
MOUTH_CORNERS = (61, 291)  # 입꼬리


@dataclass
class PrepError:
    """실패 사유. message 는 **그대로 TTS 로 내보낼 수 있는** 한국어 문장이다."""

    code: str
    message: str


@dataclass
class Prepared:
    image: Image.Image
    #  지표용 — 검출된 입술/얼굴 폭(px). MIN_W 임계값 조정의 유일한 근거다.
    box_w: int
    #  아래 셋은 메이크업 경로 전용. 모델이 좌표를 내므로 그것을 부위 이름으로
    #  옮기려면 학습 때와 같은 재료가 필요하다. 립 경로는 전부 None.
    rect: tuple | None = None        # 원본에서 잘라낸 사각형 (ox, oy, cw, ch)
    landmarks: list | None = None    # 원본 픽셀 좌표계 FaceMesh 468점
    face_box: tuple | None = None    # 원본 좌표계 얼굴 박스 (넓은 범위 판정용)
    hull: object | None = None       # 얼굴 볼록껍질 (얼굴 밖 예측 걸러내기)
    #  원본 스냅샷 크기 — 모델 좌표를 0..1 로 정규화할 때 쓴다. 정규화해 두면
    #  앱이 자기 원본(비율 동일, 해상도만 다름)에 그대로 얹을 수 있다.
    img_w: int = 0
    img_h: int = 0


_local = threading.local()


def _detector():
    """스레드당 하나. 최초 생성에 ~470ms 걸리므로 헬스체크 전에 워밍업한다."""
    d = getattr(_local, "detector", None)
    if d is None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        # 경로가 아니라 바이트로 넘긴다. mediapipe 0.10.x 는 윈도우 절대경로를
        # site-packages 기준 상대경로로 붙여버려서 파일을 못 찾는다.
        with open(config.LANDMARKER_TASK, "rb") as f:
            blob = f.read()
        d = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=blob),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
            )
        )
        _local.detector = d
        log.info("FaceLandmarker 생성 (thread=%s)", threading.current_thread().name)
    return d


def load_image(src, mirrored: bool) -> Image.Image:
    """바이트/경로/PIL 을 받아 RGB 로. 회전·미러 보정까지 여기서 끝낸다."""
    if isinstance(src, Image.Image):
        im = src
    elif isinstance(src, (bytes, bytearray)):
        im = Image.open(BytesIO(src))
    else:
        im = Image.open(src)
    # 세로 셀카는 EXIF 회전이 붙어 온다. 적용하지 않으면 누운 얼굴이 들어간다.
    im = ImageOps.exif_transpose(im).convert("RGB")
    if mirrored:
        # 전면 카메라 프리뷰 반전이 픽셀에 구워진 경우. EXIF 로는 안 잡힌다.
        im = ImageOps.mirror(im)
    return im


def _landmarks(im: Image.Image):
    """(landmarks, w, h) 또는 (PrepError, 0, 0)."""
    import mediapipe as mp

    a = np.asarray(im)
    if a.mean() < config.TOO_DARK_MEAN:
        return PrepError("too_dark", "사진이 너무 어두워요. 밝은 곳에서 다시 찍어 주세요."), 0, 0

    res = _detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=a))
    if not res.face_landmarks:
        return (
            PrepError(
                "no_face",
                "얼굴이 잘 안 보여요. 휴대폰을 얼굴에서 한 뼘 정도 떨어뜨리고 정면으로 들어 주세요.",
            ),
            0,
            0,
        )
    h, w = a.shape[:2]
    return res.face_landmarks[0], w, h


def _crop_box(x0, y0, x1, y1, pad_x, pad_y, w, h):
    bw, bh = x1 - x0, y1 - y0
    return (
        max(0, int(x0 - bw * pad_x)),
        max(0, int(y0 - bh * pad_y)),
        min(w, int(x1 + bw * pad_x)),
        min(h, int(y1 + bh * pad_y)),
    )


def _apply_crop(im, box, max_side):
    out = im.crop(box)
    if max(out.size) > max_side:
        k = max_side / max(out.size)
        out = out.resize((round(out.width * k), round(out.height * k)), Image.LANCZOS)
    return out


def _crop(im, x0, y0, x1, y1, pad_x, pad_y, w, h, max_side):
    return _apply_crop(im, _crop_box(x0, y0, x1, y1, pad_x, pad_y, w, h), max_side)


def prepare_lip(im: Image.Image):
    """입술 크롭. (Prepared, None) 또는 (None, PrepError).

    프롬프트와 few-shot 예시가 전부 입술 크롭 기준으로 검증됐다. 얼굴 전체를
    그대로 넣으면 입술이 전체 픽셀의 2~5% 라 신호가 뭉개진다.
    """
    lm, w, h = _landmarks(im)
    if isinstance(lm, PrepError):
        return None, lm

    xs = [lm[i].x * w for i in LIP_OUTER]
    ys = [lm[i].y * h for i in LIP_OUTER]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if (x1 - x0) < config.LIP_MIN_W:
        return None, PrepError(
            "lip_too_small",
            "입술이 너무 작게 나왔어요. 휴대폰을 얼굴 쪽으로 조금 더 가까이 가져와 주세요.",
        )
    crop = _crop(im, x0, y0, x1, y1, config.LIP_PAD_X, config.LIP_PAD_Y, w, h,
                 config.QUERY_MAX_SIDE)
    return Prepared(crop, int(x1 - x0)), None


# ---------------------------------------------------------------- 메이크업 경로
# 학습 코드(makeup_inference.py)와 **같은 검출기·같은 여백**을 써야 한다.
# 모델이 본 이미지는 face_detection 박스 기준 25% 여백 크롭이고, 좌표도 그 크롭
# 기준으로 학습됐다. 랜드마크 최대/최소로 만든 박스는 프레이밍이 달라서
# 조용히 분포를 벗어난다 (결과물은 여전히 "얼굴 사진"이라 눈으로는 안 보인다).
_legacy = threading.local()


def _face_detection():
    d = getattr(_legacy, "det", None)
    if d is None:
        import mediapipe as mp

        d = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        _legacy.det = d
    return d


def _face_mesh():
    m = getattr(_legacy, "mesh", None)
    if m is None:
        import mediapipe as mp

        m = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
        )
        _legacy.mesh = m
    return m


def prepare_face(im: Image.Image):
    """얼굴 크롭 + 좌표 해석에 필요한 재료. 메이크업(립 외) 경로용."""
    import numpy as np

    a = np.asarray(im)
    if a.mean() < config.TOO_DARK_MEAN:
        return None, PrepError("too_dark", "사진이 너무 어두워요. 밝은 곳에서 다시 찍어 주세요.")

    h, w = a.shape[:2]
    res = _face_detection().process(a)
    if not res.detections:
        return None, PrepError(
            "no_face",
            "얼굴이 잘 안 보여요. 휴대폰을 얼굴에서 한 뼘 정도 떨어뜨리고 정면으로 들어 주세요.",
        )
    b = res.detections[0].location_data.relative_bounding_box
    fx1, fy1 = b.xmin * w, b.ymin * h
    fx2, fy2 = (b.xmin + b.width) * w, (b.ymin + b.height) * h
    if (fx2 - fx1) < config.FACE_MIN_W:
        return None, PrepError(
            "face_too_small",
            "얼굴이 너무 작게 나왔어요. 휴대폰을 얼굴 쪽으로 조금 더 가까이 가져와 주세요.",
        )

    # 학습과 동일: 얼굴 박스 기준 상하좌우 FACE_PAD(0.25) 여백
    m = config.FACE_PAD
    fw, fh = fx2 - fx1, fy2 - fy1
    x1 = max(0, int(fx1 - fw * m))
    y1 = max(0, int(fy1 - fh * m))
    x2 = min(w, int(fx2 + fw * m))
    y2 = min(h, int(fy2 + fh * m))
    if x2 <= x1 or y2 <= y1:
        return None, PrepError("no_face", "얼굴이 잘 안 보여요. 다시 찍어 주세요.")
    rect = (x1, y1, x2 - x1, y2 - y1)

    crop = _apply_crop(im, (x1, y1, x2, y2), config.QUERY_MAX_SIDE)

    # 부위 판정용 랜드마크 (원본 픽셀 좌표계 — 학습 코드와 동일)
    mesh = _face_mesh().process(a)
    pts = None
    hull = None
    if mesh.multi_face_landmarks:
        lm = mesh.multi_face_landmarks[0].landmark
        pts = [(p.x * w, p.y * h) for p in lm]
        try:
            import cv2

            hull = cv2.convexHull(
                np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32)
            )
        except Exception:
            hull = None   # cv2 없으면 얼굴 밖 필터만 생략 (안내는 그대로 나간다)

    return Prepared(crop, int(fx2 - fx1), rect=rect, landmarks=pts,
                    face_box=(fx1, fy1, fx2, fy2), hull=hull, img_w=w, img_h=h), None


@dataclass
class Framing:
    """찍기 전 프레이밍 판정. GPU 를 쓰지 않는다."""

    ok: bool
    code: str          # ok | too_dark | no_face | too_small | cut_off | lip_small
    guidance: str      # 그대로 읽어 줄 짧은 한국어
    face_w: int = 0
    lip_w: int = 0


def check_framing(im: Image.Image) -> Framing:
    """지금 화면이 분석에 쓸 만한지 본다.

    카운트다운 전에 부른다. 찍고 나서 되돌아오면 사용자는 이미 3초를 버린 뒤이고,
    무엇을 고쳐야 하는지도 한 박자 늦게 안다. 여기서 미리 잡으면 고칠 시간이 생긴다.

    안내는 **방향**으로 준다 — 앞이 보이지 않는 사용자에게 "중앙에 맞추세요" 는
    실행할 수 없는 말이다. "조금 더 가까이" 는 몸으로 할 수 있다.

    순서가 중요하다. "가까이" 와 "멀리" 가 서로 부딪히면 안 된다.
    입술을 보려면 가까이 와야 하는데, 그러면 이마는 프레임 밖으로 나간다.
    이마가 잘리는 건 문제가 아니고 **입·턱이 잘리는 것**만 문제다 — 두 크롭 다 거기가 필요하다.
    """
    import numpy as np

    a = np.asarray(im)
    if a.mean() < config.TOO_DARK_MEAN:
        return Framing(False, "too_dark", "조금 더 밝은 곳으로 가 주세요.")

    h, w = a.shape[:2]
    res = _face_detection().process(a)
    if not res.detections:
        return Framing(False, "no_face", "얼굴이 안 보여요. 휴대폰을 정면으로 들어 주세요.")

    b = res.detections[0].location_data.relative_bounding_box
    face_w = int(b.width * w)

    mesh = _face_mesh().process(a)
    if not mesh.multi_face_landmarks:
        return Framing(False, "no_face", "얼굴이 또렷하지 않아요. 정면을 봐 주세요.",
                       face_w=face_w)
    lm = mesh.multi_face_landmarks[0].landmark

    # 입·턱이 프레임 밖이면 두 경로 다 못 쓴다 (이마가 잘리는 건 괜찮다)
    chin_y = lm[CHIN].y * h
    mouth_x = [lm[i].x * w for i in MOUTH_CORNERS]
    if chin_y > h or min(mouth_x) < 0 or max(mouth_x) > w:
        return Framing(False, "cut_off", "입과 턱이 화면 밖으로 나갔어요. 조금 멀리 떨어뜨려 주세요.",
                       face_w=face_w)

    xs = [lm[i].x * w for i in LIP_OUTER]
    lip_w = int(max(xs) - min(xs))

    if face_w < config.FACE_MIN_W:
        return Framing(False, "too_small", "조금 더 가까이 가져와 주세요.",
                       face_w=face_w, lip_w=lip_w)
    if lip_w < config.LIP_MIN_W:
        # 립 질문이면 못 쓰고 베이스 질문이면 쓸 수 있다. 막지 않고 권하기만 한다.
        return Framing(False, "lip_small", "조금만 더 가까이 오면 입술까지 볼 수 있어요.",
                       face_w=face_w, lip_w=lip_w)

    return Framing(True, "ok", "좋아요.", face_w=face_w, lip_w=lip_w)


def on_face(bbox, hull, min_inside: float = 0.45) -> bool:
    """박스를 3x3 격자로 찍어 얼굴 윤곽 안에 든 비율이 기준 이상이면 True.

    실패 사례에서 립글로스 브러시에 묻은 제품을 결함으로 잡는 일이 있었다.
    윤곽을 못 만들었으면 판단 불가이므로 통과시킨다 — 멀쩡한 예측을 지우는 게 더 나쁘다.
    """
    if hull is None:
        return True
    try:
        import cv2
    except ImportError:
        return True
    x1, y1, x2, y2 = bbox
    inside = 0
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            px = x1 + (x2 - x1) * i / 4.0
            py = y1 + (y2 - y1) * j / 4.0
            if cv2.pointPolygonTest(hull, (float(px), float(py)), False) >= 0:
                inside += 1
    return inside / 9.0 >= min_inside


def fit(im: Image.Image, side: int) -> Image.Image:
    """긴 변 기준 축소 (few-shot 예시 이미지 토큰 절약용)."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > side:
        k = side / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    return im


ready = False   # 검출기를 실제로 만들 수 있는지. /health 가 읽는다.


def warmup(pool) -> bool:
    """스레드풀의 모든 스레드에 검출기를 미리 만들어 둔다.

    스레드 수의 2배는 태워야 풀이 다 채워진다. 헬스체크 통과 전에 부른다.

    실패해도 예외를 올리지 않는다. 여기서 죽으면 컨테이너가 재시작을 반복하며
    로그만 흘러가는데, 그보다는 뜬 채로 /health 가 unhealthy 라고 말해 주는 편이
    원인을 찾기 쉽다. (mediapipe 버전·프로토버프 충돌이 흔한 원인)
    """
    global ready
    blank = Image.new("RGB", (320, 320), (128, 128, 128))
    n = config.PREPROCESS_WORKERS * 2
    try:
        list(pool.map(lambda _: _landmarks(blank), range(n)))
        ready = True
        log.info("전처리 워밍업 완료 (%d회)", n)
    except Exception:
        ready = False
        log.exception("전처리 워밍업 실패 — 얼굴 검출이 동작하지 않는다")
    return ready
