# -*- coding: utf-8 -*-
"""BeautyTalk 백엔드 — 모델 라우터.

앱과의 프로토콜은 그대로 둔다(WS analyze → analysis_result). 바뀐 것은 안쪽이다.

    사진 + 질문
      → router.pick(질문)        "립" 이 보이면 lip, 아니면 makeup
      → 경로별 MediaPipe 크롭     [CPU 스레드풀 — GPU 락 밖]
          ├ 실패 → 재촬영 안내를 그대로 TTS. GPU 안 씀
          └ 성공 ↓
      → 경로별 프롬프트 + 어댑터로 생성  [GPU 락 안]
      → 새니타이즈 → 앱

전처리를 GPU 큐 앞에 두는 게 중요하다. 앞이 안 보이는 사용자는 프레이밍 실패로
재촬영을 반복하는 게 정상 사용 패턴이라, 이 왕복 지연이 체감 품질을 좌우한다.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from . import config, router, routes, vision
from .engine import engine
from .sanitize import FALLBACK, action_of, sanitize

logging.basicConfig(
    level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("bt")

app = FastAPI(title="beautytalk-router")
_pool = ThreadPoolExecutor(
    max_workers=config.PREPROCESS_WORKERS, thread_name_prefix="prep"
)
_gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


@app.on_event("startup")
def _startup() -> None:
    t0 = time.time()
    rs = routes.load()
    vision.warmup(_pool)  # 검출기 최초 생성 ~470ms — 첫 사용자가 내지 않게
    engine.load([r.adapter for r in rs.values()])
    # 첫 추론은 CUDA 커널 준비 때문에 20~40초 더 걸린다. 그 비용을 첫 사용자가
    # 내지 않도록 여기서 미리 태운다. 경로마다 어댑터 상태가 달라서 둘 다 돌린다.
    for r in rs.values():
        engine.warmup(r)
    log.info("기동 완료 %.1fs", time.time() - t0)


@app.on_event("shutdown")
def _shutdown() -> None:
    _pool.shutdown(wait=False)
    _gpu.shutdown(wait=False)


# ---------------------------------------------------------------- 분석
class AnalysisItem(BaseModel):
    """결과를 부위별로 쪼갠 것. 화면이 목록으로 그린다.

    음성은 아래 Analysis.message 를 그대로 읽는다 — 이 항목들을 다시 이어 붙여
    문장을 만들면 낭독과 화면이 어긋난다.
    """

    region: str   # 왼쪽 볼 아래
    state: str    # 베이스가 고르지 않게 발려 있어요.
    #  문제 위치 — 원본 스냅샷 기준 0..1 (l, t, r, b). 없을 수도 있다.
    box: list[float] | None = None
    action: str   # 퍼프로 가볍게 두드려 정리해 주세요.
    type: str     # 덜발림 | 경계 | 불균형


class Analysis(BaseModel):
    status: str          # ok | retake | error
    message: str         # 그대로 TTS 로 읽어 줄 문장
    route: str = ""
    reason: str = ""     # 라우팅 사유 (로그·지표용)
    action: str = ""     # ok | wipe | blend (지표용)
    prep_ms: int = 0
    infer_ms: int = 0
    # 화면 표시용. 립 경로는 문장만 오므로 비어 있다.
    items: list[AnalysisItem] = []


def _prepare_sync(image_bytes: bytes, route, mirrored: bool):
    """CPU 풀에서 돈다. (Prepared, None) 또는 (None, Analysis).

    여기서 실패하면 GPU 를 건드리지 않고 바로 돌아간다.
    """
    try:
        im = vision.load_image(image_bytes, mirrored)
        return route.prepare(im)
    except Exception:
        # PrepError 로 안 잡히는 예외가 있다 — 잘린 업로드, HEIC 미지원 등
        log.exception("전처리 예외")
        return None, vision.PrepError("unreadable", "사진을 읽지 못했어요. 다시 한 번 찍어 주세요.")


def _generate_sync(route, image, question: str) -> str | None:
    """GPU 풀(1스레드)에서 돈다. 전처리를 통과한 요청만 여기까지 온다."""
    try:
        return engine.generate(route, image, question)
    except Exception:
        log.exception("생성 실패")
        return None


async def _analyze(image_b64: str, question: str, mirrored: bool) -> Analysis:
    if not image_b64:
        return Analysis(status="error", message="사진이 없어요. 다시 찍어 주세요.")
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except (binascii.Error, ValueError):
        return Analysis(status="error", message="사진을 읽지 못했어요. 다시 한 번 찍어 주세요.")

    name, reason = router.pick(question)
    route = routes.get(name)
    text = (route.default_question if route.fixed_question
            else (question.strip() or route.default_question))
    loop = asyncio.get_running_loop()

    # ── 전처리 (CPU 풀) — GPU 큐 앞에서 끝낸다 ────────────────────
    t0 = time.time()
    prepared, err = await loop.run_in_executor(_pool, _prepare_sync, raw, route, mirrored)
    prep_ms = int((time.time() - t0) * 1000)

    if err:
        # 재촬영 안내. GPU 슬롯을 잡지 않고 즉시 나간다.
        log.info("[%s] 재촬영 %s (%dms) — %s", name, err.code, prep_ms, reason)
        status = "error" if err.code == "unreadable" else "retake"
        return Analysis(status=status, message=err.message, route=name,
                        reason=reason, action=err.code, prep_ms=prep_ms)

    # ── 생성 (GPU 풀, 1스레드 직렬) ───────────────────────────────
    t1 = time.time()
    try:
        gen = await asyncio.wait_for(
            loop.run_in_executor(_gpu, _generate_sync, route, prepared.image, text),
            timeout=config.GENERATE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("분석 타임아웃 %.0fs", config.GENERATE_TIMEOUT)
        gen = None
    infer_ms = int((time.time() - t1) * 1000)

    if gen is None:
        return Analysis(status="error", message=FALLBACK, route=name,
                        reason=reason, prep_ms=prep_ms, infer_ms=infer_ms)

    # 경로에 따라 모델 출력이 문장일 수도 좌표 JSON 일 수도 있다.
    items: list[AnalysisItem] = []
    if route.postprocess:
        msg, detail, raw_items = route.postprocess(gen, prepared)
        items = [AnalysisItem(**i) for i in raw_items]
        if not msg:
            log.warning("[%s] 후처리 실패(%s) — 원문 %r", name, detail, (gen or "")[:120])
            return Analysis(status="error", message=FALLBACK, route=name,
                            reason=reason, action=detail,
                            prep_ms=prep_ms, infer_ms=infer_ms)
        msg, act = sanitize(msg), detail
    else:
        msg = sanitize(gen)
        act = action_of(msg)
    # 지표: 경로 / 라우팅 사유 / 동작 분포 / bbox 폭 / 지연. 인식 문장 자체는 안 남긴다.
    log.info("[%s] %s bbox=%dpx prep=%dms infer=%dms %d자 — %s",
             name, act, prepared.box_w, prep_ms, infer_ms, len(msg), reason)
    return Analysis(status="ok", message=msg, route=name, reason=reason,
                    action=act, prep_ms=prep_ms, infer_ms=infer_ms, items=items)


# ---------------------------------------------------------------- HTTP
@app.get("/health")
def health() -> dict:
    # 앱이 8100 → 8000 순으로 훑으면서 이 형태를 본다. analyzer 는 하단 표시용.
    ok = engine.ready and vision.ready
    return {
        "status": "healthy" if ok else "loading",
        "preprocess": "ready" if vision.ready else "failed",
        "analyzer": "mock" if config.MOCK else "qwen-router",
        "device": engine.device,
        "base_model": config.BASE_MODEL,
        "routes": {n: (r.adapter or "base") for n, r in routes.load().items()},
    }


@app.get("/routes")
def route_list() -> dict:
    """어떤 문장이 어디로 가는지 눈으로 확인할 때."""
    from . import makeup_regions as mr

    return {
        n: {
            "adapter": r.adapter or "(베이스)",
            "system_chars": len(r.system),
            "fewshot": len(r.shots),
            "crop": r.prepare.__name__,
            "postprocess": "좌표→한국어" if r.postprocess else "없음(원문)",
            # 히스토그램이 안 실리면 부위 이름이 조용히 뭉개진다 — 여기서 확인한다
            **({"fine_hist": f"{len(mr.FINE_HIST)}종/{sum(mr.FINE_HIST.values())}박스"}
               if r.postprocess else {}),
        }
        for n, r in routes.load().items()
    }


@app.get("/route-test")
def route_test(q: str) -> dict:
    """모델 없이 라우팅만 확인. /route-test?q=립 어때"""
    name, reason = router.pick(q)
    return {"question": q, "route": name, "reason": reason}


class AnalyzeBody(BaseModel):
    question: str = ""
    image_b64: str = ""
    mirrored: bool | None = None


@app.post("/analyze")
async def analyze_http(body: AnalyzeBody) -> Analysis:
    """WS 없이 한 방에 — curl 로 시험할 때 쓴다."""
    m = config.MIRRORED_DEFAULT if body.mirrored is None else body.mirrored
    return await _analyze(body.image_b64, body.question, m)


class FramingBody(BaseModel):
    image_b64: str = ""
    mirrored: bool | None = None


class FramingOut(BaseModel):
    ok: bool
    code: str
    guidance: str
    face_w: int = 0
    lip_w: int = 0
    ms: int = 0


def _framing_sync(raw: bytes, mirrored: bool):
    try:
        return vision.check_framing(vision.load_image(raw, mirrored))
    except Exception:
        log.exception("프레이밍 확인 예외")
        return vision.Framing(False, "unreadable", "사진을 읽지 못했어요. 다시 한 번 찍어 주세요.")


@app.post("/framing")
async def framing(body: FramingBody) -> FramingOut:
    """찍기 전 프레이밍 확인. **GPU 를 쓰지 않는다** — 수십 ms 안에 돌아온다.

    앱이 카운트다운 전에 부른다. 자세가 틀렸으면 안내만 하고 촬영을 미뤄서,
    3초를 버리고 20초짜리 분석까지 돌린 뒤에야 "다시 찍어 주세요" 를 듣는 일을 없앤다.
    """
    if not body.image_b64:
        return FramingOut(ok=False, code="empty", guidance="사진이 없어요. 다시 찍어 주세요.")
    try:
        raw = base64.b64decode(body.image_b64, validate=False)
    except (binascii.Error, ValueError):
        return FramingOut(ok=False, code="unreadable",
                          guidance="사진을 읽지 못했어요. 다시 한 번 찍어 주세요.")
    m = config.MIRRORED_DEFAULT if body.mirrored is None else body.mirrored
    t0 = time.time()
    loop = asyncio.get_running_loop()
    f = await loop.run_in_executor(_pool, _framing_sync, raw, m)
    ms = int((time.time() - t0) * 1000)
    log.info("[framing] %s face=%dpx lip=%dpx (%dms)", f.code, f.face_w, f.lip_w, ms)
    return FramingOut(ok=f.ok, code=f.code, guidance=f.guidance,
                      face_w=f.face_w, lip_w=f.lip_w, ms=ms)


@app.post("/warmup")
async def warmup() -> dict:
    """GPU 깨우기. 앱이 가이드 탭에서 주기적으로 부른다."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_gpu, engine.warmup, routes.get(router.MAKEUP))
    return {"status": "warm", "device": engine.device}


# ---------------------------------------------------------------- WS
@app.websocket("/ws/{session_id}")
async def ws(sock: WebSocket, session_id: str) -> None:
    """앱이 쓰는 경로. 프로토콜은 기존 그대로 유지한다."""
    await sock.accept()
    log.info("WS 연결 %s", session_id)
    try:
        while True:
            msg = await sock.receive_json()
            if msg.get("type") != "analyze":
                await sock.send_json({"type": "error", "message": f"알 수 없는 요청: {msg.get('type')}"})
                continue

            req_id = str(msg.get("request_id", ""))
            mirrored = msg.get("mirrored")
            r = await _analyze(
                msg.get("image_b64", "") or "",
                msg.get("question", "") or "",
                config.MIRRORED_DEFAULT if mirrored is None else bool(mirrored),
            )
            await sock.send_json({
                "type": "analysis_result",
                "request_id": req_id,
                "status": r.status,
                "message": r.message,
                # 앱은 message 만 읽지만, 로그를 맞춰 보려면 이쪽도 필요하다
                "route": r.route,
                "action": r.action,
                "prep_ms": r.prep_ms,
                "infer_ms": r.infer_ms,
                # 화면용. 기존 클라이언트는 message 만 읽으므로 추가해도 안 깨진다.
                "items": [i.model_dump() for i in r.items],
            })
    except WebSocketDisconnect:
        log.info("WS 종료 %s", session_id)
    except Exception:
        log.exception("WS 오류 %s", session_id)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level=config.LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
