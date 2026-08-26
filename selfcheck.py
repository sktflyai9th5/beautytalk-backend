# -*- coding: utf-8 -*-
"""배포 자가진단 — 팀 노트북에서 서버를 띄운 뒤 돌린다.

    python selfcheck.py                       # 기본 http://127.0.0.1:8100
    python selfcheck.py --base http://... --image selfie.jpg

사진을 주면 두 경로(립/메이크업)로 실제 추론까지 돌려 본다. 안 주면 라우팅과
헬스체크만 본다. 사람 얼굴이 나온 사진이어야 전처리를 통과한다.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

OK, BAD = "  [OK]", "  [!!]"
fails = 0


def note(ok: bool, msg: str) -> None:
    global fails
    if not ok:
        fails += 1
    print(f"{OK if ok else BAD} {msg}")


def get(base: str, path: str, timeout: float = 10):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(base: str, path: str, payload: dict, timeout: float = 180):
    req = urllib.request.Request(
        base + path, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--image", help="얼굴이 나온 사진 (없으면 추론 검사 생략)")
    a = ap.parse_args()
    base = a.base.rstrip("/")

    print(f"\n대상: {base}\n")

    print("1) 헬스체크")
    try:
        h = get(base, "/health")
    except urllib.error.URLError as e:
        note(False, f"서버에 붙지 못했다: {e}")
        return 1
    note(h.get("status") == "healthy", f"status={h.get('status')}")
    note(h.get("preprocess") == "ready", f"preprocess={h.get('preprocess')}  (MediaPipe)")
    dev = h.get("device", "")
    note(dev.startswith("cuda"), f"device={dev}  (cpu 면 추론이 수 분 걸린다)")
    print(f"       base={h.get('base_model')}")
    for n, ad in (h.get("routes") or {}).items():
        print(f"       경로 {n:7} → {ad}")

    print("\n2) 라우팅")
    cases = [
        ("립 어때?", "lip"), ("입술 봐줘", "lip"), ("틴트 번졌어?", "lip"),
        ("눈썹 괜찮아?", "makeup"), ("지금 어때?", "makeup"), ("립 말고 눈", "makeup"),
    ]
    for q, want in cases:
        r = get(base, "/route-test?q=" + urllib.parse.quote(q))
        note(r["route"] == want, f"{q!r:16} → {r['route']:7} ({r['reason']})")

    if not a.image:
        print("\n3) 추론 — 건너뜀 (--image 로 얼굴 사진을 주면 실제로 돌려 본다)")
        return 0 if fails == 0 else 1

    print("\n3) 추론")
    b64 = base64.b64encode(open(a.image, "rb").read()).decode()
    for q, want in [("립 어때?", "lip"), ("눈썹이랑 피부 어때?", "makeup")]:
        t0 = time.time()
        try:
            r = post(base, "/analyze", {"question": q, "image_b64": b64})
        except Exception as e:
            note(False, f"{q!r} 실패: {e}")
            continue
        took = time.time() - t0
        note(r["route"] == want, f"{q!r} → 경로 {r['route']} (기대 {want})")
        note(r["status"] in ("ok", "retake"), f"status={r['status']} {took:.1f}s "
             f"(전처리 {r['prep_ms']}ms / 추론 {r['infer_ms']}ms)")
        print(f"       답변: {r['message']}")
        if r["status"] == "retake":
            print("       ↑ 재촬영 안내다. 얼굴·입술이 크게 나온 사진으로 다시 돌려볼 것")
        else:
            bad = [c for c in "*_#`[](){}<>" if c in r["message"]]
            note(not bad, f"TTS 안전 문자 검사 {'통과' if not bad else bad}")

    print("\n4) 좌우 확인 — 사람이 직접 볼 것")
    print("   한쪽 입꼬리에만 립을 번지게 하고 찍어서, 답변의 좌우가")
    print("   사용자 기준으로 맞는지 확인한다. 틀리면 BT_MIRRORED_DEFAULT 를 뒤집는다.")
    print("   (전면 카메라 미러링은 EXIF 로 잡히지 않아 코드로는 알 수 없다)")

    print(f"\n{'모두 통과' if fails == 0 else f'실패 {fails}건'}\n")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
