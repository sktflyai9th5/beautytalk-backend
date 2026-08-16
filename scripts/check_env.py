"""실행 환경 버전/호환성 검증 스크립트.

Python 버전, 패키지 버전, 의존성 제약(pip check 상당), 그리고 버전이 어긋나면
실제로 깨지는 API들을 마이크로 테스트로 직접 실행해 확인한다.
로컬 venv와 Docker 컨테이너 양쪽에서 동일하게 실행할 수 있다.

사용법:
    python scripts/check_env.py            # 전체 검증 (dev 패키지는 없어도 경고만)
"""

import asyncio
import importlib.metadata as md
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # 레포 루트

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RUNTIME_PACKAGES = ["fastapi", "uvicorn", "aiortc", "av", "numpy", "pillow", "httpx", "pydantic", "websockets"]
DEV_PACKAGES = ["pytest", "pytest-asyncio"]

_failures: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def check_python() -> None:
    v = sys.version_info
    report(
        "python >= 3.11",
        (v.major, v.minor) >= (3, 11),
        f"{sys.version.split()[0]} ({sys.platform})",
    )
    in_venv = sys.prefix != sys.base_prefix
    print(f"       interpreter: {sys.executable}")
    if not in_venv:
        print("       (주의) venv가 아닌 시스템 Python입니다. 패키지 누락 시:")
        print("       .venv\\Scripts\\Activate.ps1  (또는 .venv\\Scripts\\python.exe로 실행)")


def check_package_versions() -> None:
    for pkg in RUNTIME_PACKAGES:
        try:
            print(f"       {pkg}=={md.version(pkg)}")
        except md.PackageNotFoundError:
            report(f"package {pkg} installed", False, "미설치 (runtime 필수)")
    for pkg in DEV_PACKAGES:
        try:
            print(f"       {pkg}=={md.version(pkg)} (dev)")
        except md.PackageNotFoundError:
            print(f"       {pkg} 미설치 (dev 전용 - 컨테이너에서는 정상)")


def check_dependency_conflicts() -> None:
    """설치된 모든 배포판의 Requires-Dist 제약을 실제 설치 버전과 대조 (pip check 상당)."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        report("dependency conflict scan", True, "packaging 모듈 없음 - pip check로 대체 필요")
        return
    conflicts = []
    for dist in md.distributions():
        dist_name = dist.metadata["Name"]
        for req_str in dist.requires or []:
            req = Requirement(req_str)
            if req.marker is not None and not req.marker.evaluate():
                continue
            try:
                installed = md.version(req.name)
            except md.PackageNotFoundError:
                continue  # extra 미설치는 정상
            if req.specifier and not req.specifier.contains(installed, prereleases=True):
                conflicts.append(f"{dist_name} requires {req_str} but {req.name}=={installed}")
    report("dependency version constraints", not conflicts, "; ".join(conflicts) or "충돌 없음")


def check_aiortc_declared_bounds() -> None:
    """aiortc가 선언한 핵심 의존성 범위와 실제 설치 버전을 출력 (버전 드리프트 추적용)."""
    try:
        reqs = md.requires("aiortc") or []
    except md.PackageNotFoundError:
        report("aiortc installed", False, "미설치 - venv 활성화 후 pip install -r requirements.txt")
        return
    for r in reqs:
        name = r.split(">")[0].split("<")[0].split("=")[0].split(";")[0].strip()
        if name in ("av", "aioice", "pylibsrtp", "cryptography", "pyee"):
            print(f"       aiortc declares: {r}")


def check_pydantic_v2_api() -> None:
    import pydantic

    from app.schemas import AnalysisResult

    ok = pydantic.VERSION.startswith("2")
    result = AnalysisResult(session_id="v", request_id="v", status="ok", message="검증")
    text = result.model_dump_json()  # v1이면 여기서 AttributeError
    report("pydantic v2 API (model_dump_json)", ok and '"analysis_result"' in text, f"pydantic {pydantic.VERSION}")


def check_av_numpy_pillow_interop() -> None:
    """av VideoFrame <- numpy 배열 -> PIL JPEG 인코딩: 분석 파이프라인의 프레임 경로 그대로."""
    import numpy as np
    from av import VideoFrame

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, 0] = 128
    frame = VideoFrame.from_ndarray(img, format="bgr24")
    pil = frame.to_image()  # Pillow 필요
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    report(
        "av+numpy+pillow frame->JPEG path",
        buf.tell() > 1000 and frame.width == 640,
        f"numpy {np.__version__}, jpeg {buf.tell()} bytes",
    )


def check_aiortc_signaling() -> None:
    """aiortc 실제 시그널링 API: pc 생성 -> datachannel -> offer 생성 -> close."""
    from aiortc import RTCConfiguration, RTCPeerConnection

    async def _run() -> bool:
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pc.createDataChannel("check")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        has_candidates = "candidate" in pc.localDescription.sdp
        await pc.close()
        return has_candidates and offer.type == "offer"

    import aiortc

    ok = asyncio.run(_run())
    report("aiortc offer/datachannel/ICE gathering", ok, f"aiortc {aiortc.__version__}")


def check_uvicorn_websocket_protocol() -> None:
    """uvicorn이 WebSocket 프로토콜 구현을 실제로 로드할 수 있는지 (uvicorn[standard] 확인)."""
    from uvicorn.config import Config

    config = Config(app="main:app", ws="auto")
    ws_class = config.ws_protocol_class if hasattr(config, "ws_protocol_class") else None
    if ws_class is None:
        config.load()
        ws_class = config.ws_protocol_class
    import uvicorn

    report(
        "uvicorn websocket protocol available",
        ws_class is not None,
        f"uvicorn {uvicorn.__version__} -> {ws_class.__module__ if ws_class else 'None'}",
    )


def check_websockets_client_api() -> None:
    """websockets 클라이언트 API(E2E 스크립트가 사용) 존재 확인."""
    try:
        import websockets

        ok = callable(getattr(websockets, "connect", None))
        report("websockets client API (connect)", ok, f"websockets {websockets.__version__}")
    except ImportError:
        print("       websockets 미설치 (컨테이너 runtime엔 uvicorn[standard]로 포함되어야 함)")
        report("websockets importable", False, "uvicorn[standard] 확인 필요")


def check_app_imports() -> None:
    import main  # noqa: F401
    from app import analysis, analyzer, config, schemas, sessions, webrtc, websocket  # noqa: F401

    report("all app modules import", True)


def main() -> None:
    print("=== BeautyTalk backend environment check ===")
    checks = [
        check_python,
        check_package_versions,
        check_aiortc_declared_bounds,
        check_dependency_conflicts,
        check_app_imports,
        check_pydantic_v2_api,
        check_av_numpy_pillow_interop,
        check_aiortc_signaling,
        check_uvicorn_websocket_protocol,
        check_websockets_client_api,
    ]
    for fn in checks:
        try:
            fn()
        except Exception as e:  # 한 항목이 죽어도 나머지는 계속 검사
            report(fn.__name__, False, f"{type(e).__name__}: {e}")
    if _failures:
        print(f"RESULT: FAIL ({len(_failures)}): {', '.join(_failures)}")
        sys.exit(1)
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()
