"""이미지 파일을 실제 Qwen 모델로 직접 분석한다 (WebRTC 경유 없이 모델만 검증).

사용법 (PowerShell):
    $env:QWEN_API_URL='http://127.0.0.1:11434/v1'
    $env:QWEN_MODEL='qwen3-vl:8b'
    .\.venv\Scripts\python.exe scripts\test_model_direct.py 립사진.jpg --question "립 봐줘잉"

PNG 등 다른 포맷도 자동으로 JPEG 변환 후 전송한다.
"""

import argparse
import asyncio
import io
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image

from app.analyzer import QwenAnalyzer
from app.config import load_settings


def to_jpeg(path: pathlib.Path) -> bytes:
    image = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


async def run(image_path: str, question: str) -> None:
    settings = load_settings()
    if settings.analyzer_mock or not settings.qwen_api_url:
        sys.exit("QWEN_API_URL 환경변수를 설정하세요 (예: http://127.0.0.1:11434/v1)")
    jpeg = to_jpeg(pathlib.Path(image_path))
    print(f"[*] model={settings.qwen_model} url={settings.qwen_api_url}")
    print(f"[*] image={image_path} ({len(jpeg)} bytes as JPEG) question={question!r}")
    analyzer = QwenAnalyzer(settings)
    try:
        started = time.perf_counter()
        payload = await analyzer.analyze(jpeg, question)
        elapsed = time.perf_counter() - started
    finally:
        await analyzer.aclose()
    print(json.dumps(payload.model_dump(), ensure_ascii=False, indent=2))
    print(f"[+] {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="분석할 이미지 파일 경로")
    parser.add_argument("--question", default="립 봐줘잉")
    args = parser.parse_args()
    asyncio.run(run(args.image, args.question))


if __name__ == "__main__":
    main()
