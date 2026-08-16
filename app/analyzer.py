"""메이크업 분석기.

- QwenAnalyzer: OpenAI 호환 API(vLLM/Ollama 등)로 Qwen3-VL 호출
- MockAnalyzer: 모델 없이도 전체 파이프라인을 검증할 수 있는 결정적 mock
"""

import asyncio
import base64
import json
import logging
import re
from typing import Optional, Protocol

import httpx

from .config import Settings
from .logging_utils import log_event
from .schemas import AnalysisDetail, AnalysisPayload, Issue, Region

logger = logging.getLogger("beautytalk.analyzer")


class AnalyzerError(Exception):
    """분석 실패 (모델 응답 파싱 불가, API 오류 등)."""


class NoFrameError(AnalyzerError):
    """분석할 프레임이 아직 수신되지 않음."""


class MakeupAnalyzer(Protocol):
    name: str

    async def analyze(self, image_jpeg: Optional[bytes], question: str) -> AnalysisPayload: ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------- 공통 헬퍼

_REGION_KEYWORDS: dict[Region, tuple[str, ...]] = {
    "lips": ("립", "입술", "틴트", "루즈", "lip"),
    "eyes": ("눈", "아이", "섀도", "쉐도", "라이너", "마스카라", "eye"),
    "cheeks": ("볼", "블러셔", "블러쉬", "치크", "cheek"),
}


def guess_region(question: str) -> Region:
    for region, keywords in _REGION_KEYWORDS.items():
        if any(k in question for k in keywords):
            return region
    return "overall"


# ---------------------------------------------------------------- Mock

_MOCK_PAYLOADS: dict[Region, AnalysisPayload] = {
    "lips": AnalysisPayload(
        status="ok",
        region="lips",
        verdict="needs_fix",
        message="입술 오른쪽 아래 라인이 살짝 번졌어요. 면봉으로 입꼬리 바깥쪽을 한 번만 가볍게 닦아 주세요.",
        detail=AnalysisDetail(
            issues=[
                Issue(
                    area="입술 오른쪽 아래",
                    problem="립 라인이 경계 밖으로 번짐",
                    suggestion="면봉으로 바깥쪽을 한 번 닦아내기",
                )
            ]
        ),
    ),
    "eyes": AnalysisPayload(
        status="ok",
        region="eyes",
        verdict="needs_fix",
        message="왼쪽 눈꼬리 아이라인이 조금 두꺼워요. 면봉으로 눈꼬리 끝을 살짝 정리해 주세요.",
        detail=AnalysisDetail(
            issues=[
                Issue(
                    area="왼쪽 눈꼬리",
                    problem="아이라인이 두껍게 뭉침",
                    suggestion="면봉으로 눈꼬리 끝을 얇게 정리하기",
                )
            ]
        ),
    ),
    "cheeks": AnalysisPayload(
        status="ok",
        region="cheeks",
        verdict="bad_color",
        message="오른쪽 볼 블러셔 색이 조금 진해요. 퍼프로 볼 바깥쪽을 두 번 가볍게 눌러 톤을 낮춰 주세요.",
        detail=AnalysisDetail(
            issues=[
                Issue(
                    area="오른쪽 볼",
                    problem="블러셔 발색이 과함",
                    suggestion="퍼프로 두 번 눌러 색을 덜어내기",
                )
            ]
        ),
    ),
    "overall": AnalysisPayload(
        status="ok",
        region="overall",
        verdict="perfect",
        message="지금 완벽해요! 전체적으로 균형이 잘 맞아요. 그대로 마무리하셔도 좋아요.",
        detail=AnalysisDetail(issues=[]),
    ),
}


class MockAnalyzer:
    """질문 키워드에 따라 결정적인 결과를 돌려주는 mock 분석기."""

    name = "mock"

    async def analyze(self, image_jpeg: Optional[bytes], question: str) -> AnalysisPayload:
        await asyncio.sleep(0.05)  # 실제 분석 지연 흉내 (소요 시간 로그 확인용)
        region = guess_region(question)
        payload = _MOCK_PAYLOADS[region].model_copy(deep=True)
        log_event(
            logger,
            "mock_analysis",
            region=region,
            has_frame=image_jpeg is not None,
            frame_bytes=len(image_jpeg) if image_jpeg else 0,
        )
        return payload

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------- Qwen3-VL

_SYSTEM_PROMPT = """당신은 시각장애인의 메이크업을 돕는 전문 뷰티 어시스턴트입니다.
사용자의 얼굴 사진과 질문을 보고 화장 상태를 분석하세요.

반드시 아래 JSON 형식으로만 응답하세요. 마크다운, 코드펜스, 설명 문장을 붙이지 마세요.
{"status": "ok" 또는 "no_face",
 "region": "lips" | "eyes" | "cheeks" | "overall",
 "verdict": "perfect" | "needs_fix" | "bad_color",
 "message": "사용자에게 음성으로 읽어줄 한국어 문장",
 "issues": [{"area": "위치", "problem": "문제", "suggestion": "수정 방법"}]}

message 작성 규칙 (앱이 이 문장을 그대로 TTS로 읽습니다):
1. 사용자는 화면을 볼 수 없습니다. 위치는 "오른쪽 입꼬리 아래", "왼쪽 볼 바깥쪽"처럼
   방향과 신체 기준으로 구체적으로 말하세요. "여기", "이 부분" 같은 지시어는 금지.
2. 가장 중요한 문제 딱 하나만 골라 1~2문장으로 말하세요. 문제 + 손으로 고치는 방법 순서로.
3. 존댓말로 친절하게. 문제가 없으면 verdict를 "perfect"로 하고 칭찬해 주세요.
4. 사진에 얼굴이 없거나 너무 어두우면 status를 "no_face"로 하고,
   message에 카메라를 어떻게 움직이면 되는지 안내하세요.
5. issues 배열에는 발견한 문제를 중요한 순서대로 담되, message는 첫 번째 문제만 다루세요."""


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _extract_json(text: str) -> dict:
    text = _strip_think_blocks(text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start < 0:
        raise AnalyzerError(f"모델 응답에서 JSON을 찾지 못함: {text[:200]!r}")
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise AnalyzerError(f"모델 응답 JSON 파싱 실패: {text[:200]!r}") from e
    if not isinstance(obj, dict):
        raise AnalyzerError(f"모델 응답이 JSON 객체가 아님: {text[:200]!r}")
    return obj


def parse_model_output(text: str, question: str) -> AnalysisPayload:
    """모델 텍스트 응답 → AnalysisPayload. 필드가 어긋나면 합리적 기본값으로 보정."""
    obj = _extract_json(text)

    message = str(obj.get("message") or "").strip()
    if not message:
        raise AnalyzerError("모델 응답에 message가 없음")

    status = obj.get("status")
    if status not in ("ok", "no_face"):
        status = "ok"

    region = obj.get("region")
    if region not in ("lips", "eyes", "cheeks", "overall"):
        region = guess_region(question)

    verdict = obj.get("verdict")
    if verdict not in ("perfect", "needs_fix", "bad_color"):
        verdict = None if status != "ok" else "needs_fix"

    issues = []
    raw_issues = obj.get("issues")
    if isinstance(raw_issues, list):
        for raw in raw_issues:
            if not isinstance(raw, dict):
                continue
            issues.append(
                Issue(
                    area=str(raw.get("area") or ""),
                    problem=str(raw.get("problem") or ""),
                    suggestion=str(raw.get("suggestion") or ""),
                )
            )

    return AnalysisPayload(
        status=status,
        region=region,
        verdict=verdict,
        message=message,
        detail=AnalysisDetail(issues=issues),
    )


class QwenAnalyzer:
    """OpenAI 호환 API(chat/completions)로 Qwen3-VL을 호출하는 분석기."""

    name = "qwen"

    def __init__(self, settings: Settings):
        if not settings.qwen_api_url:
            raise ValueError("QWEN_API_URL이 설정되지 않았습니다 (또는 ANALYZER_MOCK=true 사용)")
        self._model = settings.qwen_model
        self._system_prompt = _SYSTEM_PROMPT
        if settings.qwen_no_think:
            # Qwen3 thinking 모드가 max_tokens를 잠식해 응답이 비는 것을 방지
            self._system_prompt += "\n/no_think"
        self._client = httpx.AsyncClient(
            base_url=settings.qwen_api_url,
            headers={"Authorization": f"Bearer {settings.qwen_api_key}"},
            timeout=httpx.Timeout(settings.analysis_timeout, connect=10.0),
        )

    async def analyze(self, image_jpeg: Optional[bytes], question: str) -> AnalysisPayload:
        if image_jpeg is None:
            raise NoFrameError("분석할 프레임이 없습니다")
        image_b64 = base64.b64encode(image_jpeg).decode("ascii")
        body = {
            "model": self._model,
            "temperature": 0.3,
            "max_tokens": 4000,  # thinking 모델도 JSON이 잘리지 않도록 여유 (instruct는 조기 종료라 무해)
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": f"사용자 질문: {question}"},
                    ],
                },
            ],
        }
        try:
            response = await self._client.post("/chat/completions", json=body)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            raise AnalyzerError(f"Qwen API 호출 실패: {e}") from e
        except (KeyError, IndexError, ValueError) as e:
            raise AnalyzerError(f"Qwen API 응답 형식 오류: {e}") from e
        return parse_model_output(content, question)

    async def aclose(self) -> None:
        await self._client.aclose()


def get_analyzer(settings: Settings) -> MakeupAnalyzer:
    if settings.analyzer_mock:
        return MockAnalyzer()
    return QwenAnalyzer(settings)
