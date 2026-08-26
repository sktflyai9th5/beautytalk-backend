# -*- coding: utf-8 -*-
"""Qwen3-VL 추론 — 베이스 1개 + LoRA 어댑터 핫스왑.

두 경로가 같은 베이스 체크포인트를 쓴다는 점이 이 설계의 전부다.
  - 립     : 베이스 그대로 (프롬프트 + few-shot 만으로 검증됨)
  - 메이크업: 베이스 + LoRA 어댑터

모델을 두 벌 올리면 VRAM 이 두 배로 드는데, 어댑터만 갈아 끼우면 7.7GB 한 벌로
둘 다 처리한다. 어댑터 전환은 포인터 교체 수준이라 사실상 공짜다.

GPU 1장 기준이므로 생성은 락으로 직렬화한다. 전처리는 이 락 **밖**에서 끝내야
한다 — 재촬영 안내가 GPU 를 기다리면 안 된다(HANDOFF.md §4).
"""
from __future__ import annotations

import logging
import threading
import time

from PIL import Image

from . import config
from .routes import Route

log = logging.getLogger("bt.engine")


class Engine:
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._lock = threading.Lock()  # GPU 1장 → 생성 직렬화
        self._active_adapter = ""      # 지금 켜져 있는 어댑터
        self._loaded: set[str] = set() # 이미 올린 어댑터 이름
        self.device = "cpu"
        self.ready = False

    # ------------------------------------------------------------ 적재
    def load(self, adapters: list[str]) -> None:
        """부팅 시 1회. 베이스 + 필요한 어댑터를 전부 올려 둔다.

        어댑터를 요청 시점에 받아오면 첫 사용자가 다운로드를 기다린다.
        """
        if config.MOCK:
            log.warning("BT_MOCK=1 — 모델을 올리지 않는다 (라우팅·전처리만 동작)")
            self.ready = True
            return

        import torch

        t0 = time.time()
        log.info("베이스 적재: %s", config.BASE_MODEL)
        if config.USE_UNSLOTH:
            self._load_unsloth()
        else:
            self._load_transformers()
        self._model.eval()
        self.device = str(getattr(self._model, "device", "cuda:0"))
        log.info("베이스 적재 완료 %.1fs (device=%s, loader=%s)",
                 time.time() - t0, self.device,
                 "unsloth" if config.USE_UNSLOTH else "transformers")

        for name in [a for a in dict.fromkeys(adapters) if a]:
            self._load_adapter(name)

        if torch.cuda.is_available():
            log.info("VRAM %.1fGB 사용 중", torch.cuda.memory_allocated() / 1024**3)
        self.ready = True

    def _load_unsloth(self) -> None:
        """인계 문서·학습 코드가 실제로 쓴 경로.

        순정 transformers 로도 "돌아간다"고 적혀 있었지만 실제로는 안 됐다.
        unsloth 사전 양자화 체크포인트를 순정 로더로 올리면 비전 타워의 4bit
        레이어에 양자화 상태가 안 붙어서, 생성 때 bitsandbytes 가
        assert module.weight.shape[1] == 1 로 죽는다.
        """
        from unsloth import FastVisionModel   # transformers 를 패치하므로 먼저 import

        # use_gradient_checkpointing 은 학습용 플래그다. 인계 문서 예제가 그걸
        # 켜고 있어서 그대로 따랐다가 추론이 15~85초 나왔다. 추론에서는 메모리를
        # 아낄 이유가 없고 순전파를 다시 계산하게 만들어 느려지기만 한다.
        model, processor = FastVisionModel.from_pretrained(
            config.BASE_MODEL,
            load_in_4bit=True,
            dtype=None,
            use_gradient_checkpointing=False,
        )
        FastVisionModel.for_inference(model)
        self._model, self._processor = model, processor
        # few-shot 3턴 + 이미지 4장이 들어가므로 기본 길이로는 잘린다
        self._processor.tokenizer.model_max_length = config.TOKENIZER_MAX_LEN

    def _load_transformers(self) -> None:
        """폴백. unsloth 를 못 쓰는 환경용 — 이 체크포인트로는 아직 생성이 안 된다."""
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as VLModel
        except ImportError:
            from transformers import AutoModelForImageTextToText as VLModel

        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        self._model = VLModel.from_pretrained(
            config.BASE_MODEL, dtype="auto", device_map=device_map
        )
        self._processor = AutoProcessor.from_pretrained(config.BASE_MODEL)
        self._processor.tokenizer.model_max_length = config.TOKENIZER_MAX_LEN

    def _load_adapter(self, name: str) -> None:
        t0 = time.time()
        try:
            # PeftModel 로 감싸는 대신 transformers 의 어댑터 API 를 쓴다.
            # 이러면 어댑터를 여러 개 올려 두고 이름으로 전환할 수 있다.
            self._model.load_adapter(name, adapter_name=name)
            self._loaded.add(name)
            log.info("어댑터 적재: %s (%.1fs)", name, time.time() - t0)
        except Exception:
            log.exception("어댑터 적재 실패: %s — 이 경로는 베이스로 동작한다", name)

    # ------------------------------------------------------------ 전환
    def _use(self, adapter: str) -> None:
        """락 안에서만 부른다. 어댑터 상태는 모델 전역이라 경쟁하면 섞인다."""
        if adapter == self._active_adapter:
            return
        if adapter and adapter in self._loaded:
            # set_adapter() 는 "어느 어댑터를 쓸지"만 바꾼다. disable_adapters() 로
            # 꺼둔 상태를 되돌리지 않으므로 먼저 켜야 한다.
            # 이걸 빠뜨리면 립 요청(베이스 사용) 한 번 뒤로는 메이크업 요청이
            # 전부 파인튜닝 없이 도는데, 답이 그럴듯해서 눈치채기 어렵다.
            enable = getattr(self._model, "enable_adapters", None)
            if enable:
                enable()
            self._model.set_adapter(adapter)
        else:
            # 베이스로 — 어댑터가 하나도 안 올라갔으면 disable 자체가 없다
            disable = getattr(self._model, "disable_adapters", None)
            if disable and self._loaded:
                disable()
        self._active_adapter = adapter if adapter in self._loaded else ""
        log.info("어댑터 전환 → %s (활성=%s)", self._active_adapter or "(베이스)",
                 self._adapters_enabled())

    def _adapters_enabled(self) -> bool:
        """실제로 LoRA 층이 켜져 있는지 확인 (조용히 꺼지는 것을 잡기 위한 확인용)."""
        try:
            from peft.tuners.tuners_utils import BaseTunerLayer

            for _, m in self._model.named_modules():
                if isinstance(m, BaseTunerLayer):
                    return not getattr(m, "_disable_adapters", False)
        except Exception:
            pass
        return False

    # ------------------------------------------------------------ 생성
    def generate(self, route: Route, image: Image.Image, question: str) -> str:
        if config.MOCK:
            return f"[mock:{route.name}] 확인했어요. 지금은 모델 없이 도는 중이에요."

        import torch

        messages, images = route.build_messages(image, question)
        with self._lock:
            self._use(route.adapter)
            prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(
                text=[prompt], images=images, add_special_tokens=False, return_tensors="pt"
            ).to(self._model.device)
            n_in = inputs["input_ids"].shape[1]
            extra = {}
            if route.stop_on_json:
                from transformers import StoppingCriteriaList

                extra["stopping_criteria"] = StoppingCriteriaList(
                    [_json_array_stopper(self._processor.tokenizer, n_in)]
                )
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    do_sample=False,          # 재현성 — 같은 사진에 같은 답
                    max_new_tokens=route.max_new_tokens or config.MAX_NEW_TOKENS,
                    use_cache=True,
                    **extra,
                )
            n_out = out.shape[1] - n_in
            text = self._processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        # 프롬프트/생성 토큰 수. 느릴 때 prefill 문제인지 decode 문제인지 여기서 갈린다.
        log.info("[%s] 토큰 입력=%d 생성=%d", route.name, n_in, n_out)
        return text.strip()

    def warmup(self, route: Route) -> None:
        """GPU 를 깨워 둔다. 노트북 GPU 는 절전에서 첫 추론이 30~60초 걸린다."""
        if config.MOCK:
            return
        try:
            blank = Image.new("RGB", (224, 224), (200, 180, 180))
            t0 = time.time()
            saved, config.MAX_NEW_TOKENS = config.MAX_NEW_TOKENS, 1
            try:
                self.generate(route, blank, "warm")
            finally:
                config.MAX_NEW_TOKENS = saved
            log.info("GPU warm-up %.1fs", time.time() - t0)
        except Exception:
            log.exception("warm-up 실패 (무시)")


def _json_array_stopper(tokenizer, start_len):
    """대괄호가 닫히면 생성을 멈춘다.

    좌표 모델은 배열을 닫은 뒤에도 설명을 계속 붙이는 경우가 있어서 생성 상한까지
    가곤 했다(128토큰). 배열이 닫히면 더 읽을 것이 없으므로 거기서 끊는다.
    bnb 4bit 디코딩이 토큰당 200ms 수준이라 이 낭비가 그대로 지연이 된다.
    """
    from transformers import StoppingCriteria

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            text = tokenizer.decode(input_ids[0][start_len:], skip_special_tokens=True)
            depth = 0
            opened = False
            for ch in text:
                if ch == "[":
                    depth += 1
                    opened = True
                elif ch == "]":
                    depth -= 1
                if opened and depth == 0:
                    return True
            return False

    return _Stop()


engine = Engine()
