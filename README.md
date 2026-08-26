# BeautyTalk 백엔드 — 모델 라우터

사진과 질문을 받아 **읽어줄 한국어 한 문단**을 돌려주는 서버.
사용자가 뭘 물었는지 보고 **처리 경로를 갈라 태운다.**

```
사진 + 질문
  │
  ├─ "립 어때?"     ─→ 입술 크롭   ─→ 베이스 Qwen + 립 프롬프트 + few-shot 3장
  │
  └─ "눈썹 어때?"   ─→ 얼굴 크롭   ─→ 베이스 Qwen + 메이크업 LoRA 어댑터
                                          │
                                     TTS 로 읽을 한국어 한 문단
```

앱은 어느 경로로 갔는지 몰라도 된다 — 프로토콜은 한쪽뿐이다(WS `analyze` → `analysis_result`).

앱 저장소: [sktflyai9th5/beautytalk-app](https://github.com/sktflyai9th5/beautytalk-app)

`FastAPI` · `Qwen3-VL 8B (4bit)` · `PEFT/LoRA` · `MediaPipe` · `Docker/CUDA`

---

## 목차

- [핵심 설계](#핵심-설계)
- [요구사항](#요구사항)
- [띄우기](#띄우기)
- [확인하기](#확인하기)
- [API](#api)
- [설정](#설정)
- [경로 바꾸기](#경로-바꾸기)
- [미러링 — 배포 전 결정할 것](#미러링--배포-전-결정할-것)
- [구조](#구조)
- [알려진 제약](#알려진-제약)

---

## 핵심 설계

### 베이스는 한 벌만 올린다

두 경로가 **같은 체크포인트**를 쓴다.

| 경로 | 베이스 | 어댑터 | 프롬프트 | few-shot |
| --- | --- | --- | --- | --- |
| `lip` | Qwen3-VL 8B Instruct (4bit 사전 양자화) | 없음 | `assets/lip/system_prompt.txt` | 3장 |
| `makeup` | 〃 | 메이크업 LoRA | `assets/makeup/system_prompt.txt` | 0장 |

메이크업 어댑터는 **LoRA 가중치만** 들어 있고 그 베이스가 립 경로가 검증한 것과 같다.
그래서 모델을 두 벌 올리지 않고 **7.7GB 한 벌 + 어댑터 전환**으로 처리한다.
전환은 `set_adapter()` 로 고르고, 립 경로에서는 `disable_adapters()` 로 통째로 꺼서 순수
베이스로 돌린다 — `set_adapter()` 만으로는 꺼지지 않는다. 포인터 교체 수준이라 사실상 공짜다.

립 경로에 어댑터가 없는 것은 의도다 — 원 인계 자료가 "파인튜닝 없음, 시스템 프롬프트 +
few-shot 3장으로만 동작"이라고 명시한다.

> **모델을 4B 로 내리지 말 것.** 추론이 대략 절반으로 줄지만, 립 프롬프트와 few-shot 3장이
> 8B 기준으로 작성·검증된 세트다. 바꾸면 그 검증이 무효가 되고 사용자는 답이 나빠진 걸
> 스스로 확인할 수 없다. 메이크업 어댑터도 8B 위에서 학습돼 애초에 바꿀 수 없다.

### 전처리는 GPU 락 밖에서

```
업로드 → MediaPipe 크롭 [CPU 스레드풀, 중앙값 13ms]
           ├ 실패 → 재촬영 안내를 그대로 반환. GPU 안 씀
           └ 성공 → GPU 큐 [1장, 직렬] → 생성 → 새니타이즈
```

앞이 안 보이는 사용자는 프레이밍 실패로 재촬영을 여러 번 반복하는 것이 **정상 사용
패턴**이다. 전처리를 GPU 워커 안에 넣으면 어차피 실패할 요청이 GPU 슬롯을 잡아먹고,
그 왕복 지연이 체감 품질을 좌우한다.

재촬영 안내 문구는 가공하지 않고 그대로 나간다.

| `code` | 조건 | 사용자가 듣는 말 |
| --- | --- | --- |
| `too_dark` | 평균 밝기 40 미만 | 사진이 너무 어두워요. 밝은 곳에서 다시 찍어 주세요. |
| `no_face` | 얼굴 미검출 | 얼굴이 잘 안 보여요. 휴대폰을 한 뼘 정도 떨어뜨리고 정면으로 들어 주세요. |
| `lip_too_small` | 입술 폭 60px 미만 | 입술이 너무 작게 나왔어요. 조금 더 가까이 가져와 주세요. |
| `face_too_small` | 얼굴 폭 120px 미만 | 얼굴이 너무 작게 나왔어요. 조금 더 가까이 가져와 주세요. |

---

## 요구사항

| 항목 | 값 |
| --- | --- |
| GPU | NVIDIA, VRAM 10GB 이상 (RTX 4090에서 검증) |
| 디스크 | 베이스 가중치 약 16GB |
| 런타임 | Docker + NVIDIA Container Toolkit, 또는 Python 3.11 + CUDA 12.8 |

정확한 패키지 버전은 [`requirements.txt`](requirements.txt) 에 고정돼 있다.
`transformers` 가 5.x 라 4.x 기준 예제 코드는 맞지 않는다.

---

## 띄우기

```bash
docker compose up -d --build
docker compose logs -f
```

첫 실행은 **수 분** 걸린다 — 베이스 16GB를 받는다. 가중치는 이미지에 굽지 않고
`hf-cache` 볼륨에 둬서, 컨테이너를 다시 만들어도 다시 받지 않는다.

적재가 끝나기 전에는 `/health` 가 `loading` 이고 컨테이너는 unhealthy다. 정상이다
(`start-period` 15분).

### GPU 없이 (라우팅·전처리만)

```bash
docker compose -f docker-compose.yml -f docker-compose.mock.yml up -d --build
```

모델을 올리지 않고 뜬다. 라우팅 규칙과 크롭을 확인할 때 쓴다.

### Docker 없이

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m beautytalk.main
```

`torchvision` 은 선택이 아니다 — 프로세서가 비디오 프로세서를 함께 들고 있어서 없으면
로드 자체가 `ImportError` 로 죽는다.

---

## 확인하기

```bash
python selfcheck.py --image selfie.jpg
```

헬스체크 → 라우팅 6종 → 두 경로 실제 추론까지 돌리고 결과를 보여준다.
`--image` 를 빼면 라우팅까지만 본다. 사진은 **얼굴이 크게 나온 것**이어야 한다.

빠르게 눈으로만:

```bash
curl http://127.0.0.1:8100/health
```

```bash
curl -G --data-urlencode "q=립 어때" http://127.0.0.1:8100/route-test
```

테스트(GPU 불필요):

```bash
python -m pytest tests/ -q
```

---

## API

| 메서드 | 경로 | 용도 |
| --- | --- | --- |
| `WS` | `/ws/{session_id}` | **앱이 실제로 쓰는 경로** |
| `GET` | `/health` | 앱이 8100 → 8000 순으로 훑으며 본다 |
| `POST` | `/framing` | 찍기 전 프레이밍 판정 (GPU 안 씀, 수십 ms) |
| `POST` | `/warmup` | GPU 깨우기 |
| `POST` | `/analyze` | 한 방에 (curl 시험용) |
| `GET` | `/routes` | 경로별 어댑터·프롬프트 길이·few-shot 수 |
| `GET` | `/route-test?q=...` | 모델 없이 라우팅만 확인 |

실물 스키마는 [`openapi.json`](openapi.json).

### WebSocket

```jsonc
// 앱 → 서버
{ "type": "analyze", "request_id": "app-1",
  "question": "립 어때?", "image_b64": "...", "mirrored": false }

// 서버 → 앱
{ "type": "analysis_result", "request_id": "app-1",
  "status": "ok",              // ok | retake | error
  "message": "읽어줄 문장",
  "route": "lip", "action": "wipe", "prep_ms": 14, "infer_ms": 3120 }
```

앱은 `message` 만 읽는다. 나머지는 로그를 맞춰 볼 때 쓴다. `status` 가 `retake` 여도
`message` 는 그대로 읽어 주면 된다 — 재촬영 안내 문장이다.

**대기 시간은 앱이 서버보다 길어야 한다**(앱 70초 > 서버 60초). 앱이 먼저 포기하면 다 만든
답을 버리고, 그 요청이 GPU 한 자리를 계속 잡는다.

---

## 설정

전부 환경변수이고 기본값이 있다. 자주 건드리는 것만:

| 변수 | 기본 | 뜻 |
| --- | --- | --- |
| `BT_MIRRORED_DEFAULT` | `false` | 전면 카메라 미러 되돌리기 (아래 참고) |
| `BT_BASE_MODEL` | unsloth 4bit Qwen3-VL 8B | 베이스 체크포인트 |
| `BT_MAKEUP_ADAPTER` | 메이크업 LoRA | 어댑터 저장소 |
| `BT_LIP_ADAPTER` | (없음) | 립 경로에도 어댑터를 붙이고 싶을 때 |
| `BT_USE_UNSLOTH` | `true` | 로더 선택 (아래 참고) |
| `BT_MOCK` | `false` | 모델 없이 라우팅·전처리만 |
| `BT_PREPROCESS_WORKERS` | `3` | 전처리 스레드 (스레드당 검출기 1개) |
| `BT_MAX_NEW_TOKENS` | `200` | 생성 길이 |
| `BT_GENERATE_TIMEOUT` | `60` | 생성 제한(초) |

기본 로더는 **unsloth** 다. 이 체크포인트가 사전 양자화본이라 순정 `transformers` 로더로는
비전 타워의 4bit 상태가 제대로 붙지 않는다. `BT_USE_UNSLOTH=false` 로 순정 경로를 쓸 수는
있지만 검증된 조합이 아니다.

립 크롭 상수(`BT_LIP_PAD_X=0.50`, `BT_LIP_PAD_Y=0.70`, `BT_LIP_MIN_W=60`)는
**검증에 쓴 값이다.** 바꾸면 재측정 대상이다.

---

## 경로 바꾸기

### 프롬프트만 고칠 때

`beautytalk/assets/<경로>/system_prompt.txt` 를 고치고 재시작한다.
compose 가 assets 를 볼륨으로 붙여 놔서 **이미지 재빌드가 필요 없다.**

```bash
docker compose restart
```

### 트리거 단어를 바꿀 때

`beautytalk/router.py` 의 `LIP_KEYWORDS` / `LIP_NEGATIONS`. 바꿨으면 `pytest tests/` 를 돌린다 —
여기서 잘못 고르면 크롭도 프롬프트도 어댑터도 전부 어긋나는데, 사용자는 화면을 못 보므로
**조용히** 틀린다. 키워드에 오타 변형(입슬·입쑬·맆)이 들어 있는 것은 음성 인식 오인식 대응이다.

### 경로를 추가할 때

1. `beautytalk/assets/<이름>/system_prompt.txt` 작성
2. (선택) `assets/<이름>/fewshot/fewshot.json` — 있으면 자동으로 실린다
3. `beautytalk/routes.py` 의 `load()` 에 `Route(...)` 한 줄
4. `beautytalk/router.py` 에 트리거 단어

### few-shot 예시

립 경로의 3장은 **세트다.** 교체·추가·순서 변경 시 성능이 크게 흔들린다 —
예시 1장 교체로 정상 판별이 10/14 → 7/14 로 무너진 사례가 있다.

이미지 3장과 목록(`fewshot.json`)은 `beautytalk/assets/lip/fewshot/` 에 함께 들어 있다.
**실제 인물의 입술 사진이므로 저장소를 공개로 돌리기 전에 다시 판단할 것.**

---

## 미러링 — 배포 전 결정할 것

전면 카메라 프리뷰의 좌우 반전은 보통 **픽셀에 구워져** 저장되고 EXIF 에는 기록되지 않는다.
그래서 코드로는 알 수 없다.

틀리면 **좌우 안내가 통째로 뒤집힌다.** "오른쪽 입꼬리"라고 했는데 실제로는 왼쪽이다.
그리고 이건 사용자가 스스로 확인할 수 없는 종류의 오류다.

1. 한쪽 입꼬리에만 립을 번지게 하고 앱으로 찍는다
2. 답변의 좌우가 **본인 몸 기준**으로 맞는지 본다
3. 뒤집혀 있으면 `BT_MIRRORED_DEFAULT=true` 로 바꾸고 `docker compose up -d`

앱이 요청에 `mirrored` 를 실어 보내면 그 값이 이 기본값을 덮어쓴다.

---

## 구조

```
server/
├── beautytalk/
│   ├── main.py        FastAPI — WS·HTTP, 전처리풀/GPU풀 분리
│   ├── router.py      질문 → 경로  ← 여기서 틀리면 조용히 다 어긋난다
│   ├── routes.py      경로 정의 (전처리 + 프롬프트 + 어댑터)
│   ├── engine.py      베이스 1벌 + LoRA 핫스왑, GPU 직렬화
│   ├── vision.py      MediaPipe (스레드로컬) + 립/얼굴 크롭
│   ├── sanitize.py    TTS 안전 필터 + 폴백
│   ├── config.py      환경변수
│   └── assets/
│       ├── face_landmarker.task
│       ├── lip/       system_prompt.txt · fewshot/
│       └── makeup/    system_prompt.txt
├── tests/             라우팅·새니타이즈 (GPU 불필요)
├── selfcheck.py       배포 후 자가진단
├── Dockerfile · docker-compose.yml · docker-compose.mock.yml
└── requirements.txt   검증된 버전 고정
```

---

## 알려진 제약

- **메이크업 경로의 시스템 프롬프트가 임시본이다.** 어댑터 저장소에 학습에 쓴 프롬프트가
  들어 있지 않아, 립 프롬프트의 원칙을 얼굴 전체로 넓혀 새로 썼다. 원문을 받으면
  `assets/makeup/system_prompt.txt` 를 덮어써야 한다.
- **메이크업 경로의 얼굴 크롭 패딩(0.25)은 검증된 값이 아니다.** 립 크롭과 달리 기준
  데이터가 없다.
- **립 경로의 "덜 펴발림" 판정 4/11 은 알려진 한계다.** 프롬프트로는 상한이고 개선은
  파인튜닝 영역.
- **얼굴 사진 보존 정책이 아직 없다.** 지금은 아무것도 저장하지 않는다. 재측정하려면
  크롭을 남겨야 하고, 남기려면 동의가 필요하다.
- **동시 요청은 직렬 처리된다.** GPU 1장 기준 설계라 여러 명이 동시에 쓰면 순서대로 기다린다.
