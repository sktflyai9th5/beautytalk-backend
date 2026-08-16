# BeautyTalk Backend API 명세서

시각장애인 메이크업 도우미 BeautyTalk의 백엔드 API.
카메라 영상/스냅샷을 받아 Qwen3-VL 비전 모델로 화장 상태를 분석하고,
TTS로 읽어줄 한국어 피드백 문장을 WebSocket으로 push한다.

- 기본 접속 (Tailscale 내부망): `http://100.91.201.104:8000` / `ws://100.91.201.104:8000`
- 검증 환경: Galaxy S24 실기기 + Ollama qwen3-vl:8b-instruct (RTX 4090, warm 분석 3~11초)

## 아키텍처

```
[Flutter 앱 / 브라우저]
  ── WS /ws/{session_id} 연결 (결과 push 채널 + 시그널링 + 트리거) ──▶ [백엔드]
  ── WebRTC video track (실시간 영상, UDP 가능 망)          ──────▶ [최신 프레임 1장 버퍼]
  ── analyze 트리거 (+ image_b64 스냅샷 폴백)              ──────▶ [프레임 선택 → Qwen3-VL 분석]
  ◀── analysis_started / analysis_result ─────────────────────── [백엔드]
```

프레임 소스 우선순위: **최근(3초 내) WebRTC 프레임 → 트리거에 첨부된 스냅샷 → 오래된 WebRTC 프레임**.
UDP가 막힌 망(AP 격리 등)에서는 스냅샷 폴백만으로 전체 기능이 동작한다.

---

## 1. HTTP API

### `GET /`
헬스 겸 서비스 식별. → `{"service": "beautytalk-backend", "status": "ok"}`

### `GET /health`
```json
{"status": "healthy", "analyzer": "qwen | mock", "active_sessions": 0,
 "webrtc_sessions": 0, "ws_connections": 0}
```

### `GET /test`
브라우저용 테스트 페이지. 카메라/합성영상 → WebRTC/WS 전체 흐름을 UI로 검증.

### `POST /webrtc/offer`
WebRTC 시그널링(HTTP 경로). WS 경로(`webrtc_offer` 메시지)와 동일 기능.

| 항목 | 내용 |
|---|---|
| 요청 | `{"session_id": "고유 문자열(1~128자)", "sdp": "...", "type": "offer"}` |
| 응답 200 | `{"session_id": "...", "sdp": "...", "type": "answer"}` |
| 409 | 시그널링 중 세션이 종료됨 (재연결 필요) |
| 503 | 동시 세션 수 상한(`MAX_SESSIONS`) 초과 |
| 422 | 스키마 오류 (type은 "offer"만 허용) |

같은 `session_id`로 재요청하면 기존 피어를 정리하고 교체한다(재시그널링).
ICE는 host candidate만 사용 (`iceServers: []` 권장, Tailscale/LAN 내부망 전제).

---

## 2. WebSocket API — `WS /ws/{session_id}`

연결 유지형 채널. 같은 `session_id`의 WebRTC 피어와 매핑된다.
연결 직후 서버가 `connected`를 보낸다. 메시지는 모두 JSON 텍스트.

### 2-1. 클라이언트 → 서버

**ping**
```json
{"type": "ping"}
```

**analyze — 분석 트리거** (WebRTC data channel로 보내도 동일 동작)
```json
{
  "type": "analyze",
  "question": "지금 립 어때?",          // 생략 시 기본 질문. 사용자 발화 원문 권장
  "request_id": "선택. 미지정 시 서버가 생성",
  "image_b64": "선택. 트리거 순간 카메라 스냅샷(JPEG). data URL 또는 순수 base64, 8MB 이하"
}
```
- `question`의 키워드(립/입술, 눈/아이, 볼/블러셔)로 분석 부위가 정해진다.
- 분석이 이미 진행 중이면 즉시 `analysis_result(status=error, message="아직 이전 분석이 진행 중이에요...")`로 거절된다.

**webrtc_offer — WS 경유 시그널링** (file:// WebView처럼 fetch가 막힌 환경용)
```json
{"type": "webrtc_offer", "sdp": "...", "sdp_type": "offer"}
```

### 2-2. 서버 → 클라이언트

| type | 시점 | 필드 |
|---|---|---|
| `connected` | WS 연결 직후 | `session_id`, `timestamp` |
| `pong` | ping 수신 시 | `timestamp` |
| `webrtc_answer` | webrtc_offer 처리 후 | `session_id`, `sdp`, `sdp_type`, `timestamp` |
| `analysis_started` | 분석 시작 | `session_id`, `request_id`, `timestamp` |
| `analysis_result` | 분석 완료/실패 | 아래 참조 |
| `error` | 잘못된 메시지 등 | `message`, `timestamp` |

**analysis_result — 핵심 응답** (`message`는 앱이 그대로 TTS로 읽는다)
```json
{
  "type": "analysis_result",
  "session_id": "...",
  "request_id": "...",
  "status": "ok | no_face | error",
  "region": "lips | eyes | cheeks | overall",
  "verdict": "perfect | needs_fix | bad_color",   // status가 ok가 아니면 null
  "message": "입술의 색상과 윤곽이 자연스럽고 완벽해요.",
  "detail": {"issues": [{"area": "...", "problem": "...", "suggestion": "..."}]},
  "timestamp": "2026-08-16T13:03:31.051+00:00"
}
```
- `status=no_face`: 얼굴이 프레임에 없음. `message`에 카메라 위치 안내가 담긴다 → 앱은 안내 후 재시도 유도.
- `status=error`: 프레임 없음/타임아웃/모델 오류/분석 중 중복 트리거. `message`는 항상 사용자에게 읽어줄 수 있는 문장.
- **클라이언트 파서는 `verdict`를 nullable로 처리해야 한다.**

### 2-3. 세션 수명 규칙

- WS 연결 종료 → 세션 전체 정리 (WebRTC 피어 포함)
- WebRTC 실패/종료 → **WS가 살아있으면 피어만 정리**하고 세션 유지 (스냅샷 폴백 계속 동작)
- 유휴 세션(기본 300초 무활동) 자동 회수, 동시 세션 상한 기본 50 (초과 시 offer 503, WS 1013)

---

## 3. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `QWEN_API_URL` | (없음) | OpenAI 호환 API base URL. 예: `http://127.0.0.1:11434/v1` (Ollama) |
| `QWEN_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | 모델명. **Ollama는 `qwen3-vl:8b-instruct`** (기본 `8b` 태그는 thinking 전용이라 빈 응답) |
| `QWEN_API_KEY` | `EMPTY` | 로컬 서버는 불필요 |
| `ANALYZER_MOCK` | 자동 | `true`=mock. 미설정+URL 없음이면 자동 mock(기동 시 WARNING 로그) |
| `ANALYSIS_TIMEOUT` | `60` | 분석 1회 제한(초). 모델 콜드 로드 감안 시 120 권장 |
| `SESSION_IDLE_TIMEOUT` | `300` | 유휴 세션 회수 기준(초) |
| `MAX_SESSIONS` | `50` | 동시 세션 상한 |
| `DEBUG_SAVE_FRAMES` | `false` | `true`면 분석 프레임을 `debug_frames/`에 JPEG 저장 |
| `QWEN_NO_THINK` | `true` | Qwen3 thinking 억제 프롬프트 추가 |

---

## 4. 테스트 명령어 리스트 (전부 실행 검증됨)

팀 노트북 기준. venv 활성화 대신 venv python 직접 호출(실행 정책 무관):

```powershell
cd C:\portable\beautytalk-backend

# 1) 환경/버전 검증 (Python·패키지 버전, 의존성 충돌, 핵심 API 실호출)
.\.venv\Scripts\python.exe scripts\check_env.py

# 2) 단위/통합 테스트 (55개)
.\.venv\Scripts\python.exe -m pytest -q

# 3) 서버 실행 - mock 모드 (모델 없이)
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8100

# 3') 서버 실행 - 실제 모델 (Ollama 필요: ollama pull qwen3-vl:8b-instruct)
$env:ANALYZER_MOCK='false'; $env:QWEN_API_URL='http://127.0.0.1:11434/v1'
$env:QWEN_MODEL='qwen3-vl:8b-instruct'; $env:ANALYSIS_TIMEOUT='120'; $env:DEBUG_SAVE_FRAMES='true'
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8100

# 4) 헬스체크
curl http://127.0.0.1:8100/health

# 5) WebSocket E2E (접속-ping-트리거-결과 수신)
.\.venv\Scripts\python.exe scripts\test_ws_client.py --url ws://127.0.0.1:8100

# 6) WebRTC 전체 E2E (합성영상 송신-시그널링-프레임 버퍼-양경로 트리거-결과)
.\.venv\Scripts\python.exe scripts\test_webrtc_client.py --http-url http://127.0.0.1:8100 --ws-url ws://127.0.0.1:8100
# 실제 모델 서버 대상이면 --real-model 추가 (합성영상엔 no_face가 정답)

# 7) 이미지 파일을 모델에 직접 분석 (WebRTC 없이 모델 품질 검증)
$env:QWEN_API_URL='http://127.0.0.1:11434/v1'; $env:QWEN_MODEL='qwen3-vl:8b-instruct'
.\.venv\Scripts\python.exe scripts\test_model_direct.py 사진.jpg --question "립 봐줘잉"

# 8) 브라우저 테스트: http://<서버IP>:8100/test  (합성영상 모드는 어디서나 동작)

# 9) 실시간 로그 모니터링 (별도 창)
Get-Content logs\server.log -Wait -Tail 5 -Encoding utf8 |
  Select-String "analyze_trigger|analysis_done|frame_source|debug_frame|webrtc"
```

앱(Flutter) 빌드/폰 테스트 절차는 앱 레포의 `TESTING.md` 참고.

---

## 5. 배포 노트

- `main` push → GitHub Actions → Docker 이미지 빌드/푸시 → 팀 노트북에 SSH 자동 배포(포트 8000).
- **컨테이너에서 실제 모델을 쓰려면** compose에 env 필요 (없으면 자동 mock 모드로 뜸):
  ```yaml
  environment:
    - ANALYZER_MOCK=false
    - QWEN_API_URL=http://host.docker.internal:11434/v1   # 호스트의 Ollama
    - QWEN_MODEL=qwen3-vl:8b-instruct
    - ANALYSIS_TIMEOUT=120
  ```
- **WebRTC 실시간 영상은 컨테이너 배포에서 제약이 있음** (README "WebRTC + Docker 주의사항").
  스냅샷 폴백(WS)은 컨테이너에서도 완전 동작하므로 현재 앱 플로우는 8000 배포만으로 사용 가능.
