# beautytalk-backend

시각장애인 메이크업 도우미 BeautyTalk의 백엔드.
Flutter 앱이 WebRTC로 보내는 카메라 영상을 수신하고, 음성 트리거가 오면
최신 프레임을 Qwen3-VL로 분석해 WebSocket으로 한국어 피드백을 push한다.

```
[Flutter 앱]
  ── (a) POST /webrtc/offer 시그널링 ──────────────▶ [백엔드]
  ── (b) WebRTC video track (카메라 영상) ─────────▶ [백엔드: 최신 프레임 1장 버퍼]
  ── (c) analyze 트리거 (data channel 또는 WS) ────▶ [백엔드: 프레임 캡처 → Qwen3 분석]
  ◀── (d) WS /ws/{session_id} 로 analysis_result ── [백엔드]
```

## 실행

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Tailscale 내부망 기준 접속 주소:
- HTTP: `http://100.91.201.104:8000`
- WebSocket: `ws://100.91.201.104:8000/ws/{session_id}`

### 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `QWEN_API_URL` | (없음) | OpenAI 호환 API base URL (예: `http://127.0.0.1:8001/v1`) |
| `QWEN_API_KEY` | `EMPTY` | API 키 (로컬 vLLM은 보통 불필요) |
| `QWEN_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | 모델 이름 |
| `ANALYZER_MOCK` | 자동 | `true`면 mock 분석기. 미설정 시 `QWEN_API_URL` 없으면 자동 mock |
| `SESSION_IDLE_TIMEOUT` | `300` | 유휴 세션 정리 기준(초) |
| `ANALYSIS_TIMEOUT` | `60` | 분석 1회 제한 시간(초) |
| `MAX_SESSIONS` | `50` | 동시 세션 수 상한 (초과 시 offer는 503, WS는 1013으로 거절) |

> `ANALYZER_MOCK`을 설정하지 않고 `QWEN_API_URL`도 없으면 mock 모드로 자동 기동하며
> 시작 로그에 `analyzer_mock_auto_enabled` WARNING이 남는다. 실서비스 배포 시에는
> compose에 `QWEN_API_URL`(+`ANALYZER_MOCK=false`)을 반드시 넣을 것.

추가 env: `DEBUG_SAVE_FRAMES=true`(분석 프레임을 `debug_frames/`에 저장),
`QWEN_NO_THINK`(기본 true, Qwen3 thinking 억제 시도).

### 실제 모델 실행 (Ollama, 팀 노트북 RTX 4090에서 검증됨)

```powershell
ollama pull qwen3-vl:8b-instruct   # thinking 없는 instruct 버전을 쓸 것!
```

> 주의: `qwen3-vl:8b`(기본 태그)은 thinking 모델이라 응답이 수십 초 걸리고
> max_tokens를 thinking이 전부 소모해 빈 응답이 온다. 반드시 `-instruct` 태그 사용.
> 검증 결과 warm 상태 분석 속도 약 3초.

서버 실행 (mock 대신 실제 모델):

```powershell
$env:ANALYZER_MOCK='false'
$env:QWEN_API_URL='http://127.0.0.1:11434/v1'
$env:QWEN_MODEL='qwen3-vl:8b-instruct'
$env:DEBUG_SAVE_FRAMES='true'
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

모델만 따로 검증: `python scripts\test_model_direct.py 사진.jpg --question "립 봐줘잉"`
파이프라인 E2E(실모델): `python scripts\test_webrtc_client.py ... --real-model`

## API

### `POST /webrtc/offer`

WebRTC 시그널링. 앱이 offer SDP를 보내면 answer SDP를 돌려주고 세션을 만든다.
Tailscale 내부망 전제로 ICE 서버 없이 host candidate만 사용한다 (앱도 `iceServers: []` 권장).

```json
// 요청
{"session_id": "any-unique-id", "sdp": "...", "type": "offer"}
// 응답
{"session_id": "any-unique-id", "sdp": "...", "type": "answer"}
```

앱은 video track을 추가해서 offer를 만들고, 트리거용 data channel(`control` 등 라벨 자유)을 함께 열 수 있다.

### `WS /ws/{session_id}`

분석 결과 push 채널. offer와 같은 `session_id`를 쓰면 WebRTC 피어와 매핑된다.
연결 직후 서버가 `{"type": "connected", ...}`를 보낸다.

앱 → 서버 (WebSocket 또는 data channel, 동일 JSON):

```json
{"type": "ping"}                                            // → {"type": "pong"}
{"type": "analyze", "question": "립 봐줘잉", "request_id": "선택"}
```

서버 → 앱:

```json
{"type": "analysis_started", "session_id": "...", "request_id": "...", "timestamp": "..."}
{
  "type": "analysis_result",
  "session_id": "...",
  "request_id": "...",
  "status": "ok | no_face | error",
  "region": "lips | eyes | cheeks | overall",
  "verdict": "perfect | needs_fix | bad_color",
  "message": "TTS로 그대로 읽어줄 한국어 피드백 문장",
  "detail": {"issues": [{"area": "...", "problem": "...", "suggestion": "..."}]},
  "timestamp": "ISO8601"
}
```

`message`는 가장 중요한 문제 하나만 다루는 1~2문장이므로 앱이 그대로 TTS로 읽으면 된다.

주의: `status`가 `error`거나 `no_face`면 `verdict`는 `null`이다. 앱 파서는 verdict를
nullable로 처리해야 한다. 분석이 이미 진행 중일 때 온 트리거는
`status: "error"` + "아직 이전 분석이 진행 중이에요..." 메시지로 즉시 거절된다.

### `GET /health`

```json
{"status": "healthy", "analyzer": "mock", "active_sessions": 0, "webrtc_sessions": 0, "ws_connections": 0}
```

## 테스트

패키지는 전부 `.venv` 안에 있다. PowerShell 실행 정책 때문에 `Activate.ps1`이 막히면
(기본 Restricted 정책의 `PSSecurityException`) **활성화 없이 venv python을 직접 호출**하면 된다:

```powershell
cd C:\portable\beautytalk-backend
# venv가 없으면: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

.\.venv\Scripts\python.exe scripts\check_env.py     # 버전/호환성 검증 (로컬·컨테이너 공용)
.\.venv\Scripts\python.exe -m pytest -q

# 서버 실행:
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000

# 서버 띄운 뒤 E2E (별도 창):
.\.venv\Scripts\python.exe scripts\test_ws_client.py --url ws://127.0.0.1:8000
.\.venv\Scripts\python.exe scripts\test_webrtc_client.py --http-url http://127.0.0.1:8000 --ws-url ws://127.0.0.1:8000
```

굳이 activate를 쓰고 싶으면 현재 창에서만 정책을 풀거나(`Set-ExecutionPolicy -Scope Process Bypass`)
cmd에서 `.venv\Scripts\activate.bat`를 사용.

### 웹 테스트 페이지 (`GET /test`)

앱 완성 전 브라우저만으로 전체 흐름(WebRTC 영상 송신 → 트리거 → 결과 수신/TTS)을
검증하는 페이지. 다른 노트북(같은 Tailscale 네트워크)에서
`http://100.91.201.104:<port>/test` 접속 → "합성 영상으로 시작" → "립 봐줘" 클릭.

- **합성 영상 모드**: 어디서나 동작 (plain http에서도 가능)
- **카메라 모드**: plain http에서는 브라우저가 getUserMedia를 차단하므로 크롬
  `chrome://flags/#unsafely-treat-insecure-origin-as-secure`에 주소 등록 필요
- 브라우저는 카메라 권한이 없으면 ICE candidate를 mDNS(.local)로 익명화한다.
  이 경우에도 **서버가 호스트에서 직접 실행 중이면** 서버 candidate(Tailscale IP)로
  연결이 성립한다. Docker 컨테이너 서버로는 브라우저 WebRTC가 실패한다 (아래 참고).

## 배포

`main` push → GitHub Actions가 Docker 이미지를 빌드/푸시 → 팀 노트북이 Tailscale 경유 SSH로
`docker compose pull && up -d` 실행 (`.github/workflows/deploy.yml`).

### WebRTC + Docker 주의사항 (2026-08-16 실측)

시그널링(HTTP/WS)은 8000/tcp 포트 매핑으로 문제없지만, **WebRTC 미디어는 UDP**라서
Linux 컨테이너(Windows 호스트) 안에서 돌면 answer SDP의 host candidate가 컨테이너 내부
IP(172.17.x.x)로 광고된다. 실측 결과:

- aiortc 파이썬 클라이언트(실제 IP candidate 사용) ↔ 컨테이너: **연결 성공**
  (컨테이너 outbound UDP + peer-reflexive 승격)
- **브라우저 ↔ 컨테이너: 연결 실패** (`connectionState=failed`). 브라우저는 카메라 권한
  없이는 candidate를 mDNS로 익명화하는데 컨테이너가 mDNS를 못 풀어 양방향 모두 도달 불가
- **브라우저 ↔ 호스트 직접 실행 서버: 연결 성공** (서버 candidate가 실제 Tailscale IP)

결론: **WebRTC를 쓰는 한 서버는 호스트에서 직접 실행하는 것을 권장**한다. 컨테이너 배포를
유지하려면 컨테이너에 Tailscale을 넣거나(sidecar) 고정 UDP 포트 publish + candidate 조정이
필요하다. 앱(flutter_webrtc, 카메라 권한 보유)은 실제 IP candidate를 쓰므로 컨테이너와
연결될 가능성이 있으나, 실기기 검증 전까지 보장하지 말 것.

외부 접근 시 방화벽: venv python(`.venv\Scripts\python.exe`)에 대한 인바운드 허용 규칙이
필요할 수 있다 (기존 규칙은 시스템 python 경로에만 존재). 관리자 PowerShell에서:

```powershell
New-NetFirewallRule -DisplayName "BeautyTalk backend" -Direction Inbound -Action Allow -Program "C:\portable\beautytalk-backend\.venv\Scripts\python.exe"
```
