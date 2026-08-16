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

```bash
pip install -r requirements-dev.txt
pytest

# 서버 띄운 뒤 E2E:
python scripts/test_ws_client.py --url ws://127.0.0.1:8000
python scripts/test_webrtc_client.py --http-url http://127.0.0.1:8000 --ws-url ws://127.0.0.1:8000
```

## 배포

`main` push → GitHub Actions가 Docker 이미지를 빌드/푸시 → 팀 노트북이 Tailscale 경유 SSH로
`docker compose pull && up -d` 실행 (`.github/workflows/deploy.yml`).

### WebRTC + Docker 주의사항

시그널링(HTTP/WS)은 8000/tcp 포트 매핑으로 문제없지만, **WebRTC 미디어는 UDP**라서
Linux 컨테이너(Windows 호스트) 안에서 돌면 answer SDP의 host candidate가 컨테이너 내부
IP(172.17.x.x)로 광고된다. 이 경우 연결은 컨테이너 → 폰 방향의 outbound UDP + peer-reflexive
승격에 의존하게 되어 환경에 따라 실패할 수 있다. 실기기 연동 전에 반드시 다른 Tailscale
노드에서 `scripts/test_webrtc_client.py`로 컨테이너 상대 E2E를 확인하고, 실패하면
(1) 서버를 호스트에서 직접 실행하거나, (2) 고정 UDP 포트 범위를 publish하거나,
(3) 컨테이너에 Tailscale을 넣는 방식으로 전환할 것.
