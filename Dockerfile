# BeautyTalk 백엔드 (모델 라우터) — CUDA 12.8 / Python 3.11
#
# 베이스 이미지는 런타임만 담는다(devel 은 6GB 더 크고 컴파일러가 필요 없다).
# 모델 가중치는 이미지에 굽지 않는다 — HF 캐시를 볼륨으로 붙여 재빌드마다
# 16GB 를 다시 받지 않게 한다.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/hf

# python3.11 + mediapipe 가 요구하는 런타임 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        libgl1 libglib2.0-0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 먼저 — 레이어를 나눠 두면 requirements 만 바뀔 때 3GB 재설치를 피한다.
# torchvision 은 선택이 아니다 — Qwen3VLProcessor 가 비디오 프로세서를 함께 들고
# 있어서, 없으면 프로세서 로드 자체가 ImportError 로 죽는다.
RUN pip install --no-cache-dir torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY beautytalk/ ./beautytalk/

EXPOSE 8100

# 모델 적재(수 분) 동안은 unhealthy 로 두고, 준비되면 healthy 로 바뀐다.
# start-period 를 넉넉히 주지 않으면 적재 중에 컨테이너가 죽는다.
HEALTHCHECK --interval=20s --timeout=5s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request,json,sys; \
        d=json.load(urllib.request.urlopen('http://127.0.0.1:8100/health',timeout=4)); \
        sys.exit(0 if d.get('status')=='healthy' else 1)"

CMD ["python", "-m", "beautytalk.main"]
