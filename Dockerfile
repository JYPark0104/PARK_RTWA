# ============================================================================
# Dockerfile — twin_minji_verPARK (옵션 A: 컨테이너 1개 + venv 여러 개)
# ----------------------------------------------------------------------------
# 목적   : 호스트(dclserver78)에서 venv 여러 개로 나뉘어 돌던 환경을, OS/CUDA/
#          시스템 계층은 컨테이너로 고정하고 그 안에 venv 를 그대로 재현하여
#          twin_minji_verPARK '전체' 와 호환되게 한다.
#          + PARK_2_RayTracing_Agent_v2 의 RT/OptiX·H100 안정화 설정 승계.
#
# 왜 멀티 venv 인가 (호환성 검토 결과)
#   본 워크스페이스는 단일 환경이 아니며 상호 충돌하는 스택을 포함:
#     - venv-webagent : TF 2.21 / keras3 / numpy 2.2 / sionna 1.2.2  (torch 없음)
#                       → 260528 Web_Agent + P1A~P1Q 파이프라인 + 구 스냅샷
#     - venv-torch    : torch 2.12+cu130 / scikit-learn / seaborn / numpy 2.2
#                       → 260604 metric 도출 / 260606 test / 260428 VAE
#   TF(cu12) 와 torch(cu130) 는 각자 CUDA 사용자 라이브러리를 휠에 포함하므로
#   한 컨테이너 안에서 venv 로 분리하면 안전하게 공존한다(libcuda 만 드라이버 공유).
#
# 실행 환경 (작성 기준, 2026-06-06)
#   - 호스트 : dclserver78 (165.132.192.78), NVIDIA H100 NVL x2 (sm_90),
#              드라이버 580.159.03 (CUDA 13 지원)
#   - Python : 3.10
#   - Node   : 20 LTS (Web_Agent 프런트엔드)
# GPU 전제 : 런타임에 nvidia-container-toolkit + `--gpus all` 로 libcuda 주입.
#            OptiX(libnvoptix.so.1) 는 NVIDIA_DRIVER_CAPABILITIES=all 일 때 주입.
# ============================================================================

# ── 베이스: CUDA 12.3 + cuDNN9 devel (PARK_2 승계, OptiX/Mitsuba RT 지원) ─────
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Seoul

# ── 1) 시스템 패키지 ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        llvm \
        libgl1 \
        libegl1 \
        libgomp1 \
        libglib2.0-0 \
        libx11-6 \
        libx11-dev \
        libxrender1 \
        libxext6 \
        git \
        curl \
        wget \
        htop \
        net-tools \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ── 2) Node.js 20 LTS (Web_Agent 프런트엔드 빌드 / dev) ──────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    node --version && npm --version

# ── 3) venv 들의 공통 위치 ───────────────────────────────────────────────────
#   /opt/venvs/webagent : TF/sionna 스택
#   /opt/venvs/torch    : torch/sklearn 스택
ENV VENV_ROOT=/opt/venvs
RUN mkdir -p ${VENV_ROOT}

# ── 4) venv-webagent (TF 2.21 / sionna 1.2.2 / numpy 2.2) ────────────────────
#   requirements-docker.txt = 호스트 .venv-webagent 의 pip freeze 고정본
#   (tensorflow 만 [and-cuda] 치환 → CUDA 사용자 라이브러리 포함)
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m venv ${VENV_ROOT}/webagent && \
    ${VENV_ROOT}/webagent/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    ${VENV_ROOT}/webagent/bin/pip install --no-cache-dir -r /tmp/requirements-docker.txt

# ── 5) venv-torch (torch 2.12+cu130 / scikit-learn / seaborn) ────────────────
#   torch 는 cu130 인덱스에서, 나머지는 PyPI 에서 설치.
COPY requirements-torch.txt /tmp/requirements-torch.txt
RUN python -m venv ${VENV_ROOT}/torch && \
    ${VENV_ROOT}/torch/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    ${VENV_ROOT}/torch/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.12.0 && \
    ${VENV_ROOT}/torch/bin/pip install --no-cache-dir -r /tmp/requirements-torch.txt

# ── 6) GPU / RT 환경 변수 (PARK_2 승계) ──────────────────────────────────────
#   런타임 스레드/단일·멀티GPU/NCCL 데드락 방지 변수는 devcontainer.json 에서 설정.
ENV WEBAGENT_USE_GPU=1 \
    WEBAGENT_USE_GPU_RT=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# ── 7) 작업 디렉토리 ─────────────────────────────────────────────────────────
#   코드/데이터는 빌드 시 COPY 하지 않고 런타임에 /workspace 로 마운트.
WORKDIR /workspace

# 포트: 8800(백엔드) / 5173(Vite dev) / 8000(빌드 통합) / 7651(PARK_2 RT 승계)
EXPOSE 8800 5173 8000 7651

# 사용 예 (컨테이너 안)
#   Web_Agent 백엔드 : VENV_DIR=/opt/venvs/webagent bash "260528 Web_Agent/scripts/run_gpu_backend.sh"
#   torch 계열 실행   : /opt/venvs/torch/bin/python "260606 test/P2G_DistTerms_2606v1.py"
CMD ["/bin/bash"]
