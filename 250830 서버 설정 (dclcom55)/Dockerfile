FROM tensorflow/tensorflow:2.17.0-gpu-jupyter
# 안되면 2.18.0-gpu-jupyter
# 기본 개발 도구 및 라이브러리 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    llvm \
    libegl1 \
    libgl1 \
    libgomp1 \
    libx11-6 \
    libx11-dev \
    libffi-dev \
    git \
    wget \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# RUN pip install --upgrade sionna
RUN pip install sionna==1.0.2
# RUN pip install --upgrade pre sionna

RUN pip uninstall -y tensorflow tensorflow-gpu
RUN pip install tensorflow==2.17.0

# 패키지 설치

RUN pip install --upgrade  --ignore-installed \
    ipykernel \
    jupyterlab \
    geopandas \
    shapely \
    open3d \
    #open3d-cpu \
    contextily \
    tensorboard-plugin-profile \
    cupy-cuda12x

# numpy and scipy version control for compatibility (250726)
RUN pip uninstall -y numpy scipy protobuf
RUN pip install numpy==1.24.3
RUN pip install scipy==1.14.1 --no-deps
RUN pip install protobuf==4.25.3 --no-deps
RUN pip install seaborn==0.12.2

# --- VS Code Server pre-installation ---
# devcontainer.json에서 빌드 인자로 받은 VS Code Server 버전을 사용합니다.
ARG VSCODE_SERVER_COMMIT
# 'vscode-server-linux-x64.tar.gz' 파일이 프로젝트 루트에 있다고 가정합니다.
COPY vscode-server-linux-x64.tar.gz /tmp/
# 서버를 설치할 디렉토리를 만들고 압축을 해제합니다.
RUN if [ -n "$VSCODE_SERVER_COMMIT" ]; then \
    mkdir -p /root/.vscode-server/bin/$VSCODE_SERVER_COMMIT && \
    tar -xzf /tmp/vscode-server-linux-x64.tar.gz -C /root/.vscode-server/bin/$VSCODE_SERVER_COMMIT --strip-components 1 && \
    rm /tmp/vscode-server-linux-x64.tar.gz; \
    fi
# --- End of VS Code Server pre-installation ---

#RUN pip install -U tensorboard-nightly

# 작업 디렉토리 설정
WORKDIR /workspace

# 컨테이너 실행 시 실행할 명령어
CMD ["jupyter", "lab", "--port=8889", "--no-browser", "--NotebookApp.token=''", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.allow_origin='*'"]