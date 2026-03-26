FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
# Install uv binary directly from official image
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# ── System packages ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --fix-missing \
        build-essential clang git wget curl ca-certificates ninja-build unzip ffmpeg colmap \
        libstdc++6 xvfb \
        # OpenGL / display stack
        libglm-dev libgl1-mesa-glx libgl1-mesa-dri libgles2-mesa \
        libglvnd0 libglvnd-dev libgl1 libglx0 libegl1 libnvidia-gl-470-server\
        libglib2.0-0 libsm6 libxrender1 libxext6 libx11-6 \
        # PyBullet / Open3D GUI dependencies
        freeglut3-dev libglu1-mesa libxi-dev libxmu-dev \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA / OpenGL / CUDA env
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYOPENGL_PLATFORM=egl

ENV CUDA_HOME=/usr/local/cuda-11.7
ENV PATH=${CUDA_HOME}/bin:/app/.venv/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CUDAHOSTCXX=/usr/bin/g++
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

RUN ln -sfn /usr/local/cuda-11.7 /usr/local/cuda && \
    mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

# ── Create Python 3.10 virtual environment with uv ──────────
WORKDIR /app

# Copy only dependency manifests first if available
COPY . /app/

# ── Python 3.9 + local project venv (standard style) ───────
RUN uv python install 3.9 \
    && uv venv --python 3.9 /app/.venv

    # Standard venv activation style for Docker
ENV PATH="/app/.venv/bin:${PATH}"

# ── Bootstrap core Python deps (cache-friendly grouped install) ─────────────
RUN uv pip install --no-cache-dir \
    torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 && \
    uv pip install --no-cache-dir \
    "setuptools==61.1.0" wheel "packaging==23.2" \
    dotmap pyhocon chardet opencv-python-headless gpustat ipdb sentencepiece \
    xformers==0.0.18 && \
    uv pip install --no-cache-dir "numpy==1.23.5" && \
    uv pip install --no-cache-dir -r requirements.txt

RUN uv pip install --no-cache-dir torchmetrics==0.6.0

# The codebase (and some third_party deps) import the legacy `pytorch_lightning.metrics.*` API.
# Pin a compatible PyTorch Lightning 1.x to match torchmetrics==0.6.0.
RUN uv pip install --no-cache-dir --no-build-isolation "pytorch-lightning==1.5.10"

# visdom setup.py imports pkg_resources; avoid isolated build env surprises
RUN uv pip install --no-cache-dir --no-build-isolation visdom

# 1. Install pytorch3d dependencies (as in instructions)
# Clone first to keep sources in a separate layer
RUN git clone https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d

# Ensure packaging/setuptools/wheel are recent (fixes metadata build issues)
# RUN conda install -n manigaussian -y -c fvcore -c iopath -c conda-forge fvcore iopath
RUN uv pip install --no-cache-dir --no-build-isolation /tmp/pytorch3d

# 2. Install CLIP and open-clip
RUN git clone https://github.com/openai/CLIP.git /tmp/CLIP
RUN uv pip install --no-cache-dir -e /tmp/CLIP
RUN uv pip install --no-cache-dir open-clip-torch

# 3. Download coppeliasim (for PyRep and RLBench) --- IGNORE ---
# --- CoppeliaSim must exist before building PyRep ---
RUN tar -xvf /app/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C /opt \
    && ln -sfn /opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 /opt/CoppeliaSim

ENV COPPELIASIM_ROOT=/opt/CoppeliaSim
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}
ENV QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}

# Install detectron2 + ODISE
RUN git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2
RUN uv pip install --no-cache-dir --no-build-isolation /tmp/detectron2
RUN uv pip install --no-cache-dir --no-build-isolation /app/third_party/ODISE

# Install third-party local packages
RUN uv pip install --no-cache-dir --no-build-isolation -r /app/third_party/PyRep/requirements.txt
# RUN uv pip install --no-cache-dir cffi
RUN uv pip install --no-cache-dir --no-build-isolation /app/third_party/PyRep
RUN uv pip install --no-cache-dir --no-build-isolation -r /app/third_party/RLBench/requirements.txt
RUN uv pip install --no-cache-dir --no-build-isolation /app/third_party/RLBench
RUN uv pip install --no-cache-dir --no-build-isolation -r /app/third_party/YARR/requirements.txt
RUN uv pip install --no-cache-dir --no-build-isolation /app/third_party/YARR

# Fix some possible problems
RUN uv pip install --no-cache-dir opencv-python-headless
RUN uv pip install --no-cache-dir "numpy==1.23.5"

# Install gaussian splatting renderer submodules
RUN uv pip install --no-cache-dir --no-build-isolation /app/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
RUN uv pip install --no-cache-dir --no-build-isolation /app/third_party/gaussian-splatting/submodules/simple-knn

# Install Lightning Fabric
# NOTE: Do not install `lightning==2.x` here; it can conflict with the legacy `pytorch-lightning` API
# used by repo dependencies (e.g. `pytorch_lightning.metrics`).
# RUN uv pip install --no-cache-dir --no-build-isolation lightning==2.0.9.post0
RUN uv pip install --no-cache-dir --no-build-isolation transformers==4.30.2
RUN uv pip install --no-cache-dir --no-build-isolation wandb
# Use the project venv by default in shell sessions
RUN echo "export PATH=/app/.venv/bin:\$PATH" >> /root/.bashrc

CMD ["/bin/bash", "-lc", "python -V && nvidia-smi || true && bash"]
