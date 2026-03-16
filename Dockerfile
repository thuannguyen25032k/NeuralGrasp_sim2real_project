# ============================================================
# NeuralGrasp Docker Image (uv version)
# Based on CUDA 11.6 + Ubuntu 20.04 to support gcc-10 and the
# CUDA extensions required by diff-gaussian-rasterization and
# simple-knn (which break on Ubuntu 22.04+ / gcc-12+).
# ============================================================
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Install uv binary directly from official image
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# ── System packages ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y \
        build-essential git wget curl ca-certificates ninja-build unzip ffmpeg colmap \
        # OpenGL / display stack
        libgl1-mesa-glx libgl1-mesa-dri libgles2-mesa \
        libglvnd0 libglvnd-dev libgl1 libglx0 libegl1 \
        libglib2.0-0 libsm6 libxrender1 libxext6 libx11-6 \
        # PyBullet / Open3D GUI dependencies
        freeglut3-dev libglu1-mesa libxi-dev libxmu-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tell NVIDIA container toolkit to expose graphics + display capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
# Use EGL (offscreen) as fallback when no X display is available
ENV PYOPENGL_PLATFORM=egl

# ── Create Python 3.10 virtual environment with uv ──────────
WORKDIR /app
COPY ./ ./

# ── Python 3.10 + local project venv (standard style) ───────
RUN uv python install 3.10 \
    && uv venv --python 3.10 /app/.venv

# Standard venv activation style for Docker
ENV PATH="/app/.venv/bin:${PATH}"

# ── Bootstrap packaging tools ────────────────────────────────
RUN uv pip install --upgrade pip setuptools wheel packaging

# ── NumPy pin early to avoid torch ABI issues during build ───
RUN uv pip install "numpy<2"

# ── PyTorch (CUDA 11.6 build) ────────────────────────────────
RUN uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu116 \
    torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1

# ── Patch torch 1.13's cpp_extension.py ──────────────────────
RUN TORCH_DIR=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))") \
    && sed -i \
       's/from pkg_resources import packaging/import packaging/' \
       "${TORCH_DIR}/utils/cpp_extension.py"

# ── CUDA arch flags (covers Turing / Ampere / older cards) ───
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV FORCE_CUDA=1

# Force GCC/CUDA toolchain for torch CUDA extensions
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ── CUDA extension submodules ─────────────────────────────────
RUN uv pip install --no-build-isolation \
        submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/diff-gaussian-rasterization

RUN uv pip install --no-build-isolation \
        submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/simple-knn \
        submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/fused-ssim

# ── Pure-Python / easy submodules ────────────────────────────
RUN uv pip install \
        submodules/pybullet-URDF-models \
        submodules/pybullet-playground-wrapper/ \
        submodules/ghalton

RUN uv pip install -e submodules/gaussian-splatting-wrapper
RUN uv pip install -e submodules/gello_software

# Skip hardware/GUI-only packages that can't build headless in Docker:
#   PyQt6, pyrealsense2, ur-rtde, pure-python-adb, xarm, xarm-python-sdk, pyspacemouse
RUN grep -vE '^\s*(PyQt6|pyrealsense2|ur-rtde|pure-python-adb|xarm|xarm-python-sdk|pyspacemouse)' \
        submodules/gello_software/requirements.txt \
    > /tmp/gello_requirements.filtered.txt \
    && uv pip install -r /tmp/gello_requirements.filtered.txt

RUN uv pip install -e submodules/gello_software/third_party/DynamixelSDK/python

RUN uv pip install --no-build-isolation submodules/simple-knn/

# ── Project requirements & package ───────────────────────────
RUN uv pip install -r requirements.txt

# ── Final numpy guard ─────────────────────────────────────────
RUN uv pip install --force-reinstall "numpy<2"

# ── Default entry-point ───────────────────────────────────────
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["bash"]