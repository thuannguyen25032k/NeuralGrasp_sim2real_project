# Installation

The following guidance works well for a machine with 4090 GPU | cuda 11.7 | ubuntu 22.04, a machine with 3090 GPU | cuda 11.6 | ubuntu 20.04, a machine with 3090 GPU | cuda 11.7 | ubuntu 18.04, ~~a machine with 4060 GPU | cuda 11.7 | wsl2~~, and more machines.

For possible errors, please see [ERROR_CATCH.md](ERROR_CATCH.md). Our repo is mainly built upon [GNFactor](https://github.com/YanjieZe/GNFactor), so you can also refer to [GNFactor's installation instruction](https://github.com/YanjieZe/GNFactor/blob/main/docs/INSTALL.md). If you encounter any other problem, feel free to open an issue.

---

# 🐳 Docker Installation (Recommended)

A `Dockerfile` and `compose.yaml` are provided at the project root for a fully reproducible environment. The Docker image is based on `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04` and uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

## Prerequisites

- Docker Engine ≥ 24 with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host.
- `CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz` placed at the **project root** (same directory as `Dockerfile`). Download it from https://www.coppeliarobotics.com/previousVersions.

## Step 1 — Build the image

```bash
# From the project root
docker build -t manigaussian:cu117 .
```

Or use Docker Compose (builds automatically on first `up`):

```bash
docker compose up --build -d
```

## Step 2 — Launch a container

**With Docker Compose (recommended):**
```bash
docker compose up -d          # start in background
docker compose exec manigaussian bash   # attach a shell
```

**With plain Docker:**
```bash
docker run --gpus all -it --rm \
  -e DISPLAY=:4 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  --device /dev/dri/card0:/dev/dri/card0 \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  manigaussian:cu117 bash
```

## Step 3 — Verify the environment

Inside the container, run:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import numpy; print(numpy.__version__)"
nvidia-smi
```

Expected output: `2.0.0+cu117 True`, numpy `1.23.5`, and a valid GPU table.

## Step 4 — Generate demonstrations

The CoppeliaSim environment is pre-configured inside the container at `/opt/CoppeliaSim`. From the container shell:
```bash
cd /app
bash scripts/gen_demonstrations_all.sh
```

## Notes

- **Port `2403:6001`** is exposed for Visdom visualization. Open `http://localhost:2403` in your browser.
- The `.venv` is persisted in a named Docker volume (`manigaussian_venv`). Destroying this volume forces a full reinstall of Python packages.
- If you only need to iterate on source code (no new packages), `docker compose exec manigaussian bash` reuses the existing venv — no rebuild needed.
- For headless/server environments without a GPU-attached display, set `DISPLAY=:4` and ensure `Xvfb` is running (`Xvfb :4 -screen 0 1280x1024x24 &`).

---

# Manual (Conda) Installation

The steps below describe the original conda-based setup. Skip these if you are using Docker.

## 0 clone the repo and create env
```
git clone https://github.com/GuanxingLu/ManiGaussian.git

# [Optional] We have wrapped (modified) third party packages into this repo, so it might be oversized. To address this, run:
# git config --global http.postBuffer 104857600 

conda remove -n manigaussian --all
conda create -n manigaussian python=3.9
conda activate manigaussian
```
Install pytorch
```
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

# 1 install pytorch3d
```
cd ..
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -e .
cd ../ManiGaussian
```

# 2 install CLIP
```
cd ..
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install -e .
cd ../ManiGaussian

pip install open-clip-torch
```

# 3 download coppeliasim 
Download CoppeliaSim from https://www.coppeliarobotics.com/previousVersions, e.g., CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz
```
tar -xvf CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz
rm CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz
```

# 4 add following lines to your `~/.bashrc` file. 
Remember to source your bashrc (source ~/.bashrc) and reopen a new terminal then.

You should replace the path here with your own path to the coppeliasim installation directory.
```
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT

export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

# 5 install PyRep
```
cd third_party/PyRep
pip install -r requirements.txt
pip install .
cd ../..
```

# 6 install RLBench
```
cd third_party/RLBench
pip install -r requirements.txt
python setup.py develop
cd ../..
```

# 7 install YARR
```
cd third_party/YARR
pip install -r requirements.txt
python setup.py develop
cd ../..
```

# 8 install ManiGaussian requirements
```
pip install -r requirements.txt
```

# 9 install other utility packages
```
pip install packaging==21.3 dotmap pyhocon wandb==0.14.0 chardet opencv-python-headless gpustat ipdb visdom sentencepiece
```

# 10 install odise
Install xformers (this version is a must to avoid errors from detectron2)
```
pip install xformers==0.0.18 
```
Install detectron2:
```
cd ..
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ../ManiGaussian
```
Install ODISE packages
```
cd third_party/ODISE
pip install -e .
cd ../..
```

# 11 fix some possible problems
Since a lot of packages are installed, there are some possible bugs. Use these commands first before running the code.
```
# update torch
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install hydra-core==1.1
pip install opencv-python-headless
pip install numpy==1.23.5
```

# 12 install Gaussian Splatting Renderer
```
cd third_party/gaussian-splatting/
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
cd ../..
```

# 13 install Lightning Fabric
In ManiGaussian, we use Lightning Fabric to conduct DDP training for Gaussian renderer, rather than the vanilla pytorch DDP. Reference: [gaussian-splatting's issue](https://github.com/graphdeco-inria/gaussian-splatting/issues/218)
```
pip install lightning
```



