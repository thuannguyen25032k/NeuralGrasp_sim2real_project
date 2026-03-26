
# Error Catching

I have recorded the errors I encountered during the installation. If you have any questions, please feel free to open an issue.

---

## 🐳 Docker-specific Errors

- **`CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz: No such file or directory`** during `docker build`
```
# Place the CoppeliaSim archive at the project root before building:
cp /path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz .
docker build -t manigaussian:cu117 .
```

- **`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`**
```
# NVIDIA Container Toolkit is not installed or not configured. Install it:
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

- **`libEGL warning: failed to open /dev/dri/renderD128: Permission denied`** inside container
```
# Add the DRI devices to your docker run / compose.yaml:
#   --device /dev/dri/card0:/dev/dri/card0
#   --device /dev/dri/renderD128:/dev/dri/renderD128
# OR grant the current user access on the host:
sudo chmod a+rw /dev/dri/renderD128
```
The `compose.yaml` already mounts these devices. If you are using plain `docker run`, add the `--device` flags shown above.

- **`libGL error: failed to load driver: swrast`** inside container
```
# Make sure NVIDIA_DRIVER_CAPABILITIES includes graphics:
# Set in docker run:
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ...
# Or in compose.yaml / Dockerfile (already set):
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

- **`No display found` / `Xvfb` errors when running headless**
```bash
# Start a virtual display inside the container before running scripts:
Xvfb :4 -screen 0 1280x1024x24 &
export DISPLAY=:4
```
The `compose.yaml` sets `DISPLAY=:4` automatically.

- **`CUDA extension build fails: clang++ not found`** during `docker build`
```
# Ensure CC/CXX are forced to gcc in the Dockerfile (already set):
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CUDAHOSTCXX=/usr/bin/g++
```

- **`No virtual environment found` during `uv pip install`**
```
# The venv must be created before any uv pip install call.
# The Dockerfile already runs:
#   uv python install 3.9 && uv venv --python 3.9 /app/.venv
# If you add a custom RUN layer, reference the venv explicitly:
uv pip install --python /app/.venv/bin/python <package>
```

- **Port `6001` already in use (Visdom)**
```
# Change the host-side port in compose.yaml:
ports:
  - "2404:6001"   # use any free host port
```

---

## Manual Installation Errors

- PyRender error.
```
# please add following to bashrc:
export DISPLAY=:0
export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=egl
```

- libGL error: failed to load driver: swrast
```
conda install -c conda-forge gcc
```

- torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGKILL
```
pip install h5py==3.3.0
```

- PyYAML (>=5.1.*)
```
pip install setuptools==61.1.0
```

- ERROR: Could not build wheels for mask2former, which is required to install pyproject.toml-based projects
```
See https://github.com/NVlabs/ODISE/issues/19#issuecomment-1592580278
```

- wandb 'run = wi.init()' error
```
pip install wandb==0.14.0
```

- ImportError: cannot import name 'get_num_classes' from 'torchmetrics.utilities.data' 
```
pip install torchmetrics==0.6.0
```

- [glm/glm.hpp no such file or directory](https://github.com/GuanxingLu/ManiGaussian/issues/3)
```
sudo apt-get install libglm-dev
```

- The call failed on the V-REP side. 
```
pip uninstall rlbench

# then follow the instruction to reinstall the correct RLBench version please
cd third_party/RLBench
pip install -r requirements.txt
python setup.py develop
```

- libEGL warning: failed to open /dev/dri/renderD128: Permission denied
```
# Ref: https://github.com/google-deepmind/dm_control/issues/214
sudo apt install libnvidia-gl-470-server
```
Maybe a simple `chown' or `chmod' also work.

You can also refer to [GNFactor's error catching](https://github.com/YanjieZe/GNFactor/blob/main/docs/ERROR_CATCH.md) for more error types.
