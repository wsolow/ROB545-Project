# Installation
1. Clone this repo: ```https://github.com/wsolow/ROB545-Project.git```
2. Create conda environment: ```conda create -n rob545 python=3.12``` and ````conda activate rob545``` and ```cd ROB545-Project```
3. Install dependencies ```pip install -e .```
4. Install Kortex dependencies ```python -m pip install kinova_lfd/envs/base/kortex/kortex_api-2.6.0.post3-py3-none-any.whl```
5. Install mujoco and kortex project specific dependencies ```pip install -e ."[mujoco,kortex]"```

# Testing
After installing the above project dependencies, try running 
```python3 scripts/mj_teleop.py```

To view the wirst and tabletop camera run which will bring up cv2 windows:
```python3 scripts/mj_teleop.py --wrist_camera --tabletop_camera```

# Image-Based Visual Servoing
To run image based visual servoing in Mujoco:
```python3 scripts/mj_ibvs.py```
This will move the robot to a default position with the cube in view, servo to the setpoint recorded,
and then execute a pre-defined grasp. 

# Imitation Learning
Imitation learning is built upon the LeRobot package which provides nice support for dataset configs and is widely used. Right now we are assuming that LeRobot is v0.6.2. To install: 
```git clone https://github.com/huggingface/lerobot.git```
and then install with pip
```pip install -e 'lerobot[training,dataset,diffusion]'```

To test the recording script in Mujoco, run:
```python3 scripts/record_gen3.py``` 
which will print all the keyboard commands for recording using the SE3 keyboard. 



On the first run, this should take a few minutes to configure the CUDA kernel with Mujoco Warp. After that a window should pop up (see window.png for example) with the kinova arm. If you press "k" you should see the gripper open and close.

# Human Demonstrations: Object Tracking

Human demos are recorded with the RealSense into a LeRobot dataset plus a raw RGB-D folder:
```bash
PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/record/record_human_demo.py \
    --dataset-name human_pick_pink_block --task "Pick the pink block" --num-episodes 5
```
Each episode is saved to `_data/<name>/` (LeRobot: RGB video + intrinsics, no actions) and
`_data/<name>/raw/episode_XXXXXX/{rgb/*.png, depth/*.png (uint16 mm), cam_K.txt}`.

The object tracking pipeline (`kinova_lfd/data_collection/human_demo/perception/`) segments objects
with **SAM3** (text prompts) and estimates their 6D pose with **FoundationPose**. Both models run as
local servers that keep their weights on the GPU.

## Installation
All steps run inside the `rob545` env. Model source goes in `third_party/` and weights in
`kinova_lfd/.cache/models/` (both git-ignored).

1. Python dependencies:
```bash
pip install -e ".[perception]"
```

2. CUDA toolkit matching PyTorch (torch is built for CUDA 13.0; only the driver is installed system-wide).
   This is needed to build nvdiffrast:
```bash
conda install -c nvidia cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 libcublas-dev \
    libcusparse-dev libcusolver-dev libcurand-dev cuda-nvrtc-dev=13.0 cuda-profiler-api=13.0
```

3. SAM3 (installed with `--no-deps`: it pins `numpy<2`, which would break the env). Request access
   to the [checkpoints](https://huggingface.co/facebook/sam3) and log in with `hf auth login` first:
```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
pip install --no-deps -e third_party/sam3
python -c "from huggingface_hub import hf_hub_download as d; [d('facebook/sam3', f, local_dir='kinova_lfd/.cache/models/sam3') for f in ('config.json', 'sam3.pt')]"
```

4. FoundationPose, nvdiffrast, pytorch3d, and FoundationPose's `mycpp` extension. Do not install
   FoundationPose's `requirements.txt` directly: its `pyrender` pin downgrades PyOpenGL. The
   `perception` extra already covers what the pose pipeline needs.
```bash
git clone https://github.com/NVlabs/FoundationPose.git third_party/FoundationPose
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH TORCH_CUDA_ARCH_LIST="12.0"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
# Build mycpp against pip's pybind11 (the system pybind11 2.11 does not support numpy 2 and segfaults)
cd third_party/FoundationPose/mycpp && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -m pybind11 --cmakedir) && cmake --build . -j
cd ../../../..
```
   `TORCH_CUDA_ARCH_LIST="12.0"` targets the RTX 5070; change it for other GPUs. mycpp needs
   `libboost-dev` and `libeigen3-dev` (already listed under the Kortex C++ requirements below).

5. FoundationPose weights (refiner `2023-10-28-18-33-37`, scorer `2024-01-11-20-02-45`). The server
   links `third_party/FoundationPose/weights` to this folder on startup:
```bash
gdown --folder https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i \
    -O kinova_lfd/.cache/models/foundationpose
```

## Running
Start both servers (each in its own terminal; leave them running between runs):
```bash
sam3_server            # http://127.0.0.1:8001, add --offload-video-to-cpu for long episodes
foundationpose_server  # http://127.0.0.1:8002
```
Then track the objects listed in `kinova_lfd/data_collection/human_demo/perception/objects.yaml`
(name, SAM3 text prompt, box size or mesh, symmetry) through one or more datasets:
```bash
PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/perception/track_object_pose.py \
    --dataset _data/human_demo --episodes 0 1 --overlay
```
Outputs, next to each dataset's `raw/`:
- `segmentation/episode_XXXXXX.npz`: per-object masks (packed bits), boxes, SAM3 scores.
- `6d_pose/episode_XXXXXX.npz`: `T_C_O` (object pose in the camera frame), `valid` (frame has a SAM3
  mask), `registered` (pose from a re-registration), `scores`, `frame_index`, `timestamps`, `intrinsic_k`.
- `6d_pose/meta.json`: objects and configs used; `--overlay` also writes `episode_XXXXXX_overlay.mp4`.

Notes:
- One SAM3 instance is kept per text prompt (highest average score over the episode).
- FoundationPose re-registers when the object reappears or tracking disagrees with the SAM3 mask.
  Re-registrations are snapped to the symmetric equivalent closest to the last pose, so symmetric
  objects (`symmetry: cube` / `box`) do not flip orientation.
- SAM3 keeps decoded frames on the GPU (about 6 MB per frame), so episodes of more than about
  a minute need `sam3_server --offload-video-to-cpu` on a 12 GB GPU.

# Kortex UDP bridge with C++ bindings

To do real-time control at 1kHz, it is recommended to use C++ as the Python3 API cannot run sufficiently fast. To avoid using ROS2, we install the Kinova Kortex C++ API below

1. Install essential build requirements
```bash
sudo apt update
sudo apt install build-essential cmake git libeigen3-dev libboost-all-dev liburdfdom-dev libconsole-bridge-dev
```

2. Clone and install Pinocchio for fast FK/IK computation 

```bash
git clone --recursive https://github.com/stack-of-tasks/pinocchio.git
cd pinocchio
git checkout v4.1.0
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -DBUILD_WITH_COLLISION_SUPPORT=OFF \
    -DBUILD_TESTING=OFF
cmake --build . -j
sudo cmake --install .
cd ../..
```

3. Download and extract the Kinova Kortex C++ API:
```bash
mkdir -p kinova_lfd/envs/base/kortex/kortex_cpp_api
curl -L --fail -o /tmp/kortex_api_cpp_2.6.0_linux_x86_64.zip \
    https://artifactory.kinovaapps.com/artifactory/generic-public/kortex/API/2.6.0/linux_x86_64_gcc_5.4.zip
unzip -q -o /tmp/kortex_api_cpp_2.6.0_linux_x86_64.zip \
    -d kinova_lfd/envs/base/kortex/kortex_cpp_api
```

4. Compile the Kinova UDP Bridge in C++

```bash
cmake -S kinova_lfd/envs/base/kortex/kortex_udp/cpp \
    -B kinova_lfd/envs/base/kortex/kortex_udp/cpp/build \
    -DKORTEX_DIR="$PWD/kinova_lfd/envs/base/kortex/kortex_cpp_api" \
    -DPINOCCHIO_DIR=/usr/local
cmake --build kinova_lfd/envs/base/kortex/kortex_udp/cpp/build
```
