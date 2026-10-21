# Installation
1. Clone this repo: ```https://github.com/wsolow/ROB545-Project.git```
2. Create conda environment: ```conda create -n rob545 python=3.12``` and ````conda activate rob545``` and ```cd ROB545-Project```
3. Install dependencies ```pip install -e .```
4. Install Kortex dependencies ```python -m pip install kinova_lfd/envs/base/kortex/kortex_api-2.6.0.post3-py3-none-any.whl```
5. Install mujoco and kortex project specific dependencies ```pip install -e ."[mujoco,kortex]"```

# Testing
After installing the above project dependencies, try running 
```scripts/mj_teleop.py```

To view the wirst and tabletop camera run which will bring up cv2 windows:
```scripts/mj_teleop.py --wrist_camera --tabletop_camera

On the first run, this should take a few minutes to configure the CUDA kernel with Mujoco Warp. After that a window should pop up (see window.png for example) with the kinova arm. If you press "k" you should see the gripper open and close.

# Kortex UDP bridge

The 1 kHz UDP bridge is a separate C++ program. `pip install` does not compile it, and the build does not use ROS 2. Install Eigen, Boost, and urdfdom from apt, build Pinocchio 3 from source, and point CMake at the Kinova Kortex C++ API.

```bash
sudo apt update
sudo apt install build-essential cmake git libeigen3-dev libboost-all-dev liburdfdom-dev libconsole-bridge-dev
```

Pinocchio 4 is the C++ kinematics library. Build it with the Python bindings and collision support turned off so the install does not pull EigenPy or Coal:

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
```

`KORTEX_DIR` is the extracted Kortex C++ API. That directory contains `include/` and `lib/release/libKortexApiCpp.a`. From the repository root:

```bash
cmake -S kinova_lfd/envs/base/kortex/kortex_udp/cpp \
    -B kinova_lfd/envs/base/kortex/kortex_udp/cpp/build \
    -DKORTEX_DIR=/path/to/kortex_api \
    -DPINOCCHIO_DIR=/usr/local
cmake --build kinova_lfd/envs/base/kortex/kortex_udp/cpp/build
```

The executable is `kinova_lfd/envs/base/kortex/kortex_udp/cpp/build/kortex_udp_bridge`. The `cpp/build` directory is generated and is not part of the git repository. If Pinocchio is installed in `/usr` or `/usr/local`, `-DPINOCCHIO_DIR` can be omitted. 