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
```scripts/mj_teleop.py --wrist_camera --tabletop_camera```

On the first run, this should take a few minutes to configure the CUDA kernel with Mujoco Warp. After that a window should pop up (see window.png for example) with the kinova arm. If you press "k" you should see the gripper open and close.

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
