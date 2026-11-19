# Installation
1. Clone this repo: 
```bash
git clone https://github.com/wsolow/ROB545-Project.git
```
2. Create conda environment: 
```bash
conda create -n rob545 python=3.12
conda activate rob545
cd ROB545-Project
```
3. Install dependencies 
```bash 
pip install -e .
python -m pip install kinova_lfd/envs/base/kortex/kortex_api-2.6.0.post3-py3-none-any.whl
pip install -e ."[mujoco,kortex,imitation]"
```

### Testing
After installing the above project dependencies, try running 
```python3 scripts/mj_teleop.py --wrist_camera --tabletop_camera```
n the first run, this should take a few minutes to configure the CUDA kernel with Mujoco Warp.

# Image-Based Visual Servoing
To run image based visual servoing in Mujoco:
```bash
python3 scripts/mj_ibvs.py
```
This will move the robot to a default position with the cube in view, servo to the setpoint recorded, and then execute a pre-defined grasp. 

# Imitation Learning

To test the recording script in Mujoco, run:
```bash 
python3 scripts/record_gen3.py
``` 


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
