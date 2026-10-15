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