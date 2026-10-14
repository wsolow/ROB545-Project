# Installation
1. Clone this repo: ```https://github.com/wsolow/ROB545-Project.git```
2. Create conda environment: ```conda create -n rob545 python=3.12``` and ````conda activate rob545```
3. Install dependencies ```pip install -e .```
4. Install Kortex dependencies ```python -m pip installkinova_lfd/envs/base/kortex/kortex_api-2.6.0.post3-py3-none-any.whl```
5. Install mujoco and kortex project specific dependencies ```pip install -e ."[mujoco,kortex]"```