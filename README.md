# Installation
1. Clone this repo: ```https://github.com/wsolow/ROB545-Project.git```
2. Create conda environment: ```conda create -n rob545 python=3.12``` and ````conda activate rob545```
3. Install dependencies ```pip install -e .```
4. Install mujoco and kortex specific dependencies ```pip install -e ."[mujoco,kortex]"```