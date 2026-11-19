#!/bin/bash
#SBATCH -J demo10_eecs
#SBATCH -o output/demo10_eecs.out
#SBATCH -e output/demo10_eecs.err
#SBATCH -p gpu,eecs
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1

python3 scripts/imitation/train_diffusion.py --dataset gen3_demo_adm --batch-size 32
