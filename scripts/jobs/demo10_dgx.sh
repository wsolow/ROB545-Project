#!/bin/bash
#SBATCH -J demo10_dgx2
#SBATCH -o output/demo10_dgx2.out
#SBATCH -e output/demo10_dgx2.err
#SBATCH -p dgx2
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G 

python3 scripts/imitation/train_diffusion.py --dataset gen3_demo_adm --batch-size 32
