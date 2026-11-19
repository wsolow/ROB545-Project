#!/bin/bash
#SBATCH -J demo100_dgx2
#SBATCH -o output/demo100_dgx2.out
#SBATCH -e output/demo100_dgx2.err
#SBATCH -p dgx2
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G 

python3 scripts/imitation/train_diffusion.py --dataset gen3_demo --batch-size 32
