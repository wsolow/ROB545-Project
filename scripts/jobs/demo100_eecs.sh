#!/bin/bash
#SBATCH -J demo100_eecs
#SBATCH -o output/demo100_eecs.out
#SBATCH -e output/demo100_eecs.err
#SBATCH -p gpu,eecs
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G 

python3 scripts/imitation/train_diffusion.py --dataset gen3_demo --batch-size 32
