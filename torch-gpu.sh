#!/bin/sh

#SBATCH -p Gpu
#SBATCH --gres gpu:4
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1

python3 src/main.py 3 ml-1m --num_feature 1 --gpuid 4 
