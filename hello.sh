#!/usr/bin/bash

#SBATCH -J 0930
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 0-4
#SBATCH -o logs/slurm-%A.out 

pwd
which python
hostname

python metrics.py --source_path /local_datasets/dataset/nerf_llff_data/room -m ./output/room --iteration 2000

exit 0

