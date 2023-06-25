#!/bin/bash
#SBATCH --mem=16G
#SBATCH -t 0:05:00
#SBATCH -J dc_test
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o out/slurm-dc_test-%j.out

module load anaconda/2022.05
module load cuda
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate deepcompose

cd ..

python3 -u exp_propositions.py -a smallnet -ds exclude23b -e 5

conda deactivate