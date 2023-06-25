#!/bin/bash
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J dc_lenet_ex23_lr1e-6
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -o out/slurm-dc_lenet_ex23_lr1e-6-%j.out

module load anaconda/2022.05
module load cuda
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate deepcompose

cd ..

python3 -u exp_propositions.py -a lenet -ds exclude23b -e 50 -s True

conda deactivate