#!/bin/bash
#SBATCH --mem=16G
#SBATCH -t 1:30:00
#SBATCH -J dc_crazy_ex14
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -o out/slurm-dc_crazy_ex14-%j.out

module load anaconda/2022.05
module load cuda
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate deepcompose

cd ..

python3 -u exp_propositions.py -a crazy -ds exclude14b -e 50 -lr 0.00001 -s True

conda deactivate