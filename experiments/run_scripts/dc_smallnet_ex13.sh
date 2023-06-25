#!/bin/bash
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J dc_smallnet_ex13
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -o out/slurm-dc_smallnet_ex13-%j.out

module load anaconda/2022.05
module load cuda
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate deepcompose

cd ..

python3 -u exp_propositions.py -a smallnet -ds exclude13b -e 50

conda deactivate