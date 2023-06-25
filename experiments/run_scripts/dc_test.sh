#!/bin/bash
module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate deepcompose

python3 exp_propositions.py -a smallnet -ds exclude23b

conda deactivate