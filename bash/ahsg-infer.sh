#!/bin/bash
#PBS -l select=ncpus=8:mem=16gb:ngpus=1
#PBS -q gpu
module load cuda
module load anaconda/3
source activate torchenv
cd AHSG-InterIIT/
python src/inference.py "ahsg-epoch-1.pt" --batch_size "16"
python src/inference.py "ahsg-epoch-2.pt" --batch_size "16"
source deactivate
