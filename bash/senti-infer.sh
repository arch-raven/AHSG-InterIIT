#!/bin/bash
#PBS -l select=ncpus=8:mem=16gb:ngpus=1
#PBS -q gpu
module load cuda
module load anaconda/3
source activate torchenv
cd AHSG-InterIIT/
#python src2/inference.py "models" --batch_size=16 --base_path "distilbert-base-uncased" --num_labels 3
python src2/inference.py "dont" --batch_size=16 --base_path "ganeshkharad/gk-hinglish-sentiment" --num_labels 3
source deactivate
