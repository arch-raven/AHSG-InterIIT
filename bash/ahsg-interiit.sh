#!/bin/bash
#PBS -l select=ncpus=8:mem=16gb:ngpus=1
#PBS -q gpu
module load cuda
module load anaconda/3
source activate torchenv
cd AHSG-InterIIT/
python src/train.py --run_name "xlm-roberta-baseline" --gpus 1 --logger --max_epochs 3 --accumulate_grad_batches 2 --batch_size 8
source deactivate