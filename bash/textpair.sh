#!/bin/bash
#PBS -l select=ncpus=8:mem=16gb:ngpus=1
#PBS -q gpu
module load cuda
module load anaconda/3
source activate torchenv
cd AHSG-InterIIT/
python src2/sentiment_classification.py --base_path "distilbert-base-uncased" --model_name "distilbert-en" --num_labels 3 --val_check_interval 0.33 --linear_lr 5e-5 --bert_output_used 'cls' --gpus 1 --logger --max_epochs 3 --accumulate_grad_batches 1 --batch_size 16


