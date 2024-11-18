#!/bin/bash -l
# Job name:
#SBATCH --job-name=torchRLN_train
#
# QoS: must be savio_long for jobs &gt; 3 days
#SBATCH --qos=normal
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=4
#
# Processors per task:
#SBATCH --cpus-per-task=32
#
# Wall clock limit (7 days in this case):
#SBATCH --time=1-00:00:00
#
## Command(s) to run (example):

conda activate project

for dataset_name in "url"; do
	for model_name in "torchrln"; do
		python -m mlc.run.train_search --model=$model_name --dataset_name=$dataset_name --save_best_model=1
		python -m mlc.run.train_best --model=$model_name --dataset_name=$dataset_name --device=""
	done
done
