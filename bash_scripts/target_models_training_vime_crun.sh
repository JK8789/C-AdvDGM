#!/bin/bash -l
# Job name:
#SBATCH --job-name=vime_train2
#
# QoS: must be savio_long for jobs &gt; 3 days
#SBATCH --qos=normal
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=100
#
# Wall clock limit (7 days in this case):
#SBATCH --time=1-00:00:00
#
## Command(s) to run (example):

conda activate project

for dataset_name in "url"; do
	for model_name in "vime"; do
		srun --cpus-per-task $SLURM_CPUS_PER_TASK  python -m mlc.run.train_search --model=$model_name --dataset_name=$dataset_name --save_best_model=1
		srun --cpus-per-task $SLURM_CPUS_PER_TASK python -m mlc.run.train_best --model=$model_name --dataset_name=$dataset_name --device=""
	done
done