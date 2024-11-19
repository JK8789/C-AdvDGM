#!/bin/bash -l
# Job name:
#SBATCH --job-name=postprocessing_torch
#
# QoS: must be savio_long for jobs &gt; 3 days
#SBATCH --qos=normal
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=32
#
# Processors per task:
#SBATCH --cpus-per-task=128
#
# Wall clock limit (7 days in this case):
#SBATCH --time=2-00:00:00

conda activate project

model="WGAN"
target_models="torchrln"
scaler_type="TabScalerOHE"

use_cases="url"
for use_case in $use_cases ;
do
   for target in $target_models;
      do
         python run_postprocessing/post.py  ${use_case} ${model} --target_model=$target --scaler_type=$scaler_type
      done
    done


use_cases="url"
for use_case in $use_cases ;
do
   for target in $target_models;
      do
         python run_postprocessing/post.py  ${use_case} ${model} --target_model=$target --scaler_type=$scaler_type --attacked_class 1
      done
    done
