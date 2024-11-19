#!/bin/bash -l
# Job name:
#SBATCH --job-name=wgan_tabtransformer
#
# QoS: must be savio_long for jobs &gt; 3 days
#SBATCH --qos=normal
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=16
#
# Processors per task:
#SBATCH --cpus-per-task=128
#
# Wall clock limit (7 days in this case):
#SBATCH --time=2-00:00:00

conda activate project

# Default params
default_rep=5
default_pac=1
default_ordering="random"


########### Params to be updated ############

model="WGAN"
use_case="url"
wandbp="AdvCDGM_hyper_search_${model}"
version="constrained"
scaler_type="TabScalerOHE"
target_model="tabtransformer"
seeds="1"
eps=40


#lrs="0.005 0.001 0.0005 0.0001"
lrs="0.001"
batch_sizes="256"
for bs in $batch_sizes ;
do
    echo "Varying the learning rate for ${default_optimiser}"
    for lr in $lrs ;
    do 
      #pert_scales="1 5 10 20 50"
      pert_scales="5"
      for pert_scale in $pert_scales ;
      do
        for seed in $seeds ;
        do
        CUDA_VISIBLE_DEVICES=-1 python run_advdgm/main_wgan.py ${use_case} --scaler_type=$scaler_type --target_model=$target_model --wandb_project=$wandbp --seed=$seed --epochs=$eps --disc_repeats=${default_rep}  --d_lr=${lr} --g_lr=${lr} --pac=${default_pac} --batch_size=${bs} --version=${version}  --label_ordering=${default_ordering} --pert_scale=$pert_scale --adv_scale 1 --hyperparam_search
        done
      done


      #adv_scales="5 10 20 50"
      #for adv_scale in $adv_scales ;
      #do
      #  for seed in $seeds ;
      #  do
      #  CUDA_VISIBLE_DEVICES=-1 python run_dgm/main_wgan.py ${use_case} --scaler_type=$scaler_type --target_model=$target_model --wandb_project=$wandbp --seed=$seed --epochs=$eps --disc_repeats=${default_rep}  --d_lr=${lr} --g_lr=${lr} --pac=${default_pac} --batch_size=${bs} --version=${version}  --label_ordering=${default_ordering} --pert_scale 1 --adv_scale=$adv_scale --hyperparam_search
      #  done
      #done
    done
done
