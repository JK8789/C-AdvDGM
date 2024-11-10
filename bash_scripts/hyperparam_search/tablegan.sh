#!/bin/bash -l

# Default params
default_optimiser="adam"
default_ordering="random"
default_random_dim=100




########### Params to be updated ############
model="TableGAN"
use_case="faults"
wandbp="AdvCDGM_hyper_search_${model}"
seeds="1"
eps=40
default_bs=512
scaler_type="TabScaler"
target_model="vime"
version="constrained"

lrs="0.005 0.001 0.0005 0.0001"
batch_sizes="256"

for bs in $batch_sizes ;
do
    echo "Varying the learning rate for ${default_optimiser}"
    for lr in $lrs ;
    do 
      pert_scales="1 5 10 20 50"
      for pert_scale in $pert_scales ;
      do
        for seed in $seeds ;
        do
        python run_advdgm/main_tableGAN.py ${use_case} --scaler_type=$scaler_type --target_model=$target_model --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=$default_optimiser --lr=${lr} --batch_size=${bs} --random_dim=${default_random_dim} --version=$version --label_ordering=${default_ordering} --hyperparam_search  --pert_scale=$pert_scale 
        done
      done

      adv_scales="5 10 20 50"
      for adv_scale in $adv_scales ;
      do
        for seed in $seeds ;
        do
        python run_dgm/main_tableGAN.py ${use_case} --scaler_type=$scaler_type --target_model=$target_model --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=$default_optimiser --lr=${lr} --batch_size=${bs} --random_dim=${default_random_dim} --version=$version  --label_ordering=${default_ordering} --hyperparam_search  --adv_scale=$adv_scale 
        done
      done
    done
done

