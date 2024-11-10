#!/bin/bash -l




# Default params
default_optimiser="adam"
default_decay=0.0001
default_pac=1
default_ordering="random"



########### Params to be updated ############
model="CTGAN"
use_case="faults"
wandbp="AdvCDGM_hyper_search_${model}"
scaler_type="TabScaler"
target_model="vime"
version="unconstrained"
eps=40
seeds="1"


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
        CUDA_VISIBLE_DEVICES=-1 python run_advdgm/main_ctgan.py  ${use_case} --scaler_type=$scaler_type --target_model=$target_model --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version=$version  --label_ordering=${default_ordering} --pert_scale=$pert_scale --adv_scale 1 --hyperparam_search
        done
      done


      adv_scales="5 10 20 50"
      for adv_scale in $adv_scales ;
      do
        for seed in $seeds ;
        do
        CUDA_VISIBLE_DEVICES=-1 python run_dgm/main_ctgan.py  ${use_case} --scaler_type=$scaler_type --target_model=$target_model --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version=$version  --label_ordering=${default_ordering} --pert_scale 1 --adv_scale=$adv_scale --hyperparam_search
        done
      done
    done
done
