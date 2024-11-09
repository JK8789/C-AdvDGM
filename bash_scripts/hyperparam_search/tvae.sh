#!/bin/bash -l


# Default params
default_optimiser="adam"
l2scale=0.00001
default_ordering="random"


########### Params to be updated ############
model="TVAE"
use_case="faults"
wandbp="AdvCDGM_hyper_search_${model}"

seeds="1"
eps=40
scaler_type="TabScaler"
target_model="vime"


lrs="2 2.5 3"
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
        CUDA_VISIBLE_DEVICES=-1 python run_dgm/main_tvae.py  ${use_case} --scaler_type=$scaler_type --target_model=$target_model --wandb_project=$wandbp --seed=$seed --epochs=$eps  --batch_size=${bs} --l2scale=${l2scale} --loss_factor=${lr} --version=$version --label_ordering=${default_ordering} --pert_scale=$pert_scale --adv_scale 1 --hyperparam_search
        done
      done


      adv_scales="5 10 20 50"
      for adv_scale in $adv_scales ;
      do
        for seed in $seeds ;
        do
        CUDA_VISIBLE_DEVICES=-1 python run_advdgm/main_tvae.py  ${use_case} --scaler_type=$scaler_type --wandb_project=$wandbp --seed=$seed --epochs=$eps  --batch_size=${bs} --l2scale=${l2scale} --loss_factor=${lr} --version=$version --label_ordering=${default_ordering} --pert_scale 1 --adv_scale=$adv_scale --hyperparam_search
        done
      done
    done
done


