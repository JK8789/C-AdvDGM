#!/bin/bash -l

model="WGAN"
target_models="tabtransformer"
scaler_type="TabScaler"

use_cases="faults wids"
for use_case in $use_cases ;
do
   for target in $target_models;
      do
         python run_postprocessing/post.py  ${use_case} ${model} --target_model=$target --scaler_type=$scaler_type
      done
    done


use_cases="url heloc"
for use_case in $use_cases ;
do
   for target in $target_models;
      do
         python run_postprocessing/post.py  ${use_case} ${model} --target_model=$target --scaler_type=$scaler_type --attacked_class 1
      done
    done
