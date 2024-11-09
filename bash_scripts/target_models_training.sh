#!/bin/bash -l

conda activate project

for dataset_name in "url"; do
	for model_name in "vime" "tabtransformer" "torchrln"; do
		python -m mlc.run.train_search --model=$model_name --dataset_name=$dataset_name --save_best_model=1
		python -m mlc.run.train_best --model=$model_name --dataset_name=$dataset_name --device=""
	done
done
