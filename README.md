
## Deep generative models as an adversarial attack strategy for tabular machine learning

This repository contains the code to reproduce the experiments for the paper ["Deep generative models as an adversarial attack strategy for tabular machine learning"](https://arxiv.org/abs/2409.12642) accepted at ICMLC 2024 (International Conference on Machine Learning and Cybernetics). 


## Setting up

The basic required packages for the python environment are listed on requirements.txt file. The python version for the environment is 3.9.20

You can download the datasets [here](https://figshare.com/articles/dataset/The_data_for_ICMLC_2024_paper_Deep_generative_models_as_an_adversarial_attack_strategy_for_tabular_machine_learning_/27241575?file=49831404) and place them in the data directory


## Running experiments

### 1. Training target models
To train target models use mlc/run/train_search.py for hyperparameter search and mlc/run/train_best.py to train the final model. 

### 2. Applying C-AdvDGM and P-AdvDGM

To apply C-AdvDGM you can first perform hyperparameter search for each model by using the script files bash_scripts/hyperparam_search
Then for the best hyperparameters you need to run the attack. 
To apply postprocessing on the already generated data from the unconstrained models you can follow the example of bash_scripts/postprocessing.sh

The experiments log the results on wandb, therefore you need to set your credentials on the code.



