
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


Here is the changes summary:

    Add bash scripts to the repo (remove one from .gitignore).
    Add some extra auxiliary bash scripts.
    Update pathes to the datasets in the "data" folder.
    Add script for target models preparation.
    Use python=3.9.20
    Replace "from mlc.transformers.ctgan.data_transformer import DataTransformer"
    by "from cdgm.data_processors"
    Remove wrong parameter train_search for function save of model vime (no such
    parameter for this function of the movel vime).
    Fix ctgan.data_transformer path for tran_best script
    Fix train_best.py for torchRLN
    Use given model scaler in eval_asr instead hardcoded torch.
    Use according scaler during the evaluation.

```
 .gitignore                                       |  4 +-
 README.md                                        |  2 +-
 bash_scripts/hyperparam_search/ctgan.sh          |  0
 bash_scripts/hyperparam_search/tablegan.sh       |  0
 bash_scripts/hyperparam_search/tvae.sh           |  0
 bash_scripts/hyperparam_search/wgan.sh           | 46 ++++++++++++-----
 bash_scripts/hyperparam_search/wgan_tabular.sh   | 65 ++++++++++++++++++++++++
 bash_scripts/hyperparam_search/wgan_vime.sh      | 65 ++++++++++++++++++++++++
 bash_scripts/postprocessing.sh                   | 27 ++++++++--
 bash_scripts/postprocessing_torch.sh             | 40 +++++++++++++++
 bash_scripts/target_models_training.sh           | 10 ++++
 bash_scripts/target_models_training_tab.sh       | 26 ++++++++++
 bash_scripts/target_models_training_torch.sh     | 26 ++++++++++
 bash_scripts/target_models_training_vime.sh      | 26 ++++++++++
 bash_scripts/target_models_training_vime_crun.sh | 26 ++++++++++
 evaluation/eval_asr.py                           |  4 +-
 mlc/datasets/samples/faults.py                   |  4 +-
 mlc/datasets/samples/heloc.py                    |  4 +-
 mlc/datasets/samples/url.py                      |  4 +-
 mlc/datasets/samples/wids.py                     |  4 +-
 mlc/logging/comet_config.py                      |  4 +-
 mlc/run/train_best.py                            |  5 +-
 mlc/run/train_search.py                          |  7 ++-
 run_advdgm/main_ctgan.py                         |  7 +--
 run_postprocessing/post.py                       | 26 +++++-----
 26 files changed, 378 insertions(+), 55 deletions(-)
```
