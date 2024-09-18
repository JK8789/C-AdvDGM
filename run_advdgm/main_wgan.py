import datetime
import time
import time
import os
import sys
import warnings
from argparse import ArgumentParser
sys.path.append('.')

import pandas as pd
import torch
import joblib
import wandb
import numpy as np


from cdgm.synthesizers.WGAN.wgan import  train_model
from utils import _load_json, read_csv, set_seed, load_model_and_weights
from utils import get_max_decimal_places, set_pac_val
from evaluation.eval_asr import attack_asr_time
from evaluation.cons_check import cons_sat_check
from cdgm.synthesizers.WGAN.wgan import sample as sample_wgan

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)


DATETIME = datetime.datetime.now()

def get_args():
    args = ArgumentParser()
    args.add_argument("--model", default="WGAN", type=str)
    args.add_argument("use_case", type=str, choices=["url","wids","heloc","faults"])
    args.add_argument("--version", type=str, default='unconstrained', choices=['unconstrained','constrained', "postprocessing"])
    args.add_argument("--scaler_type", default="TabScalerOHE", type=str, choices=["TabScalerOHE", "TabScaler", "CTGAN"])
    args.add_argument("--target_model", default="torchrln", type=str, choices=["torchrln", "tabtransformer", "vime"])
    args.add_argument("--label_ordering", default='random', choices=['random'])
    args.add_argument("--hyperparam_search", default=False, action='store_true')
    args.add_argument("--wandb_project", default="Adv_WGAN", type=str)
    args.add_argument("--wandb_mode", default="online", type=str, choices=['online', 'disabled', 'offline'])
    args.add_argument("--num_samples", default=5, type=int)
    args.add_argument("--attacked_class", default=None, type=int)
    args.add_argument("--seed", default=7, type=int)
    args.add_argument('-e', '--epochs', default=300, type=int)
    args.add_argument("--save_every_n_epochs", default=200, type=int)
    args.add_argument("--pert_scale", default=1, type=float)
    args.add_argument("--adv_scale", default=1, type=float)
    args.add_argument("--pac", default=1, type=int)
    args.add_argument("--batch_size", default=70, type=int)
    args.add_argument("--disc_repeats", default=2, type=int)
    args.add_argument("--gp_weight", default=10, type=float)
    args.add_argument("--d_lr", default=0.00005, type=float)
    args.add_argument("--g_lr", default=0.00005, type=float)
    args.add_argument("--alpha", default=0.9, type=float)
    args.add_argument("--weight_decay", default=0, type=float)
    args.add_argument("--momentum", default=0, type=float)  # 0.00005
    args.add_argument("--clamp", default=None, type=float)  # 0.01
    return args.parse_args()


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    ######################################################################
    dataset_info = _load_json("data/datasets_info.json")[args.use_case]
    print(dataset_info)
    ######################################################################
    X_train, (_, cat_idx) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case, dataset_info["manual_inspection_categorical_cols_idx"])
    X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
    X_val = pd.read_csv(f"data/{args.use_case}/val_data.csv")

    path = f"models/best_models/{args.target_model}_{args.use_case}_default_{args.scaler_type}.model"
    target_scaler = joblib.load(path+"/scaler.joblib")
    metadata = pd.read_csv(f"data/{args.use_case}/{args.use_case}_metadata.csv")
    not_modifiable = metadata.iloc[:-1,][metadata["mutable"]==False].index.to_list()
    target_model, weight_path = load_model_and_weights(args.use_case, args.target_model, path, metadata.iloc[:-1,:], target_scaler, args.scaler_type, "cpu")
    target_model.eval()

    
    num_labels = X_train.iloc[:,-1].nunique()
    round_decs = []
    for col in X_train.columns:
        dec = get_max_decimal_places(X_train[col])
        round_decs.append(dec)
    args.round_decs = round_decs
    columns = X_train.columns.values.tolist()
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes
    exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{args.disc_repeats}_{args.gp_weight}_{args.d_lr}_{args.g_lr}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    
    if args.hyperparam_search:
        path_name = f"outputs/WGAN_out/{args.use_case}/hyperparam/{args.version}/{args.target_model}/{exp_id}"
    else:
        path_name = f"outputs/WGAN_out/{args.use_case}/{args.version}/{args.target_model}/{exp_id}"
    exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{args.disc_repeats}_{args.gp_weight}_{args.d_lr}_{args.g_lr}_{DATETIME:%d-%m-%y--%H-%M-%S}"

    # if not os.path.exists(path_name):
    # Create a new directory for the output
    os.makedirs(path_name)
    args.path_name = path_name


    # set args.pac:
    args = set_pac_val(args)

    ######################################################################
    wandb_run = wandb.init(project=args.wandb_project, id=exp_id, reinit=True,  mode=args.wandb_mode)
    for k,v in args._get_kwargs():
        wandb_run.config[k] = v
    ######################################################################
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
    ######################################################################

    print("Created new directory {:}, the experiment is starting".format(args.path_name))
    start_time = time.time()
    adv_model = train_model(args, target_model, target_scaler, cat_idx, not_modifiable, args.path_name, X_train,  num_labels)
    end_time = time.time()
    runtime_t = end_time - start_time
    wandb.log({"TrainRuntime": runtime_t})

    # ########################################################################################
    # ################################  EVALUATION ###########################################
    # ########################################################################################
    if args.hyperparam_search:
        adv_cand = X_val
    else:
        adv_cand = X_test

    if args.attacked_class is not None:
        adv_cand = adv_cand[adv_cand[dataset_info["target_col"]]==args.attacked_class]

    args.cat_idx = cat_idx
    attack_asr_time("WGAN", adv_model, target_model, target_scaler, adv_cand, args,  num_labels, X_test.dtypes[:-1], args.path_name)

