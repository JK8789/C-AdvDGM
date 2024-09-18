"""CLI."""
import argparse
import datetime
import os
import sys
from timeit import default_timer as timer
import time
sys.path.append('.')

import numpy as np
import pandas as pd
import wandb
import joblib
from cdgm.synthesizers.TableGAN.tableGAN import TableGAN
from utils import _load_json, read_csv, set_seed, load_model_and_weights, get_max_decimal_places
from train_target.faults_nn_pl import MLP
from evaluation.eval_asr import attack_asr_time

DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='TableGAN Command Line Interface')
    parser.add_argument("--model", default="TableGAN", type=str)
    parser.add_argument("use_case", type=str, choices=["url","wids","heloc","faults"])
    parser.add_argument("--version", type=str, default='unconstrained', choices=['unconstrained','constrained', "postprocessing"])
    parser.add_argument("--scaler_type", default="TabScalerOHE", type=str, choices=["TabScalerOHE", "TabScaler", "CTGAN"])
    parser.add_argument("--target_model", default="torchrln", type=str, choices=["torchrln", "tabtransformer", "vime"])
    parser.add_argument("--label_ordering", default='random', choices=['random'])
    parser.add_argument("--hyperparam_search", default=False, action='store_true')
    parser.add_argument("--wandb_project", default="Adv_TableGAN", type=str)
    parser.add_argument("--wandb_mode", default="online", type=str, choices=['online', 'disabled', 'offline'])
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--attacked_class", default=None, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument('-e', '--epochs', default=300, type=int)
    parser.add_argument("--save_every_n_epochs", default=200, type=int)
    parser.add_argument("--pert_scale", default=1, type=float)
    parser.add_argument("--adv_scale", default=1, type=float)
    parser.add_argument('--random_dim', type=int, default=100)
    parser.add_argument('--num_channels', type=int, default=64,
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--l2scale', type=float, default=1e-5, help='')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--optimiser', type=str, default="adam", choices=['adam','rmsprop','sgd'], help='')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')
    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    set_seed(args.seed)
    exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.optimiser}_{args.epochs}_{args.batch_size}_{args.random_dim}_{args.num_channels}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    if args.hyperparam_search:
        path = f"outputs/TableGAN_out/{args.use_case}/hyperparam/{args.version}/{args.target_model}/{args.target_model}/{exp_id}"
    else:
        path = f"outputs/TableGAN_out/{args.use_case}/{args.version}/{args.target_model}/{args.target_model}/{exp_id}"

    args.exp_path = path
    os.makedirs(path)

    ######################################################################
    wandb_run = wandb.init(project=args.wandb_project, id=exp_id, reinit=True,  mode=args.wandb_mode)
    for k,v in args._get_kwargs():
        wandb_run.config[k] = v
    ######################################################################
    ######################################################################
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
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
    not_modifiable = metadata[metadata["mutable"]==False].index.to_list()
    target_model, weight_path = load_model_and_weights(args.use_case, args.target_model, path, metadata.iloc[:-1,:], target_scaler, args.scaler_type, "cpu")
    target_model.eval()
    num_labels = X_train.iloc[:,-1].nunique()

    round_decs = []
    for col in X_train.columns:
        dec = get_max_decimal_places(X_train[col])
        round_decs.append(dec)
    args.round_decs = round_decs
    args.train_data_cols = X_train.columns.values.tolist()
    args.dtypes = X_train.dtypes
    if args.load:
        adv_model = TableGAN.load(args.load)
    else:
        adv_model = TableGAN(target_model, target_scaler, 
            random_dim=64, num_channels=args.num_channels,
            l2scale=args.l2scale, batch_size=args.batch_size,
            epochs=args.epochs, pert_scale=args.pert_scale, adv_scale=args.adv_scale, 
            path=args.exp_path,  version=args.version, not_modifiable=not_modifiable)


    start_time = time.time()
    adv_model.fit(args, X_train, cat_idx, num_labels)
    end_time = time.time()
    runtime_t = end_time - start_time
    wandb.log({"TrainRuntime": runtime_t})
    if args.save is not None:
        adv_model.save(args.save)


    # ########################################################################################
    # ################################  EVALUATION ###########################################
    # ########################################################################################
    if args.hyperparam_search:
        adv_cand = X_val
    else:
        adv_cand = X_test

    if args.attacked_class is not None:
        adv_cand = adv_cand[adv_cand[dataset_info["target_col"]]==args.attacked_class]

    attack_asr_time("TableGAN", adv_model, target_model, target_scaler, adv_cand, args,  X_test.columns[:-1], X_test.dtypes[:-1], args.exp_path)


if __name__ == '__main__':

    main()
