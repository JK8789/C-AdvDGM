"""CLI."""
import argparse
import datetime
import os
import sys
import joblib
import time

import pandas as pd
import wandb
sys.path.append('.')

from cdgm.synthetizers.CTGAN.ctgan import CTGAN
from evaluation.eval_asr import attack_asr_time
from utils import _load_json, set_pac_val, read_csv, set_seed, load_model_and_weights, get_max_decimal_places

DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument("--model", default="CTGAN", type=str)
    parser.add_argument("use_case", type=str, choices=["url","wids","heloc","faults"])
    parser.add_argument("--version", type=str, default='unconstrained', choices=['unconstrained','constrained', "postprocessing"])
    parser.add_argument("--scaler_type", default="TabScalerOHE", type=str, choices=["TabScalerOHE", "TabScaler", "CTGAN"])
    parser.add_argument("--target_model", default="torchrln", type=str, choices=["torchrln", "tabtransformer", "vime"])
    parser.add_argument("--label_ordering", default='random', choices=['random'])
    parser.add_argument("--hyperparam_search", default=False, action='store_true')
    parser.add_argument("--wandb_project", default="Adv_CTGAN", type=str)
    parser.add_argument("--wandb_mode", default="online", type=str, choices=['online', 'disabled', 'offline'])
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--attacked_class", default=None, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument('-e', '--epochs', default=300, type=int)
    parser.add_argument("--save_every_n_epochs", default=200, type=int)
    parser.add_argument("--pert_scale", default=1, type=float)
    parser.add_argument("--adv_scale", default=1, type=float)
    parser.add_argument("--pac", default=10, type=int)
    parser.add_argument('--generator_lr', type=float, default=2e-4,
                        help='Learning rate for the generator.')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4,
                        help='Learning rate for the discriminator.')
    parser.add_argument('--generator_decay', type=float, default=1e-6,
                        help='Weight decay for the generator.')
    parser.add_argument('--discriminator_decay', type=float, default=0,
                        help='Weight decay for the discriminator.')
    parser.add_argument('--optimiser', type=str, default="adam", choices=['adam','rmsprop','sgd'], help='')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input z to the generator.')
    parser.add_argument('--generator_dim', type=str, default='256,256',
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--discriminator_dim', type=str, default='256,256',
                        help='Dimension of each discriminator layer. '
                        'Comma separated integers with no whitespaces.')
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
    exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{args.discriminator_lr}_{args.generator_lr}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    
    if args.hyperparam_search:
        path = f"outputs/CTGAN_out/{args.use_case}/hyperparam/{args.version}/{args.target_model}/{exp_id}"
    else:
        path = f"outputs/CTGAN_out/{args.use_case}/{args.version}/{args.target_model}/{exp_id}"

    args.exp_path = path
    os.makedirs(path)


    args = set_pac_val(args)

    ######################################################################
    wandb_run = wandb.init(project=args.wandb_project, id=exp_id, reinit=True,  mode=args.wandb_mode)
    for k,v in args._get_kwargs():
        wandb_run.config[k] = v
    ######################################################################
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
    ######################################################################
    dataset_info = _load_json("data/datasets_info.json")[args.use_case]
    print(dataset_info)
    ######################################################################

    X_train, (cat_cols, cat_idx) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case, dataset_info["manual_inspection_categorical_cols_idx"])
    X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
    X_val = pd.read_csv(f"data/{args.use_case}/val_data.csv")

    args.train_data_cols = X_train.columns.values.tolist()
    args.dtypes = X_train.dtypes
    args.cat_idx = cat_idx
    if args.scaler_type == "TabScalerOHE":
        path = f"models/best_models/{args.target_model}_{args.use_case}_default.model"
    else:
        path = f"models/best_models/{args.target_model}_{args.use_case}_default_{args.scaler_type}.model"
        print("Model path", path)
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

    if args.load:
        model = CTGAN.load(args.load)
    else:
        generator_dim = [int(x) for x in args.generator_dim.split(',')]
        discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
        model = CTGAN(target_model, target_scaler, X_test,
                    embedding_dim=args.embedding_dim, generator_dim=generator_dim,
                    discriminator_dim=discriminator_dim, generator_lr=args.generator_lr,
                    generator_decay=args.generator_decay, discriminator_lr=args.discriminator_lr,
                    discriminator_decay=args.discriminator_decay, batch_size=args.batch_size,
                    epochs=args.epochs, path=args.exp_path, bin_cols_idx=cat_idx, version=args.version, pac=args.pac, 
                    pert_scale=args.pert_scale, adv_scale=args.adv_scale, not_modifiable=not_modifiable,
                    feats_in_constraints=dataset_info["feats_in_constraints"])

    model.set_random_state(args.seed)
    start_time = time.time()
    model.fit(args, X_train,  num_labels, cat_cols)
    end_time = time.time()
    runtime_t = end_time - start_time
    wandb.log({"TrainRuntime": runtime_t})

    if args.save is not None:
        model.save(args.save)

    # ########################################################################################
    # ################################  EVALUATION ###########################################
    # ########################################################################################
    if args.hyperparam_search:
        adv_cand = X_val
    else:
        adv_cand = X_test

    if args.attacked_class is not None:
        adv_cand = adv_cand[adv_cand[dataset_info["target_col"]]==args.attacked_class]
        
    attack_asr_time("CTGAN", model, target_model, target_scaler, adv_cand, args,  X_test.columns[:-1], X_test.dtypes[:-1], args.exp_path)


if __name__ == '__main__':

    main()
