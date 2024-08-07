import warnings
from pathlib import Path
import sys
sys.path.append('.')
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import wandb
from cdgm.constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from cdgm.constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from cdgm.constraints_code.feature_orderings import set_ordering
from cdgm.constraints_code.parser import parse_constraints_file
from evaluation.eval_asr import  calc_adv_success_rate
from utils import read_csv, set_seed, _load_json
from utils import _load_json, set_pac_val, read_csv, set_seed, load_model_and_weights, get_max_decimal_places
from cdgm.synthetizers.utils import get_sets_constraints, round_func_BPDA

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=3, suppress=True)
from argparse import ArgumentParser
import pickle as pkl

set_seed(0)


def get_args():
    args = ArgumentParser()
    args.add_argument("use_case", type=str)
    args.add_argument("model_type", type=str)
    args.add_argument("--scaler_type", default="TabScalerOHE", type=str, choices=["TabScalerOHE", "TabScaler", "CTGAN"])
    args.add_argument("--target_model", default="torchrln", type=str, choices=["torchrln", "tabtransformer", "vime"])
    args.add_argument("--attacked_class", default=None, type=int)
    args.add_argument("--prefix", default='./', help='the path prefix to experiment dir')
    args.add_argument("--version", type=str, default="postprocessing")
    args.add_argument("--postprocessing_label_ordering", default='random', choices=['random'])
    return args.parse_args()


def get_model_paths(args):
    existing_model_paths = []
    existing_model_paths.extend(glob.glob(f"outputs/{args.model_type}_out/{args.use_case}/unconstrained/{args.target_model}/*19-07-24*"))
    existing_model_paths.extend(glob.glob(f"outputs/{args.model_type}_out/{args.use_case}/unconstrained/{args.target_model}/*20-07-24*"))
    existing_model_paths.extend(glob.glob(f"outputs/{args.model_type}_out/{args.use_case}/unconstrained/{args.target_model}/*21-07-24*"))
    existing_model_paths.extend(glob.glob(f"outputs/{args.model_type}_out/{args.use_case}/unconstrained/{args.target_model}/*22-07-24*"))
    existing_model_paths.extend(glob.glob(f"outputs/{args.model_type}_out/{args.use_case}/unconstrained/{args.target_model}/*23-07-24*"))
    existing_model_paths.extend(glob.glob(f"outputs/{args.model_type}_out/{args.use_case}/unconstrained/{args.target_model}/*26-07-24*"))
    return existing_model_paths


def postprocess(args, adv_cand_init, generated_data):
    constraints, sets_of_constr, ordering = get_sets_constraints(args.model_type, args.use_case, args.postprocessing_label_ordering, args.constraints_file)
    generated_data = correct_preds(generated_data, ordering, sets_of_constr)
    if args.not_modifiable:
        generated_data[:,args.not_modifiable] = adv_cand_init[:,args.not_modifiable]
    for i in range(generated_data.shape[1]):
        generated_data[:,i] = round_func_BPDA(generated_data[:,i], args.round_decs[i])
    constrained_data = generated_data.detach().numpy()
    return constrained_data 


def evaluate(args, target_model, target_scaler, adv_cand, generated_data):
    misclass_rates, cons_rates, dist_rates, success_rates = [], [], [], []
    if isinstance(adv_cand, pd.DataFrame):
        adv_cand_torch = torch.tensor(adv_cand.values.astype('float32'))
    if isinstance(adv_cand, np.ndarray):
        adv_cand_torch = torch.tensor(adv_cand.astype('float32'))
    for gen_data_i in generated_data:
        gen_data_i = torch.from_numpy(gen_data_i)
        cons_data_i = postprocess(args, adv_cand_torch, gen_data_i)
        mis, cons, dist, asr = calc_adv_success_rate(args, adv_cand, cons_data_i, target_model, target_scaler)
        misclass_rates.append(mis)
        cons_rates.append(cons)
        dist_rates.append(dist)
        success_rates.append(asr)

    columns = [".01", ".05", ".1", ".2", ".3", ".4", ".5"]

    misclass_rates =  pd.DataFrame(misclass_rates, columns=columns)
    cons_rates = pd.DataFrame(cons_rates, columns=columns)
    dist_rates =  pd.DataFrame(dist_rates, columns=columns)
    success_rates =  pd.DataFrame(success_rates, columns=columns)

    wandb.log({"Attack_results/Results/ASR_means": success_rates.mean(axis=0).to_frame(name="Means").transpose()})
    wandb.log({"Attack_results/Results/ASR_stds": success_rates.std(axis=0).to_frame(name="Stds").transpose()})

    wandb.log({"Attack_results/Detailed/Missclassification": misclass_rates,
               "Attack_results/Detailed/Cons_satisfaction": cons_rates,
             "Attack_results/Detailed/Distance": dist_rates,
             "Attack_results/Detailed/ASR": success_rates})


 
def main():
    args = get_args()


    ######################################################################
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
    ######################################################################
    dataset_info = _load_json("data/datasets_info.json")[args.use_case]
    print(dataset_info)
    ######################################################################

    if args.scaler_type == "TabScalerOHE":
        path = f"models/best_models/{args.target_model}_{args.use_case}_default.model"
    else:
        path = f"models/best_models/{args.target_model}_{args.use_case}_default_{args.scaler_type}.model"
        print("Model path", path)
    target_scaler = joblib.load(path+"/scaler.joblib")
    metadata = pd.read_csv(f"data/{args.use_case}/{args.use_case}_metadata.csv")

    args.not_modifiable = metadata.iloc[:-1,][metadata["mutable"]==False].index.to_list()
    target_model, weight_path = load_model_and_weights(args.use_case, args.target_model, path, metadata.iloc[:-1,:], target_scaler, args.scaler_type, "cpu")
    target_model.eval()

    X_train = pd.read_csv(f"data/{args.use_case}/train_data.csv")
    X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
    adv_cand = X_test
    round_decs = []
    for col in X_train.columns:
        dec = get_max_decimal_places(X_train[col])
        round_decs.append(dec)
    args.round_decs = round_decs


    upper_model = args.model_type.upper()
    args.wandb_project = f"AdvCDGM_attack_{upper_model}_hpc"
    args.path_names = get_model_paths(args)
    for path_name in args.path_names:
        args.exp_path = path_name
        exp_id = Path(path_name).parts[-1]

        ######################################################################
        wandb_run = wandb.init(project=args.wandb_project, id=exp_id+(f'{args.postprocessing_label_ordering}'))
        for k,v in args._get_kwargs():
            wandb_run.config[k] = v
        ######################################################################

        generated_data = pkl.load(open(f'{args.exp_path}/adv_data.pkl', 'rb'))
 
        if args.attacked_class is not None:
            adv_cand = adv_cand[adv_cand[dataset_info["target_col"]]==args.attacked_class]
        for gen_data_i in generated_data:
            print(gen_data_i.shape)

        evaluate(args, target_model, target_scaler, adv_cand, generated_data)
        wandb.finish()


if __name__ == "__main__":
    main()