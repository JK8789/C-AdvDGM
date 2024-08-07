"""CLI."""
import argparse
import datetime
import sys
import joblib

import pandas as pd
import numpy as np

sys.path.append('.')

from utils import _load_json, set_seed, load_model_and_weights
from utils import get_model_preds

DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str)
    args.add_argument("--scaler_type", default="TabScalerOHE", type=str, choices=["TabScalerOHE", "TabScaler", "CTGAN"])
    args.add_argument("--target_model", default="torchrln", type=str, choices=["torchrln", "tabtransformer", "vime"])
    parser.add_argument("use_case", type=str, choices=["url","wids","heloc","faults"])
    parser.add_argument("--attacked_class", default=None, type=int)
    return parser.parse_args()


def calc_accuracy(adv_cand, target_model, target_scaler):
    pred_class_adv, _ = get_model_preds(adv_cand.iloc[:,:-1].to_numpy(), target_model, target_scaler)
    idx_correct = np.argwhere(pred_class_adv == adv_cand.iloc[:,-1].to_numpy()).squeeze().flatten()
    print("Acccuracy of the model:", idx_correct.shape[0]/adv_cand.shape[0])

def main():
    args = _parse_args()
    dataset_info = _load_json("data/datasets_info.json")[args.use_case]
    print(dataset_info)
        
    X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")

    path = f"models/best_models/{args.target_model}_{args.use_case}_default_{args.scaler_type}.model"
    target_scaler = joblib.load(path+"/scaler.joblib")
    metadata = pd.read_csv(f"data/{args.use_case}/{args.use_case}_metadata.csv")
    not_modifiable = metadata.iloc[:-1,][metadata["mutable"]==False].index.to_list()
    target_model, weight_path = load_model_and_weights(args.use_case, args.target_model, path, metadata.iloc[:-1,:], target_scaler, args.scaler_type, "cpu")
    target_model.eval()
    adv_cand = X_test

    if args.attacked_class is not None:
        adv_cand = adv_cand[adv_cand[dataset_info["target_col"]]==args.attacked_class]

    calc_accuracy(adv_cand, target_model, target_scaler)

if __name__ == "__main__":
    main()