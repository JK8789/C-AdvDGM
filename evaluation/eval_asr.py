import time

import joblib
import wandb
import pandas as pd
import numpy as np
import torch
import pickle

from cdgm.synthesizers.WGAN.wgan import sample as sample_wgan
from cdgm.synthesizers.utils import get_modes_idx
from evaluation.cons_check import cons_sat_check
from utils import get_model_preds


def compute_distance(x_1, x_2, norm):
    if norm in ["inf", np.inf]:
        distance = np.linalg.norm(x_1 - x_2, ord=np.inf, axis=-1)
    elif norm in ["2", 2]:
        distance = np.linalg.norm(x_1 - x_2, ord=2, axis=-1)
        print("Distance", np.mean(distance))
    else:
        raise NotImplementedError
    return distance


def calc_adv_success_rate(args, adv_cand, sampled_data, target_model, target_scaler):
    # if args.scaler_type == "CTGAN":
    #     sampled_data_df = pd.DataFrame(sampled_data, columns=adv_cand.iloc[:, :-1].columns.tolist())
    #     sampled_data = torch.from_numpy(sampled_data).float()
    #     adv_cand_scaled =  target_scaler.transform(adv_cand.iloc[:, :-1], None)
    #     # adv_cand_scaled =  args.adv_scaler.transform(adv_cand.iloc[:, :-1], None)

    #     pred_class_adv = get_model_preds(sampled_data, target_model, target_scaler)
    #     sampled_data_scaled =  target_scaler.transform(sampled_data, None).detach().numpy()
    #     # sampled_data_scaled =  args.adv_scaler.transform(sampled_data_df, None)

    #     modes_idx = get_modes_idx(target_scaler)
    #     # modes_idx = np.array(modes_idx)
    #     # mask = np.ones(adv_cand_scaled.shape[1], dtype=bool)
    #     # mask[modes_idx] = False
    #     adv_cand_scaled[:,modes_idx]=sampled_data_scaled[:,modes_idx]
    #     dist = compute_distance(adv_cand_scaled, sampled_data_scaled, norm="2")

    # else: 
    if args.scaler_type == "CTGAN":
        sampled_data_torch = torch.from_numpy(sampled_data).float()
        pred_class_adv = get_model_preds(sampled_data_torch, target_model, target_scaler)

    else:
        pred_class_adv = get_model_preds(sampled_data, target_model, target_scaler)

    # if args.scaler_type == "TabScalerOHE":
    #     path = f"models/best_models/{args.target_model}_{args.use_case}_default.model"
    # else:
    #     path = f"models/best_models/{args.target_model}_{args.use_case}_default_{args.scaler_type}.model"

    dist_scaler_path = f"models/best_models/torchrln_{args.use_case}_default.model"
    dist_scaler = joblib.load(dist_scaler_path+"/scaler.joblib")

    adv_cand_scaled =  dist_scaler.transform(adv_cand.iloc[:, :-1].to_numpy())
    sampled_data_scaled =  dist_scaler.transform(sampled_data)
    dist = compute_distance(adv_cand_scaled, sampled_data_scaled, norm="2")

    cons_rates, misclass_rates, dist_rates, success_rates = [], [], [], []
    
    ## Apparently no easy way in python to get a range of floats
    epsilons = list(np.arange(0.1, 0.6, 0.1))
    epsilons = [0.01, 0.05] + epsilons
    idx_mis = np.argwhere(pred_class_adv != adv_cand.iloc[:,-1].to_numpy()).squeeze().flatten()
    samples_sat_constr = cons_sat_check(args, sampled_data)
    idx_sat_constr = torch.nonzero(samples_sat_constr).squeeze().flatten()
    print("Satisfaction", idx_sat_constr.shape[0])

    for eps in epsilons:
        idx_dist = np.argwhere(dist<eps).squeeze().flatten()
        if eps==0.5:
            print("Dist", idx_dist.shape[0])

        idx_dist_sat = np.intersect1d(idx_dist, idx_sat_constr)
        if eps==0.5:
            print("Dist Sat", idx_dist_sat.shape[0])

        idx_adv = np.intersect1d(idx_dist_sat, idx_mis)
        if eps==0.5:
            print("Dist adv", idx_adv.shape[0])

        misclass_rate = idx_mis.shape[0]/adv_cand.shape[0]
        cons_rate = idx_sat_constr.shape[0]/adv_cand.shape[0]
        dist_rate = idx_dist.shape[0]/adv_cand.shape[0]
        success_rate = idx_adv.shape[0]/adv_cand.shape[0]

        misclass_rates.append(misclass_rate)
        cons_rates.append(cons_rate)
        dist_rates.append(dist_rate)
        success_rates.append(success_rate)
    return misclass_rates, cons_rates, dist_rates, success_rates


def attack_asr_time(dgm_type, adv_model, target_model, target_scaler, adv_cand, args, columns, dtypes, exp_path):

    misclass_rates, cons_rates, dist_rates, success_rates = [], [], [], []
    run_times = []
    sampled_data_ls = []



    for i in range(args.num_samples):
        start = time.time()
        # adv_cand_scaled = target_scaler.transform(adv_cand)
        if dgm_type in ["TableGAN", "CTGAN", "TVAE"]:
            sampled_data = adv_model.sample(adv_cand)
        elif dgm_type == "WGAN":
            sampled_data = sample_wgan(adv_model, adv_cand)
        elif dgm_type == "GOGGLE":
            sampled_data = adv_model.sample(adv_cand, args.X_test, inverse_transformed=True)
        
        # if dgm_type in ["CTGAN", "TVAE"]:
        #     args.adv_scaler = adv_model._transformer

        end = time.time()
        sampling_duration = end - start
        sampled_data_ls.append(sampled_data)
        # sampled_data_df = pd.DataFrame(sampled_data, columns=columns)
        # sampled_data_df_cast = sampled_data_df.astype(dtypes)
        # mis, dist, asr = calc_adv_success_rate(args, adv_cand, sampled_data_df_cast.to_numpy(dtype=np.float32), target_model, target_scaler)
        mis, cons, dist, asr = calc_adv_success_rate(args, adv_cand, sampled_data, target_model, target_scaler)
        misclass_rates.append(mis)
        cons_rates.append(cons)
        dist_rates.append(dist)
        success_rates.append(asr)
        run_times.append(sampling_duration)

    with open(f'{exp_path}/adv_data.pkl', 'wb') as f:
        pickle.dump(sampled_data_ls, f)

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
    cols = list(map(str,np.arange(args.num_samples)))
    print(type(cols[0]))

    run_times = pd.DataFrame([run_times], columns=cols)
    run_times["Avg"] = run_times.mean(axis=1)
    run_times["Std"] = run_times.std(axis=1)

    wandb.log({"Attack_results/RunTimes/Sampling": run_times})
 