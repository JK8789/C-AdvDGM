import torch
import pandas as pd
from cdgm.constraints_code.parser import parse_constraints_file
from cdgm.constraints_code.classes import get_missing_mask



def cons_sat_check(args, data):
    _, constraints = parse_constraints_file(args.constraints_file)

    sat_rate_per_constr = {i: [] for i in range(len(constraints))}
    percentage_of_samples_sat_constraints = []
    mask_out_missing_values = True
    samples_sat_constr = torch.ones(data.shape[0]) == 1.
    data = torch.tensor(data)
    for j, constr in enumerate(constraints):
        sat_per_datapoint = constr.single_inequality.check_satisfaction(data)
        if mask_out_missing_values:
            missing_values_mask = get_missing_mask(constr.single_inequality.body, data)
        else:
            missing_values_mask = torch.ones(data.shape[0]) == 0.
        sat_per_datapoint[missing_values_mask] = True
        samples_sat_constr = samples_sat_constr & sat_per_datapoint
    return samples_sat_constr