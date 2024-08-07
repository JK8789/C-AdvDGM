# Standard imports
import json
import os
import random
import sys

# 3rd party
import wandb
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from mlc.models.model_factory import load_model


class NumpyArrayIterator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_samples = data.shape[0]
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.current_batch = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_batch < self.num_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = min((self.current_batch + 1) * self.batch_size, self.num_samples)
            batch_data = self.data[start_idx:end_idx]
            self.current_batch += 1
            return batch_data
        else:
            raise StopIteration()

def all_div_gt_n(n, m):
    for i in range(n, m // 2):
         if m % i == 0:
            return i
    return 1


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(X_train, X_val, batch_size, seed):
    if X_val is None:
        X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=seed)

    train_dataset = TensorDataset(torch.Tensor(X_train.values))
    val_dataset = TensorDataset(torch.Tensor(X_val.values))

    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)

    dataloader = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        ),
    }
    return dataloader


def get_roundable_data(df):
    _is_roundable = ((df%1)==0).all(axis=0)
    roundable_cols = df.columns[_is_roundable]
    roundable_idx = [df.columns.get_loc(c) for c in roundable_cols]
    round_digits = df.iloc[:,roundable_idx].apply(get_round_decimals)
    return roundable_idx, round_digits


def single_value_cols(df):
    a = df.to_numpy()
    single_value = (a[0] == a).all(0)
    return df.columns[single_value].to_list()



def read_csv(csv_filename, use_case="", manual_inspection_cat_cols_idx=[]):
    """Read a csv file."""
    data = pd.read_csv(csv_filename)
    single_val_col = single_value_cols(data)

    # TODO: Create configuration files
    # TODO: Unify data loading for WGAN and the rest of the models
    cat_cols_names = data.columns[manual_inspection_cat_cols_idx].values.tolist()
    for col in single_val_col:
        try:
            cat_cols_names.remove(col)
        except Exception as e:
            pass
    cat_cols_idx = [data.columns.get_loc(c) for c in cat_cols_names if c in data]

    if len(cat_cols_idx) == 0:
        cat_cols_idx = []
        cat_cols_names = []
    return data, (cat_cols_names, cat_cols_idx)


def _load_json(path):
    print(os.system('pwd'))
    with open(path) as json_file:
        return json.load(json_file)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def round_func_BPDA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def metrics_to_wandb(metrics):
    columns = []
    data = []
    for k_prim, v_prim in metrics.items():
        for k_sec, v_sec in v_prim.items():
            columns.append(f"{k_prim}.{k_sec}")
            data.append(v_sec)
    df = pd.DataFrame(data=[data], columns=columns, index=[0])
    return df

def log_dict(name, results, logger):
    logger.info(f'\n{name}')
    for key in results:
        log_result = f"{key}: {results[key][0]}"
        logger.info(log_result)

def prepare_data_for_goggle(columns, X_original, ct=None, regression=False):  # ct is the column_transformer
    if regression:
        X = X_original

        if ct is None:
            ind = list(range(len(X.columns)))
            col_list = X.columns[ind]
            ct = ColumnTransformer([("scaler", StandardScaler(), col_list)], remainder="passthrough")
            X_ = ct.fit_transform(X)
        else:
            X_ = ct.transform(X)

        X = pd.DataFrame(X_, index=X.index, columns=X.columns)
        # target_col = X_original.columns[-1]
        # X = X.rename(columns={target_col: 'target'})

    else:
        X = X_original.drop(columns=columns[-1])
        y = X_original.drop(columns=columns[:-1])

        if ct is None:
            ind = list(range(len(X.columns)))
            col_list = X.columns[ind]
            ct = ColumnTransformer([("scaler", StandardScaler(), col_list)], remainder="passthrough")
            X_ = ct.fit_transform(X)
        else:
            X_ = ct.transform(X)

        X = pd.DataFrame(X_, index=X.index, columns=X.columns)
        X["target"] = y

    print(X.head())
    return X, ct



def get_model_preds(X, target_model, target_scaler):
    target_model.eval()
    # X_scaled = target_scaler.transform(X)
    probs = target_model.wrapper_model(X)
    pred_class = torch.argmax(probs, dim=1).int().detach().numpy()

    #probs = torch.nn.functional.softmax(probs)
    # probs = target_model.get_logits(X, False)
    # pred_class_logits = torch.argmax(probs, dim=1).int().detach().numpy()
    return pred_class

def get_max_decimal_places(col):
    decimal_places = col.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0)
    return decimal_places.max()

def get_round_decimals(col):
    MAX_DECIMALS = sys.float_info.dig - 1
    if (col == col.round(MAX_DECIMALS)).all():
        for decimal in range(MAX_DECIMALS + 1):
         if (col == col.round(decimal)).all():
             return decimal

def load_model_and_weights(
    dataset_name,
    model_name,
    custom_path,
    metadata,
    scaler,
    scaler_type,
    device,):
    print("###############")
    print(model_name)
    print(custom_path)

    # Load model
    model_class = load_model(model_name)
    print(model_class)
    weight_path = custom_path
    print(metadata)
    print(scaler_type)
    if not os.path.exists(weight_path):
        print("{} not found. Skipping".format(weight_path))
        return

    force_device = device if device != "" else None
    model = model_class.load_class(
        weight_path,
        x_metadata=metadata,
        scaler=scaler,
        scaler_type=scaler_type,
        force_device=force_device,
    )

    return model, weight_path

def set_pac_val(args):
        # set args.pac:
    if args.pac != 1:
        if args.batch_size % args.pac != 0:
            original_pac = args.pac
            args.pac = all_div_gt_n(original_pac, args.batch_size)
            print(f'Changed pac {original_pac} to {args.pac}')
    return args