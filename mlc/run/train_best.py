import os
import sys

sys.path.append(".")
from mlc.logging import XP
import joblib
import torch
import optuna
from optuna.trial import TrialState
from mlc.datasets.dataset_factory import load_dataset
from mlc.models.model_factory import load_model
from argparse import ArgumentParser, Namespace
from mlc.transformers.tab_scaler import TabScaler
from cdgm.data_processors.ctgan.data_transformer import DataTransformer
from mlc.metrics.compute import compute_metric, compute_metrics
from mlc.metrics.metric_factory import create_metric
from mlc.dataloaders import get_custom_dataloader

# Torch config to avoid crash on HPC
torch.multiprocessing.set_sharing_strategy('file_system')

CUSTOM_DATALOADERS = ["default"]


def run(dataset_name: str = "lcld_v2_iid", model_name: str = "tabtransformer",  scaler_type: str = "TabScalerOHE", n_trials: int = 100, subset: int = 1000,
        train_batch_size: int = 256, val_batch_size: int = 1024, epochs: int = 100, verbose: int = 0, device="cuda",
        custom_dataloaders: list = CUSTOM_DATALOADERS, num_classes: int = 2) -> None:
    print("Train hyperparameter optimization for {} on {}".format(model_name, dataset_name))
    dataset = load_dataset(dataset_name)
    metadata = dataset.get_metadata(only_x=True)
    common_model_params = {
        "x_metadata": metadata,
        "objective": "classification",
        "use_gpu": True,
        "batch_size": train_batch_size,
        "num_classes": num_classes,
        "early_stopping_rounds": 5,
        "val_batch_size": val_batch_size,
        "class_weight": "balanced",
        "custom_dataloader": ""
    }

    x, y = dataset.get_x_y()
    splits = dataset.get_splits()

    x_train = x.iloc[splits["train"]]
    y_train = y[splits["train"]]

    if subset > 0:
        x_train = x_train[:subset]
        y_train = y_train[:subset]

    model_class = load_model(model_name)

    study_name = model_name + "_" + dataset_name + "_" + scaler_type
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction="maximize",
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)

    n_completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
    print(f"{n_completed}/{n_trials} already completed")

    if scaler_type == "TabScalerOHE":
        scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
        scaler.fit(
        torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"]
    )
    elif scaler_type == "TabScaler":
        scaler = TabScaler(num_scaler="min_max", one_hot_encode=False)
        scaler.fit(
        torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"]
    )
    elif scaler_type == "CTGAN":
        scaler = DataTransformer()
        scaler.fit(x.iloc[:,:-1], metadata[metadata["type"]=="cat"].index.to_list())

    else:
        raise ValueError("Wrong value for scaler_type. Accepted choices are TabScalerOHE, TabScaler and CTGAN")

    args = {
        **common_model_params,
        "scaler_type": scaler_type,
        "num_classes": num_classes,
        "depth": 12,
        "heads": 2,  # 4,
        "weight_decay": -3,  # -2,
        "learning_rate": -3,  # -2,
        "dropout": 0.5,  # 0.3,
        "data_parallel": False,
        "epochs": epochs,
        "model_name": model_name,
        "dataset": dataset_name,
        "early_stopping_rounds": 5,
        "num_splits": 5,
        "seed": 42,
        "shuffle": True,
        "metrics": ["auc"]
    }
    
    if model_name == "torchrln":
        args["weight_decay"] = 0
        args["n_layers"] = 5
        args["hidden_dim"] = 100
        args["learning_rate"] = 0.001

    best_args = {**args, "task": "train", "trial": "best"}
    if len(study.trials) > 1:
        best_trial = sorted(study.trials, key=lambda d: d.value if d.values is not None else -1)[-1]
        print(f"Best trial parameters: {best_trial.params} with best performance: {best_trial.value}")
        best_args = {**best_args, **study.best_trial.params}
    else:
        best_trial = Namespace(params={},value=0, number=-1)
        print("There is no best trial, using default params!!!")

    best_args["x_metadata"] = None

    for custom_dataloader in custom_dataloaders:
        best_args = {**best_args, **best_trial.params, "best_trial_value": best_trial.value,
                     "best_trial_index": best_trial.number, "custom_dataloader": custom_dataloader}
        experiment = XP(best_args, project_name="mlc_best_v2")

        model = model_class(**{**args, **best_trial.params, "early_stopping_rounds": args["epochs"],
                               "force_device":device if device!="" else None}, scaler=scaler)

        x_test = x.iloc[splits["test"]]
        y_test = y[splits["test"]]

        x_dataloader = x.iloc[splits["val"]].values if custom_dataloader=="dist" else x_train.values
        y_dataloader = y[splits["val"]] if custom_dataloader == "dist" else y_train

        custom_train_dataloader = get_custom_dataloader(custom_dataloader, dataset, model, scaler, {}, verbose=verbose,
                                                        x=x_dataloader, y=y_dataloader, train=True,
                                                        batch_size=train_batch_size)
        model.fit(x_train, y_train, x_test, y_test, custom_train_dataloader=custom_train_dataloader)
        model.eval()

        if num_classes > 2:
            metrics = ["accuracy", "f1_multiclass", "f1_weighted", "auc"]
        else:
            metrics = ["accuracy", "f1", "f1_weighted", "precision", "recall", "auc"]
        metrics_obj = [create_metric(m) for m in metrics]
        res = compute_metrics(model, metrics_obj, x_test.values, y_test)

        for i, e in enumerate(metrics_obj):
            print(f"Test {metrics[i]}: {res[i]}")
            experiment.log_metric(metrics[i], res[i])
        experiment.log_table("metrics.csv",
                     [["Experiment"]+metrics,
                      [experiment.experiment_name]+res.tolist()])

        save_path = os.path.join(".", "data", "best_models", "{}_{}_{}_{}.model".format(model_name, dataset_name,
                                                                                     custom_dataloader, scaler_type))
        model.save(save_path)
        joblib.dump(scaler, save_path+"/scaler.joblib")

        experiment.log_model("{}_{}_{}.model".format(model_name, dataset_name, custom_dataloader), save_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training with Hyper-parameter optimization"
    )
    parser.add_argument("--dataset_name", type=str, default="lcld_v2_iid",
                        )
    parser.add_argument("--model_name", type=str, default="tabtransformer",
                        )
    parser.add_argument("--scaler_type", type=str, default="TabScalerOHE",
                        )
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--val_batch_size", type=int, default=2048)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda",
                        )
    parser.add_argument("--custom_dataloaders", type=str, default="default",
                        )

    args = parser.parse_args()

    run(dataset_name=args.dataset_name, model_name=args.model_name, scaler_type=args.scaler_type, n_trials=args.n_trials, subset=args.subset,
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, verbose=args.verbose,
        epochs=args.epochs, device=args.device, custom_dataloaders=args.custom_dataloaders.split("+"), num_classes=args.num_classes)
