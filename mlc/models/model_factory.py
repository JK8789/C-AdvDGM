import importlib
from typing import Any, Dict, Type, Union

from mlc.models.model import Model
from mlc.models.samples import load_model


def get_model_from_config(config: Dict[str, Any]) -> Type[Model]:
    name = config["name"]

    # If it is a known dataset
    if len(config.keys()) == 1:
        return load_model(name)

    # Else load and do
    models = importlib.import_module(config["source"]).models
    models_out = list(filter(lambda e: e[0] in [name], models))
    models_out = [e[1] for e in models_out]

    if len(models_out) != 1:
        raise NotImplementedError("At least one model is not available.")
    return models_out[0]


def get_model(config: Union[Dict[str, Any], str]) -> Type[Model]:
    if isinstance(config, str):
        config = {"name": config}
    return get_model_from_config(config)
