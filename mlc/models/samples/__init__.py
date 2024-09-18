from typing import List, Tuple, Type, Union

from mlc.models.model import Model
from mlc.models.tabsurvey import models as tabsurvey_models

models: List[Tuple[str, Type[Model]]] = (
   tabsurvey_models
)


def load_model(model_name: str) -> Type[Model]:
    return load_models(model_name)[0]


def load_models(model_names: Union[str, List[str]]) -> List[Type[Model]]:

    if isinstance(model_names, str):
        model_names = [model_names]
    models_out = list(filter(lambda e: e[0] in model_names, models))
    models_out = [e[1] for e in models_out]

    if len(models_out) != len(model_names):
        raise NotImplementedError("At least one model is not available.")

    return models_out
