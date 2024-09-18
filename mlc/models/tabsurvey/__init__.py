from typing import List, Tuple, Type

from mlc.models.model import Model
from mlc.models.tabsurvey.mlp_rln import models as mlp_rln_models
from mlc.models.tabsurvey.tabtransformer import models as tabtransformer_models
from mlc.models.tabsurvey.vime import models as vime_models

models: List[Tuple[str, Type[Model]]] = (
    tabtransformer_models
    + mlp_rln_models
    + vime_models
)
