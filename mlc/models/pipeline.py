from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from mlc.models.model import Model
from mlc.transformers.transformer import Transformer


class Pipeline(Model):
    def __init__(self, steps: List[Union[Transformer, Model]]):
        super().__init__(name=steps[-1].name, objective=steps[-1].objective)
        self.steps = steps
        # For compatibility

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ):
        for step in self.steps[:-1]:
            x = step.fit_transform(x, y)
        self.steps[-1].fit(x, y, x_val, y_val)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = self.transform(x)
        return self.steps[-1].predict_proba(x)

    def predict(
        self, x: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        for step in self.steps[:-1]:
            x = step.transform(x)
        return self.steps[-1].predict(x)

    def load(self, path):
        for i, step in enumerate(self.steps):
            step.load(f"{path}/{i}_{step.name}.model")

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        for i, step in enumerate(self.steps):
            step.save(f"{path}/{i}_{step.name}.model")

    def clone(self):
        return self.__class__([step.clone() for step in self.steps])

    def __getitem__(self, item):
        return self.steps[item]

    def transform(self, x):
        for step in self.steps[:-1]:
            x = step.transform(x)
        return x
