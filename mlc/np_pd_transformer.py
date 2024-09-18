from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NpPdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.columns = []

    def fit(self, x: pd.DataFrame, y=None):
        self.columns = x.columns
        return self

    def transform(self, x: Union[np.array, pd.DataFrame], y=None):
        if isinstance(x, pd.DataFrame):
            return x

        return pd.DataFrame(x, columns=self.columns)
