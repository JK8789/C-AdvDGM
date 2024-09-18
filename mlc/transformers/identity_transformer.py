from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from mlc.transformers.transformer import Transformer


class IdentityTransformer(Transformer):
    def fit(
        self,
        x: Union[npt.NDArray[np.float_], pd.DataFrame],
        y: Union[npt.NDArray[np.int_], pd.Series] = None,
    ) -> Transformer:
        return self  # Nothing to fit

    def transform(
        self, x: Union[npt.NDArray[np.float_], pd.DataFrame]
    ) -> Union[npt.NDArray[np.float_], pd.DataFrame]:
        return x

    def load(self, path: str) -> None:
        # Nothing to load
        pass

    def save(self, path: str) -> None:
        # Nothing to save
        pass
