from __future__ import annotations

import abc
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import TransformerMixin


class Transformer(TransformerMixin):
    def __init__(self, **kwargs: Any) -> None:
        self.name = kwargs["name"]
        self.constructor_kwargs = kwargs

    @abc.abstractmethod
    def fit(
        self,
        x: Union[npt.NDArray[np.float_], pd.DataFrame],
        y: Union[npt.NDArray[np.int_], pd.Series] = None,
    ) -> Transformer:
        pass

    @abc.abstractmethod
    def transform(
        self, x: Union[npt.NDArray[np.float_], pd.DataFrame]
    ) -> Union[npt.NDArray[np.float_], pd.DataFrame]:
        pass

    def clone(self) -> Transformer:
        return self.__class__(**self.constructor_kwargs)

    @abc.abstractmethod
    def load(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        pass
