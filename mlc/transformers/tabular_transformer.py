from typing import Any, Dict, Union

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlc.transformers.tokennizer import TokenizerTransformer
from mlc.transformers.transformer import Transformer


class TabTransformer(Transformer):
    def __init__(
        self,
        metadata: pd.DataFrame,
        scale: bool,
        one_hot_encode: bool,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            name="tab_transformer",
            metadata=metadata,
            scale=scale,
            one_hot_encode=one_hot_encode,
            **kwargs,
        )
        cat_index = metadata["type"] == "cat"
        cat_features = metadata[cat_index]["feature"]
        num_index = (metadata["type"] == "int") | (metadata["type"] == "real")
        num_features = metadata[num_index]["feature"]
        text_index = metadata["type"] == "text"
        text_features = metadata[text_index]["feature"]

        cat_min = metadata[cat_index]["min"].to_list()
        cat_max = metadata[cat_index]["max"].to_list()
        cat_range = [
            np.arange(int(cat_min[i]), int(cat_max[i]) + 1)
            for i in range(len(cat_min))
        ]
        transformers = []
        if scale:
            transformers.append(("num", StandardScaler(), num_features))
        if one_hot_encode:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(
                        sparse=False,
                        handle_unknown="ignore",
                        drop="if_binary",
                        categories=cat_range,
                    ),
                    cat_features,
                )
            )

        if len(metadata[text_index]["max"]) > 0:
            self.max_token_length = int(metadata[text_index]["max"].max())
            transformers.append(
                (
                    "text",
                    TokenizerTransformer(self.max_token_length),
                    text_features,
                )
            )

        self.transformer = ColumnTransformer(
            transformers=transformers,
            sparse_threshold=0,
            remainder="passthrough",
            n_jobs=-1,
        )

    def fit(
        self,
        x: Union[npt.NDArray[np.float_], pd.DataFrame],
        y: Union[npt.NDArray[np.int_], pd.Series] = None,
    ) -> Transformer:
        self.transformer.fit(x, y)
        return self

    def transform(
        self, x: Union[npt.NDArray[np.float_], pd.DataFrame]
    ) -> Union[npt.NDArray[np.float_], pd.DataFrame]:
        return self.transformer.transform(x)

    def load(self, path: str) -> None:
        text_pending_idx = []
        for i, e in enumerate(self.transformer.transformers):
            if e[0] == "text":
                text_pending_idx.append(i)

        text_pending = [
            self.transformer.transformers[i] for i in text_pending_idx
        ]
        self.transformer = joblib.load(path)
        for i in range(len(text_pending_idx)):
            self.transformer.transformers[text_pending_idx[i]] = text_pending[
                i
            ]
            self.transformer.transformers_[text_pending_idx[i]] = text_pending[
                i
            ]

    def save(self, path: str) -> None:
        text_pending_idx = []
        for i, e in enumerate(self.transformer.transformers):
            if e[0] == "text":
                text_pending_idx.append(i)

        text_pending = [
            self.transformer.transformers[i] for i in text_pending_idx
        ]
        for i in text_pending_idx:
            self.transformer.transformers[i] = None
            self.transformer.transformers_[i] = None

        joblib.dump(self.transformer, path)

        for i in range(len(text_pending_idx)):
            self.transformer.transformers[text_pending_idx[i]] = text_pending[
                i
            ]
            self.transformer.transformers_[text_pending_idx[i]] = text_pending[
                i
            ]
