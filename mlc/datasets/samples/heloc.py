from typing import Dict, List

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from mlc.constraints.relation_constraint import (
    BaseRelationConstraint,
    Constant,
    Feature,
    OrConstraint,
)
from mlc.datasets.dataset import (
    CsvDataSource,
    Dataset,
    DataSource,
    DefaultIndexSorter,
    DownloadFileDataSource,
    Splitter,
    Task,
)


class HelocSplitter(Splitter):
    def get_splits(self, dataset: Dataset) -> Dict[str, npt.NDArray[np.int_]]:
        _, y = dataset.get_x_y()
        i = np.arange(len(y))
        i_train = i[:7897]
        i_val = i[7897:9652]
        i_test = i[9652:]

        return {"train": i_train, "val": i_val, "test": i_test}


def get_relation_constraints(
    metadata: DataSource,
) -> List[BaseRelationConstraint]:
    def apply_if_a_supp_zero_than_b_supp_zero(
        a: Feature, b: Feature
    ) -> OrConstraint:
        return (Constant(0) <= a) | (Constant(0) < b)

    g1 = Feature(2) - Feature(1)
    g2 = Feature(6) - Feature(5)
    g3 = apply_if_a_supp_zero_than_b_supp_zero(Feature(4), Feature(7))
    g4 = apply_if_a_supp_zero_than_b_supp_zero(Feature(8), Feature(10))
    g5 = apply_if_a_supp_zero_than_b_supp_zero(Feature(15), Feature(14))
    g6 = Feature(4) - Feature(11)
    g7 = Feature(12) - Feature(11)
    g8 = Feature(4) - Feature(11)*Feature(7)
    g9 = Feature(19) - Feature(11)
    g10 = Feature(20) - Feature(11)
    g11 = Feature(20) - Feature(11)*Feature(13)
    g12 = Feature(21) - Feature(11)
    g13 = Feature(19) + Feature(20) - Feature(22)*Feature(11)

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13]


def create_dataset() -> Dataset:
    data_source = DownloadFileDataSource(
        url="",
        file_data_source=CsvDataSource(
            path="./data/mlc/heloc/heloc.csv"
        ),
    )
    metadata_source = DownloadFileDataSource(
        url="",
        file_data_source=CsvDataSource(
            path="./data/mlc/heloc/heloc_metadata.csv"
        ),
    )
    tasks = [
        Task(
            name="is_at_risk",
            task_type="classification",
            evaluation_metric="f1_score",
        )
    ]
    sorter = DefaultIndexSorter()
    splitter = HelocSplitter()
    relation_constraints = get_relation_constraints(metadata_source)

    heloc = Dataset(
        name="heloc",
        data_source=data_source,
        metadata_source=metadata_source,
        tasks=tasks,
        sorter=sorter,
        splitter=splitter,
        relation_constraints=relation_constraints,
    )
    return heloc


datasets = [
    {
        "name": "heloc",
        "fun_create": create_dataset,
    }
]
