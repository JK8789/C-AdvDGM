from typing import Dict, List

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from mlc.constraints.relation_constraint import BaseRelationConstraint
from mlc.constraints.relation_constraint import Constant as Co
from mlc.constraints.relation_constraint import Feature as Fe
from mlc.datasets.dataset import (
    CsvDataSource,
    Dataset,
    DataSource,
    DefaultIndexSorter,
    DownloadFileDataSource,
    Splitter,
    Task,
)


class FaultsSplitter(Splitter):
    def get_splits(self, dataset: Dataset) -> Dict[str, npt.NDArray[np.int_]]:
        _, y = dataset.get_x_y()
        i = np.arange(len(y))
        i_train = i[:1553]
        i_val = i[1553:1746]
        i_test = i[1746:]

        return {"train": i_train, "val": i_val, "test": i_test}


def get_relation_constraints(
    metadata: DataSource,
) -> List[BaseRelationConstraint]:

    g1 = Fe(0) - Fe(1)
    g2 = Fe(2) - Fe(3)
    g3 = Fe(8) - Fe(9)
    g4 = Fe(9) - Fe(7)
    return [g1, g2, g3, g4]


def create_dataset() -> Dataset:
    data_source = DownloadFileDataSource(
        url="",
        file_data_source=CsvDataSource(
            path="./data/faults/train_data.csv"
        ),
    )
    metadata_source = DownloadFileDataSource(
        url="",
        file_data_source=CsvDataSource(
            path="./data/faults/faults_metadata.csv"
        ),
    )
    tasks = [
        Task(
            name="faults",
            task_type="classification",
            evaluation_metric="f1_score",
        )
    ]
    sorter = DefaultIndexSorter()
    splitter = FaultsSplitter()
    relation_constraints = get_relation_constraints(metadata_source)

    faults = Dataset(
        name="faults",
        data_source=data_source,
        metadata_source=metadata_source,
        tasks=tasks,
        sorter=sorter,
        splitter=splitter,
        relation_constraints=relation_constraints,
    )
    return faults


datasets = [
    {
        "name": "faults",
        "fun_create": create_dataset,
    }
]
