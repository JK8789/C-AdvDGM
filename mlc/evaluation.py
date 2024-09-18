import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_score, threshold=None) -> dict:
    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = (y_score[:, 1] >= threshold).astype(int)

    metrics = {
        **classification_report(y_true, y_pred, output_dict=True),
        **{
            "roc_auc_score": roc_auc_score(y_true, y_score[:, 1]),
            "precision_score": precision_score(y_true, y_pred),
            "recall_score": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        },
    }
    return metrics
