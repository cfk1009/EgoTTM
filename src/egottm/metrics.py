"""Evaluation metrics matching the original Ego4D TTM implementation."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class EvaluationMetrics:
    """Two-class AP, mean AP, threshold accuracy, and confusion counts."""

    ap_non_ttm: float
    ap_ttm: float
    map: float
    accuracy: float
    tp: int
    fn: int
    fp: int
    tn: int


def compute_average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute VOC-style area under a monotonically smoothed PR curve."""

    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)

    if precision.ndim != 1 or recall.ndim != 1:
        raise ValueError("precision and recall must be one-dimensional")
    if len(precision) != len(recall):
        raise ValueError("precision and recall must have the same length")
    if not precision.size:
        return 0.0
    if np.any((precision < 0) | (precision > 1)):
        raise ValueError("precision must be in [0, 1]")
    if np.any((recall < 0) | (recall > 1)):
        raise ValueError("recall must be in [0, 1]")

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])

    changing_recall = np.where(recall[1:] != recall[:-1])[0] + 1

    return float(
        np.sum(
            (recall[changing_recall] - recall[changing_recall - 1])
            * precision[changing_recall]
        )
    )


def _class_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    class_id: int,
) -> float:
    class_scores = scores if class_id == 1 else 1.0 - scores
    is_positive = labels == class_id
    positive_count = int(is_positive.sum())

    if positive_count == 0:
        raise ValueError("labels must contain both TTM classes")

    order = np.argsort(-class_scores, kind="stable")
    true_positives = np.cumsum(is_positive[order], dtype=np.float64)
    precision = true_positives / np.arange(
        1,
        len(labels) + 1,
        dtype=np.float64,
    )
    recall = true_positives / float(positive_count)

    return compute_average_precision(precision, recall)


def evaluate_binary_scores(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Evaluate TTM scores using the original two-class mAP definition."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if len(labels) == 0:
        raise ValueError("labels and scores cannot be empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)

    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if not np.all(np.isin(label_array, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")
    if (
        not np.all(np.isfinite(score_array))
        or np.any((score_array < 0) | (score_array > 1))
    ):
        raise ValueError("scores must be finite values in [0, 1]")

    ap_non_ttm = _class_average_precision(
        label_array,
        score_array,
        class_id=0,
    )
    ap_ttm = _class_average_precision(
        label_array,
        score_array,
        class_id=1,
    )

    predicted_positive = score_array >= threshold
    actual_positive = label_array == 1

    tp = int(np.sum(predicted_positive & actual_positive))
    fn = int(np.sum(~predicted_positive & actual_positive))
    fp = int(np.sum(predicted_positive & ~actual_positive))
    tn = int(np.sum(~predicted_positive & ~actual_positive))

    accuracy = float((tp + tn) / len(label_array))

    return EvaluationMetrics(
        ap_non_ttm=ap_non_ttm,
        ap_ttm=ap_ttm,
        map=float((ap_non_ttm + ap_ttm) / 2.0),
        accuracy=accuracy,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
    )


def evaluate_prediction_rows(
    rows: Iterable[Mapping],
    threshold: float = 0.5,
) -> Optional[EvaluationMetrics]:
    """Evaluate inference rows, or return None when labels are unavailable."""

    rows = list(rows)

    if not rows:
        return None

    labels = [row.get("label") for row in rows]

    if any(label is None or label == "" for label in labels):
        return None

    return evaluate_binary_scores(
        labels=[int(label) for label in labels],
        scores=[float(row["score"]) for row in rows],
        threshold=threshold,
    )


def format_evaluation(
    metrics: EvaluationMetrics,
    threshold: float = 0.5,
) -> str:
    """Format final infer metrics as two concise log lines."""

    return (
        "[infer] metrics: AP(non-TTM)={:.6f} AP(TTM)={:.6f} "
        "mAP={:.6f} Accuracy@{:.3f}={:.6f}\n"
        "[infer] confusion: TP={} FN={} FP={} TN={}"
    ).format(
        metrics.ap_non_ttm,
        metrics.ap_ttm,
        metrics.map,
        threshold,
        metrics.accuracy,
        metrics.tp,
        metrics.fn,
        metrics.fp,
        metrics.tn,
    )
