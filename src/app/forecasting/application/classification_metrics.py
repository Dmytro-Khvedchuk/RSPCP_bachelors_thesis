"""Classification metrics with confidence-based abstention for direction forecasters.

Provides standalone metrics (accuracy, precision/recall/F1, AUC-ROC),
confidence-based abstention curves, classifier reliability diagrams (ECE),
economic accuracy weighted by absolute returns, and asymmetric class
weighting that penalises missed crashes.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-12
"""Small constant to avoid division by zero."""

_DEFAULT_ABSTENTION_THRESHOLDS: tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7)
"""Default confidence thresholds for abstention analysis."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class ClassificationMetrics(BaseModel, frozen=True):
    """Overall classification metrics for a direction forecaster.

    Attributes:
        accuracy: Fraction of correct direction predictions.
        precision_up: Precision for class +1 (long).
        recall_up: Recall for class +1 (long).
        f1_up: F1 score for class +1 (long).
        precision_down: Precision for class -1 (short).
        recall_down: Recall for class -1 (short).
        f1_down: F1 score for class -1 (short).
        auc_roc: Area under the ROC curve.
        n_samples: Number of samples used for computation.
    """

    accuracy: float
    """Fraction of correct direction predictions."""

    precision_up: float
    """Precision for class +1 (long)."""

    recall_up: float
    """Recall for class +1 (long)."""

    f1_up: float
    """F1 score for class +1 (long)."""

    precision_down: float
    """Precision for class -1 (short)."""

    recall_down: float
    """Recall for class -1 (short)."""

    f1_down: float
    """F1 score for class -1 (short)."""

    auc_roc: float
    """Area under the ROC curve."""

    n_samples: int
    """Number of samples used."""


class AbstentionResult(BaseModel, frozen=True):
    """Metrics for a single confidence abstention threshold.

    Attributes:
        threshold: Confidence threshold applied.
        accuracy: Directional accuracy on retained samples.
        coverage: Fraction of total samples retained.
        n_retained: Number of samples retained at this threshold.
        n_total: Total number of samples before filtering.
    """

    threshold: float
    """Confidence threshold applied."""

    accuracy: float
    """Directional accuracy on retained samples."""

    coverage: float
    """Fraction of total samples retained."""

    n_retained: int
    """Number of samples retained at this threshold."""

    n_total: int
    """Total number of samples before filtering."""


class AbstentionCurve(BaseModel, frozen=True):
    """Accuracy-coverage tradeoff across multiple confidence thresholds.

    Attributes:
        results: One ``AbstentionResult`` per threshold.
        thresholds: The confidence threshold levels used.
    """

    results: tuple[AbstentionResult, ...]
    """One AbstentionResult per threshold."""

    thresholds: tuple[float, ...]
    """The confidence threshold levels used."""


class ClassificationReliabilityResult(BaseModel, frozen=True):
    """Reliability diagram data for a direction classifier.

    Bins predictions by confidence and compares mean predicted confidence
    against observed accuracy in each bin.  Reports Expected Calibration
    Error (ECE) as the weighted deviation.

    Attributes:
        bin_edges: Probability bin edges of shape ``(n_bins + 1,)``.
        bin_accuracies: Observed accuracy per bin of shape ``(n_bins,)``.
        bin_confidences: Mean predicted confidence per bin of shape ``(n_bins,)``.
        bin_counts: Number of samples per bin of shape ``(n_bins,)``.
        n_bins: Number of bins used.
        n_samples: Total number of samples.
        ece: Expected Calibration Error.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bin_edges: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_bins + 1,)`` — probability bin edges."""

    bin_accuracies: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_bins,)`` — observed accuracy per bin."""

    bin_confidences: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_bins,)`` — mean predicted confidence per bin."""

    bin_counts: np.ndarray[tuple[int], np.dtype[np.int64]]
    """Shape ``(n_bins,)`` — number of samples per bin."""

    n_bins: int
    """Number of bins used."""

    n_samples: int
    """Total number of samples."""

    ece: float
    """Expected Calibration Error."""


class EconomicAccuracyResult(BaseModel, frozen=True):
    """Accuracy weighted by absolute return magnitude.

    Correct predictions on large price moves contribute more than correct
    predictions on small moves, reflecting economic significance.

    Attributes:
        economic_accuracy: Return-weighted accuracy.
        standard_accuracy: Unweighted accuracy for comparison.
        n_samples: Number of samples used.
    """

    economic_accuracy: float
    """Return-weighted accuracy."""

    standard_accuracy: float
    """Unweighted accuracy for comparison."""

    n_samples: int
    """Number of samples used."""


class AsymmetricMetrics(BaseModel, frozen=True):
    """Classification metrics with asymmetric class weights.

    Penalises missed crashes (class -1) more heavily than missed rallies,
    reflecting the negative skewness of crypto return distributions.

    Attributes:
        weighted_f1: F1 with asymmetric class weights.
        weighted_precision: Precision with asymmetric class weights.
        weighted_recall: Recall with asymmetric class weights.
        crash_weight: Weight applied to the down class (-1).
        rally_weight: Weight applied to the up class (+1).
    """

    weighted_f1: float
    """F1 with asymmetric class weights."""

    weighted_precision: float
    """Precision with asymmetric class weights."""

    weighted_recall: float
    """Recall with asymmetric class weights."""

    crash_weight: Annotated[
        float,
        PydanticField(description="Weight applied to the down class (-1)"),
    ]
    """Weight applied to the down class (-1)."""

    rally_weight: Annotated[
        float,
        PydanticField(description="Weight applied to the up class (+1)"),
    ]
    """Weight applied to the up class (+1)."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_binary_arrays(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> int:
    """Validate that y_true and y_pred are non-empty, same-length, and contain only {-1, +1}.

    Args:
        y_true: True direction labels.
        y_pred: Predicted direction labels.

    Returns:
        Number of samples.

    Raises:
        ValueError: If arrays are empty, different lengths, or contain invalid values.
    """
    n: int = y_true.shape[0]
    if n == 0:
        msg: str = "y_true must contain at least one sample"
        raise ValueError(msg)
    if y_pred.shape[0] != n:
        msg = f"y_pred length {y_pred.shape[0]} != y_true length {n}"
        raise ValueError(msg)

    valid_values: set[float] = {-1.0, 1.0}
    unique_true: set[float] = set(np.unique(y_true).tolist())
    unique_pred: set[float] = set(np.unique(y_pred).tolist())
    if not unique_true.issubset(valid_values):
        msg = f"y_true must contain only {{-1, +1}}, got unique values {unique_true}"
        raise ValueError(msg)
    if not unique_pred.issubset(valid_values):
        msg = f"y_pred must contain only {{-1, +1}}, got unique values {unique_pred}"
        raise ValueError(msg)

    return n


def _precision_recall_f1(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
    positive_label: float,
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for a single class.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        positive_label: The class to treat as positive (+1 or -1).

    Returns:
        Tuple of (precision, recall, f1).
    """
    tp: int = int(np.sum((y_pred == positive_label) & (y_true == positive_label)))
    fp: int = int(np.sum((y_pred == positive_label) & (y_true != positive_label)))
    fn: int = int(np.sum((y_pred != positive_label) & (y_true == positive_label)))

    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1: float = 2.0 * precision * recall / (precision + recall) if (precision + recall) > _EPSILON else 0.0

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_proba_positive: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> ClassificationMetrics:
    """Compute core classification metrics for a direction forecaster.

    Metrics computed:

    - **Accuracy**: fraction of correct direction predictions
    - **Per-class precision / recall / F1**: for both up (+1) and down (-1)
    - **AUC-ROC**: discrimination ability using P(class=+1)

    For AUC-ROC, ``y_proba_positive`` must be the predicted probability of
    class +1.  This is converted to {0, 1} labels internally for the
    trapezoidal ROC integration.

    Args:
        y_true: True direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        y_pred: Predicted direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        y_proba_positive: Predicted probability for class +1 of shape ``(n_samples,)``,
            values in [0, 1].

    Returns:
        ClassificationMetrics with accuracy, per-class metrics, and AUC-ROC.

    Raises:
        ValueError: If arrays are empty, lengths differ, or contain invalid values.
    """
    n: int = _validate_binary_arrays(y_true, y_pred)
    if y_proba_positive.shape[0] != n:
        msg: str = f"y_proba_positive length {y_proba_positive.shape[0]} != y_true length {n}"
        raise ValueError(msg)

    # Accuracy
    accuracy: float = float(np.mean(y_true == y_pred))

    # Per-class precision, recall, F1
    precision_up: float
    recall_up: float
    f1_up: float
    precision_up, recall_up, f1_up = _precision_recall_f1(y_true, y_pred, positive_label=1.0)

    precision_down: float
    recall_down: float
    f1_down: float
    precision_down, recall_down, f1_down = _precision_recall_f1(y_true, y_pred, positive_label=-1.0)

    # AUC-ROC via trapezoidal rule
    # Convert {-1, +1} → {0, 1} for ROC computation
    y_true_binary: np.ndarray[tuple[int], np.dtype[np.float64]] = ((y_true + 1.0) / 2.0).astype(np.float64)
    auc_roc: float = _compute_auc_roc(y_true_binary, y_proba_positive)

    logger.debug(
        "Classification metrics (n={}): acc={:.4f}, F1_up={:.4f}, F1_down={:.4f}, AUC={:.4f}",
        n,
        accuracy,
        f1_up,
        f1_down,
        auc_roc,
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_up=precision_up,
        recall_up=recall_up,
        f1_up=f1_up,
        precision_down=precision_down,
        recall_down=recall_down,
        f1_down=f1_down,
        auc_roc=auc_roc,
        n_samples=n,
    )


def _compute_auc_roc(
    y_true_binary: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_scores: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    """Compute AUC-ROC via the trapezoidal rule (no sklearn dependency).

    Args:
        y_true_binary: True binary labels of shape ``(n_samples,)``, values in {0, 1}.
        y_scores: Predicted scores (probabilities) of shape ``(n_samples,)``.

    Returns:
        Area under the ROC curve.
    """
    n: int = y_true_binary.shape[0]
    n_pos: int = int(np.sum(y_true_binary == 1.0))
    n_neg: int = n - n_pos

    # Degenerate: only one class present
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by decreasing score
    desc_indices: np.ndarray[tuple[int], np.dtype[np.intp]] = np.argsort(-y_scores)
    y_sorted: np.ndarray[tuple[int], np.dtype[np.float64]] = y_true_binary[desc_indices]

    # Build ROC curve: accumulate TPR and FPR
    tps: np.ndarray[tuple[int], np.dtype[np.float64]] = np.cumsum(y_sorted).astype(np.float64)
    fps: np.ndarray[tuple[int], np.dtype[np.float64]] = (np.arange(1, n + 1) - tps).astype(np.float64)

    tpr: np.ndarray[tuple[int], np.dtype[np.float64]] = (tps / n_pos).astype(np.float64)
    fpr: np.ndarray[tuple[int], np.dtype[np.float64]] = (fps / n_neg).astype(np.float64)

    # Prepend (0, 0) for the origin
    tpr_full: np.ndarray[tuple[int], np.dtype[np.float64]] = np.concatenate([np.array([0.0], dtype=np.float64), tpr])
    fpr_full: np.ndarray[tuple[int], np.dtype[np.float64]] = np.concatenate([np.array([0.0], dtype=np.float64), fpr])

    # Trapezoidal integration
    auc: float = float(np.trapezoid(tpr_full, fpr_full))
    return auc


# ---------------------------------------------------------------------------
# Confidence-based abstention
# ---------------------------------------------------------------------------


def compute_abstention_curve(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
    confidences: np.ndarray[tuple[int], np.dtype[np.float64]],
    thresholds: tuple[float, ...] = _DEFAULT_ABSTENTION_THRESHOLDS,
) -> AbstentionCurve:
    """Compute accuracy-coverage tradeoff at multiple confidence thresholds.

    At each threshold, only samples where ``confidence >= threshold`` are
    retained.  This models the classifier as a filter: a model with
    DA=56% on 30% of bars is vastly more useful than DA=51% on 100%.

    Args:
        y_true: True direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        y_pred: Predicted direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        confidences: Predicted probability for the chosen direction of shape
            ``(n_samples,)``, values in [0, 1].
        thresholds: Confidence threshold levels to evaluate.

    Returns:
        AbstentionCurve with one AbstentionResult per threshold.

    Raises:
        ValueError: If arrays are empty, lengths differ, or thresholds are empty.
    """
    n: int = _validate_binary_arrays(y_true, y_pred)
    if confidences.shape[0] != n:
        msg: str = f"confidences length {confidences.shape[0]} != y_true length {n}"
        raise ValueError(msg)
    if len(thresholds) == 0:
        msg = "thresholds must not be empty"
        raise ValueError(msg)

    results: list[AbstentionResult] = []
    for threshold in thresholds:
        mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = confidences >= threshold
        n_retained: int = int(np.sum(mask))

        if n_retained == 0:
            acc: float = 0.0
        else:
            acc = float(np.mean(y_true[mask] == y_pred[mask]))

        coverage: float = n_retained / n

        results.append(
            AbstentionResult(
                threshold=threshold,
                accuracy=acc,
                coverage=coverage,
                n_retained=n_retained,
                n_total=n,
            )
        )

    logger.debug(
        "Abstention curve (n={}): {} thresholds, coverage range [{:.2f}, {:.2f}]",
        n,
        len(thresholds),
        results[-1].coverage if results else 0.0,
        results[0].coverage if results else 0.0,
    )

    return AbstentionCurve(
        results=tuple(results),
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Classifier reliability diagram (ECE)
# ---------------------------------------------------------------------------


def compute_classification_reliability(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
    confidences: np.ndarray[tuple[int], np.dtype[np.float64]],
    n_bins: int = 10,
) -> ClassificationReliabilityResult:
    """Compute a reliability diagram and Expected Calibration Error (ECE).

    Bins predictions by confidence level and compares the mean predicted
    confidence against the actual accuracy in each bin.  A well-calibrated
    model should have ``bin_accuracies ≈ bin_confidences`` in every bin.

    ECE is the weighted average of ``|accuracy_b - confidence_b|`` across
    bins, weighted by the fraction of samples in each bin.

    Args:
        y_true: True direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        y_pred: Predicted direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        confidences: Predicted probability for the chosen direction of shape
            ``(n_samples,)``, values in [0, 1].
        n_bins: Number of equal-width bins for the reliability diagram.

    Returns:
        ClassificationReliabilityResult with bin-level data and ECE.

    Raises:
        ValueError: If arrays are empty, lengths differ, or n_bins < 1.
    """
    n: int = _validate_binary_arrays(y_true, y_pred)
    if confidences.shape[0] != n:
        msg: str = f"confidences length {confidences.shape[0]} != y_true length {n}"
        raise ValueError(msg)
    if n_bins < 1:
        msg = f"n_bins must be >= 1, got {n_bins}"
        raise ValueError(msg)

    bin_edges: np.ndarray[tuple[int], np.dtype[np.float64]] = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    bin_accuracies: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(n_bins, dtype=np.float64)
    bin_confidences: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(n_bins, dtype=np.float64)
    bin_counts: np.ndarray[tuple[int], np.dtype[np.int64]] = np.zeros(n_bins, dtype=np.int64)

    correct: np.ndarray[tuple[int], np.dtype[np.bool_]] = y_true == y_pred

    # Digitize: bin index for each sample (1-indexed, clip to [1, n_bins])
    bin_indices: np.ndarray[tuple[int], np.dtype[np.intp]] = np.digitize(confidences, bin_edges[1:-1])

    for b in range(n_bins):
        mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = bin_indices == b
        count: int = int(np.sum(mask))
        bin_counts[b] = count
        if count > 0:
            bin_accuracies[b] = float(np.mean(correct[mask]))
            bin_confidences[b] = float(np.mean(confidences[mask]))

    # ECE: weighted average of |accuracy - confidence| per bin
    weights: np.ndarray[tuple[int], np.dtype[np.float64]] = (bin_counts / n).astype(np.float64)
    ece: float = float(np.sum(weights * np.abs(bin_accuracies - bin_confidences)))

    logger.debug(
        "Classification reliability (n={}, bins={}): ECE={:.4f}",
        n,
        n_bins,
        ece,
    )

    return ClassificationReliabilityResult(
        bin_edges=bin_edges,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        n_bins=n_bins,
        n_samples=n,
        ece=ece,
    )


# ---------------------------------------------------------------------------
# Economic accuracy
# ---------------------------------------------------------------------------


def compute_economic_accuracy(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
    actual_returns: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> EconomicAccuracyResult:
    """Compute return-weighted accuracy (economic accuracy).

    Each prediction is weighted by the absolute value of the actual return.
    Correct predictions on large price moves contribute more than correct
    predictions during low-volatility periods.

    Formula::

        economic_accuracy = sum(|return_i| * correct_i) / sum(|return_i|)

    Args:
        y_true: True direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        y_pred: Predicted direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        actual_returns: Actual returns of shape ``(n_samples,)``.

    Returns:
        EconomicAccuracyResult with economic and standard accuracy.

    Raises:
        ValueError: If arrays are empty, lengths differ, or all returns are zero.
    """
    n: int = _validate_binary_arrays(y_true, y_pred)
    if actual_returns.shape[0] != n:
        msg: str = f"actual_returns length {actual_returns.shape[0]} != y_true length {n}"
        raise ValueError(msg)

    correct: np.ndarray[tuple[int], np.dtype[np.float64]] = (y_true == y_pred).astype(np.float64)
    abs_returns: np.ndarray[tuple[int], np.dtype[np.float64]] = np.abs(actual_returns).astype(np.float64)

    total_weight: float = float(np.sum(abs_returns))
    if total_weight < _EPSILON:
        msg = "actual_returns must not all be zero"
        raise ValueError(msg)

    economic_accuracy: float = float(np.sum(abs_returns * correct)) / total_weight
    standard_accuracy: float = float(np.mean(correct))

    logger.debug(
        "Economic accuracy (n={}): econ_acc={:.4f}, std_acc={:.4f}",
        n,
        economic_accuracy,
        standard_accuracy,
    )

    return EconomicAccuracyResult(
        economic_accuracy=economic_accuracy,
        standard_accuracy=standard_accuracy,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# Asymmetric class weighting
# ---------------------------------------------------------------------------


def compute_asymmetric_metrics(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
    crash_weight: float = 1.5,
    rally_weight: float = 1.0,
) -> AsymmetricMetrics:
    """Compute precision, recall, F1 with asymmetric class weights.

    Applies higher weight to the down class (-1) to penalise missed
    crashes, reflecting the negative skewness (-0.36) of crypto return
    distributions where tail losses are more severe than tail gains.

    The weighted metrics are computed as::

        weighted_metric = (w_down * metric_down * n_down + w_up * metric_up * n_up)
                         / (w_down * n_down + w_up * n_up)

    Args:
        y_true: True direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        y_pred: Predicted direction labels of shape ``(n_samples,)``, values in {-1, +1}.
        crash_weight: Weight for the down class (-1).  Default 1.5.
        rally_weight: Weight for the up class (+1).  Default 1.0.

    Returns:
        AsymmetricMetrics with weighted F1, precision, and recall.

    Raises:
        ValueError: If arrays are empty, lengths differ, or weights are non-positive.
    """
    if crash_weight <= 0.0:
        msg: str = f"crash_weight must be positive, got {crash_weight}"
        raise ValueError(msg)
    if rally_weight <= 0.0:
        msg = f"rally_weight must be positive, got {rally_weight}"
        raise ValueError(msg)

    _validate_binary_arrays(y_true, y_pred)

    # Per-class metrics
    precision_up: float
    recall_up: float
    f1_up: float
    precision_up, recall_up, f1_up = _precision_recall_f1(y_true, y_pred, positive_label=1.0)

    precision_down: float
    recall_down: float
    f1_down: float
    precision_down, recall_down, f1_down = _precision_recall_f1(y_true, y_pred, positive_label=-1.0)

    # Class counts
    n_up: int = int(np.sum(y_true == 1.0))
    n_down: int = int(np.sum(y_true == -1.0))

    # Weighted combination
    total_weight: float = rally_weight * n_up + crash_weight * n_down
    if total_weight < _EPSILON:
        # Degenerate: no samples
        weighted_precision: float = 0.0
        weighted_recall: float = 0.0
        weighted_f1: float = 0.0
    else:
        weighted_precision = (
            rally_weight * precision_up * n_up + crash_weight * precision_down * n_down
        ) / total_weight
        weighted_recall = (rally_weight * recall_up * n_up + crash_weight * recall_down * n_down) / total_weight
        weighted_f1 = (rally_weight * f1_up * n_up + crash_weight * f1_down * n_down) / total_weight

    logger.debug(
        "Asymmetric metrics: wF1={:.4f}, wPrec={:.4f}, wRec={:.4f}, crash_w={:.2f}, rally_w={:.2f}",
        weighted_f1,
        weighted_precision,
        weighted_recall,
        crash_weight,
        rally_weight,
    )

    return AsymmetricMetrics(
        weighted_f1=weighted_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        crash_weight=crash_weight,
        rally_weight=rally_weight,
    )
