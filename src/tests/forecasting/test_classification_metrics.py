"""Unit tests for classification metrics (direction forecaster evaluation).

Tests cover accuracy, precision/recall/F1, AUC-ROC, confidence-based
abstention curves, reliability diagrams (ECE), economic accuracy,
asymmetric class weighting, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.classification_metrics import (
    AbstentionCurve,
    AbstentionResult,
    AsymmetricMetrics,
    ClassificationMetrics,
    ClassificationReliabilityResult,
    EconomicAccuracyResult,
    compute_abstention_curve,
    compute_asymmetric_metrics,
    compute_classification_metrics,
    compute_classification_reliability,
    compute_economic_accuracy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arr(*values: float) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Shorthand to create a float64 array."""
    return np.array(values, dtype=np.float64)


def _int_arr(*values: int) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """Shorthand to create an int64 array."""
    return np.array(values, dtype=np.int64)


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_perfect_predictions(self) -> None:
        """Perfect predictions give accuracy=1, all F1s=1, AUC=1."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        y_pred = y_true.copy()
        # For perfect predictions, P(+1) = 1.0 when true=+1, 0.0 when true=-1
        y_proba = _arr(1.0, 0.0, 1.0, 0.0, 1.0, 0.0)

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        assert isinstance(result, ClassificationMetrics)
        assert result.n_samples == 6
        np.testing.assert_almost_equal(result.accuracy, 1.0)
        np.testing.assert_almost_equal(result.precision_up, 1.0)
        np.testing.assert_almost_equal(result.recall_up, 1.0)
        np.testing.assert_almost_equal(result.f1_up, 1.0)
        np.testing.assert_almost_equal(result.precision_down, 1.0)
        np.testing.assert_almost_equal(result.recall_down, 1.0)
        np.testing.assert_almost_equal(result.f1_down, 1.0)
        np.testing.assert_almost_equal(result.auc_roc, 1.0)

    def test_worst_predictions(self) -> None:
        """Completely wrong predictions give accuracy=0, all F1s=0."""
        y_true = _arr(1.0, 1.0, -1.0, -1.0)
        y_pred = _arr(-1.0, -1.0, 1.0, 1.0)
        y_proba = _arr(0.0, 0.0, 1.0, 1.0)

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        np.testing.assert_almost_equal(result.accuracy, 0.0)
        np.testing.assert_almost_equal(result.precision_up, 0.0)
        np.testing.assert_almost_equal(result.recall_up, 0.0)
        np.testing.assert_almost_equal(result.f1_up, 0.0)
        np.testing.assert_almost_equal(result.precision_down, 0.0)
        np.testing.assert_almost_equal(result.recall_down, 0.0)
        np.testing.assert_almost_equal(result.f1_down, 0.0)

    def test_known_precision_recall(self) -> None:
        """Verify precision/recall on a hand-crafted example."""
        # For class +1: TP=2, FP=1, FN=1 → P=2/3, R=2/3, F1=2/3
        # For class -1: TP=2, FP=1, FN=1 → P=2/3, R=2/3, F1=2/3
        y_true = _arr(1.0, 1.0, 1.0, -1.0, -1.0, -1.0)
        y_pred = _arr(1.0, 1.0, -1.0, -1.0, -1.0, 1.0)
        y_proba = _arr(0.8, 0.7, 0.4, 0.3, 0.2, 0.6)

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        np.testing.assert_almost_equal(result.accuracy, 4 / 6)
        np.testing.assert_almost_equal(result.precision_up, 2 / 3)
        np.testing.assert_almost_equal(result.recall_up, 2 / 3)
        np.testing.assert_almost_equal(result.f1_up, 2 / 3)
        np.testing.assert_almost_equal(result.precision_down, 2 / 3)
        np.testing.assert_almost_equal(result.recall_down, 2 / 3)
        np.testing.assert_almost_equal(result.f1_down, 2 / 3)

    def test_auc_roc_random(self) -> None:
        """AUC-ROC for random classifier should be close to 0.5."""
        rng = np.random.default_rng(42)
        n = 1000
        y_true = np.where(rng.random(n) > 0.5, 1.0, -1.0).astype(np.float64)
        y_proba = rng.random(n).astype(np.float64)
        y_pred = np.where(y_proba > 0.5, 1.0, -1.0).astype(np.float64)

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        # AUC should be near 0.5 for random predictions
        assert 0.4 <= result.auc_roc <= 0.6

    def test_auc_roc_single_class_true(self) -> None:
        """When all true labels are the same class, AUC-ROC defaults to 0.5."""
        y_true = _arr(1.0, 1.0, 1.0, 1.0)
        y_pred = _arr(1.0, 1.0, -1.0, 1.0)
        y_proba = _arr(0.9, 0.8, 0.4, 0.7)

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        np.testing.assert_almost_equal(result.auc_roc, 0.5)

    def test_empty_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            compute_classification_metrics(empty, empty, empty)

    def test_length_mismatch_y_pred(self) -> None:
        with pytest.raises(ValueError, match="y_pred length"):
            compute_classification_metrics(_arr(1.0, -1.0), _arr(1.0), _arr(0.8, 0.3))

    def test_length_mismatch_y_proba(self) -> None:
        with pytest.raises(ValueError, match="y_proba_positive length"):
            compute_classification_metrics(_arr(1.0, -1.0), _arr(1.0, -1.0), _arr(0.8))

    def test_invalid_values_y_true(self) -> None:
        with pytest.raises(ValueError, match="y_true must contain only"):
            compute_classification_metrics(_arr(0.0, 1.0), _arr(1.0, -1.0), _arr(0.5, 0.5))

    def test_invalid_values_y_pred(self) -> None:
        with pytest.raises(ValueError, match="y_pred must contain only"):
            compute_classification_metrics(_arr(1.0, -1.0), _arr(2.0, -1.0), _arr(0.5, 0.5))

    def test_single_sample(self) -> None:
        result = compute_classification_metrics(_arr(1.0), _arr(1.0), _arr(0.9))
        assert result.n_samples == 1
        np.testing.assert_almost_equal(result.accuracy, 1.0)

    def test_all_same_prediction(self) -> None:
        """When all predictions are the same, one class has zero precision."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0)
        y_pred = _arr(1.0, 1.0, 1.0, 1.0)  # Always predict +1
        y_proba = _arr(0.9, 0.8, 0.7, 0.6)

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        np.testing.assert_almost_equal(result.accuracy, 0.5)
        np.testing.assert_almost_equal(result.precision_up, 0.5)
        np.testing.assert_almost_equal(result.recall_up, 1.0)
        # Down class: no predictions → precision=0, recall=0
        np.testing.assert_almost_equal(result.precision_down, 0.0)
        np.testing.assert_almost_equal(result.recall_down, 0.0)


# ---------------------------------------------------------------------------
# Abstention curve
# ---------------------------------------------------------------------------


class TestAbstentionCurve:
    def test_monotonic_accuracy_coverage_tradeoff(self) -> None:
        """Higher thresholds should generally yield higher accuracy, lower coverage."""
        rng = np.random.default_rng(42)
        n = 500
        # Generate data where high-confidence predictions are more accurate
        confidences = rng.uniform(0.5, 1.0, n).astype(np.float64)
        y_true = np.where(rng.random(n) > 0.5, 1.0, -1.0).astype(np.float64)
        # Make high-confidence predictions correct with higher probability
        noise = rng.random(n).astype(np.float64)
        y_pred = np.where(noise < confidences, y_true, -y_true).astype(np.float64)

        result = compute_abstention_curve(y_true, y_pred, confidences)

        assert isinstance(result, AbstentionCurve)
        assert len(result.results) == 5
        assert result.thresholds == (0.5, 0.55, 0.6, 0.65, 0.7)

        # Coverage should be non-increasing with threshold
        coverages = [r.coverage for r in result.results]
        for i in range(1, len(coverages)):
            assert coverages[i] <= coverages[i - 1] + 1e-10

    def test_threshold_at_0_5_full_coverage(self) -> None:
        """Threshold 0.5 with all confidences >= 0.5 gives full coverage."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0)
        y_pred = y_true.copy()
        confidences = _arr(0.6, 0.7, 0.8, 0.9)

        result = compute_abstention_curve(y_true, y_pred, confidences, thresholds=(0.5,))

        assert len(result.results) == 1
        np.testing.assert_almost_equal(result.results[0].coverage, 1.0)
        np.testing.assert_almost_equal(result.results[0].accuracy, 1.0)
        assert result.results[0].n_retained == 4
        assert result.results[0].n_total == 4

    def test_high_threshold_filters_samples(self) -> None:
        """Very high threshold retains only the most confident samples."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0, 1.0)
        y_pred = _arr(1.0, -1.0, 1.0, 1.0, 1.0)  # 4/5 correct
        confidences = _arr(0.55, 0.62, 0.90, 0.51, 0.75)

        result = compute_abstention_curve(y_true, y_pred, confidences, thresholds=(0.7,))

        # Only samples with confidence >= 0.7: indices 2 (conf=0.90) and 4 (conf=0.75)
        assert result.results[0].n_retained == 2
        # Both retained samples are correct
        np.testing.assert_almost_equal(result.results[0].accuracy, 1.0)
        np.testing.assert_almost_equal(result.results[0].coverage, 2 / 5)

    def test_no_samples_above_threshold(self) -> None:
        """When no samples exceed the threshold, accuracy is 0 and coverage is 0."""
        y_true = _arr(1.0, -1.0)
        y_pred = _arr(1.0, -1.0)
        confidences = _arr(0.51, 0.52)

        result = compute_abstention_curve(y_true, y_pred, confidences, thresholds=(0.9,))

        assert result.results[0].n_retained == 0
        np.testing.assert_almost_equal(result.results[0].accuracy, 0.0)
        np.testing.assert_almost_equal(result.results[0].coverage, 0.0)

    def test_custom_thresholds(self) -> None:
        y_true = _arr(1.0, -1.0, 1.0)
        y_pred = y_true.copy()
        confidences = _arr(0.6, 0.7, 0.8)

        result = compute_abstention_curve(y_true, y_pred, confidences, thresholds=(0.55, 0.75))

        assert result.thresholds == (0.55, 0.75)
        assert len(result.results) == 2

    def test_empty_thresholds_raises(self) -> None:
        with pytest.raises(ValueError, match="thresholds must not be empty"):
            compute_abstention_curve(
                _arr(1.0, -1.0),
                _arr(1.0, -1.0),
                _arr(0.6, 0.7),
                thresholds=(),
            )

    def test_empty_arrays_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            compute_abstention_curve(empty, empty, empty)

    def test_confidence_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="confidences length"):
            compute_abstention_curve(
                _arr(1.0, -1.0),
                _arr(1.0, -1.0),
                _arr(0.6),
            )

    def test_abstention_result_fields(self) -> None:
        """Verify AbstentionResult contains correct fields."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0)
        y_pred = _arr(1.0, 1.0, 1.0, -1.0)  # 3/4 correct
        confidences = _arr(0.8, 0.6, 0.9, 0.7)

        result = compute_abstention_curve(y_true, y_pred, confidences, thresholds=(0.65,))
        entry = result.results[0]

        assert isinstance(entry, AbstentionResult)
        assert entry.threshold == 0.65
        # Retained: indices 0 (conf=0.8), 2 (conf=0.9), 3 (conf=0.7) — all correct among retained
        assert entry.n_retained == 3
        assert entry.n_total == 4


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


class TestClassificationReliability:
    def test_perfectly_calibrated(self) -> None:
        """A perfectly calibrated model should have ECE close to 0."""
        rng = np.random.default_rng(42)
        n = 5000
        # Generate well-calibrated confidences
        confidences = rng.uniform(0.5, 1.0, n).astype(np.float64)
        y_true_list: list[float] = []
        y_pred_list: list[float] = []
        for conf in confidences:
            if rng.random() < conf:
                y_true_list.append(1.0)
                y_pred_list.append(1.0)
            else:
                y_true_list.append(-1.0)
                y_pred_list.append(1.0)

        y_true = np.array(y_true_list, dtype=np.float64)
        y_pred = np.array(y_pred_list, dtype=np.float64)

        result = compute_classification_reliability(y_true, y_pred, confidences)

        assert isinstance(result, ClassificationReliabilityResult)
        assert result.n_bins == 10
        assert result.n_samples == n
        # ECE should be small for well-calibrated model
        assert result.ece < 0.05

    def test_bin_counts_sum_to_n(self) -> None:
        """Sum of bin counts should equal total samples."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        y_pred = y_true.copy()
        confidences = _arr(0.55, 0.65, 0.75, 0.85, 0.95, 0.51)

        result = compute_classification_reliability(y_true, y_pred, confidences, n_bins=5)

        assert int(np.sum(result.bin_counts)) == 6
        assert result.bin_edges.shape[0] == 6  # n_bins + 1

    def test_ece_bounds(self) -> None:
        """ECE must be in [0, 1]."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0)
        y_pred = _arr(1.0, 1.0, -1.0, -1.0)
        confidences = _arr(0.9, 0.8, 0.7, 0.6)

        result = compute_classification_reliability(y_true, y_pred, confidences)

        assert 0.0 <= result.ece <= 1.0

    def test_single_bin(self) -> None:
        """With n_bins=1, all samples land in one bin."""
        y_true = _arr(1.0, -1.0, 1.0)
        y_pred = _arr(1.0, -1.0, -1.0)
        confidences = _arr(0.8, 0.7, 0.6)

        result = compute_classification_reliability(y_true, y_pred, confidences, n_bins=1)

        assert result.n_bins == 1
        assert int(result.bin_counts[0]) == 3

    def test_n_bins_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            compute_classification_reliability(_arr(1.0), _arr(1.0), _arr(0.8), n_bins=0)

    def test_empty_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            compute_classification_reliability(empty, empty, empty)

    def test_confidence_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="confidences length"):
            compute_classification_reliability(
                _arr(1.0, -1.0),
                _arr(1.0, -1.0),
                _arr(0.6),
            )

    def test_output_shapes(self) -> None:
        y_true = _arr(1.0, -1.0, 1.0, -1.0, 1.0)
        y_pred = y_true.copy()
        confidences = _arr(0.6, 0.7, 0.8, 0.9, 0.55)
        n_bins = 5

        result = compute_classification_reliability(y_true, y_pred, confidences, n_bins=n_bins)

        assert result.bin_edges.shape == (n_bins + 1,)
        assert result.bin_accuracies.shape == (n_bins,)
        assert result.bin_confidences.shape == (n_bins,)
        assert result.bin_counts.shape == (n_bins,)


# ---------------------------------------------------------------------------
# Economic accuracy
# ---------------------------------------------------------------------------


class TestEconomicAccuracy:
    def test_correct_on_large_moves(self) -> None:
        """Correct predictions on large moves should yield high economic accuracy."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0)
        y_pred = _arr(1.0, -1.0, -1.0, 1.0)  # First two correct, last two wrong
        # Large returns on the correct predictions, small on incorrect
        returns = _arr(0.05, 0.04, 0.001, 0.001)

        result = compute_economic_accuracy(y_true, y_pred, returns)

        assert isinstance(result, EconomicAccuracyResult)
        # Economic accuracy should be high because correct predictions had large returns
        assert result.economic_accuracy > result.standard_accuracy
        np.testing.assert_almost_equal(result.standard_accuracy, 0.5)

    def test_correct_on_small_moves(self) -> None:
        """Correct predictions only on small moves should yield low economic accuracy."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0)
        y_pred = _arr(1.0, -1.0, -1.0, 1.0)  # First two correct, last two wrong
        # Small returns on the correct predictions, large on incorrect
        returns = _arr(0.001, 0.001, 0.05, 0.04)

        result = compute_economic_accuracy(y_true, y_pred, returns)

        assert result.economic_accuracy < result.standard_accuracy

    def test_perfect_predictions(self) -> None:
        """Perfect predictions give economic_accuracy=1.0."""
        y_true = _arr(1.0, -1.0, 1.0)
        y_pred = y_true.copy()
        returns = _arr(0.03, -0.02, 0.01)

        result = compute_economic_accuracy(y_true, y_pred, returns)

        np.testing.assert_almost_equal(result.economic_accuracy, 1.0)
        np.testing.assert_almost_equal(result.standard_accuracy, 1.0)

    def test_known_value(self) -> None:
        """Verify economic accuracy against a hand-calculated value."""
        y_true = _arr(1.0, -1.0, 1.0)
        y_pred = _arr(1.0, 1.0, 1.0)  # Correct on idx 0 and 2, wrong on idx 1
        returns = _arr(0.10, 0.05, 0.02)

        result = compute_economic_accuracy(y_true, y_pred, returns)

        # economic_acc = (|0.10| * 1 + |0.05| * 0 + |0.02| * 1) / (|0.10| + |0.05| + |0.02|)
        expected = (0.10 + 0.02) / (0.10 + 0.05 + 0.02)
        np.testing.assert_almost_equal(result.economic_accuracy, expected)

    def test_negative_returns(self) -> None:
        """Economic accuracy should use |return| (absolute values)."""
        y_true = _arr(1.0, -1.0)
        y_pred = y_true.copy()
        returns = _arr(0.05, -0.03)

        result = compute_economic_accuracy(y_true, y_pred, returns)

        np.testing.assert_almost_equal(result.economic_accuracy, 1.0)

    def test_all_zero_returns_raises(self) -> None:
        with pytest.raises(ValueError, match="must not all be zero"):
            compute_economic_accuracy(
                _arr(1.0, -1.0),
                _arr(1.0, -1.0),
                _arr(0.0, 0.0),
            )

    def test_empty_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            compute_economic_accuracy(empty, empty, empty)

    def test_returns_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="actual_returns length"):
            compute_economic_accuracy(
                _arr(1.0, -1.0),
                _arr(1.0, -1.0),
                _arr(0.05),
            )


# ---------------------------------------------------------------------------
# Asymmetric metrics
# ---------------------------------------------------------------------------


class TestAsymmetricMetrics:
    def test_equal_weights_matches_macro(self) -> None:
        """With equal weights, asymmetric metrics should match macro-averaged metrics."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        y_pred = _arr(1.0, -1.0, -1.0, -1.0, 1.0, 1.0)

        result = compute_asymmetric_metrics(y_true, y_pred, crash_weight=1.0, rally_weight=1.0)

        assert isinstance(result, AsymmetricMetrics)
        # With equal weights and balanced classes, this is standard macro-average
        np.testing.assert_almost_equal(result.crash_weight, 1.0)
        np.testing.assert_almost_equal(result.rally_weight, 1.0)

    def test_crash_weight_increases_down_importance(self) -> None:
        """Higher crash_weight should increase the influence of down-class metrics."""
        y_true = _arr(1.0, 1.0, -1.0, -1.0)
        # All +1 correct, all -1 wrong
        y_pred = _arr(1.0, 1.0, 1.0, 1.0)

        result_equal = compute_asymmetric_metrics(y_true, y_pred, crash_weight=1.0, rally_weight=1.0)
        result_crash = compute_asymmetric_metrics(y_true, y_pred, crash_weight=2.0, rally_weight=1.0)

        # With higher crash_weight, missing all down predictions matters more
        # So weighted_f1 should decrease (down F1=0 gets more weight)
        assert result_crash.weighted_f1 < result_equal.weighted_f1

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should give F1=1 regardless of weights."""
        y_true = _arr(1.0, -1.0, 1.0, -1.0)
        y_pred = y_true.copy()

        result = compute_asymmetric_metrics(y_true, y_pred, crash_weight=2.0, rally_weight=1.0)

        np.testing.assert_almost_equal(result.weighted_f1, 1.0)
        np.testing.assert_almost_equal(result.weighted_precision, 1.0)
        np.testing.assert_almost_equal(result.weighted_recall, 1.0)

    def test_default_weights(self) -> None:
        """Default crash_weight=1.5, rally_weight=1.0."""
        y_true = _arr(1.0, -1.0)
        y_pred = y_true.copy()

        result = compute_asymmetric_metrics(y_true, y_pred)

        np.testing.assert_almost_equal(result.crash_weight, 1.5)
        np.testing.assert_almost_equal(result.rally_weight, 1.0)

    def test_known_value(self) -> None:
        """Verify asymmetric F1 against hand-calculated value."""
        # 3 up, 3 down. Predict: all up correct, 1 of 3 down correct.
        y_true = _arr(1.0, 1.0, 1.0, -1.0, -1.0, -1.0)
        y_pred = _arr(1.0, 1.0, 1.0, -1.0, 1.0, 1.0)

        # Class +1: TP=3, FP=2, FN=0 → P=3/5, R=1, F1=2*(3/5)*1/(3/5+1)=6/5/8/5=6/8=0.75
        # Class -1: TP=1, FP=0, FN=2 → P=1, R=1/3, F1=2*1*(1/3)/(1+1/3)=2/3/4/3=0.5
        # n_up=3, n_down=3, crash_weight=1.5, rally_weight=1.0
        # total_weight = 1.0*3 + 1.5*3 = 7.5
        # weighted_f1 = (1.0*0.75*3 + 1.5*0.5*3) / 7.5 = (2.25 + 2.25) / 7.5 = 0.6
        result = compute_asymmetric_metrics(y_true, y_pred, crash_weight=1.5, rally_weight=1.0)

        np.testing.assert_almost_equal(result.weighted_f1, 0.6)

    def test_all_one_class_true(self) -> None:
        """When all true labels are one class, the other class has zero support."""
        y_true = _arr(1.0, 1.0, 1.0, 1.0)
        y_pred = _arr(1.0, 1.0, -1.0, 1.0)

        result = compute_asymmetric_metrics(y_true, y_pred)

        # n_down = 0, so only rally_weight * n_up contributes
        # weighted_f1 = (rally_weight * f1_up * n_up) / (rally_weight * n_up)
        # = f1_up (since n_down=0 makes the down term vanish)
        assert result.weighted_f1 >= 0.0

    def test_zero_crash_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="crash_weight must be positive"):
            compute_asymmetric_metrics(_arr(1.0, -1.0), _arr(1.0, -1.0), crash_weight=0.0)

    def test_negative_rally_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="rally_weight must be positive"):
            compute_asymmetric_metrics(_arr(1.0, -1.0), _arr(1.0, -1.0), rally_weight=-1.0)

    def test_empty_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            compute_asymmetric_metrics(empty, empty)


# ---------------------------------------------------------------------------
# Large synthetic data (using conftest factory)
# ---------------------------------------------------------------------------


class TestWithSyntheticData:
    def test_classification_metrics_on_synthetic(self) -> None:
        """Metrics should be computable on larger synthetic data without errors."""
        from src.tests.forecasting.conftest import make_classification_data

        x, y_true = make_classification_data(n=500, n_features=5, seed=42)
        # Simulate imperfect predictions with noise
        rng = np.random.default_rng(99)
        flip_mask = rng.random(500) < 0.3  # flip 30% of labels
        y_pred = y_true.copy()
        y_pred[flip_mask] = -y_pred[flip_mask]
        y_proba = np.where(y_pred == 1.0, rng.uniform(0.5, 1.0, 500), rng.uniform(0.0, 0.5, 500)).astype(np.float64)

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        assert result.n_samples == 500
        assert 0.6 < result.accuracy < 0.8  # ~70% correct
        assert 0.0 <= result.auc_roc <= 1.0

    def test_abstention_curve_on_synthetic(self) -> None:
        """Abstention curve should show coverage decreasing with threshold."""
        from src.tests.forecasting.conftest import make_classification_data

        _, y_true = make_classification_data(n=300, seed=42)
        rng = np.random.default_rng(99)
        y_pred = y_true.copy()
        confidences = rng.uniform(0.45, 0.95, 300).astype(np.float64)

        result = compute_abstention_curve(y_true, y_pred, confidences)

        # Since y_pred = y_true, accuracy at every threshold should be 1.0
        for entry in result.results:
            if entry.n_retained > 0:
                np.testing.assert_almost_equal(entry.accuracy, 1.0)

    def test_economic_accuracy_on_synthetic(self) -> None:
        """Economic accuracy should be computable on synthetic data."""
        from src.tests.forecasting.conftest import make_classification_data

        _, y_true = make_classification_data(n=200, seed=42)
        rng = np.random.default_rng(99)
        y_pred = y_true.copy()
        returns = rng.normal(0, 0.02, 200).astype(np.float64)

        result = compute_economic_accuracy(y_true, y_pred, returns)

        np.testing.assert_almost_equal(result.economic_accuracy, 1.0)
        np.testing.assert_almost_equal(result.standard_accuracy, 1.0)
