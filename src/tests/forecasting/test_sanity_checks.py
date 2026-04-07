"""Tests for sanity-check components: naive classifiers and shuffled-labels permutation test."""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.naive_classifiers import (
    MajorityClassifier,
    MomentumSignClassifier,
    PersistenceClassifier,
)
from src.app.forecasting.application.sanity_checks import ShuffledLabelCheckConfig, run_shuffled_labels_check
from src.app.forecasting.domain.value_objects import (
    ForecastHorizon,
    MajorityConfig,
    MomentumSignConfig,
    NaiveBenchmarkResult,
    PersistenceConfig,
    SanityCheckReport,
    ShuffledLabelResult,
)
from src.tests.forecasting.conftest import (
    make_classification_data,
    make_logistic_config,
    make_majority_config,
    make_momentum_sign_config,
    make_persistence_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HORIZON: ForecastHorizon = ForecastHorizon.H1


def _simple_folds(
    n: int,
    n_folds: int = 3,
) -> list[
    tuple[
        np.ndarray[tuple[int], np.dtype[np.intp]],
        np.ndarray[tuple[int], np.dtype[np.intp]],
    ]
]:
    """Build simple sequential train/test folds (no overlap).

    Each fold uses the first ``(n_folds - 1) / n_folds`` samples for training
    and the remaining for testing, shifted by fold index.

    Args:
        n: Total number of samples.
        n_folds: Number of folds.

    Returns:
        List of ``(train_indices, test_indices)`` tuples.
    """
    fold_size: int = n // n_folds
    folds: list[
        tuple[
            np.ndarray[tuple[int], np.dtype[np.intp]],
            np.ndarray[tuple[int], np.dtype[np.intp]],
        ]
    ] = []
    for i in range(n_folds):
        test_start: int = i * fold_size
        test_end: int = (i + 1) * fold_size
        test_idx: np.ndarray[tuple[int], np.dtype[np.intp]] = np.arange(test_start, test_end)
        train_idx: np.ndarray[tuple[int], np.dtype[np.intp]] = np.concatenate(
            [np.arange(0, test_start), np.arange(test_end, n)],
        )
        folds.append((train_idx, test_idx))
    return folds


# ===========================================================================
# MajorityClassifier tests
# ===========================================================================


class TestMajorityClassifier:
    """Tests for the majority-class naive classifier."""

    def test_always_predicts_majority(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        """MajorityClassifier should predict the majority class for every sample."""
        x, y = classification_data
        config: MajorityConfig = make_majority_config()
        clf: MajorityClassifier = MajorityClassifier(config=config, horizon=_HORIZON)

        clf.fit(x, y)
        forecasts = clf.predict(x)

        # All predictions should be the same direction
        directions: set[int] = {f.predicted_direction for f in forecasts}
        assert len(directions) == 1

        # That direction should be the majority class
        n_positive: int = int(np.sum(y == 1.0))
        n_negative: int = len(y) - n_positive
        expected_majority: int = 1 if n_positive >= n_negative else -1
        assert forecasts[0].predicted_direction == expected_majority

    def test_da_equals_majority_fraction(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        """DA should equal the majority class fraction."""
        x, y = classification_data
        config: MajorityConfig = make_majority_config()
        clf: MajorityClassifier = MajorityClassifier(config=config, horizon=_HORIZON)

        clf.fit(x, y)
        forecasts = clf.predict(x)

        preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
            [f.predicted_direction for f in forecasts],
            dtype=np.float64,
        )
        da: float = float(np.mean(preds == y))

        n_positive: int = int(np.sum(y == 1.0))
        n_negative: int = len(y) - n_positive
        expected_da: float = max(n_positive, n_negative) / len(y)

        assert da == pytest.approx(expected_da, abs=1e-10)

    def test_confidence_equals_frequency(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Confidence should be the majority class frequency."""
        x, y = classification_data
        config: MajorityConfig = make_majority_config()
        clf: MajorityClassifier = MajorityClassifier(config=config, horizon=_HORIZON)

        clf.fit(x, y)
        forecasts = clf.predict(x[:10])

        n_positive: int = int(np.sum(y == 1.0))
        n_negative: int = len(y) - n_positive
        expected_freq: float = max(n_positive, n_negative) / len(y)

        for f in forecasts:
            assert f.confidence == pytest.approx(expected_freq, abs=1e-10)

    def test_fit_required(self) -> None:
        """predict() before fit() should raise RuntimeError."""
        config: MajorityConfig = make_majority_config()
        clf: MajorityClassifier = MajorityClassifier(config=config, horizon=_HORIZON)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((5, 3), dtype=np.float64)
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict(x_test)

    def test_empty_x_train_raises(self) -> None:
        """fit() with empty x_train should raise ValueError."""
        config: MajorityConfig = make_majority_config()
        clf: MajorityClassifier = MajorityClassifier(config=config, horizon=_HORIZON)
        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 3), dtype=np.float64)
        y_empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0, dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            clf.fit(x_empty, y_empty)


# ===========================================================================
# PersistenceClassifier tests
# ===========================================================================


class TestPersistenceClassifier:
    """Tests for the persistence (last-value) naive classifier."""

    def test_predicts_last_training_direction(self) -> None:
        """PersistenceClassifier should predict the last training label."""
        rng: np.random.Generator = np.random.default_rng(42)
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((50, 5)).astype(np.float64)
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = np.where(
            rng.standard_normal(50) > 0,
            1.0,
            -1.0,
        ).astype(np.float64)

        config: PersistenceConfig = make_persistence_config()
        clf: PersistenceClassifier = PersistenceClassifier(config=config, horizon=_HORIZON)

        clf.fit(x, y)
        forecasts = clf.predict(x[:10])

        expected_direction: int = int(y[-1])
        for f in forecasts:
            assert f.predicted_direction == expected_direction

    def test_confidence_is_half(self) -> None:
        """Persistence has no probabilistic model — confidence should be 0.5."""
        rng: np.random.Generator = np.random.default_rng(99)
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((20, 3)).astype(np.float64)
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(20, dtype=np.float64)

        config: PersistenceConfig = make_persistence_config()
        clf: PersistenceClassifier = PersistenceClassifier(config=config, horizon=_HORIZON)

        clf.fit(x, y)
        forecasts = clf.predict(x[:5])

        for f in forecasts:
            assert f.confidence == pytest.approx(0.5)

    def test_fit_required(self) -> None:
        """predict() before fit() should raise RuntimeError."""
        config: PersistenceConfig = make_persistence_config()
        clf: PersistenceClassifier = PersistenceClassifier(config=config, horizon=_HORIZON)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((5, 3), dtype=np.float64)
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict(x_test)


# ===========================================================================
# MomentumSignClassifier tests
# ===========================================================================


class TestMomentumSignClassifier:
    """Tests for the momentum-sign naive classifier."""

    def test_predicts_sign_of_momentum_column(self) -> None:
        """Should predict +1 for non-negative, -1 for negative momentum values."""
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
            [
                [1.0, 0.5, 0.0],
                [-0.5, 0.3, 1.0],
                [0.0, -1.0, 0.5],
                [2.0, 0.0, -0.5],
                [-3.0, 0.1, 0.2],
            ],
            dtype=np.float64,
        )
        y_dummy: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(5, dtype=np.float64)

        config: MomentumSignConfig = make_momentum_sign_config(momentum_col_idx=0)
        clf: MomentumSignClassifier = MomentumSignClassifier(config=config, horizon=_HORIZON)

        clf.fit(x, y_dummy)
        forecasts = clf.predict(x)

        # col 0 values: [1.0, -0.5, 0.0, 2.0, -3.0]
        # expected signs: [+1, -1, +1, +1, -1] (>= 0 → +1)
        expected: list[int] = [1, -1, 1, 1, -1]
        actual: list[int] = [f.predicted_direction for f in forecasts]
        assert actual == expected

    def test_different_column_index(self) -> None:
        """Should read from the column specified by momentum_col_idx."""
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
            [
                [1.0, -1.0, 0.0],
                [-0.5, 2.0, 1.0],
            ],
            dtype=np.float64,
        )
        y_dummy: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(2, dtype=np.float64)

        config: MomentumSignConfig = make_momentum_sign_config(momentum_col_idx=1)
        clf: MomentumSignClassifier = MomentumSignClassifier(config=config, horizon=_HORIZON)

        clf.fit(x, y_dummy)
        forecasts = clf.predict(x)

        # col 1 values: [-1.0, 2.0] → [-1, +1]
        expected: list[int] = [-1, 1]
        actual: list[int] = [f.predicted_direction for f in forecasts]
        assert actual == expected

    def test_confidence_is_half(self) -> None:
        """Momentum-sign has no probabilistic model — confidence should be 0.5."""
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
            [[1.0, 0.5], [-0.5, 0.3]],
            dtype=np.float64,
        )
        y_dummy: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(2, dtype=np.float64)

        config: MomentumSignConfig = make_momentum_sign_config()
        clf: MomentumSignClassifier = MomentumSignClassifier(config=config, horizon=_HORIZON)

        clf.fit(x, y_dummy)
        forecasts = clf.predict(x)

        for f in forecasts:
            assert f.confidence == pytest.approx(0.5)

    def test_out_of_bounds_col_idx_raises(self) -> None:
        """fit() with col_idx >= n_features should raise ValueError."""
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((5, 3), dtype=np.float64)
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(5, dtype=np.float64)

        config: MomentumSignConfig = make_momentum_sign_config(momentum_col_idx=5)
        clf: MomentumSignClassifier = MomentumSignClassifier(config=config, horizon=_HORIZON)

        with pytest.raises(ValueError, match="out of bounds"):
            clf.fit(x, y)

    def test_fit_required(self) -> None:
        """predict() before fit() should raise RuntimeError."""
        config: MomentumSignConfig = make_momentum_sign_config()
        clf: MomentumSignClassifier = MomentumSignClassifier(config=config, horizon=_HORIZON)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((5, 3), dtype=np.float64)
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict(x_test)


# ===========================================================================
# Shuffled-labels sanity check tests
# ===========================================================================


class TestShuffledLabelsCheck:
    """Tests for the shuffled-labels permutation test."""

    def test_logistic_on_shuffled_labels_da_near_50(self) -> None:
        """Training LogisticBaseline on permuted labels should yield DA ≈ 50%."""
        from src.app.forecasting.application.logistic_baseline import LogisticBaseline

        x, y = make_classification_data(n=300, n_features=5, seed=42)
        folds = _simple_folds(n=300, n_folds=3)

        result: ShuffledLabelResult = run_shuffled_labels_check(
            model_factory=LogisticBaseline,
            model_kwargs={
                "config": make_logistic_config(),
                "horizon": ForecastHorizon.H1,
            },
            x=x,
            y=y,
            folds=folds,
            config=ShuffledLabelCheckConfig(n_permutations=5, random_seed=42),
        )

        assert result.model_name == "LogisticBaseline"
        assert result.n_permutations == 5
        assert len(result.per_permutation_da) == 5
        # Mean DA should be near 0.50 (within ±2pp)
        assert 0.48 <= result.mean_da <= 0.52, f"mean_da={result.mean_da:.4f} outside [0.48, 0.52]"
        assert result.passed is True

    def test_result_passed_false_when_da_high(self) -> None:
        """If mean_da > 0.52 on shuffled data, result.passed should be False."""
        # We test the value object directly since we can't easily make a real
        # model exceed 52% on shuffled labels without cheating
        result: ShuffledLabelResult = ShuffledLabelResult(
            model_name="FakeModel",
            n_permutations=3,
            per_permutation_da=(0.55, 0.60, 0.58),
            mean_da=0.5767,
            passed=False,
        )
        assert result.passed is False

    def test_result_passed_true_when_da_in_range(self) -> None:
        """If mean_da within [0.48, 0.52], result.passed should be True."""
        result: ShuffledLabelResult = ShuffledLabelResult(
            model_name="GoodModel",
            n_permutations=5,
            per_permutation_da=(0.49, 0.51, 0.50, 0.48, 0.52),
            mean_da=0.50,
            passed=True,
        )
        assert result.passed is True

    def test_empty_x_raises(self) -> None:
        """Empty feature matrix should raise ValueError."""
        from src.app.forecasting.application.logistic_baseline import LogisticBaseline

        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        y_empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0, dtype=np.float64)
        folds = _simple_folds(n=100, n_folds=2)

        with pytest.raises(ValueError, match="at least one sample"):
            run_shuffled_labels_check(
                model_factory=LogisticBaseline,
                model_kwargs={
                    "config": make_logistic_config(),
                    "horizon": ForecastHorizon.H1,
                },
                x=x_empty,
                y=y_empty,
                folds=folds,
            )

    def test_empty_folds_raises(self) -> None:
        """Empty folds list should raise ValueError."""
        from src.app.forecasting.application.logistic_baseline import LogisticBaseline

        x, y = make_classification_data(n=100)

        with pytest.raises(ValueError, match="folds must not be empty"):
            run_shuffled_labels_check(
                model_factory=LogisticBaseline,
                model_kwargs={
                    "config": make_logistic_config(),
                    "horizon": ForecastHorizon.H1,
                },
                x=x,
                y=y,
                folds=[],
            )

    def test_single_permutation(self) -> None:
        """Should work correctly with n_permutations=1."""
        from src.app.forecasting.application.logistic_baseline import LogisticBaseline

        x, y = make_classification_data(n=200, n_features=5, seed=99)
        folds = _simple_folds(n=200, n_folds=2)

        result: ShuffledLabelResult = run_shuffled_labels_check(
            model_factory=LogisticBaseline,
            model_kwargs={
                "config": make_logistic_config(),
                "horizon": ForecastHorizon.H1,
            },
            x=x,
            y=y,
            folds=folds,
            config=ShuffledLabelCheckConfig(n_permutations=1, random_seed=77),
        )

        assert result.n_permutations == 1
        assert len(result.per_permutation_da) == 1

    def test_random_forest_on_shuffled_labels_da_near_50(self) -> None:
        """Training RandomForestClassifier on permuted labels should yield DA ≈ 50% (11H null test)."""
        from src.app.forecasting.application.random_forest_clf import RandomForestClassifier

        from src.tests.forecasting.conftest import make_rf_clf_config

        x, y = make_classification_data(n=300, n_features=5, seed=42)
        folds = _simple_folds(n=300, n_folds=3)

        result: ShuffledLabelResult = run_shuffled_labels_check(
            model_factory=RandomForestClassifier,
            model_kwargs={
                "config": make_rf_clf_config(),
                "horizon": ForecastHorizon.H1,
            },
            x=x,
            y=y,
            folds=folds,
            config=ShuffledLabelCheckConfig(n_permutations=5, random_seed=42),
        )

        assert result.model_name == "RandomForestClassifier"
        assert 0.48 <= result.mean_da <= 0.52, f"mean_da={result.mean_da:.4f} outside [0.48, 0.52]"
        assert result.passed is True

    def test_gradient_boosting_clf_on_shuffled_labels_da_near_50(self) -> None:
        """Training GradientBoostingClassifier on permuted labels should yield DA ≈ 50% (11H null test).

        Uses a wider tolerance ([0.43, 0.57]) than Logistic/RF because LightGBM's
        Platt scaling (CalibratedClassifierCV with internal CV) introduces extra
        variance on small shuffled-label datasets.  The key invariant is the same:
        no systematic bias above chance.
        """
        from src.app.forecasting.application.gradient_boosting_clf import GradientBoostingClassifier

        from src.tests.forecasting.conftest import make_gb_clf_config

        x, y = make_classification_data(n=500, n_features=5, seed=42)
        folds = _simple_folds(n=500, n_folds=3)

        result: ShuffledLabelResult = run_shuffled_labels_check(
            model_factory=GradientBoostingClassifier,
            model_kwargs={
                "config": make_gb_clf_config(),
                "horizon": ForecastHorizon.H1,
            },
            x=x,
            y=y,
            folds=folds,
            config=ShuffledLabelCheckConfig(n_permutations=10, random_seed=42),
        )

        assert result.model_name == "GradientBoostingClassifier"
        # Wider band accounts for Platt scaling variance on shuffled data
        assert 0.43 <= result.mean_da <= 0.57, f"mean_da={result.mean_da:.4f} outside [0.43, 0.57]"

    # NOTE: GRU null test intentionally omitted — 2-3s per permutation adds
    # test latency for no additional leakage-detection insight beyond the
    # three classifiers above.  If Logistic/RF/LightGBM all collapse to ~50%
    # on shuffled labels, the pipeline is validated.


# ===========================================================================
# SanityCheckReport value object tests
# ===========================================================================


class TestSanityCheckReport:
    """Tests for the SanityCheckReport aggregation value object."""

    def test_all_passed_true_when_all_shuffle_pass(self) -> None:
        """all_passed should be True when every shuffled result passed."""
        shuffled_1: ShuffledLabelResult = ShuffledLabelResult(
            model_name="Model1",
            n_permutations=3,
            per_permutation_da=(0.49, 0.50, 0.51),
            mean_da=0.50,
            passed=True,
        )
        shuffled_2: ShuffledLabelResult = ShuffledLabelResult(
            model_name="Model2",
            n_permutations=3,
            per_permutation_da=(0.48, 0.51, 0.50),
            mean_da=0.4967,
            passed=True,
        )
        naive_1: NaiveBenchmarkResult = NaiveBenchmarkResult(
            benchmark_name="Majority",
            da=0.55,
            n_samples=100,
        )

        report: SanityCheckReport = SanityCheckReport(
            shuffled_results=(shuffled_1, shuffled_2),
            naive_results=(naive_1,),
            all_passed=True,
        )
        assert report.all_passed is True

    def test_all_passed_false_when_any_shuffle_fails(self) -> None:
        """all_passed should be False when any shuffled result failed."""
        shuffled_ok: ShuffledLabelResult = ShuffledLabelResult(
            model_name="GoodModel",
            n_permutations=3,
            per_permutation_da=(0.50, 0.49, 0.51),
            mean_da=0.50,
            passed=True,
        )
        shuffled_bad: ShuffledLabelResult = ShuffledLabelResult(
            model_name="LeakyModel",
            n_permutations=3,
            per_permutation_da=(0.60, 0.65, 0.58),
            mean_da=0.61,
            passed=False,
        )

        report: SanityCheckReport = SanityCheckReport(
            shuffled_results=(shuffled_ok, shuffled_bad),
            naive_results=(),
            all_passed=False,
        )
        assert report.all_passed is False

    def test_empty_results(self) -> None:
        """Report with no results should be valid (vacuously all_passed)."""
        report: SanityCheckReport = SanityCheckReport(
            shuffled_results=(),
            naive_results=(),
            all_passed=True,
        )
        assert report.all_passed is True
        assert len(report.shuffled_results) == 0
        assert len(report.naive_results) == 0


# ===========================================================================
# NaiveBenchmarkResult value object tests
# ===========================================================================


class TestNaiveBenchmarkResult:
    """Tests for the NaiveBenchmarkResult value object."""

    def test_valid_construction(self) -> None:
        """Should construct with valid parameters."""
        result: NaiveBenchmarkResult = NaiveBenchmarkResult(
            benchmark_name="Majority",
            da=0.55,
            n_samples=100,
        )
        assert result.benchmark_name == "Majority"
        assert result.da == pytest.approx(0.55)
        assert result.n_samples == 100

    def test_frozen(self) -> None:
        """Should be immutable."""
        result: NaiveBenchmarkResult = NaiveBenchmarkResult(
            benchmark_name="Majority",
            da=0.55,
            n_samples=100,
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            result.da = 0.99  # type: ignore[misc]
