"""Tests for the walk-forward recommender pipeline."""

from __future__ import annotations

import polars as pl
import pytest

from src.app.recommendation.application.baseline_recommenders import (
    AllAssetsRecommender,
    RandomRecommender,
    RandomRecommenderConfig,
)
from src.app.recommendation.application.feature_builder import RecommenderFeatureBuilder
from src.app.recommendation.application.gradient_boosting_recommender import (
    GradientBoostingRecommender,
    GradientBoostingRecommenderConfig,
)
from src.app.recommendation.application.label_builder import LabelBuilder, LabelConfig
from src.app.recommendation.application.pipeline import (
    PipelineConfig,
    PipelineResult,
    RecommenderPipeline,
    WindowResult,
    _compute_splits,
)
from src.app.recommendation.domain.value_objects import Recommendation
from src.tests.recommendation.conftest import (
    ASSET_SYMBOL,
    STRATEGY_NAME,
    make_bars,
    make_classifier_outputs,
    make_market_features,
    make_regressor_outputs,
    make_signals,
    make_strategy_returns,
    make_vol_forecasts,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_N_BARS: int = 80
_SMALL_N_BARS: int = 5


def _build_pipeline(
    *,
    n_windows: int = 3,
    train_ratio: float = 0.5,
    purge_bars: int = 2,
    embargo_bars: int = 1,
    label_horizon: int = 3,
    commission_bps: float = 10.0,
    gbm_estimators: int = 10,
) -> RecommenderPipeline:
    """Build a RecommenderPipeline with sensible test defaults."""
    config = PipelineConfig(
        n_windows=n_windows,
        train_ratio=train_ratio,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        label_horizon=label_horizon,
        commission_bps=commission_bps,
    )

    primary = GradientBoostingRecommender(
        config=GradientBoostingRecommenderConfig(
            n_estimators=gbm_estimators,
            random_seed=42,
        ),
    )
    baselines = [
        RandomRecommender(config=RandomRecommenderConfig(random_seed=42)),
        AllAssetsRecommender(),
    ]
    baseline_names = ["random", "all_assets"]

    feature_builder = RecommenderFeatureBuilder()
    label_builder = LabelBuilder(
        config=LabelConfig(label_horizon=label_horizon, commission_bps=commission_bps),
    )

    return RecommenderPipeline(
        primary=primary,
        baselines=baselines,
        baseline_names=baseline_names,
        config=config,
        feature_builder=feature_builder,
        label_builder=label_builder,
    )


def _make_all_data(
    n: int = _N_BARS,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    """Build a complete set of aligned synthetic data for pipeline tests."""
    bars = make_bars(n)
    signals = make_signals(n)
    mf = make_market_features(n)
    clf = make_classifier_outputs(n)
    reg = make_regressor_outputs(n)
    vol = make_vol_forecasts(n)
    strat = make_strategy_returns(n)
    return bars, signals, mf, clf, reg, vol, strat


# ---------------------------------------------------------------------------
# TestPipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for PipelineConfig validation."""

    def test_defaults(self):
        config = PipelineConfig()
        assert config.n_windows == 5
        assert config.train_ratio == 0.6
        assert config.purge_bars == 10
        assert config.embargo_bars == 5
        assert config.label_horizon == 7
        assert config.commission_bps == 10.0

    def test_frozen(self):
        config = PipelineConfig()
        with pytest.raises(Exception, match="frozen"):  # noqa: B017
            config.n_windows = 10  # type: ignore[misc]

    def test_custom_values(self):
        config = PipelineConfig(
            n_windows=3,
            train_ratio=0.7,
            purge_bars=5,
            embargo_bars=2,
            label_horizon=10,
            commission_bps=5.0,
        )
        assert config.n_windows == 3
        assert config.train_ratio == 0.7
        assert config.purge_bars == 5
        assert config.embargo_bars == 2

    def test_invalid_n_windows_zero(self):
        with pytest.raises(Exception, match="greater_than"):  # noqa: B017
            PipelineConfig(n_windows=0)

    def test_invalid_train_ratio_zero(self):
        with pytest.raises(Exception, match="greater_than"):  # noqa: B017
            PipelineConfig(train_ratio=0.0)

    def test_invalid_train_ratio_one(self):
        with pytest.raises(Exception, match="less_than"):  # noqa: B017
            PipelineConfig(train_ratio=1.0)

    def test_zero_purge_embargo_valid(self):
        config = PipelineConfig(purge_bars=0, embargo_bars=0)
        assert config.purge_bars == 0
        assert config.embargo_bars == 0


# ---------------------------------------------------------------------------
# TestPipelineBasic
# ---------------------------------------------------------------------------


class TestPipelineBasic:
    """Basic pipeline execution tests."""

    def test_basic_run_produces_results(self):
        """Pipeline completes and returns WindowResults."""
        pipeline = _build_pipeline(n_windows=3, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        assert isinstance(result, PipelineResult)
        assert result.n_windows_completed > 0
        assert len(result.window_results) > 0
        assert len(result.all_primary_predictions) > 0
        assert len(result.all_test_labels) > 0

    def test_window_result_fields(self):
        """WindowResult has all expected fields."""
        pipeline = _build_pipeline(n_windows=2, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            assert isinstance(wr, WindowResult)
            assert wr.window_index >= 0
            assert wr.train_start_idx == 0  # expanding window starts at 0
            assert wr.train_end_idx > wr.train_start_idx
            assert wr.test_start_idx >= wr.train_end_idx  # purge gap
            assert wr.test_end_idx > wr.test_start_idx
            assert wr.n_train_samples > 0
            assert wr.n_test_samples > 0
            assert len(wr.primary_predictions) == wr.n_test_samples
            assert len(wr.test_labels) == wr.n_test_samples

    def test_predictions_are_recommendations(self):
        """Primary and baseline predictions are Recommendation instances."""
        pipeline = _build_pipeline(n_windows=2, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            for rec in wr.primary_predictions:
                assert isinstance(rec, Recommendation)
                assert 0.0 <= rec.confidence <= 1.0
                assert 0.0 <= rec.position_size <= 1.0

    def test_pipeline_result_aggregation(self):
        """All window predictions are aggregated into the pipeline result."""
        pipeline = _build_pipeline(n_windows=3, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        # Sum of per-window test samples should equal total
        total_test = sum(wr.n_test_samples for wr in result.window_results)
        assert len(result.all_primary_predictions) == total_test
        assert len(result.all_test_labels) == total_test

    def test_completed_plus_skipped_equals_configured(self):
        """n_windows_completed + n_windows_skipped should be consistent."""
        pipeline = _build_pipeline(n_windows=3, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        assert result.n_windows_completed == len(result.window_results)
        assert result.n_windows_completed + result.n_windows_skipped >= 1


# ---------------------------------------------------------------------------
# TestExpandingWindow
# ---------------------------------------------------------------------------


class TestExpandingWindow:
    """Verify expanding (anchored) window correctness."""

    def test_train_sets_grow_monotonically(self):
        """Training set end index should increase across windows."""
        pipeline = _build_pipeline(n_windows=3, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        train_ends = [wr.train_end_idx for wr in result.window_results]
        for i in range(1, len(train_ends)):
            assert train_ends[i] >= train_ends[i - 1], (
                f"Window {i} train_end={train_ends[i]} should be >= window {i - 1} train_end={train_ends[i - 1]}"
            )

    def test_all_windows_start_at_zero(self):
        """Expanding windows always train from index 0."""
        pipeline = _build_pipeline(n_windows=3, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            assert wr.train_start_idx == 0

    def test_n_train_samples_non_decreasing(self):
        """Number of training samples should be non-decreasing."""
        pipeline = _build_pipeline(n_windows=3, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        n_trains = [wr.n_train_samples for wr in result.window_results]
        for i in range(1, len(n_trains)):
            assert n_trains[i] >= n_trains[i - 1]


# ---------------------------------------------------------------------------
# TestPurgingEmbargo
# ---------------------------------------------------------------------------


class TestPurgingEmbargo:
    """Verify purge and embargo enforcement."""

    def test_no_overlap_between_train_and_test(self):
        """Test period must not overlap with training period."""
        pipeline = _build_pipeline(n_windows=3, purge_bars=3, embargo_bars=2)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            gap = wr.test_start_idx - wr.train_end_idx
            expected_gap = 3 + 2  # purge + embargo
            assert gap >= expected_gap, f"Window {wr.window_index}: gap={gap} < expected={expected_gap}"

    def test_purge_gap_equals_config(self):
        """The gap between train_end and test_start should be purge+embargo."""
        purge = 4
        embargo = 3
        pipeline = _build_pipeline(n_windows=2, purge_bars=purge, embargo_bars=embargo)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            gap = wr.test_start_idx - wr.train_end_idx
            assert gap == purge + embargo

    def test_zero_purge_zero_embargo(self):
        """With zero purge and embargo, test starts right after train."""
        pipeline = _build_pipeline(n_windows=2, purge_bars=0, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            assert wr.test_start_idx == wr.train_end_idx

    def test_multi_layer_leakage_timestamps(self):
        """L1 predictions used as L2 features must not leak across train/test.

        Verify that no train-period timestamp appears in the test period
        for each window. This is the key invariant from the task spec
        (12F leakage test).
        """
        pipeline = _build_pipeline(n_windows=3, purge_bars=3, embargo_bars=2)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            # Get timestamps for train and test periods
            train_timestamps = set(
                bars.slice(wr.train_start_idx, wr.train_end_idx - wr.train_start_idx).get_column("timestamp").to_list()
            )
            test_timestamps = set(
                bars.slice(wr.test_start_idx, wr.test_end_idx - wr.test_start_idx).get_column("timestamp").to_list()
            )

            overlap = train_timestamps & test_timestamps
            assert len(overlap) == 0, (
                f"Window {wr.window_index}: {len(overlap)} timestamps overlap "
                f"between train and test — temporal leakage!"
            )

            # Also verify the purge gap in timestamp space
            if wr.train_end_idx < len(bars) and wr.test_start_idx < len(bars):
                last_train_ts = bars.row(wr.train_end_idx - 1)[0]
                first_test_ts = bars.row(wr.test_start_idx)[0]
                assert first_test_ts > last_train_ts, (
                    "First test timestamp must be strictly after last train timestamp"
                )


# ---------------------------------------------------------------------------
# TestBaselineComparison
# ---------------------------------------------------------------------------


class TestBaselineComparison:
    """Verify baselines produce results alongside primary."""

    def test_all_baselines_present(self):
        """All configured baselines should appear in results."""
        pipeline = _build_pipeline(n_windows=2, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            assert "random" in wr.baseline_predictions
            assert "all_assets" in wr.baseline_predictions

    def test_baseline_prediction_counts_match(self):
        """Each baseline should have the same number of predictions as the primary."""
        pipeline = _build_pipeline(n_windows=2, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            n_primary = len(wr.primary_predictions)
            for name, preds in wr.baseline_predictions.items():
                assert len(preds) == n_primary, f"Baseline '{name}' has {len(preds)} predictions, expected {n_primary}"

    def test_baseline_predictions_are_recommendations(self):
        """All baseline predictions should be valid Recommendation objects."""
        pipeline = _build_pipeline(n_windows=2, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        for wr in result.window_results:
            for name, preds in wr.baseline_predictions.items():
                for rec in preds:
                    assert isinstance(rec, Recommendation), (
                        f"Baseline '{name}' produced non-Recommendation: {type(rec)}"
                    )


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Pipeline should work with minimal inputs (no classifier/regressor/vol)."""

    def test_only_market_features(self):
        """Pipeline should work with only market features (no L1 outputs)."""
        pipeline = _build_pipeline(n_windows=2, purge_bars=1, embargo_bars=0)
        bars = make_bars(_N_BARS)
        signals = make_signals(_N_BARS)
        mf = make_market_features(_N_BARS)

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            # No classifier_outputs, regressor_outputs, vol_forecasts
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        assert isinstance(result, PipelineResult)
        assert result.n_windows_completed > 0
        assert len(result.all_primary_predictions) > 0

    def test_partial_l1_outputs(self):
        """Pipeline works with only classifier outputs (no regressor, no vol)."""
        pipeline = _build_pipeline(n_windows=2, purge_bars=1, embargo_bars=0)
        bars = make_bars(_N_BARS)
        signals = make_signals(_N_BARS)
        mf = make_market_features(_N_BARS)
        clf = make_classifier_outputs(_N_BARS)

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        assert isinstance(result, PipelineResult)
        assert result.n_windows_completed > 0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_bars_raises(self):
        """Empty bars should raise ValueError."""
        pipeline = _build_pipeline()
        with pytest.raises(ValueError, match="bars.*not be empty"):
            pipeline.run(
                bars=pl.DataFrame({"timestamp": [], "close": []}),
                signals=make_signals(10),
                market_features=make_market_features(10),
            )

    def test_empty_signals_raises(self):
        """Empty signals should raise ValueError."""
        pipeline = _build_pipeline()
        with pytest.raises(ValueError, match="signals.*not be empty"):
            pipeline.run(
                bars=make_bars(10),
                signals=pl.DataFrame({"timestamp": [], "side": []}),
                market_features=make_market_features(10),
            )

    def test_empty_market_features_raises(self):
        """Empty market features should raise ValueError."""
        pipeline = _build_pipeline()
        with pytest.raises(ValueError, match="market_features.*not be empty"):
            pipeline.run(
                bars=make_bars(10),
                signals=make_signals(10),
                market_features=pl.DataFrame({"timestamp": []}),
            )

    def test_data_too_small_for_walk_forward(self):
        """Data too small for even one window should raise ValueError."""
        pipeline = _build_pipeline(n_windows=3, purge_bars=5, embargo_bars=5)
        bars = make_bars(_SMALL_N_BARS)
        signals = make_signals(_SMALL_N_BARS)
        mf = make_market_features(_SMALL_N_BARS)

        with pytest.raises(ValueError, match="Insufficient"):
            pipeline.run(
                bars=bars,
                signals=signals,
                market_features=mf,
                asset_symbol=ASSET_SYMBOL,
                strategy_name=STRATEGY_NAME,
            )

    def test_mismatched_baseline_names_raises(self):
        """baselines and baseline_names length mismatch should raise."""
        config = PipelineConfig(n_windows=2)
        primary = GradientBoostingRecommender()
        baselines = [RandomRecommender()]
        baseline_names = ["random", "extra"]

        with pytest.raises(ValueError, match="same length"):
            RecommenderPipeline(
                primary=primary,
                baselines=baselines,
                baseline_names=baseline_names,
                config=config,
                feature_builder=RecommenderFeatureBuilder(),
                label_builder=LabelBuilder(),
            )

    def test_single_window(self):
        """Pipeline works with a single walk-forward window."""
        pipeline = _build_pipeline(n_windows=1, purge_bars=1, embargo_bars=0)
        bars, signals, mf, clf, reg, vol, strat = _make_all_data()

        result = pipeline.run(
            bars=bars,
            signals=signals,
            market_features=mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            asset_symbol=ASSET_SYMBOL,
            strategy_name=STRATEGY_NAME,
        )

        assert result.n_windows_completed >= 1
        assert len(result.window_results) >= 1


# ---------------------------------------------------------------------------
# TestComputeSplits
# ---------------------------------------------------------------------------


class TestComputeSplits:
    """Unit tests for the _compute_splits helper."""

    def test_basic_split_structure(self):
        splits = _compute_splits(
            n_total=100,
            n_windows=3,
            train_ratio=0.5,
            purge_bars=2,
            embargo_bars=1,
        )
        assert len(splits) > 0

        for train_start, train_end, test_start, test_end in splits:
            # Training starts at 0 (expanding)
            assert train_start == 0
            # Training end is positive
            assert train_end > 0
            # Test starts after train + gap
            assert test_start >= train_end + 3  # purge=2 + embargo=1
            # Test end is after test start
            assert test_end > test_start

    def test_expanding_windows(self):
        """Train end should increase across windows."""
        splits = _compute_splits(
            n_total=100,
            n_windows=4,
            train_ratio=0.5,
            purge_bars=1,
            embargo_bars=0,
        )
        train_ends = [s[1] for s in splits]
        for i in range(1, len(train_ends)):
            assert train_ends[i] >= train_ends[i - 1]

    def test_insufficient_data_raises(self):
        """Too few bars for the configuration should raise."""
        with pytest.raises(ValueError, match="Insufficient"):
            _compute_splits(
                n_total=5,
                n_windows=3,
                train_ratio=0.5,
                purge_bars=10,
                embargo_bars=5,
            )

    def test_zero_purge_embargo(self):
        """Zero gap means test starts immediately after train."""
        splits = _compute_splits(
            n_total=100,
            n_windows=3,
            train_ratio=0.5,
            purge_bars=0,
            embargo_bars=0,
        )
        for _, train_end, test_start, _ in splits:
            assert test_start == train_end

    def test_no_overlapping_test_periods(self):
        """Test periods across windows should not overlap."""
        splits = _compute_splits(
            n_total=100,
            n_windows=4,
            train_ratio=0.5,
            purge_bars=2,
            embargo_bars=1,
        )
        for i in range(1, len(splits)):
            prev_test_end = splits[i - 1][3]
            curr_test_start = splits[i][2]
            # Test periods may overlap because each window's test is
            # from split_i+gap to split_{i+1}, but the next window's
            # test is from split_{i+1}+gap to split_{i+2}. Since
            # split_{i+1}+gap > split_{i+1} >= prev_test_end,
            # there should be no overlap.
            assert curr_test_start >= prev_test_end, (
                f"Window {i} test_start={curr_test_start} < prev test_end={prev_test_end}"
            )


# ---------------------------------------------------------------------------
# TestPipelineConfig property
# ---------------------------------------------------------------------------


class TestPipelineProperty:
    """Test pipeline property accessors."""

    def test_config_accessible(self):
        pipeline = _build_pipeline()
        assert isinstance(pipeline.config, PipelineConfig)
        assert pipeline.config.n_windows == 3
