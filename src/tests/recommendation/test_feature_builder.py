"""Tests for the RecommenderFeatureBuilder service."""

from __future__ import annotations


import polars as pl
import pytest

from src.app.recommendation.application.feature_builder import (
    RecommenderFeatureBuilder,
    RecommenderFeatureConfig,
)
from src.tests.recommendation.conftest import (
    BASE_TS,
    ONE_HOUR,
    make_classifier_outputs,
    make_market_features,
    make_regressor_outputs,
    make_strategy_returns,
    make_vol_forecasts,
)


# ---------------------------------------------------------------------------
# RecommenderFeatureConfig tests
# ---------------------------------------------------------------------------


class TestRecommenderFeatureConfig:
    """Tests for RecommenderFeatureConfig value object."""

    def test_defaults(self):
        cfg = RecommenderFeatureConfig()
        assert cfg.rolling_window == 20
        assert cfg.vol_regime_threshold == 1.0
        assert cfg.mi_reference_vol is None

    def test_frozen(self):
        cfg = RecommenderFeatureConfig()
        with pytest.raises(Exception, match="frozen"):  # noqa: B017
            cfg.rolling_window = 30  # type: ignore[misc]

    def test_custom_values(self):
        cfg = RecommenderFeatureConfig(
            rolling_window=10,
            vol_regime_threshold=1.5,
            mi_reference_vol=0.03,
        )
        assert cfg.rolling_window == 10
        assert cfg.vol_regime_threshold == 1.5
        assert cfg.mi_reference_vol == 0.03


# ---------------------------------------------------------------------------
# Feature builder — market features only
# ---------------------------------------------------------------------------


class TestFeatureBuilderMarketOnly:
    """Tests for feature builder with only market features."""

    def test_market_features_passed_through(self):
        """Market features are preserved as-is in the output."""
        mf = make_market_features(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf)

        assert "timestamp" in result.columns
        assert "close" in result.columns
        assert "volatility" in result.columns
        assert len(result) == 10

    def test_empty_market_features_raises(self):
        mf = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "close": pl.Series([], dtype=pl.Float64),
            }
        )
        builder = RecommenderFeatureBuilder()

        with pytest.raises(ValueError, match="empty"):
            builder.build_features(mf)

    def test_missing_timestamp_raises(self):
        mf = pl.DataFrame({"close": [1.0, 2.0], "volatility": [0.01, 0.02]})
        builder = RecommenderFeatureBuilder()

        with pytest.raises(ValueError, match="timestamp"):
            builder.build_features(mf)


# ---------------------------------------------------------------------------
# Feature builder — classifier features
# ---------------------------------------------------------------------------


class TestFeatureBuilderClassifier:
    """Tests for classifier feature integration."""

    def test_classifier_columns_added(self):
        mf = make_market_features(10)
        clf = make_classifier_outputs(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, classifier_outputs=clf)

        assert "clf_direction" in result.columns
        assert "clf_confidence" in result.columns

    def test_rolling_accuracy_computed(self):
        """Rolling accuracy is computed from clf_correct column."""
        n = 10
        correct = [True, False, True, True, False, True, True, True, False, True]
        mf = make_market_features(n)
        clf = make_classifier_outputs(n, correct=correct)
        config = RecommenderFeatureConfig(rolling_window=3)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf, classifier_outputs=clf)

        assert "clf_rolling_accuracy" in result.columns
        # Non-null values should exist
        acc_col = result.get_column("clf_rolling_accuracy")
        assert acc_col.null_count() < n

    def test_no_clf_correct_column_no_rolling_accuracy(self):
        """Without clf_correct, no rolling accuracy column is produced."""
        mf = make_market_features(10)
        clf = make_classifier_outputs(10)  # no correct= argument
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, classifier_outputs=clf)

        assert "clf_rolling_accuracy" not in result.columns


# ---------------------------------------------------------------------------
# Feature builder — regressor features
# ---------------------------------------------------------------------------


class TestFeatureBuilderRegressor:
    """Tests for regressor feature integration."""

    def test_regressor_columns_added(self):
        mf = make_market_features(10)
        reg = make_regressor_outputs(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, regressor_outputs=reg)

        assert "reg_predicted_return" in result.columns
        assert "reg_prediction_std" in result.columns

    def test_optional_quantile_spread(self):
        """Quantile spread is passed through when available."""
        n = 10
        mf = make_market_features(n)
        reg = make_regressor_outputs(n)
        reg = reg.with_columns(pl.lit(0.05).alias("reg_quantile_spread"))
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, regressor_outputs=reg)

        assert "reg_quantile_spread" in result.columns


# ---------------------------------------------------------------------------
# Feature builder — vol features
# ---------------------------------------------------------------------------


class TestFeatureBuilderVol:
    """Tests for volatility forecast feature integration."""

    def test_vol_predicted_added(self):
        mf = make_market_features(10)
        vol = make_vol_forecasts(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, vol_forecasts=vol)

        assert "vol_predicted" in result.columns

    def test_vol_error_trend_computed(self):
        """Vol error trend is computed when both predicted and actual are present."""
        n = 10
        predicted = [0.02 + 0.001 * i for i in range(n)]
        actual = [0.019 + 0.001 * i for i in range(n)]
        mf = make_market_features(n)
        vol = make_vol_forecasts(n, vol_predicted=predicted, vol_actual=actual)
        config = RecommenderFeatureConfig(rolling_window=3)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf, vol_forecasts=vol)

        assert "vol_error_trend" in result.columns

    def test_vol_error_trend_not_without_actual(self):
        """Without vol_actual, vol_error_trend is not computed."""
        mf = make_market_features(10)
        vol = make_vol_forecasts(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, vol_forecasts=vol)

        assert "vol_error_trend" not in result.columns


# ---------------------------------------------------------------------------
# Feature builder — combined features
# ---------------------------------------------------------------------------


class TestFeatureBuilderCombined:
    """Tests for combined classifier + regressor features."""

    def test_forecast_agreement_computed(self):
        """Forecast agreement is computed when both clf and reg present."""
        n = 10
        mf = make_market_features(n)
        # Positive direction and positive predicted return -> agreement = 1
        clf = make_classifier_outputs(n, directions=[1] * n)
        reg = make_regressor_outputs(n, predicted_returns=[0.01] * n)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, classifier_outputs=clf, regressor_outputs=reg)

        assert "forecast_agreement" in result.columns
        # All should agree
        agreements = result.get_column("forecast_agreement").to_list()
        assert all(a == 1 for a in agreements)

    def test_forecast_disagreement(self):
        """Disagreeing direction and return sign produce agreement = 0."""
        n = 5
        mf = make_market_features(n)
        clf = make_classifier_outputs(n, directions=[1] * n)
        reg = make_regressor_outputs(n, predicted_returns=[-0.01] * n)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, classifier_outputs=clf, regressor_outputs=reg)

        agreements = result.get_column("forecast_agreement").to_list()
        assert all(a == 0 for a in agreements)

    def test_conviction_score_computed(self):
        """Conviction score = |predicted_return| * confidence."""
        n = 5
        mf = make_market_features(n)
        clf = make_classifier_outputs(n, confidences=[0.8] * n)
        reg = make_regressor_outputs(n, predicted_returns=[0.05] * n)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, classifier_outputs=clf, regressor_outputs=reg)

        assert "conviction_score" in result.columns
        expected = 0.05 * 0.8
        scores = result.get_column("conviction_score").to_list()
        assert all(abs(s - expected) < 1e-10 for s in scores)

    def test_no_combined_features_without_both(self):
        """No combined features when only one of clf/reg is provided."""
        mf = make_market_features(10)
        clf = make_classifier_outputs(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, classifier_outputs=clf)

        assert "forecast_agreement" not in result.columns
        assert "conviction_score" not in result.columns


# ---------------------------------------------------------------------------
# Feature builder — regime features
# ---------------------------------------------------------------------------


class TestFeatureBuilderRegime:
    """Tests for regime feature computation."""

    def test_vol_regime_computed(self):
        """Vol regime is computed from vol_predicted column."""
        n = 20
        # Create clearly bimodal vol: 15 low, 5 high
        vol_values = [0.01] * 15 + [0.10] * 5
        mf = make_market_features(n)
        vol = make_vol_forecasts(n, vol_predicted=vol_values)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, vol_forecasts=vol)

        assert "vol_regime" in result.columns
        regimes = result.get_column("vol_regime").to_list()
        # High-vol bars should be regime=1
        assert any(r == 1 for r in regimes)
        assert any(r == 0 for r in regimes)

    def test_mi_significant_regime(self):
        """MI significance proxy fires when vol > reference."""
        n = 10
        vol_values = [0.01] * 5 + [0.05] * 5
        mf = make_market_features(n)
        vol = make_vol_forecasts(n, vol_predicted=vol_values)
        config = RecommenderFeatureConfig(mi_reference_vol=0.03)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf, vol_forecasts=vol)

        assert "mi_significant_regime" in result.columns
        mi_flags = result.get_column("mi_significant_regime").to_list()
        # First 5 bars (vol=0.01 < 0.03) should be 0, last 5 (vol=0.05 > 0.03) should be 1
        assert mi_flags[:5] == [0] * 5
        assert mi_flags[5:] == [1] * 5

    def test_no_regime_without_vol(self):
        """No regime features without volatility data."""
        mf = make_market_features(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf)

        assert "vol_regime" not in result.columns
        assert "mi_significant_regime" not in result.columns


# ---------------------------------------------------------------------------
# Feature builder — cross-asset features
# ---------------------------------------------------------------------------


class TestFeatureBuilderCrossAsset:
    """Tests for cross-asset feature computation."""

    def test_btc_lagged_return_added(self):
        """BTC lagged return is shift(1) of btc_return."""
        n = 10
        mf = make_market_features(n)
        btc_returns = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "btc_return": [0.001 * i for i in range(n)],
            }
        )
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, btc_returns=btc_returns)

        assert "btc_lagged_return" in result.columns
        lagged = result.get_column("btc_lagged_return").to_list()
        # First value should be null (shifted)
        assert lagged[0] is None
        # Second value should be btc_return[0] = 0.0
        assert abs(lagged[1] - 0.0) < 1e-10

    def test_relative_strength_computed(self):
        """Relative strength = asset return - universe mean return."""
        n = 10
        mf = make_market_features(n, price_step=100.0)
        universe = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "universe_mean_return": [0.001] * n,
            }
        )
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, universe_returns=universe)

        assert "relative_strength" in result.columns

    def test_no_cross_asset_without_data(self):
        """No cross-asset features when inputs are None."""
        mf = make_market_features(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf)

        assert "btc_lagged_return" not in result.columns
        assert "relative_strength" not in result.columns
        assert "rolling_cross_corr" not in result.columns


# ---------------------------------------------------------------------------
# Feature builder — strategy features
# ---------------------------------------------------------------------------


class TestFeatureBuilderStrategy:
    """Tests for historical strategy feature computation."""

    def test_rolling_sharpe_computed(self):
        """Rolling Sharpe is computed from strategy returns."""
        n = 30
        returns = [0.001 * ((-1) ** i) for i in range(n)]
        mf = make_market_features(n)
        strat = make_strategy_returns(n, returns=returns)
        config = RecommenderFeatureConfig(rolling_window=5)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf, strategy_returns=strat)

        assert "rolling_strategy_sharpe" in result.columns

    def test_rolling_win_rate_computed(self):
        """Rolling win rate is computed from strategy returns."""
        n = 20
        # Alternating positive and negative returns
        returns = [0.01 if i % 2 == 0 else -0.01 for i in range(n)]
        mf = make_market_features(n)
        strat = make_strategy_returns(n, returns=returns)
        config = RecommenderFeatureConfig(rolling_window=4)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf, strategy_returns=strat)

        assert "rolling_win_rate" in result.columns
        # With alternating returns and window=4, win rate should be ~0.5
        win_rates = result.get_column("rolling_win_rate").drop_nulls().to_list()
        assert len(win_rates) > 0
        assert all(0.0 <= wr <= 1.0 for wr in win_rates)


# ---------------------------------------------------------------------------
# Feature builder — graceful degradation
# ---------------------------------------------------------------------------


class TestFeatureBuilderGracefulDegradation:
    """Tests for graceful degradation with missing upstream inputs."""

    def test_all_none_returns_market_only(self):
        """All optional inputs as None returns market features + rolling perm entropy."""
        mf = make_market_features(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf)

        # rolling_perm_entropy is computed from close when present
        assert set(result.columns) == {"timestamp", "close", "volatility", "rolling_perm_entropy"}

    def test_full_feature_set(self):
        """All inputs provided produces the richest feature set."""
        n = 30
        mf = make_market_features(n)
        clf = make_classifier_outputs(n, correct=[i % 2 == 0 for i in range(n)])
        reg = make_regressor_outputs(n)
        vol = make_vol_forecasts(n, vol_actual=[0.019] * n)
        strat = make_strategy_returns(n)
        btc = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "btc_return": [0.001 * ((-1) ** i) for i in range(n)],
            }
        )
        universe = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "universe_mean_return": [0.0005] * n,
            }
        )
        config = RecommenderFeatureConfig(
            rolling_window=5,
            mi_reference_vol=0.015,
        )
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(
            mf,
            classifier_outputs=clf,
            regressor_outputs=reg,
            vol_forecasts=vol,
            strategy_returns=strat,
            btc_returns=btc,
            universe_returns=universe,
            asset_symbol="ETHUSDT",
        )

        # Should have many feature columns
        assert len(result.columns) > 10
        assert "clf_direction" in result.columns
        assert "reg_predicted_return" in result.columns
        assert "vol_predicted" in result.columns
        assert "forecast_agreement" in result.columns
        assert "conviction_score" in result.columns
        assert "rolling_strategy_sharpe" in result.columns
        assert "rolling_win_rate" in result.columns
        assert "btc_lagged_return" in result.columns
        assert "relative_strength" in result.columns
        # New features from #114
        assert "vol_qlike_residual" in result.columns
        assert "rolling_perm_entropy" in result.columns
        assert "btc_beta" in result.columns
        assert "mi_significant_regime" in result.columns
        assert len(result) == n


# ---------------------------------------------------------------------------
# Feature builder — QLIKE residual (Phase 12C / #114)
# ---------------------------------------------------------------------------


class TestFeatureBuilderQLIKE:
    """Tests for QLIKE residual computation in vol features."""

    def test_qlike_residual_computed(self):
        """QLIKE residual is computed when both vol_predicted and vol_actual are present."""
        n = 20
        predicted = [0.02 + 0.001 * i for i in range(n)]
        actual = [0.019 + 0.001 * i for i in range(n)]
        mf = make_market_features(n)
        vol = make_vol_forecasts(n, vol_predicted=predicted, vol_actual=actual)
        config = RecommenderFeatureConfig(rolling_window=5)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf, vol_forecasts=vol)

        assert "vol_qlike_residual" in result.columns
        qlike = result.get_column("vol_qlike_residual")
        # QLIKE(sigma, sigma) = sigma/sigma - ln(sigma/sigma) - 1 = 1 - 0 - 1 = 0
        # Since predicted != actual, values should be > 0 (QLIKE property)
        non_null = qlike.drop_nulls().to_list()
        assert len(non_null) > 0
        assert all(v >= 0.0 for v in non_null)

    def test_qlike_not_without_vol_actual(self):
        """Without vol_actual, QLIKE residual is not computed."""
        mf = make_market_features(10)
        vol = make_vol_forecasts(10)
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, vol_forecasts=vol)

        assert "vol_qlike_residual" not in result.columns

    def test_qlike_handles_zero_actual(self):
        """QLIKE handles vol_actual == 0 gracefully (null, not crash)."""
        n = 10
        predicted = [0.02] * n
        actual = [0.0] * 5 + [0.02] * 5  # first 5 are zero
        mf = make_market_features(n)
        vol = make_vol_forecasts(n, vol_predicted=predicted, vol_actual=actual)
        config = RecommenderFeatureConfig(rolling_window=3)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf, vol_forecasts=vol)

        assert "vol_qlike_residual" in result.columns
        # Should not raise — zeros handled via null guard


# ---------------------------------------------------------------------------
# Feature builder — rolling permutation entropy (Phase 12C / #114)
# ---------------------------------------------------------------------------


class TestFeatureBuilderPermEntropy:
    """Tests for rolling permutation entropy in regime features."""

    def test_perm_entropy_computed_from_close(self):
        """Rolling permutation entropy is computed when close prices are present."""
        n = 30
        mf = make_market_features(n)
        config = RecommenderFeatureConfig(rolling_window=10)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf)

        assert "rolling_perm_entropy" in result.columns
        pe = result.get_column("rolling_perm_entropy")
        non_null = pe.drop_nulls().to_list()
        assert len(non_null) > 0
        # PE is normalised to [0, 1]
        assert all(0.0 <= v <= 1.0 for v in non_null)

    def test_perm_entropy_config(self):
        """Permutation entropy config parameters are respected."""
        config = RecommenderFeatureConfig(perm_entropy_dim=4, perm_entropy_delay=2)
        assert config.perm_entropy_dim == 4
        assert config.perm_entropy_delay == 2

    def test_constant_prices_low_entropy(self):
        """Constant prices (zero returns) yield near-zero entropy."""
        n = 30
        mf = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "close": [100.0] * n,
                "volatility": [0.01] * n,
            }
        )
        config = RecommenderFeatureConfig(rolling_window=10)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf)

        pe = result.get_column("rolling_perm_entropy")
        non_null = pe.drop_nulls().to_list()
        # Constant returns → only one pattern → entropy should be 0 or very low
        if len(non_null) > 0:
            assert all(v <= 0.1 for v in non_null)


# ---------------------------------------------------------------------------
# Feature builder — beta to BTC (Phase 12C / #114)
# ---------------------------------------------------------------------------


class TestFeatureBuilderBTCBeta:
    """Tests for BTC beta in cross-asset features."""

    def test_btc_beta_computed(self):
        """Beta to BTC is computed for altcoins."""
        n = 30
        mf = make_market_features(n)
        btc = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "btc_return": [0.001 * ((-1) ** i) for i in range(n)],
            }
        )
        config = RecommenderFeatureConfig(rolling_window=10)
        builder = RecommenderFeatureBuilder(config)

        result = builder.build_features(mf, btc_returns=btc, asset_symbol="ETHUSDT")

        assert "btc_beta" in result.columns
        beta = result.get_column("btc_beta")
        non_null = beta.drop_nulls().to_list()
        assert len(non_null) > 0

    def test_btc_beta_not_for_btc_self(self):
        """BTC beta is skipped when asset is BTC itself."""
        n = 30
        mf = make_market_features(n)
        btc = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "btc_return": [0.001 * i for i in range(n)],
            }
        )
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, btc_returns=btc, asset_symbol="BTCUSDT")

        assert "btc_beta" not in result.columns
        assert "btc_lagged_return" not in result.columns
        assert "rolling_cross_corr" not in result.columns


# ---------------------------------------------------------------------------
# Feature builder — asset_symbol guard (Phase 12C / #114)
# ---------------------------------------------------------------------------


class TestFeatureBuilderAssetSymbolGuard:
    """Tests for asset_symbol parameter controlling BTC self-reference."""

    def test_btc_gets_no_btc_features(self):
        """BTC asset does not get BTC-lagged return or beta."""
        n = 20
        mf = make_market_features(n)
        btc = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "btc_return": [0.001] * n,
            }
        )
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, btc_returns=btc, asset_symbol="BTCUSDT")

        assert "btc_lagged_return" not in result.columns
        assert "btc_beta" not in result.columns

    def test_altcoin_gets_btc_features(self):
        """Altcoin (ETHUSDT) gets BTC-lagged return and beta."""
        n = 20
        mf = make_market_features(n)
        btc = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "btc_return": [0.001] * n,
            }
        )
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, btc_returns=btc, asset_symbol="ETHUSDT")

        assert "btc_lagged_return" in result.columns

    def test_no_asset_symbol_allows_btc_features(self):
        """When asset_symbol is None, BTC features are added (backwards compat)."""
        n = 20
        mf = make_market_features(n)
        btc = pl.DataFrame(
            {
                "timestamp": [BASE_TS + i * ONE_HOUR for i in range(n)],
                "btc_return": [0.001] * n,
            }
        )
        builder = RecommenderFeatureBuilder()

        result = builder.build_features(mf, btc_returns=btc)

        assert "btc_lagged_return" in result.columns
