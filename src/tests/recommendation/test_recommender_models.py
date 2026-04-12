"""Tests for recommender models (gradient boosting + baselines)."""

from __future__ import annotations

import numpy as np
import pytest

from src.app.recommendation.application.baseline_recommenders import (
    AllAssetsRecommender,
    ClassifierOnlyRecommender,
    ColumnIndexConfig,
    EqualWeightRecommender,
    RandomRecommender,
    RandomRecommenderConfig,
    RegressorOnlyRecommender,
)
from src.app.recommendation.application.gradient_boosting_recommender import (
    GradientBoostingRecommender,
    GradientBoostingRecommenderConfig,
)
from src.app.recommendation.domain.protocols import IRecommender
from src.app.recommendation.domain.value_objects import Recommendation


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_N_TRAIN: int = 200
_N_TEST: int = 50
_N_FEATURES: int = 5
_SEED: int = 42


def _make_linear_data(
    n: int = _N_TRAIN + _N_TEST,
    n_features: int = _N_FEATURES,
    seed: int = _SEED,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Generate synthetic linear data with noise for testing."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_features)).astype(np.float64)
    w = rng.standard_normal(n_features).astype(np.float64)
    y = (x @ w + rng.normal(0, 0.5, n)).astype(np.float64)
    return x, y


def _make_noise_data(
    n: int = _N_TRAIN + _N_TEST,
    n_features: int = _N_FEATURES,
    seed: int = _SEED,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Generate pure noise (no signal) for sanity checks."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_features)).astype(np.float64)
    y = rng.standard_normal(n).astype(np.float64)
    return x, y


def _assert_valid_recommendation(rec: Recommendation) -> None:
    """Validate a single recommendation's invariants."""
    assert isinstance(rec, Recommendation)
    assert 0.0 <= rec.confidence <= 1.0
    assert 0.0 <= rec.position_size <= 1.0
    assert rec.predicted_direction in {1, -1}
    assert rec.predicted_magnitude >= 0.0
    assert len(rec.asset) > 0


# ---------------------------------------------------------------------------
# GradientBoostingRecommender tests
# ---------------------------------------------------------------------------


class TestGradientBoostingRecommender:
    """Tests for the LightGBM gradient boosting recommender."""

    def test_fit_predict_round_trip(self):
        """fit + predict returns correct number of Recommendation objects."""
        x, y = _make_linear_data()
        model = GradientBoostingRecommender()
        model.fit(x[:_N_TRAIN], y[:_N_TRAIN])

        recs = model.predict(x[_N_TRAIN:])

        assert len(recs) == _N_TEST
        for rec in recs:
            _assert_valid_recommendation(rec)

    def test_protocol_conformance(self):
        """GradientBoostingRecommender satisfies IRecommender protocol."""
        model: IRecommender = GradientBoostingRecommender()
        x, y = _make_linear_data(n=30)
        model.fit(x[:20], y[:20])
        recs = model.predict(x[20:])
        assert len(recs) == 10
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_empty_x_train_raises(self):
        x_empty = np.empty((0, _N_FEATURES), dtype=np.float64)
        y_empty = np.empty(0, dtype=np.float64)
        model = GradientBoostingRecommender()

        with pytest.raises(ValueError, match="at least 2 samples"):
            model.fit(x_empty, y_empty)

    def test_single_sample_raises(self):
        """LightGBM requires at least 2 samples — single sample must fail."""
        x = np.ones((1, _N_FEATURES), dtype=np.float64)
        y = np.array([0.01], dtype=np.float64)
        model = GradientBoostingRecommender()

        with pytest.raises(ValueError, match="at least 2 samples"):
            model.fit(x, y)

    def test_empty_x_predict_raises(self):
        x, y = _make_linear_data(n=30)
        model = GradientBoostingRecommender()
        model.fit(x[:20], y[:20])

        x_empty = np.empty((0, _N_FEATURES), dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            model.predict(x_empty)

    def test_predict_before_fit_raises(self):
        model = GradientBoostingRecommender()
        x_test = np.ones((10, _N_FEATURES), dtype=np.float64)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(x_test)

    def test_shape_mismatch_raises(self):
        x = np.ones((10, _N_FEATURES), dtype=np.float64)
        y = np.ones(5, dtype=np.float64)  # wrong length
        model = GradientBoostingRecommender()

        with pytest.raises(ValueError, match="same number of samples"):
            model.fit(x, y)

    def test_deploy_threshold_logic(self):
        """Samples with predicted return <= threshold should not deploy."""
        config = GradientBoostingRecommenderConfig(min_threshold=0.5)
        x, y = _make_linear_data()
        model = GradientBoostingRecommender(config=config)
        model.fit(x[:_N_TRAIN], y[:_N_TRAIN])

        recs = model.predict(x[_N_TRAIN:])

        for rec in recs:
            if rec.deploy:
                assert rec.predicted_strategy_return > 0.5
            else:
                assert rec.predicted_strategy_return <= 0.5

    def test_position_sizing_clamped(self):
        """Position size should always be in [0.0, cap]."""
        config = GradientBoostingRecommenderConfig(position_size_cap=0.5)
        x, y = _make_linear_data()
        model = GradientBoostingRecommender(config=config)
        model.fit(x[:_N_TRAIN], y[:_N_TRAIN])

        recs = model.predict(x[_N_TRAIN:])

        for rec in recs:
            assert 0.0 <= rec.position_size <= 0.5

    def test_non_deployed_has_zero_position_size(self):
        """When deploy=False, position_size should be 0.0."""
        config = GradientBoostingRecommenderConfig(min_threshold=100.0)
        x, y = _make_linear_data()
        model = GradientBoostingRecommender(config=config)
        model.fit(x[:_N_TRAIN], y[:_N_TRAIN])

        recs = model.predict(x[_N_TRAIN:])

        # With huge threshold, nothing should deploy
        for rec in recs:
            assert not rec.deploy
            assert rec.position_size == 0.0

    def test_feature_importances_after_fit(self):
        """Feature importances should be available after fit."""
        x, y = _make_linear_data()
        model = GradientBoostingRecommender()
        model.fit(x[:_N_TRAIN], y[:_N_TRAIN])

        importances = model.feature_importances
        assert importances is not None
        assert importances.shape == (_N_FEATURES,)
        assert np.all(importances >= 0)

    def test_feature_importances_none_before_fit(self):
        model = GradientBoostingRecommender()
        assert model.feature_importances is None

    def test_get_feature_importances_dict(self):
        x, y = _make_linear_data()
        model = GradientBoostingRecommender()
        model.fit(x[:_N_TRAIN], y[:_N_TRAIN])

        imp_dict = model.get_feature_importances()
        assert len(imp_dict) == _N_FEATURES
        assert all(k.startswith("f_") for k in imp_dict)

        # Sorted descending
        values = list(imp_dict.values())
        assert values == sorted(values, reverse=True)

    def test_get_feature_importances_with_names(self):
        x, y = _make_linear_data()
        model = GradientBoostingRecommender()
        model.fit(x[:_N_TRAIN], y[:_N_TRAIN])

        names = ["alpha", "beta", "gamma", "delta", "epsilon"]
        imp_dict = model.get_feature_importances(feature_names=names)
        assert set(imp_dict.keys()) == set(names)

    def test_get_feature_importances_before_fit_raises(self):
        model = GradientBoostingRecommender()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_feature_importances()

    def test_asset_names_passed_through(self):
        """Custom asset names should appear in recommendations."""
        x, y = _make_linear_data(n=30)
        assets = [f"ASSET{i}" for i in range(10)]
        model = GradientBoostingRecommender(asset_names=assets)
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        for i, rec in enumerate(recs):
            assert rec.asset == f"ASSET{i}"

    def test_default_asset_when_none(self):
        """Without asset_names, default placeholder is used."""
        x, y = _make_linear_data(n=30)
        model = GradientBoostingRecommender()
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        assert all(rec.asset == "UNKNOWN" for rec in recs)

    def test_direction_matches_return_sign(self):
        """predicted_direction should match the sign of predicted_strategy_return."""
        x, y = _make_linear_data()
        model = GradientBoostingRecommender()
        model.fit(x[:_N_TRAIN], y[:_N_TRAIN])

        recs = model.predict(x[_N_TRAIN:])

        for rec in recs:
            if rec.predicted_strategy_return >= 0.0:
                assert rec.predicted_direction == 1
            else:
                assert rec.predicted_direction == -1

    def test_magnitude_is_abs_return(self):
        """predicted_magnitude should equal abs(predicted_strategy_return)."""
        x, y = _make_linear_data()
        model = GradientBoostingRecommender()
        model.fit(x[:_N_TRAIN], y[:_N_TRAIN])

        recs = model.predict(x[_N_TRAIN:])

        for rec in recs:
            assert abs(rec.predicted_magnitude - abs(rec.predicted_strategy_return)) < 1e-12

    def test_is_fitted_property(self):
        model = GradientBoostingRecommender()
        assert not model.is_fitted

        x, y = _make_linear_data(n=30)
        model.fit(x[:20], y[:20])
        assert model.is_fitted

    def test_noise_sanity_check(self):
        """On pure noise, the GBM recommender should not be dramatically better than random.

        This is the key sanity test from Phase 12F: a recommender trained on
        noise that outperforms random is a red flag for data leakage or overfitting.
        """
        x, y = _make_noise_data(n=500, seed=123)
        model = GradientBoostingRecommender(
            config=GradientBoostingRecommenderConfig(n_estimators=50, min_threshold=0.0)
        )
        model.fit(x[:350], y[:350])

        recs = model.predict(x[350:])
        y_test = y[350:]

        # Compute model MAE vs naive (predict zero) MAE
        model_preds = np.array([r.predicted_strategy_return for r in recs], dtype=np.float64)
        mae_model = float(np.mean(np.abs(model_preds - y_test)))
        mae_naive = float(np.mean(np.abs(y_test)))

        # Model should not be much better than naive on pure noise
        assert mae_model > 0.5 * mae_naive, (
            f"GBM MAE={mae_model:.4f} << naive MAE={mae_naive:.4f} on pure noise — possible overfitting or leakage"
        )

    def test_config_defaults(self):
        config = GradientBoostingRecommenderConfig()
        assert config.n_estimators == 100
        assert config.learning_rate == 0.05
        assert config.max_depth == 5
        assert config.min_child_samples == 20
        assert config.min_threshold == 0.0
        assert config.position_size_cap == 1.0
        assert config.random_seed == 42

    def test_config_frozen(self):
        config = GradientBoostingRecommenderConfig()
        with pytest.raises(Exception, match="frozen"):  # noqa: B017
            config.n_estimators = 200  # type: ignore[misc]

    def test_two_sample_fit(self):
        """Edge case: fit with exactly two samples (LightGBM minimum) should succeed."""
        x = np.array([[1.0] * _N_FEATURES, [2.0] * _N_FEATURES], dtype=np.float64)
        y = np.array([0.01, -0.01], dtype=np.float64)
        model = GradientBoostingRecommender()
        model.fit(x, y)

        recs = model.predict(x)
        assert len(recs) == 2
        for rec in recs:
            _assert_valid_recommendation(rec)


# ---------------------------------------------------------------------------
# RandomRecommender tests
# ---------------------------------------------------------------------------


class TestRandomRecommender:
    """Tests for the RandomRecommender baseline."""

    def test_fit_predict_round_trip(self):
        x, y = _make_noise_data(n=30)
        model = RandomRecommender()
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        assert len(recs) == 10
        for rec in recs:
            _assert_valid_recommendation(rec)

    def test_protocol_conformance(self):
        model: IRecommender = RandomRecommender()
        x, y = _make_noise_data(n=30)
        model.fit(x[:20], y[:20])
        recs = model.predict(x[20:])
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_empty_x_train_raises(self):
        model = RandomRecommender()
        with pytest.raises(ValueError, match="at least one sample"):
            model.fit(
                np.empty((0, 5), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

    def test_empty_x_predict_raises(self):
        x, y = _make_noise_data(n=20)
        model = RandomRecommender()
        model.fit(x[:10], y[:10])

        with pytest.raises(ValueError, match="at least one sample"):
            model.predict(np.empty((0, 5), dtype=np.float64))

    def test_predict_before_fit_raises(self):
        model = RandomRecommender()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.ones((5, 3), dtype=np.float64))

    def test_deploy_probability_zero(self):
        """deploy_probability=0 means nothing deployed."""
        config = RandomRecommenderConfig(deploy_probability=0.0)
        model = RandomRecommender(config=config)
        x, y = _make_noise_data(n=30)
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        assert all(not r.deploy for r in recs)

    def test_deploy_probability_one(self):
        """deploy_probability=1.0 means everything deployed."""
        config = RandomRecommenderConfig(deploy_probability=1.0)
        model = RandomRecommender(config=config)
        x, y = _make_noise_data(n=30)
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        assert all(r.deploy for r in recs)

    def test_reproducibility_with_same_seed(self):
        """Same seed produces same recommendations."""
        config = RandomRecommenderConfig(random_seed=99)
        x, y = _make_noise_data(n=30)

        model_a = RandomRecommender(config=config)
        model_a.fit(x[:20], y[:20])
        recs_a = model_a.predict(x[20:])

        model_b = RandomRecommender(config=config)
        model_b.fit(x[:20], y[:20])
        recs_b = model_b.predict(x[20:])

        for a, b in zip(recs_a, recs_b, strict=True):
            assert a.predicted_strategy_return == b.predicted_strategy_return
            assert a.deploy == b.deploy


# ---------------------------------------------------------------------------
# AllAssetsRecommender tests
# ---------------------------------------------------------------------------


class TestAllAssetsRecommender:
    """Tests for the AllAssetsRecommender baseline."""

    def test_all_deployed(self):
        x, y = _make_noise_data(n=30)
        model = AllAssetsRecommender()
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        assert len(recs) == 10
        assert all(r.deploy for r in recs)

    def test_full_position_size(self):
        x, y = _make_noise_data(n=30)
        model = AllAssetsRecommender()
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        assert all(r.position_size == 1.0 for r in recs)

    def test_protocol_conformance(self):
        model: IRecommender = AllAssetsRecommender()
        x, y = _make_noise_data(n=30)
        model.fit(x[:20], y[:20])
        recs = model.predict(x[20:])
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_empty_x_train_raises(self):
        model = AllAssetsRecommender()
        with pytest.raises(ValueError, match="at least one sample"):
            model.fit(
                np.empty((0, 5), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

    def test_predict_before_fit_raises(self):
        model = AllAssetsRecommender()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.ones((5, 3), dtype=np.float64))

    def test_direction_from_training_mean(self):
        """Direction should be +1 when mean(y_train) >= 0, else -1."""
        x = np.ones((10, 3), dtype=np.float64)
        y_positive = np.array([0.01] * 10, dtype=np.float64)
        y_negative = np.array([-0.01] * 10, dtype=np.float64)

        model_pos = AllAssetsRecommender()
        model_pos.fit(x, y_positive)
        recs_pos = model_pos.predict(x[:5])
        assert all(r.predicted_direction == 1 for r in recs_pos)

        model_neg = AllAssetsRecommender()
        model_neg.fit(x, y_negative)
        recs_neg = model_neg.predict(x[:5])
        assert all(r.predicted_direction == -1 for r in recs_neg)


# ---------------------------------------------------------------------------
# ClassifierOnlyRecommender tests
# ---------------------------------------------------------------------------


class TestClassifierOnlyRecommender:
    """Tests for the ClassifierOnlyRecommender baseline."""

    def test_fit_predict_round_trip(self):
        config = ColumnIndexConfig(col_idx=0)
        x, y = _make_noise_data(n=30)
        model = ClassifierOnlyRecommender(config=config)
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        assert len(recs) == 10
        for rec in recs:
            _assert_valid_recommendation(rec)

    def test_protocol_conformance(self):
        config = ColumnIndexConfig(col_idx=0)
        model: IRecommender = ClassifierOnlyRecommender(config=config)
        x, y = _make_noise_data(n=30)
        model.fit(x[:20], y[:20])
        recs = model.predict(x[20:])
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_col_idx_out_of_bounds_raises(self):
        config = ColumnIndexConfig(col_idx=99)
        model = ClassifierOnlyRecommender(config=config)
        x = np.ones((10, 5), dtype=np.float64)
        y = np.ones(10, dtype=np.float64)

        with pytest.raises(ValueError, match="out of bounds"):
            model.fit(x, y)

    def test_predict_before_fit_raises(self):
        config = ColumnIndexConfig(col_idx=0)
        model = ClassifierOnlyRecommender(config=config)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.ones((5, 3), dtype=np.float64))

    def test_deploy_based_on_confidence_threshold(self):
        """Deployment should depend on |confidence| > threshold."""
        config = ColumnIndexConfig(col_idx=0, min_threshold=0.5)
        x_train = np.ones((10, 3), dtype=np.float64)
        y_train = np.ones(10, dtype=np.float64)
        model = ClassifierOnlyRecommender(config=config)
        model.fit(x_train, y_train)

        # Feature column 0 has values: [0.3, 0.7, -0.8, 0.1]
        x_test = np.array(
            [
                [0.3, 0.0, 0.0],
                [0.7, 0.0, 0.0],
                [-0.8, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        recs = model.predict(x_test)

        # |0.3| = 0.3 <= 0.5 → no deploy
        assert not recs[0].deploy
        # |0.7| = 0.7 > 0.5 → deploy
        assert recs[1].deploy
        # |-0.8| = 0.8 > 0.5 → deploy
        assert recs[2].deploy
        # |0.1| = 0.1 <= 0.5 → no deploy
        assert not recs[3].deploy


# ---------------------------------------------------------------------------
# RegressorOnlyRecommender tests
# ---------------------------------------------------------------------------


class TestRegressorOnlyRecommender:
    """Tests for the RegressorOnlyRecommender baseline."""

    def test_fit_predict_round_trip(self):
        config = ColumnIndexConfig(col_idx=0)
        x, y = _make_noise_data(n=30)
        model = RegressorOnlyRecommender(config=config)
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        assert len(recs) == 10
        for rec in recs:
            _assert_valid_recommendation(rec)

    def test_protocol_conformance(self):
        config = ColumnIndexConfig(col_idx=0)
        model: IRecommender = RegressorOnlyRecommender(config=config)
        x, y = _make_noise_data(n=30)
        model.fit(x[:20], y[:20])
        recs = model.predict(x[20:])
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_col_idx_out_of_bounds_raises(self):
        config = ColumnIndexConfig(col_idx=99)
        model = RegressorOnlyRecommender(config=config)
        x = np.ones((10, 5), dtype=np.float64)
        y = np.ones(10, dtype=np.float64)

        with pytest.raises(ValueError, match="out of bounds"):
            model.fit(x, y)

    def test_predict_before_fit_raises(self):
        config = ColumnIndexConfig(col_idx=0)
        model = RegressorOnlyRecommender(config=config)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.ones((5, 3), dtype=np.float64))

    def test_deploy_threshold_logic(self):
        """Deploys when predicted return > threshold."""
        config = ColumnIndexConfig(col_idx=0, min_threshold=0.1)
        x_train = np.ones((10, 3), dtype=np.float64)
        y_train = np.ones(10, dtype=np.float64)
        model = RegressorOnlyRecommender(config=config)
        model.fit(x_train, y_train)

        x_test = np.array(
            [
                [0.05, 0.0, 0.0],  # 0.05 <= 0.1 → no deploy
                [0.2, 0.0, 0.0],  # 0.2 > 0.1 → deploy
                [-0.5, 0.0, 0.0],  # negative → no deploy
            ],
            dtype=np.float64,
        )

        recs = model.predict(x_test)
        assert not recs[0].deploy
        assert recs[1].deploy
        assert not recs[2].deploy

    def test_position_sizing_kelly_adjacent(self):
        """Position size should follow max(r_hat - threshold, 0) / sigma."""
        config = ColumnIndexConfig(col_idx=0, min_threshold=0.0)
        x_train = np.ones((10, 3), dtype=np.float64)
        # y_train with known std
        y_train = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1], dtype=np.float64)
        model = RegressorOnlyRecommender(config=config)
        model.fit(x_train, y_train)

        # Use a known value in column 0
        x_test = np.array([[0.05, 0.0, 0.0]], dtype=np.float64)
        recs = model.predict(x_test)

        # sigma = std(y_train, ddof=1) ≈ 0.10541
        expected_sigma = float(np.std(y_train, ddof=1))
        expected_size = min(0.05 / expected_sigma, 1.0)
        assert abs(recs[0].position_size - expected_size) < 1e-6


# ---------------------------------------------------------------------------
# EqualWeightRecommender tests
# ---------------------------------------------------------------------------


class TestEqualWeightRecommender:
    """Tests for the EqualWeightRecommender baseline."""

    def test_fit_predict_round_trip(self):
        config = ColumnIndexConfig(col_idx=0)
        x, y = _make_noise_data(n=30)
        model = EqualWeightRecommender(config=config)
        model.fit(x[:20], y[:20])

        recs = model.predict(x[20:])
        assert len(recs) == 10
        for rec in recs:
            _assert_valid_recommendation(rec)

    def test_protocol_conformance(self):
        config = ColumnIndexConfig(col_idx=0)
        model: IRecommender = EqualWeightRecommender(config=config)
        x, y = _make_noise_data(n=30)
        model.fit(x[:20], y[:20])
        recs = model.predict(x[20:])
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_equal_weight_distribution(self):
        """All positive forecasts get 1/n_positive weight."""
        config = ColumnIndexConfig(col_idx=0)
        model = EqualWeightRecommender(config=config)
        x_train = np.ones((10, 3), dtype=np.float64)
        y_train = np.ones(10, dtype=np.float64)
        model.fit(x_train, y_train)

        # 3 out of 4 samples are positive
        x_test = np.array(
            [
                [0.1, 0.0, 0.0],  # positive → deployed
                [0.2, 0.0, 0.0],  # positive → deployed
                [-0.1, 0.0, 0.0],  # negative → not deployed
                [0.3, 0.0, 0.0],  # positive → deployed
            ],
            dtype=np.float64,
        )

        recs = model.predict(x_test)

        expected_weight = 1.0 / 3.0
        assert recs[0].deploy
        assert abs(recs[0].position_size - expected_weight) < 1e-12
        assert recs[1].deploy
        assert abs(recs[1].position_size - expected_weight) < 1e-12
        assert not recs[2].deploy
        assert recs[2].position_size == 0.0
        assert recs[3].deploy
        assert abs(recs[3].position_size - expected_weight) < 1e-12

    def test_all_negative_forecasts(self):
        """When all forecasts are negative, no deployment and zero position size."""
        config = ColumnIndexConfig(col_idx=0)
        model = EqualWeightRecommender(config=config)
        x_train = np.ones((10, 3), dtype=np.float64)
        y_train = np.ones(10, dtype=np.float64)
        model.fit(x_train, y_train)

        x_test = np.array(
            [
                [-0.1, 0.0, 0.0],
                [-0.2, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        recs = model.predict(x_test)
        assert all(not r.deploy for r in recs)
        assert all(r.position_size == 0.0 for r in recs)

    def test_single_positive_gets_full_weight(self):
        """A single positive forecast gets weight = 1.0."""
        config = ColumnIndexConfig(col_idx=0)
        model = EqualWeightRecommender(config=config)
        x_train = np.ones((10, 3), dtype=np.float64)
        y_train = np.ones(10, dtype=np.float64)
        model.fit(x_train, y_train)

        x_test = np.array(
            [
                [0.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        recs = model.predict(x_test)
        assert recs[0].position_size == 1.0
        assert recs[1].position_size == 0.0

    def test_col_idx_out_of_bounds_raises(self):
        config = ColumnIndexConfig(col_idx=99)
        model = EqualWeightRecommender(config=config)
        x = np.ones((10, 5), dtype=np.float64)
        y = np.ones(10, dtype=np.float64)

        with pytest.raises(ValueError, match="out of bounds"):
            model.fit(x, y)

    def test_predict_before_fit_raises(self):
        config = ColumnIndexConfig(col_idx=0)
        model = EqualWeightRecommender(config=config)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.ones((5, 3), dtype=np.float64))

    def test_zero_forecast_not_deployed(self):
        """A forecast of exactly 0.0 should not deploy (> 0, not >=)."""
        config = ColumnIndexConfig(col_idx=0)
        model = EqualWeightRecommender(config=config)
        x_train = np.ones((10, 3), dtype=np.float64)
        y_train = np.ones(10, dtype=np.float64)
        model.fit(x_train, y_train)

        x_test = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        recs = model.predict(x_test)
        assert not recs[0].deploy
        assert recs[0].position_size == 0.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestColumnIndexConfig:
    """Tests for the ColumnIndexConfig shared config."""

    def test_defaults(self):
        config = ColumnIndexConfig(col_idx=2)
        assert config.col_idx == 2
        assert config.min_threshold == 0.0
        assert config.random_seed == 42

    def test_frozen(self):
        config = ColumnIndexConfig(col_idx=0)
        with pytest.raises(Exception, match="frozen"):  # noqa: B017
            config.col_idx = 5  # type: ignore[misc]


class TestRandomRecommenderConfig:
    """Tests for RandomRecommenderConfig."""

    def test_defaults(self):
        config = RandomRecommenderConfig()
        assert config.random_seed == 42
        assert config.deploy_probability == 0.5

    def test_frozen(self):
        config = RandomRecommenderConfig()
        with pytest.raises(Exception, match="frozen"):  # noqa: B017
            config.deploy_probability = 0.9  # type: ignore[misc]
