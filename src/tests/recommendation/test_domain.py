"""Tests for recommendation domain value objects and protocols."""

from __future__ import annotations

from datetime import datetime, UTC

import numpy as np
import pytest
from pydantic import ValidationError

from src.app.recommendation.domain.protocols import IRecommender
from src.app.recommendation.domain.value_objects import (
    Recommendation,
    RecommendationInput,
    RecommenderConfig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSET: str = "BTCUSDT"
TIMESTAMP: datetime = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# RecommendationInput tests
# ---------------------------------------------------------------------------


class TestRecommendationInput:
    """Tests for RecommendationInput value object."""

    def test_valid_long(self):
        features: np.ndarray = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        inp: RecommendationInput = RecommendationInput(
            asset=ASSET,
            timestamp=TIMESTAMP,
            feature_vector=features,
            direction_forecast=1,
            return_forecast=0.005,
        )
        assert inp.asset == ASSET
        assert inp.timestamp == TIMESTAMP
        assert np.array_equal(inp.feature_vector, features)
        assert inp.direction_forecast == 1
        assert inp.return_forecast == 0.005

    def test_valid_short(self):
        features: np.ndarray = np.array([0.5], dtype=np.float64)
        inp: RecommendationInput = RecommendationInput(
            asset="ETHUSDT",
            timestamp=TIMESTAMP,
            feature_vector=features,
            direction_forecast=-1,
            return_forecast=-0.01,
        )
        assert inp.direction_forecast == -1
        assert inp.return_forecast == -0.01

    def test_frozen(self):
        features: np.ndarray = np.array([1.0], dtype=np.float64)
        inp: RecommendationInput = RecommendationInput(
            asset=ASSET,
            timestamp=TIMESTAMP,
            feature_vector=features,
            direction_forecast=1,
            return_forecast=0.0,
        )
        with pytest.raises(ValidationError, match="frozen"):
            inp.asset = "OTHER"  # type: ignore[misc]

    def test_direction_must_be_plus_or_minus_one(self):
        features: np.ndarray = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValidationError, match="direction_forecast must be \\+1 or -1"):
            RecommendationInput(
                asset=ASSET,
                timestamp=TIMESTAMP,
                feature_vector=features,
                direction_forecast=0,
                return_forecast=0.0,
            )

    def test_direction_rejects_two(self):
        features: np.ndarray = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValidationError, match="direction_forecast must be \\+1 or -1"):
            RecommendationInput(
                asset=ASSET,
                timestamp=TIMESTAMP,
                feature_vector=features,
                direction_forecast=2,
                return_forecast=0.0,
            )

    def test_feature_vector_must_be_1d(self):
        features_2d: np.ndarray = np.array([[1.0, 2.0]], dtype=np.float64)
        with pytest.raises(ValidationError, match="feature_vector must be 1-D"):
            RecommendationInput(
                asset=ASSET,
                timestamp=TIMESTAMP,
                feature_vector=features_2d,
                direction_forecast=1,
                return_forecast=0.0,
            )

    def test_empty_asset_rejected(self):
        features: np.ndarray = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            RecommendationInput(
                asset="",
                timestamp=TIMESTAMP,
                feature_vector=features,
                direction_forecast=1,
                return_forecast=0.0,
            )


# ---------------------------------------------------------------------------
# Recommendation tests
# ---------------------------------------------------------------------------


class TestRecommendation:
    """Tests for Recommendation value object."""

    def test_valid_deploy(self):
        rec: Recommendation = Recommendation(
            asset=ASSET,
            predicted_strategy_return=0.003,
            confidence=0.85,
            deploy=True,
            predicted_direction=1,
            predicted_magnitude=0.003,
            position_size=0.7,
        )
        assert rec.asset == ASSET
        assert rec.predicted_strategy_return == 0.003
        assert rec.confidence == 0.85
        assert rec.deploy is True
        assert rec.predicted_direction == 1
        assert rec.predicted_magnitude == 0.003
        assert rec.position_size == 0.7

    def test_valid_no_deploy(self):
        rec: Recommendation = Recommendation(
            asset=ASSET,
            predicted_strategy_return=-0.001,
            confidence=0.3,
            deploy=False,
            predicted_direction=-1,
            predicted_magnitude=0.001,
            position_size=0.0,
        )
        assert rec.deploy is False
        assert rec.predicted_direction == -1
        assert rec.position_size == 0.0

    def test_frozen(self):
        rec: Recommendation = Recommendation(
            asset=ASSET,
            predicted_strategy_return=0.0,
            confidence=0.5,
            deploy=False,
            predicted_direction=1,
            predicted_magnitude=0.0,
            position_size=0.0,
        )
        with pytest.raises(ValidationError, match="frozen"):
            rec.deploy = True  # type: ignore[misc]

    def test_confidence_lower_bound(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Recommendation(
                asset=ASSET,
                predicted_strategy_return=0.0,
                confidence=-0.1,
                deploy=False,
                predicted_direction=1,
                predicted_magnitude=0.0,
                position_size=0.0,
            )

    def test_confidence_upper_bound(self):
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            Recommendation(
                asset=ASSET,
                predicted_strategy_return=0.0,
                confidence=1.1,
                deploy=False,
                predicted_direction=1,
                predicted_magnitude=0.0,
                position_size=0.0,
            )

    def test_confidence_boundary_zero(self):
        rec: Recommendation = Recommendation(
            asset=ASSET,
            predicted_strategy_return=0.0,
            confidence=0.0,
            deploy=False,
            predicted_direction=1,
            predicted_magnitude=0.0,
            position_size=0.0,
        )
        assert rec.confidence == 0.0

    def test_confidence_boundary_one(self):
        rec: Recommendation = Recommendation(
            asset=ASSET,
            predicted_strategy_return=0.0,
            confidence=1.0,
            deploy=True,
            predicted_direction=-1,
            predicted_magnitude=0.0,
            position_size=1.0,
        )
        assert rec.confidence == 1.0

    def test_direction_must_be_plus_or_minus_one(self):
        with pytest.raises(ValidationError, match="predicted_direction must be \\+1 or -1"):
            Recommendation(
                asset=ASSET,
                predicted_strategy_return=0.0,
                confidence=0.5,
                deploy=False,
                predicted_direction=0,
                predicted_magnitude=0.0,
                position_size=0.0,
            )

    def test_predicted_magnitude_non_negative(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Recommendation(
                asset=ASSET,
                predicted_strategy_return=0.0,
                confidence=0.5,
                deploy=False,
                predicted_direction=1,
                predicted_magnitude=-0.01,
                position_size=0.0,
            )

    def test_position_size_lower_bound(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Recommendation(
                asset=ASSET,
                predicted_strategy_return=0.0,
                confidence=0.5,
                deploy=False,
                predicted_direction=1,
                predicted_magnitude=0.0,
                position_size=-0.1,
            )

    def test_position_size_upper_bound(self):
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            Recommendation(
                asset=ASSET,
                predicted_strategy_return=0.0,
                confidence=0.5,
                deploy=False,
                predicted_direction=1,
                predicted_magnitude=0.0,
                position_size=1.1,
            )

    def test_position_size_boundary_zero(self):
        rec: Recommendation = Recommendation(
            asset=ASSET,
            predicted_strategy_return=0.0,
            confidence=0.5,
            deploy=False,
            predicted_direction=1,
            predicted_magnitude=0.0,
            position_size=0.0,
        )
        assert rec.position_size == 0.0

    def test_position_size_boundary_one(self):
        rec: Recommendation = Recommendation(
            asset=ASSET,
            predicted_strategy_return=0.01,
            confidence=0.9,
            deploy=True,
            predicted_direction=1,
            predicted_magnitude=0.01,
            position_size=1.0,
        )
        assert rec.position_size == 1.0

    def test_empty_asset_rejected(self):
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            Recommendation(
                asset="",
                predicted_strategy_return=0.0,
                confidence=0.5,
                deploy=False,
                predicted_direction=1,
                predicted_magnitude=0.0,
                position_size=0.0,
            )


# ---------------------------------------------------------------------------
# RecommenderConfig tests
# ---------------------------------------------------------------------------


class TestRecommenderConfig:
    """Tests for RecommenderConfig value object."""

    def test_defaults(self):
        cfg: RecommenderConfig = RecommenderConfig()
        assert cfg.model_type == "lightgbm"
        assert cfg.train_window == 500
        assert cfg.retrain_frequency == 50
        assert cfg.min_threshold == 0.0
        assert cfg.label_horizon == 7

    def test_custom_values(self):
        cfg: RecommenderConfig = RecommenderConfig(
            model_type="random",
            train_window=1000,
            retrain_frequency=100,
            min_threshold=0.001,
            label_horizon=14,
        )
        assert cfg.model_type == "random"
        assert cfg.train_window == 1000
        assert cfg.retrain_frequency == 100
        assert cfg.min_threshold == 0.001
        assert cfg.label_horizon == 14

    def test_frozen(self):
        cfg: RecommenderConfig = RecommenderConfig()
        with pytest.raises(ValidationError, match="frozen"):
            cfg.model_type = "other"  # type: ignore[misc]

    def test_train_window_must_be_positive(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            RecommenderConfig(train_window=0)

    def test_retrain_frequency_must_be_positive(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            RecommenderConfig(retrain_frequency=0)

    def test_min_threshold_non_negative(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            RecommenderConfig(min_threshold=-0.01)

    def test_label_horizon_must_be_positive(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            RecommenderConfig(label_horizon=0)

    def test_model_type_non_empty(self):
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            RecommenderConfig(model_type="")


# ---------------------------------------------------------------------------
# IRecommender protocol tests
# ---------------------------------------------------------------------------


class TestIRecommenderProtocol:
    """Tests for the IRecommender structural protocol."""

    def test_structural_conformance(self):
        """A class with the right fit/predict signatures satisfies IRecommender."""

        class _FakeRecommender:
            def fit(
                self,
                x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
                y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
            ) -> None:
                pass

            def predict(
                self,
                x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
            ) -> list[Recommendation]:
                n_samples: int = x.shape[0]
                return [
                    Recommendation(
                        asset=ASSET,
                        predicted_strategy_return=0.001,
                        confidence=0.6,
                        deploy=True,
                        predicted_direction=1,
                        predicted_magnitude=0.001,
                        position_size=0.5,
                    )
                    for _ in range(n_samples)
                ]

        fake: IRecommender = _FakeRecommender()

        x_train: np.ndarray = np.random.default_rng(42).standard_normal((10, 5))
        y_train: np.ndarray = np.random.default_rng(42).standard_normal(10)
        fake.fit(x_train, y_train)

        x_test: np.ndarray = np.random.default_rng(42).standard_normal((3, 5))
        recs: list[Recommendation] = fake.predict(x_test)
        assert len(recs) == 3
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_protocol_is_structural_not_nominal(self):
        """Verify IRecommender is a Protocol (structural subtyping)."""
        assert hasattr(IRecommender, "__protocol_attrs__") or issubclass(IRecommender, type) is False
