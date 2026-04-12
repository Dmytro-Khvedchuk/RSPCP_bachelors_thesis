"""Recommendation domain protocols — structural interfaces for recommender implementations."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.app.recommendation.domain.value_objects import Recommendation


class IRecommender(Protocol):
    """Structural interface for recommender implementations.

    A recommender learns to predict expected strategy returns from assembled
    feature vectors (produced by :class:`RecommenderFeatureBuilder`) and
    realised strategy-return labels (produced by :class:`LabelBuilder`).

    The ``fit`` / ``predict`` contract mirrors the forecasting module's
    protocols, using NumPy arrays for consistency.
    """

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train on feature matrix and realised strategy returns.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Realised strategy returns of shape ``(n_samples,)``.
        """
        ...

    def predict(
        self,
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[Recommendation]:
        """Predict recommendations for input features.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of recommendations, one per sample.
        """
        ...
