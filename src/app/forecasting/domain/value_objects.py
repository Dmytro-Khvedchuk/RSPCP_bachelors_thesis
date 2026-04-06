"""Forecasting domain value objects — model configs and prediction containers."""

from __future__ import annotations

from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import model_validator


# ---------------------------------------------------------------------------
# Prediction containers
# ---------------------------------------------------------------------------


class PointPrediction(BaseModel, frozen=True):
    """Point estimate with optional uncertainty from a return regressor.

    Attributes:
        mean: Point estimate (predicted return magnitude).
        std: Standard deviation of the prediction (residual std for Ridge,
            MC Dropout std for GRU, median quantile spread for LightGBM).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_samples,)`` — point predictions."""

    std: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_samples,)`` — uncertainty estimates."""

    @model_validator(mode="after")
    def _shapes_match(self) -> Self:
        """Ensure mean and std arrays have the same length.

        Returns:
            Validated instance.

        Raises:
            ValueError: If shapes differ.
        """
        if self.mean.shape != self.std.shape:
            msg: str = f"mean shape {self.mean.shape} != std shape {self.std.shape}"
            raise ValueError(msg)
        return self


class QuantilePrediction(BaseModel, frozen=True):
    """Quantile predictions from a distributional regressor (e.g. LightGBM).

    Attributes:
        quantiles: Sorted tuple of quantile levels (e.g. ``(0.05, 0.25, 0.5, 0.75, 0.95)``).
        values: 2-D array of shape ``(n_samples, n_quantiles)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    quantiles: tuple[float, ...]
    """Sorted quantile levels."""

    values: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    """Shape ``(n_samples, n_quantiles)`` — predicted quantile values."""

    @model_validator(mode="after")
    def _width_matches_quantiles(self) -> Self:
        """Ensure the number of columns equals the number of quantiles.

        Returns:
            Validated instance.

        Raises:
            ValueError: If column count != len(quantiles).
        """
        if self.values.ndim != 2:  # noqa: PLR2004
            msg: str = f"values must be 2-D, got ndim={self.values.ndim}"
            raise ValueError(msg)
        n_q: int = len(self.quantiles)
        if self.values.shape[1] != n_q:
            msg = f"values has {self.values.shape[1]} columns but {n_q} quantiles specified"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _quantiles_sorted(self) -> Self:
        """Ensure quantiles are strictly ascending.

        Returns:
            Validated instance.

        Raises:
            ValueError: If quantiles are not strictly ascending.
        """
        for i in range(1, len(self.quantiles)):
            if self.quantiles[i] <= self.quantiles[i - 1]:
                msg: str = f"quantiles must be strictly ascending, got {self.quantiles}"
                raise ValueError(msg)
        return self


class VolatilityForecast(BaseModel, frozen=True):
    """Volatility forecast result from a vol forecasting model.

    Attributes:
        predicted_vol: Predicted volatility (sigma, not sigma-squared).
        predicted_var: Predicted variance (sigma-squared).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    predicted_vol: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_samples,)`` — predicted volatility (standard deviation)."""

    predicted_var: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_samples,)`` — predicted variance (sigma-squared)."""

    @model_validator(mode="after")
    def _shapes_match(self) -> Self:
        """Ensure vol and var arrays have the same length.

        Returns:
            Validated instance.

        Raises:
            ValueError: If shapes differ.
        """
        if self.predicted_vol.shape != self.predicted_var.shape:
            msg: str = (
                f"predicted_vol shape {self.predicted_vol.shape} != predicted_var shape {self.predicted_var.shape}"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Model configs — Return Regressors
# ---------------------------------------------------------------------------


class RidgeConfig(BaseModel, frozen=True):
    """Configuration for the Ridge regression baseline.

    Attributes:
        alpha: L2 regularisation strength.
        use_huber: If ``True``, use Huber loss instead of squared loss.
        huber_epsilon: Huber loss transition point (only used when ``use_huber=True``).
        random_seed: Seed for reproducibility.
    """

    alpha: Annotated[
        float,
        PydanticField(default=1.0, gt=0, description="L2 regularisation strength"),
    ]

    use_huber: bool = False
    """If True, use HuberRegressor instead of Ridge."""

    huber_epsilon: Annotated[
        float,
        PydanticField(
            default=1.35,
            gt=1.0,
            description="Huber loss transition point (must be > 1.0)",
        ),
    ]

    random_seed: int = 42
    """Seed for reproducibility."""


class GradientBoostingConfig(BaseModel, frozen=True):
    """Configuration for LightGBM quantile regressor.

    Attributes:
        quantiles: Quantile levels to predict.
        n_estimators: Number of boosting rounds.
        learning_rate: Step size shrinkage.
        max_depth: Maximum tree depth (-1 for no limit).
        min_child_samples: Minimum samples in a leaf.
        reg_alpha: L1 regularisation.
        reg_lambda: L2 regularisation.
        subsample: Row subsampling ratio.
        colsample_bytree: Column subsampling ratio.
        apply_isotonic: Whether to apply isotonic regression for quantile monotonicity.
        random_seed: Seed for reproducibility.
    """

    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    """Quantile levels to predict."""

    n_estimators: Annotated[
        int,
        PydanticField(default=500, ge=10, description="Number of boosting rounds"),
    ]

    learning_rate: Annotated[
        float,
        PydanticField(default=0.05, gt=0, le=1, description="Step size shrinkage"),
    ]

    max_depth: int = 6
    """Maximum tree depth (-1 for no limit)."""

    min_child_samples: Annotated[
        int,
        PydanticField(default=20, ge=1, description="Minimum samples in a leaf"),
    ]

    reg_alpha: Annotated[
        float,
        PydanticField(default=0.0, ge=0, description="L1 regularisation"),
    ]

    reg_lambda: Annotated[
        float,
        PydanticField(default=1.0, ge=0, description="L2 regularisation"),
    ]

    subsample: Annotated[
        float,
        PydanticField(default=0.8, gt=0, le=1, description="Row subsampling ratio"),
    ]

    colsample_bytree: Annotated[
        float,
        PydanticField(default=0.8, gt=0, le=1, description="Column subsampling ratio"),
    ]

    apply_isotonic: bool = True
    """Whether to apply isotonic regression for quantile monotonicity correction."""

    random_seed: int = 42
    """Seed for reproducibility."""

    @model_validator(mode="after")
    def _quantiles_valid(self) -> Self:
        """Ensure quantiles are strictly ascending and within (0, 1).

        Returns:
            Validated instance.

        Raises:
            ValueError: If quantiles are invalid.
        """
        for q in self.quantiles:
            if not 0 < q < 1:
                msg: str = f"All quantiles must be in (0, 1), got {q}"
                raise ValueError(msg)
        for i in range(1, len(self.quantiles)):
            if self.quantiles[i] <= self.quantiles[i - 1]:
                msg = f"quantiles must be strictly ascending, got {self.quantiles}"
                raise ValueError(msg)
        return self


class GRUConfig(BaseModel, frozen=True):
    """Configuration for the GRU regressor with MC Dropout.

    Attributes:
        hidden_size: GRU hidden state dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability (used for MC Dropout at inference).
        sequence_length: Number of time steps in the input sequence.
        learning_rate: Optimiser learning rate.
        n_epochs: Maximum training epochs.
        batch_size: Training mini-batch size.
        mc_samples: Number of forward passes for MC Dropout uncertainty.
        patience: Early stopping patience (epochs without improvement).
        random_seed: Seed for reproducibility.
    """

    hidden_size: Annotated[
        int,
        PydanticField(default=64, ge=8, description="GRU hidden state dimension"),
    ]

    num_layers: Annotated[
        int,
        PydanticField(default=2, ge=1, description="Number of stacked GRU layers"),
    ]

    dropout: Annotated[
        float,
        PydanticField(default=0.2, ge=0, lt=1, description="Dropout probability for MC Dropout"),
    ]

    sequence_length: Annotated[
        int,
        PydanticField(default=20, ge=2, description="Number of time steps in input sequence"),
    ]

    learning_rate: Annotated[
        float,
        PydanticField(default=1e-3, gt=0, description="Optimiser learning rate"),
    ]

    n_epochs: Annotated[
        int,
        PydanticField(default=100, ge=1, description="Maximum training epochs"),
    ]

    batch_size: Annotated[
        int,
        PydanticField(default=32, ge=1, description="Training mini-batch size"),
    ]

    mc_samples: Annotated[
        int,
        PydanticField(default=50, ge=2, description="MC Dropout forward passes at inference"),
    ]

    patience: Annotated[
        int,
        PydanticField(default=10, ge=1, description="Early stopping patience"),
    ]

    random_seed: int = 42
    """Seed for reproducibility."""


# ---------------------------------------------------------------------------
# Model configs — Volatility Forecasters
# ---------------------------------------------------------------------------


class HARRVConfig(BaseModel, frozen=True):
    """Configuration for the HAR-RV model (Corsi 2009).

    The HAR-RV decomposes realized volatility into three additive
    components at daily, weekly, and monthly horizons.

    Attributes:
        daily_lag: Number of bars for the daily RV component.
        weekly_lag: Number of bars for the weekly RV component.
        monthly_lag: Number of bars for the monthly RV component.
        fit_intercept: Whether to include a constant in OLS.
    """

    daily_lag: Annotated[
        int,
        PydanticField(default=1, ge=1, description="Daily RV lag (bars)"),
    ]

    weekly_lag: Annotated[
        int,
        PydanticField(default=5, ge=2, description="Weekly RV lag (bars)"),
    ]

    monthly_lag: Annotated[
        int,
        PydanticField(default=22, ge=5, description="Monthly RV lag (bars)"),
    ]

    fit_intercept: bool = True
    """Whether to include a constant in the OLS regression."""


class GARCHConfig(BaseModel, frozen=True):
    """Configuration for the GARCH(1,1) baseline.

    GARCH assumes equally-spaced observations, so this model should only
    be used with ``time_1h`` bars.

    Attributes:
        p: GARCH lag order for conditional variance.
        q: ARCH lag order for squared residuals.
        mean_model: Mean model specification (``"AR"`` or ``"Zero"`` or ``"Constant"``).
        ar_order: AR order when ``mean_model="AR"``.
        dist: Error distribution (``"normal"``, ``"t"``, ``"skewt"``).
        rescale: Whether to rescale returns for numerical stability.
    """

    p: Annotated[
        int,
        PydanticField(default=1, ge=1, description="GARCH lag order"),
    ]

    q: Annotated[
        int,
        PydanticField(default=1, ge=1, description="ARCH lag order"),
    ]

    mean_model: str = "Constant"
    """Mean model specification: 'AR', 'Zero', or 'Constant'."""

    ar_order: Annotated[
        int,
        PydanticField(default=1, ge=0, description="AR order when mean_model='AR'"),
    ]

    dist: str = "t"
    """Error distribution: 'normal', 't', or 'skewt' (Student-t is default per issue)."""

    rescale: bool = True
    """Whether to rescale returns for numerical stability."""
