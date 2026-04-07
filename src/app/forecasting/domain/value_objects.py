"""Forecasting domain value objects — model configs, prediction containers, and calibration results."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import model_validator


# ---------------------------------------------------------------------------
# Forecast horizon & direction enums
# ---------------------------------------------------------------------------


class ForecastHorizon(StrEnum):
    """Forecast horizon for direction and return predictions.

    Defines the look-ahead period in bars for which a forecast is made.

    Attributes:
        H1: 1-bar horizon.
        H4: 4-bar horizon.
        H24: 24-bar horizon.
    """

    H1 = "h1"
    H4 = "h4"
    H24 = "h24"


# ---------------------------------------------------------------------------
# Classification value objects
# ---------------------------------------------------------------------------


class DirectionForecast(BaseModel, frozen=True):
    """Predicted direction with confidence from a classification model.

    Attributes:
        predicted_direction: Predicted direction: +1 (long) or -1 (short).
        confidence: Predicted probability for the chosen direction, in [0, 1].
        horizon: Forecast horizon for which this prediction was made.
    """

    predicted_direction: Annotated[
        int,
        PydanticField(description="Predicted direction: +1 (long) or -1 (short)"),
    ]
    """Predicted direction: +1 (long) or -1 (short)."""

    confidence: Annotated[
        float,
        PydanticField(ge=0.0, le=1.0, description="Probability for the predicted direction"),
    ]
    """Predicted probability for the chosen direction, in [0, 1]."""

    horizon: ForecastHorizon
    """Forecast horizon for which this prediction was made."""

    @model_validator(mode="after")
    def _direction_valid(self) -> Self:
        """Ensure predicted_direction is +1 or -1.

        Returns:
            Validated instance.

        Raises:
            ValueError: If direction is not +1 or -1.
        """
        if self.predicted_direction not in {1, -1}:
            msg: str = f"predicted_direction must be +1 or -1, got {self.predicted_direction}"
            raise ValueError(msg)
        return self


class ReturnForecast(BaseModel, frozen=True):
    """Point estimate of predicted return magnitude with optional uncertainty.

    Attributes:
        predicted_return: Point estimate of the predicted return.
        prediction_std: Standard deviation (uncertainty) of the prediction.
        quantiles: Optional predicted quantile values (e.g. at 5th, 25th, ..., 95th).
        confidence_interval: Optional (lower, upper) bounds for the prediction.
    """

    predicted_return: float
    """Point estimate of the predicted return."""

    prediction_std: Annotated[
        float,
        PydanticField(ge=0.0, description="Standard deviation of the prediction"),
    ]
    """Standard deviation (uncertainty) of the prediction."""

    quantiles: tuple[float, ...] | None = None
    """Optional predicted quantile values."""

    confidence_interval: tuple[float, float] | None = None
    """Optional (lower, upper) bounds for the prediction."""

    @model_validator(mode="after")
    def _ci_bounds_valid(self) -> Self:
        """Ensure confidence interval lower <= upper when provided.

        Returns:
            Validated instance.

        Raises:
            ValueError: If lower > upper in confidence_interval.
        """
        if self.confidence_interval is not None:
            lower: float = self.confidence_interval[0]
            upper: float = self.confidence_interval[1]
            if lower > upper:
                msg: str = f"confidence_interval lower ({lower}) must be <= upper ({upper})"
                raise ValueError(msg)
        return self


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


# ---------------------------------------------------------------------------
# Model configs — Calibration & Conformal Prediction
# ---------------------------------------------------------------------------


class ACIConfig(BaseModel, frozen=True):
    """Configuration for Adaptive Conformal Inference (Gibbs & Candes 2021).

    ACI adapts the miscoverage rate alpha_t online so that prediction
    intervals maintain target coverage even under distribution shift.

    Attributes:
        target_coverage: Desired marginal coverage probability (e.g. 0.90).
        gamma: Step size controlling how fast alpha_t adapts.
        initial_alpha: Starting miscoverage rate.  Defaults to ``1 - target_coverage``.
        min_alpha: Floor for alpha_t to prevent degenerate (infinite-width) intervals.
        max_alpha: Cap for alpha_t to prevent degenerate (zero-width) intervals.
    """

    target_coverage: Annotated[
        float,
        PydanticField(default=0.90, gt=0, lt=1, description="Target marginal coverage"),
    ]

    gamma: Annotated[
        float,
        PydanticField(default=0.005, gt=0, le=1, description="ACI step size"),
    ]

    initial_alpha: float | None = None
    """Starting miscoverage rate.  ``None`` → ``1 - target_coverage``."""

    min_alpha: Annotated[
        float,
        PydanticField(default=0.01, gt=0, lt=1, description="Floor for alpha_t"),
    ]

    max_alpha: Annotated[
        float,
        PydanticField(default=0.50, gt=0, lt=1, description="Cap for alpha_t"),
    ]

    @model_validator(mode="after")
    def _alpha_bounds_valid(self) -> Self:
        """Ensure min_alpha < max_alpha and initial_alpha is within bounds.

        Returns:
            Validated instance.

        Raises:
            ValueError: If bounds are invalid.
        """
        if self.min_alpha >= self.max_alpha:
            msg: str = f"min_alpha ({self.min_alpha}) must be < max_alpha ({self.max_alpha})"
            raise ValueError(msg)
        effective_alpha: float = self.initial_alpha if self.initial_alpha is not None else (1.0 - self.target_coverage)
        if not self.min_alpha <= effective_alpha <= self.max_alpha:
            msg = f"initial_alpha ({effective_alpha}) must be in [{self.min_alpha}, {self.max_alpha}]"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Calibration & conformal prediction result containers
# ---------------------------------------------------------------------------


class ConformalInterval(BaseModel, frozen=True):
    """Prediction interval produced by a conformal predictor.

    Attributes:
        lower: Lower bounds of shape ``(n_samples,)``.
        upper: Upper bounds of shape ``(n_samples,)``.
        coverage: Empirical coverage if actuals were provided during prediction,
            otherwise ``None``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lower: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_samples,)`` — lower interval bounds."""

    upper: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_samples,)`` — upper interval bounds."""

    coverage: float | None = None
    """Empirical coverage (fraction of actuals inside interval), or ``None``."""

    @model_validator(mode="after")
    def _shapes_match(self) -> Self:
        """Ensure lower and upper arrays have the same length.

        Returns:
            Validated instance.

        Raises:
            ValueError: If shapes differ.
        """
        if self.lower.shape != self.upper.shape:
            msg: str = f"lower shape {self.lower.shape} != upper shape {self.upper.shape}"
            raise ValueError(msg)
        return self


class ReliabilityDiagramResult(BaseModel, frozen=True):
    """Reliability diagram data: expected vs observed coverage at each quantile level.

    Attributes:
        expected_coverage: Nominal quantile levels (e.g. ``[0.05, 0.25, 0.50, 0.75, 0.95]``).
        observed_coverage: Actual fraction of observations below each predicted quantile.
        n_samples: Number of test samples used to compute coverage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    expected_coverage: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Nominal quantile levels."""

    observed_coverage: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Actual coverage at each quantile level."""

    n_samples: int
    """Number of test samples."""

    @model_validator(mode="after")
    def _shapes_match(self) -> Self:
        """Ensure expected and observed arrays have the same length.

        Returns:
            Validated instance.

        Raises:
            ValueError: If shapes differ.
        """
        if self.expected_coverage.shape != self.observed_coverage.shape:
            msg: str = (
                f"expected_coverage shape {self.expected_coverage.shape} "
                f"!= observed_coverage shape {self.observed_coverage.shape}"
            )
            raise ValueError(msg)
        return self


class ResidualDiagnostics(BaseModel, frozen=True):
    """Residual diagnostics for assessing conformal prediction assumptions.

    Reports normality (Shapiro-Wilk) and homoscedasticity (Breusch-Pagan)
    of model residuals.

    Attributes:
        shapiro_stat: Shapiro-Wilk W statistic.
        shapiro_pvalue: Shapiro-Wilk p-value.
        breusch_pagan_stat: Breusch-Pagan LM statistic.
        breusch_pagan_pvalue: Breusch-Pagan p-value.
        mean_residual: Mean of the residual vector.
        std_residual: Standard deviation of the residual vector.
        is_normal: ``True`` if Shapiro-Wilk p-value >= 0.05.
        is_homoscedastic: ``True`` if Breusch-Pagan p-value >= 0.05.
    """

    shapiro_stat: float
    """Shapiro-Wilk W statistic."""

    shapiro_pvalue: float
    """Shapiro-Wilk p-value."""

    breusch_pagan_stat: float
    """Breusch-Pagan LM statistic."""

    breusch_pagan_pvalue: float
    """Breusch-Pagan p-value."""

    mean_residual: float
    """Mean of residuals."""

    std_residual: float
    """Standard deviation of residuals."""

    is_normal: bool
    """True if Shapiro-Wilk p-value >= 0.05."""

    is_homoscedastic: bool
    """True if Breusch-Pagan p-value >= 0.05."""


class RegimeCoverage(BaseModel, frozen=True):
    """Per-regime coverage statistics for conformal intervals.

    Splits samples by a volatility threshold (median) and reports
    coverage separately for high-volatility and low-volatility periods.

    Attributes:
        overall_coverage: Fraction of actuals inside the interval across all samples.
        high_vol_coverage: Coverage restricted to high-volatility samples.
        low_vol_coverage: Coverage restricted to low-volatility samples.
        high_vol_count: Number of high-volatility samples.
        low_vol_count: Number of low-volatility samples.
        vol_threshold: The volatility threshold used to split regimes (median).
    """

    overall_coverage: float
    """Fraction of actuals inside the interval (all samples)."""

    high_vol_coverage: float
    """Coverage restricted to high-vol regime."""

    low_vol_coverage: float
    """Coverage restricted to low-vol regime."""

    high_vol_count: int
    """Number of samples in the high-vol regime."""

    low_vol_count: int
    """Number of samples in the low-vol regime."""

    vol_threshold: float
    """Volatility threshold used to split regimes."""
