"""HAR-RV volatility forecaster (Corsi 2009) -- OLS on multi-horizon realized volatility."""

from __future__ import annotations

from typing import Final

import numpy as np
from loguru import logger

from src.app.forecasting.domain.value_objects import HARRVConfig, VolatilityForecast

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: Final[float] = 1e-12
"""Floor for variance predictions to avoid negative/zero values."""

_HAR_N_REGRESSORS: Final[int] = 3
"""Number of HAR regressors: daily, weekly, monthly RV."""


class HARRVForecaster:
    r"""Heterogeneous Autoregressive model of Realized Volatility (Corsi 2009).

    The HAR-RV decomposes realized volatility into three additive components
    at daily, weekly, and monthly horizons, then fits a simple OLS regression:

    .. math::

        RV_{t+1} = \beta_0 + \beta_1 \cdot RV^{(d)}_t
                  + \beta_2 \cdot RV^{(w)}_t
                  + \beta_3 \cdot RV^{(m)}_t + \varepsilon_t

    where:
        - :math:`RV^{(d)}_t` is the daily (1-bar) realized volatility,
        - :math:`RV^{(w)}_t` is the rolling mean RV over ``weekly_lag`` bars,
        - :math:`RV^{(m)}_t` is the rolling mean RV over ``monthly_lag`` bars.

    The model is intentionally simple (OLS, 3 features) and serves as the
    primary volatility baseline for alternative bars (dollar, volume, imbalance)
    where GARCH cannot be applied due to irregular spacing.

    Attributes:
        config: Frozen HAR-RV configuration.
    """

    def __init__(self, config: HARRVConfig) -> None:
        """Initialise the HAR-RV forecaster.

        Args:
            config: Model configuration specifying lag horizons and intercept.
        """
        self.config: HARRVConfig = config
        self._coefficients: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None
        self._last_rv_series: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def construct_har_features(
        rv_series: np.ndarray[tuple[int], np.dtype[np.float64]],
        config: HARRVConfig,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Build the 3-column HAR regressor matrix from a raw RV series.

        Columns are: [daily_rv, weekly_rv, monthly_rv] where weekly and
        monthly components are rolling means over the respective lag windows.

        The output has length ``len(rv_series) - monthly_lag + 1`` because
        the first ``monthly_lag - 1`` rows lack a full monthly window.

        Args:
            rv_series: 1-D array of realized volatility values.
            config: HAR-RV config with lag specifications.

        Returns:
            Feature matrix of shape ``(n_valid, 3)``.

        Raises:
            ValueError: If the series is too short for the monthly lag.
        """
        n: int = len(rv_series)
        min_length: int = config.monthly_lag + 1
        if n < min_length:
            msg: str = (
                f"rv_series length {n} is too short for monthly_lag={config.monthly_lag}; "
                f"need at least {min_length} observations"
            )
            raise ValueError(msg)

        # Compute rolling means via cumulative sum for efficiency
        # Rolling mean at index t over window w = mean(rv[t-w+1 : t+1])
        # First valid index for window w is w-1
        cumsum: np.ndarray[tuple[int], np.dtype[np.float64]] = np.concatenate(
            [np.array([0.0], dtype=np.float64), np.cumsum(rv_series)]
        )

        warmup: int = config.monthly_lag - 1
        valid_length: int = n - warmup

        # Daily RV: rolling mean over daily_lag bars (typically just rv[t])
        d: int = config.daily_lag
        daily_rv: np.ndarray[tuple[int], np.dtype[np.float64]] = (
            cumsum[warmup + 1 : n + 1] - cumsum[warmup + 1 - d : n + 1 - d]
        ) / d

        # Weekly RV: rolling mean over weekly_lag bars
        w: int = config.weekly_lag
        weekly_rv: np.ndarray[tuple[int], np.dtype[np.float64]] = (
            cumsum[warmup + 1 : n + 1] - cumsum[warmup + 1 - w : n + 1 - w]
        ) / w

        # Monthly RV: rolling mean over monthly_lag bars
        m: int = config.monthly_lag
        monthly_rv: np.ndarray[tuple[int], np.dtype[np.float64]] = (
            cumsum[warmup + 1 : n + 1] - cumsum[warmup + 1 - m : n + 1 - m]
        ) / m

        features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.column_stack(
            [daily_rv, weekly_rv, monthly_rv]
        )

        assert features.shape == (valid_length, _HAR_N_REGRESSORS)  # noqa: S101
        return features

    # ------------------------------------------------------------------
    # IVolatilityForecaster interface
    # ------------------------------------------------------------------

    def fit(
        self,
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,  # noqa: ARG002
    ) -> None:
        """Train the HAR-RV model via OLS on realized volatility.

        ``y_train`` is the realized volatility series (NOT returns). The
        method constructs the three HAR regressors internally, shifts the
        target by 1 to predict next-period RV, and fits OLS.

        Args:
            y_train: Realized volatility series of shape ``(n_samples,)``.
            x_train: Ignored for HAR-RV (regressors are built from ``y_train``).

        Raises:
            ValueError: If the series is too short for the monthly warmup.
        """
        n: int = len(y_train)
        warmup: int = self.config.monthly_lag
        min_observations: int = warmup + 2  # warmup + at least 2 rows for OLS
        if n < min_observations:
            msg: str = (
                f"y_train length {n} is too short; need at least {min_observations} "
                f"observations (monthly_lag={self.config.monthly_lag} + 2)"
            )
            raise ValueError(msg)

        # Build HAR features from the full RV series
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = self.construct_har_features(y_train, self.config)

        # Target is next-period RV: shift by 1
        # features[i] corresponds to rv_series[monthly_lag - 1 + i]
        # target[i] = rv_series[monthly_lag + i]
        x_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]] = features[:-1]
        y_target: np.ndarray[tuple[int], np.dtype[np.float64]] = y_train[warmup:]

        # Verify alignment: x_matrix rows should match y_target length
        assert x_matrix.shape[0] == y_target.shape[0], (  # noqa: S101
            f"Shape mismatch: X={x_matrix.shape[0]}, y={y_target.shape[0]}"
        )

        # Add intercept column if configured
        if self.config.fit_intercept:
            ones: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((x_matrix.shape[0], 1), dtype=np.float64)
            x_matrix = np.hstack([ones, x_matrix])

        # OLS via numpy least squares
        coefficients: np.ndarray[tuple[int], np.dtype[np.float64]]
        residuals: np.ndarray[tuple[int], np.dtype[np.float64]]
        coefficients, residuals, _, _ = np.linalg.lstsq(x_matrix, y_target, rcond=None)

        self._coefficients = coefficients
        self._last_rv_series = y_train.copy()
        self._is_fitted = True

        n_samples: int = x_matrix.shape[0]
        r_squared: float = _compute_r_squared(y_target, x_matrix @ coefficients)

        logger.info(
            "HAR-RV fitted | samples={} | warmup={} | R²={:.4f} | coefficients={}",
            n_samples,
            warmup,
            r_squared,
            np.round(coefficients, 6).tolist(),
        )

    def predict(
        self,
        n_steps: int,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
    ) -> VolatilityForecast:
        """Forecast volatility for the next ``n_steps`` periods.

        Two modes of operation:

        1. **With ``x_test``**: pre-constructed HAR feature matrix (3 columns:
           daily, weekly, monthly RV). Each row produces one forecast.
           ``n_steps`` must equal ``x_test.shape[0]``.

        2. **Without ``x_test``**: iterative 1-step-ahead forecasting using
           the last training RV values, feeding each prediction back as the
           next daily RV component.

        Args:
            n_steps: Number of periods to forecast.
            x_test: Optional pre-constructed HAR feature matrix of shape
                ``(n_steps, 3)``.

        Returns:
            Volatility forecast with predicted vol (sigma) and var (sigma^2).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted or self._coefficients is None or self._last_rv_series is None:
            msg: str = "Model must be fitted before prediction"
            raise RuntimeError(msg)

        if x_test is not None:
            return self._predict_with_features(x_test, n_steps)
        return self._predict_iterative(n_steps)

    # ------------------------------------------------------------------
    # Private prediction helpers
    # ------------------------------------------------------------------

    def _predict_with_features(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        n_steps: int,
    ) -> VolatilityForecast:
        """Predict using a pre-constructed HAR feature matrix.

        Args:
            x_test: Feature matrix of shape ``(n_steps, 3)``.
            n_steps: Expected number of forecast steps.

        Returns:
            Volatility forecast.

        Raises:
            ValueError: If ``x_test`` has unexpected shape.
        """
        if x_test.ndim != 2 or x_test.shape[1] != _HAR_N_REGRESSORS:  # noqa: PLR2004
            msg: str = f"x_test must have shape (n, {_HAR_N_REGRESSORS}), got {x_test.shape}"
            raise ValueError(msg)
        if x_test.shape[0] != n_steps:
            msg = f"x_test has {x_test.shape[0]} rows but n_steps={n_steps}"
            raise ValueError(msg)

        # Safe to assert: caller (predict) already verified _is_fitted
        assert self._coefficients is not None  # noqa: S101
        coefficients: np.ndarray[tuple[int], np.dtype[np.float64]] = self._coefficients
        x_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x_test

        if self.config.fit_intercept:
            ones: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((x_matrix.shape[0], 1), dtype=np.float64)
            x_matrix = np.hstack([ones, x_matrix])

        raw_predictions: np.ndarray[tuple[int], np.dtype[np.float64]] = x_matrix @ coefficients

        return _build_forecast(raw_predictions)

    def _predict_iterative(self, n_steps: int) -> VolatilityForecast:
        """Iterative 1-step-ahead forecasting using stored training history.

        For each step, construct HAR features from the current RV history,
        produce a 1-step forecast, append the prediction, and repeat.

        Args:
            n_steps: Number of steps to forecast iteratively.

        Returns:
            Volatility forecast.
        """
        # Safe to assert: caller (predict) already verified _is_fitted
        assert self._coefficients is not None  # noqa: S101
        assert self._last_rv_series is not None  # noqa: S101
        coefficients: np.ndarray[tuple[int], np.dtype[np.float64]] = self._coefficients
        rv_history: list[float] = self._last_rv_series.tolist()
        predictions: list[float] = []

        for _ in range(n_steps):
            rv_array: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(rv_history, dtype=np.float64)
            n_hist: int = len(rv_array)

            # Daily RV: last value
            daily: float = float(rv_array[-1])

            # Weekly RV: mean of last weekly_lag values (or all if less)
            weekly_start: int = max(0, n_hist - self.config.weekly_lag)
            weekly: float = float(np.mean(rv_array[weekly_start:]))

            # Monthly RV: mean of last monthly_lag values (or all if less)
            monthly_start: int = max(0, n_hist - self.config.monthly_lag)
            monthly: float = float(np.mean(rv_array[monthly_start:]))

            # Construct feature vector
            x_row: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([daily, weekly, monthly], dtype=np.float64)

            if self.config.fit_intercept:
                x_row = np.concatenate([[1.0], x_row])

            pred: float = float(x_row @ coefficients)
            predictions.append(pred)

            # Feed prediction back into history for next step
            rv_history.append(max(pred, _EPS))

        raw_predictions: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(predictions, dtype=np.float64)
        return _build_forecast(raw_predictions)


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _build_forecast(
    raw_predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> VolatilityForecast:
    """Convert raw RV predictions to a ``VolatilityForecast``.

    HAR-RV predicts realized variance (or a variance proxy), so:
        - ``predicted_var = abs(prediction)`` (clip negative to epsilon)
        - ``predicted_vol = sqrt(predicted_var)``

    Args:
        raw_predictions: Raw OLS predictions, shape ``(n,)``.

    Returns:
        Volatility forecast with non-negative vol and var.
    """
    predicted_var: np.ndarray[tuple[int], np.dtype[np.float64]] = np.maximum(raw_predictions, _EPS)
    predicted_vol: np.ndarray[tuple[int], np.dtype[np.float64]] = np.sqrt(predicted_var)

    return VolatilityForecast(
        predicted_vol=predicted_vol,
        predicted_var=predicted_var,
    )


def _compute_r_squared(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    """Compute the coefficient of determination (R-squared).

    Args:
        y_true: Actual target values.
        y_pred: Predicted values.

    Returns:
        R-squared value (can be negative for poor fits).
    """
    ss_res: float = float(np.sum((y_true - y_pred) ** 2))
    ss_tot: float = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < _EPS:
        return 0.0
    r_squared: float = 1.0 - ss_res / ss_tot
    return r_squared
