"""GARCH volatility forecaster -- conditional variance via the arch library."""

from __future__ import annotations

from typing import Final

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from loguru import logger

from src.app.forecasting.domain.value_objects import GARCHConfig, VolatilityForecast

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: Final[float] = 1e-12
"""Floor for variance predictions to avoid negative/zero values."""

_RESCALE_FACTOR: Final[float] = 100.0
"""Default manual rescale multiplier when config.rescale is True.

The ``arch`` library's built-in ``rescale`` argument automatically chooses a
scale factor based on the data's standard deviation.  However, we disable
the library's ``rescale`` and handle it ourselves so we can deterministically
undo the transformation.  Multiplying returns by 100 (i.e. converting to
percentage returns) is the standard convention in the GARCH literature.
"""


class ARIMAGARCHForecaster:
    """GARCH(p,q) volatility forecaster using the ``arch`` library.

    This model fits a GARCH(p,q) process with configurable mean model
    (Constant, Zero, or AR) and error distribution (normal, Student-t, or
    skewed-t) to a **return series**.  The conditional variance from the
    fitted model is used as the volatility forecast.

    .. warning::

        GARCH assumes equally-spaced observations.  This forecaster should
        **only** be used with ``time_1h`` bars.  For irregularly-spaced
        alternative bars (dollar, volume, imbalance), use :class:`HARRVForecaster`
        instead.

    Attributes:
        config: Frozen GARCH configuration.
    """

    def __init__(self, config: GARCHConfig) -> None:
        """Initialise the GARCH forecaster.

        Args:
            config: Model configuration specifying GARCH orders, mean model,
                and error distribution.
        """
        self.config: GARCHConfig = config
        self._result: ARCHModelResult | None = None
        self._scale: float = 1.0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # IVolatilityForecaster interface
    # ------------------------------------------------------------------

    def fit(
        self,
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,  # noqa: ARG002
    ) -> None:
        """Fit the GARCH model to a return series.

        ``y_train`` must be a **return series** (not realized volatility).
        GARCH models returns directly and extracts conditional variance
        as the volatility estimate.

        Args:
            y_train: Return series of shape ``(n_samples,)``.
            x_train: Ignored for GARCH (no exogenous regressors).

        Raises:
            ValueError: If the return series is too short.
        """
        n: int = len(y_train)
        min_observations: int = max(self.config.p, self.config.q) + 10
        if n < min_observations:
            msg: str = (
                f"y_train length {n} is too short for GARCH({self.config.p},{self.config.q}); "
                f"need at least {min_observations} observations"
            )
            raise ValueError(msg)

        # Manual rescale: convert to percentage returns for numerical stability
        scaled_returns: np.ndarray[tuple[int], np.dtype[np.float64]]
        if self.config.rescale:
            self._scale = _RESCALE_FACTOR
            scaled_returns = y_train * self._scale
        else:
            self._scale = 1.0
            scaled_returns = y_train.copy()

        # Build arch model -- disable library's rescale since we handle it
        lags: int | None = self.config.ar_order if self.config.mean_model == "AR" else None
        am = arch_model(
            y=scaled_returns,
            mean=self.config.mean_model,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            vol="GARCH",
            p=self.config.p,
            q=self.config.q,
            dist=self.config.dist,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            lags=lags,
            rescale=False,
        )
        result: ARCHModelResult = am.fit(disp="off")

        self._result = result
        self._is_fitted = True

        # Log summary statistics
        log_likelihood: float = float(result.loglikelihood)
        aic: float = float(result.aic)
        bic: float = float(result.bic)
        n_params: int = len(result.params)

        logger.info(
            "GARCH({},{}) fitted | dist={} | mean={} | n={} | scale={} | "
            "LL={:.2f} | AIC={:.2f} | BIC={:.2f} | params={}",
            self.config.p,
            self.config.q,
            self.config.dist,
            self.config.mean_model,
            n,
            self._scale,
            log_likelihood,
            aic,
            bic,
            n_params,
        )

    def predict(
        self,
        n_steps: int,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,  # noqa: ARG002
    ) -> VolatilityForecast:
        """Forecast conditional variance for the next ``n_steps`` periods.

        Uses the ``arch`` library's analytical multi-step variance forecast.
        The forecasts are produced from the last in-sample observation and
        represent the conditional variance path.

        Args:
            n_steps: Number of periods to forecast (horizon).
            x_test: Ignored for GARCH (no exogenous regressors).

        Returns:
            Volatility forecast with predicted vol (sigma) and var (sigma^2)
            in the **original** return scale (i.e. rescaling is undone).

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``n_steps`` is not positive.
        """
        if not self._is_fitted or self._result is None:
            msg: str = "Model must be fitted before prediction"
            raise RuntimeError(msg)
        if n_steps < 1:
            msg = f"n_steps must be >= 1, got {n_steps}"
            raise ValueError(msg)

        forecast = self._result.forecast(horizon=n_steps)

        # Extract conditional variance: shape (1, n_steps) from last row
        # The forecast DataFrame has horizon columns h.1, h.2, ..., h.n_steps
        variance_df = forecast.variance
        scaled_variance: np.ndarray[tuple[int], np.dtype[np.float64]] = variance_df.values[-1].astype(np.float64)

        # Undo rescale: variance scales quadratically
        predicted_var: np.ndarray[tuple[int], np.dtype[np.float64]] = scaled_variance / (self._scale**2)

        # Clip to epsilon to prevent negative/zero variance
        predicted_var = np.maximum(predicted_var, _EPS)

        predicted_vol: np.ndarray[tuple[int], np.dtype[np.float64]] = np.sqrt(predicted_var)

        return VolatilityForecast(
            predicted_vol=predicted_vol,
            predicted_var=predicted_var,
        )
