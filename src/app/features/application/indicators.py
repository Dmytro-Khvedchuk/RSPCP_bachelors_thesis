"""Core technical indicators -- pure functions on Polars DataFrames.

All indicator functions are stateless and operate either as Polars expression
combinators (returning ``pl.Expr``) or as DataFrame-level transforms (when
multi-column access or NumPy ``rolling_map`` callbacks are required).

The ``compute_all_indicators`` orchestrator applies every indicator group
according to the supplied :class:`IndicatorConfig` and returns the enriched
DataFrame with all feature columns appended.

Column naming convention:
    ``{indicator}_{param}`` -- e.g. ``logret_1``, ``rv_24``, ``rsi_14``,
    ``bbpctb_20_2.0``.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import polars as pl

from src.app.features.domain.value_objects import IndicatorConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: Final[float] = 1e-12
"""Epsilon for division-by-zero protection."""

_LN2: Final[float] = math.log(2.0)
"""Natural log of 2, used in Garman-Klass and Parkinson estimators."""

_MIN_RS_REGRESSION_POINTS: Final[int] = 3
"""Minimum number of (log_size, log_RS) pairs needed for a meaningful
Hurst exponent regression.  With fewer points the OLS slope is unreliable."""

_HURST_MIN_BLOCK_SIZE: Final[int] = 10
"""Minimum sub-range block size used in the R/S Hurst analysis.
Blocks smaller than this produce noisy R/S estimates."""


# ===================================================================
# 1. RETURNS
# ===================================================================


def log_return(expr: pl.Expr, periods: int = 1) -> pl.Expr:
    r"""Compute log returns over *periods* bars.

    .. math::

        r_t = \ln\!\left(\frac{P_t}{P_{t - \text{periods}}}\right)

    Args:
        expr: Price expression (typically ``pl.col("close")``).
        periods: Lag period (number of bars).

    Returns:
        Polars expression for the log return series.
    """
    return (expr / expr.shift(periods)).log()


# ===================================================================
# 2. VOLATILITY
# ===================================================================


def realized_vol(logret_expr: pl.Expr, window: int) -> pl.Expr:
    """Compute realized volatility as the rolling standard deviation of log returns.

    Args:
        logret_expr: Log-return expression (1-period returns).
        window: Rolling window size in bars.

    Returns:
        Polars expression for realized volatility.
    """
    return logret_expr.rolling_std(window_size=window, min_samples=window)


def garman_klass_vol(
    window: int,
    *,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
) -> pl.Expr:
    r"""Compute Garman-Klass volatility estimator.

    .. math::

        \sigma^2_{\text{GK}} = \frac{1}{N}\sum_{i=1}^{N}
        \left[
            \frac{1}{2}\ln^2\!\left(\frac{H_i}{L_i}\right)
            - (2\ln 2 - 1)\ln^2\!\left(\frac{C_i}{O_i}\right)
        \right]

    The function returns the square root of the rolling mean of the
    per-bar GK variance term.

    Reference:
        Garman & Klass (1980), "On the Estimation of Security Price
        Volatilities from Historical Data".

    Args:
        window: Rolling window size for the mean.
        high_col: Name of the high-price column.
        low_col: Name of the low-price column.
        open_col: Name of the open-price column.
        close_col: Name of the close-price column.

    Returns:
        Polars expression for Garman-Klass volatility.
    """
    hl_term: pl.Expr = (pl.col(high_col) / pl.col(low_col)).log().pow(2) * 0.5
    co_term: pl.Expr = (pl.col(close_col) / pl.col(open_col)).log().pow(2) * (2.0 * _LN2 - 1.0)
    gk_var: pl.Expr = hl_term - co_term
    return gk_var.rolling_mean(window_size=window, min_samples=window).sqrt()


def parkinson_vol(
    window: int,
    *,
    high_col: str = "high",
    low_col: str = "low",
) -> pl.Expr:
    r"""Compute Parkinson volatility estimator.

    .. math::

        \sigma^2_{\text{P}} = \frac{1}{4 N \ln 2}
        \sum_{i=1}^{N} \ln^2\!\left(\frac{H_i}{L_i}\right)

    The function returns the square root of the rolling mean of the
    per-bar Parkinson variance term.

    Reference:
        Parkinson (1980), "The Extreme Value Method for Estimating
        the Variance of the Rate of Return".

    Args:
        window: Rolling window size for the mean.
        high_col: Name of the high-price column.
        low_col: Name of the low-price column.

    Returns:
        Polars expression for Parkinson volatility.
    """
    hl_log_sq: pl.Expr = (pl.col(high_col) / pl.col(low_col)).log().pow(2)
    park_var: pl.Expr = hl_log_sq / (4.0 * _LN2)
    return park_var.rolling_mean(window_size=window, min_samples=window).sqrt()


def true_range(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
) -> pl.Expr:
    r"""Compute True Range (TR).

    .. math::

        TR_t = \max\bigl(H_t - L_t,\;
        |H_t - C_{t-1}|,\;|L_t - C_{t-1}|\bigr)

    Args:
        high: High price expression.
        low: Low price expression.
        close: Close price expression.

    Returns:
        Polars expression for True Range.
    """
    prev_close: pl.Expr = close.shift(1)
    tr1: pl.Expr = (high - low).abs()
    tr2: pl.Expr = (high - prev_close).abs()
    tr3: pl.Expr = (low - prev_close).abs()
    return pl.max_horizontal(tr1, tr2, tr3)


def atr(
    period: int = 14,
    *,
    wilder: bool = True,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.Expr:
    r"""Compute Average True Range (ATR).

    Uses Wilder's smoothing by default
    ($\alpha = 1 / \text{period}$).  Falls back to simple rolling
    mean when ``wilder=False``.

    Args:
        period: ATR lookback period.
        wilder: Whether to use Wilder's exponential smoothing.
        high_col: High price column name.
        low_col: Low price column name.
        close_col: Close price column name.

    Returns:
        Polars expression for ATR.
    """
    tr_expr: pl.Expr = true_range(pl.col(high_col), pl.col(low_col), pl.col(close_col))
    if wilder:
        alpha: float = 1.0 / period
        return tr_expr.ewm_mean(alpha=alpha, adjust=False)
    return tr_expr.rolling_mean(window_size=period, min_samples=period)


# ===================================================================
# 3. MOMENTUM
# ===================================================================


def ema(expr: pl.Expr, span: int) -> pl.Expr:
    r"""Compute Exponential Moving Average (EMA).

    Uses ``adjust=False`` (streaming-friendly) semantics:

    .. math::

        \text{EMA}_t = \alpha\, x_t + (1 - \alpha)\,\text{EMA}_{t-1},
        \quad \alpha = \frac{2}{\text{span} + 1}

    Args:
        expr: Polars expression (e.g. ``pl.col("close")``).
        span: EMA span controlling the decay factor.

    Returns:
        Polars expression for the EMA.
    """
    alpha: float = 2.0 / (span + 1.0)
    return expr.ewm_mean(alpha=alpha, adjust=False)


def ema_crossover(
    fast_span: int,
    slow_span: int,
    atr_period: int = 14,
    *,
    atr_wilder: bool = True,
    close_col: str = "close",
) -> pl.Expr:
    r"""Compute ATR-normalised EMA crossover signal.

    .. math::

        \text{crossover}_t =
        \frac{\text{EMA}_{\text{fast}} - \text{EMA}_{\text{slow}}}
             {\text{ATR}_t + \varepsilon}

    Normalising by ATR makes the signal comparable across assets and
    volatility regimes.

    Args:
        fast_span: Fast EMA span.
        slow_span: Slow EMA span.
        atr_period: ATR period for normalisation.
        atr_wilder: Whether to use Wilder's ATR smoothing.
        close_col: Close price column name.

    Returns:
        Polars expression for the normalised EMA crossover.
    """
    close_expr: pl.Expr = pl.col(close_col)
    ema_fast: pl.Expr = ema(close_expr, fast_span)
    ema_slow: pl.Expr = ema(close_expr, slow_span)
    atr_expr: pl.Expr = atr(atr_period, wilder=atr_wilder)
    return (ema_fast - ema_slow) / (atr_expr + _EPS)


def rsi(
    period: int = 14,
    *,
    close_col: str = "close",
) -> pl.Expr:
    r"""Compute Wilder's Relative Strength Index (RSI).

    $$RSI = 100 - \frac{100}{1 + RS},\quad
    RS = \frac{\text{EWM}(\text{gains},\alpha=1/p)}
               {\text{EWM}(\text{losses},\alpha=1/p) + \varepsilon}$$

    Uses Wilder's smoothing ($\alpha = 1 / \text{period}$) for
    consistency with the standard RSI definition.

    Args:
        period: RSI lookback period.
        close_col: Close price column name.

    Returns:
        Polars expression for RSI (continuous, 0-100 scale).
    """
    delta: pl.Expr = pl.col(close_col).diff()
    gain: pl.Expr = delta.clip(lower_bound=0.0)
    loss: pl.Expr = (-delta).clip(lower_bound=0.0)
    alpha: float = 1.0 / period
    avg_gain: pl.Expr = gain.ewm_mean(alpha=alpha, adjust=False)
    avg_loss: pl.Expr = loss.ewm_mean(alpha=alpha, adjust=False)
    rs: pl.Expr = avg_gain / (avg_loss + _EPS)
    return pl.lit(100.0) - pl.lit(100.0) / (pl.lit(1.0) + rs)


def roc(expr: pl.Expr, period: int) -> pl.Expr:
    r"""Compute Rate of Change (ROC).

    .. math::

        ROC_t = \frac{P_t}{P_{t - n}} - 1

    Args:
        expr: Price expression.
        period: Lookback period in bars.

    Returns:
        Polars expression for Rate of Change.
    """
    return expr / expr.shift(period) - pl.lit(1.0)


def rolling_slope(expr: pl.Expr, window: int) -> pl.Expr:
    r"""Compute rolling ordinary-least-squares slope.

    Fits $y = a x + b$ where $x = [0, 1, \ldots, w-1]$
    inside each rolling window and returns the slope coefficient *a*.

    Note:
        Uses ``rolling_map`` with a NumPy callback because Polars does
        not natively support rolling linear regression.  Window sizes in
        this project are small (<= 50), so the Python callback overhead
        is acceptable.

    Args:
        expr: Value expression (e.g. ``pl.col("close")``).
        window: Rolling window size.

    Returns:
        Polars expression for the slope at each bar.
    """
    x: np.ndarray[tuple[int], np.dtype[np.float64]] = np.arange(window, dtype=np.float64)

    def _slope(values: pl.Series) -> float:
        arr: np.ndarray[tuple[int], np.dtype[np.float64]] = values.to_numpy()
        if np.isnan(arr).any():
            return float("nan")
        coeffs: np.ndarray[tuple[int], np.dtype[np.float64]] = np.polyfit(x, arr, 1)
        return float(coeffs[0])

    return expr.rolling_map(function=_slope, window_size=window, min_samples=window)


# ===================================================================
# 4. VOLUME
# ===================================================================


def volume_zscore(
    window: int,
    *,
    volume_col: str = "volume",
) -> pl.Expr:
    r"""Compute rolling z-score of volume.

    .. math::

        z_t = \frac{V_t - \bar{V}_{\text{window}}}
                    {\sigma_{V,\text{window}} + \varepsilon}

    Args:
        window: Rolling window size.
        volume_col: Volume column name.

    Returns:
        Polars expression for the volume z-score.
    """
    return zscore_rolling(pl.col(volume_col), window)


def obv_slope(
    window: int,
    *,
    close_col: str = "close",
    volume_col: str = "volume",
) -> pl.Expr:
    r"""Compute slope of On-Balance Volume (OBV) over a rolling window.

    OBV is a cumulative indicator:

    $$OBV_t = OBV_{t-1} + V_t \cdot \text{sign}(\Delta C_t)$$

    where $\Delta C_t = C_t - C_{t-1}$.  The function computes
    OBV as a cumulative sum, then fits a rolling linear regression slope
    to capture the OBV trend direction and magnitude.

    Args:
        window: Rolling window for the slope computation.
        close_col: Close price column name.
        volume_col: Volume column name.

    Returns:
        Polars expression for the OBV slope.
    """
    signed_volume: pl.Expr = pl.col(volume_col) * pl.col(close_col).diff().sign()
    obv_expr: pl.Expr = signed_volume.cum_sum()
    return rolling_slope(obv_expr, window)


def amihud_illiquidity(
    window: int,
    *,
    close_col: str = "close",
    volume_col: str = "volume",
) -> pl.Expr:
    r"""Compute Amihud illiquidity ratio.

    $$\text{ILLIQ}_t = \frac{1}{N}\sum_{i=t-N+1}^{t}
    \frac{|r_i|}{\text{DollarVolume}_i + \varepsilon}$$

    where $r_i$ is the 1-bar log return and
    $\text{DollarVolume}_i = C_i \times V_i$.

    Reference:
        Amihud (2002), "Illiquidity and stock returns:
        cross-section and time-series effects".

    Args:
        window: Rolling window size.
        close_col: Close price column name.
        volume_col: Volume column name.

    Returns:
        Polars expression for the Amihud illiquidity ratio.
    """
    abs_ret: pl.Expr = log_return(pl.col(close_col), 1).abs()
    dollar_vol: pl.Expr = pl.col(close_col) * pl.col(volume_col)
    ratio: pl.Expr = abs_ret / (dollar_vol + _EPS)
    return ratio.rolling_mean(window_size=window, min_samples=window)


# ===================================================================
# 5. STATISTICAL
# ===================================================================


def _compute_block_rs(block: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float | None:
    """Compute the R/S statistic for a single contiguous block.

    Args:
        block: 1-D array of price observations for one sub-range.

    Returns:
        The rescaled range R/S for the block, or ``None`` if the block
        has near-zero standard deviation (constant prices).
    """
    mean_val: np.floating = np.mean(block)
    deviations: np.ndarray[tuple[int], np.dtype[np.float64]] = block - mean_val
    cumdev: np.ndarray[tuple[int], np.dtype[np.float64]] = np.cumsum(deviations)
    r: np.floating = np.max(cumdev) - np.min(cumdev)
    s: np.floating = np.std(block, ddof=1)
    if s < _EPS:
        return None
    return float(r / s)


def _hurst_rs(values: pl.Series) -> float:
    r"""Estimate the Hurst exponent via rescaled-range (R/S) analysis.

    This is a ``rolling_map`` callback operating on a single window.
    The R/S method splits the window into sub-ranges of increasing
    size and regresses $\ln(R/S)$ against $\ln(n)$.

    Args:
        values: A Polars Series slice from the rolling window.

    Returns:
        Estimated Hurst exponent (scalar).  Returns ``NaN`` if the
        computation is infeasible (e.g. constant series, NaN values).
    """
    arr: np.ndarray[tuple[int], np.dtype[np.float64]] = values.to_numpy().astype(np.float64)
    n_total: int = len(arr)

    if np.isnan(arr).any():
        return float("nan")

    max_k: int = n_total // 2
    sizes: list[int] = []
    rs_values: list[float] = []

    for size in range(_HURST_MIN_BLOCK_SIZE, max_k + 1):
        n_blocks: int = n_total // size
        if n_blocks < 1:
            continue

        rs_block: list[float] = [
            rs_val
            for b in range(n_blocks)
            if (rs_val := _compute_block_rs(arr[b * size : (b + 1) * size])) is not None
        ]

        if len(rs_block) > 0:
            sizes.append(size)
            rs_values.append(float(np.mean(rs_block)))

    if len(sizes) < _MIN_RS_REGRESSION_POINTS:
        return float("nan")

    log_sizes: np.ndarray[tuple[int], np.dtype[np.float64]] = np.log(np.array(sizes, dtype=np.float64))
    log_rs: np.ndarray[tuple[int], np.dtype[np.float64]] = np.log(np.array(rs_values, dtype=np.float64))

    if np.isnan(log_rs).any() or np.isinf(log_rs).any():
        return float("nan")

    coeffs: np.ndarray[tuple[int], np.dtype[np.float64]] = np.polyfit(log_sizes, log_rs, 1)
    hurst: float = float(coeffs[0])

    # Clamp to [0, 1] range -- values outside indicate estimation noise
    return max(0.0, min(1.0, hurst))


def rolling_hurst(window: int, *, close_col: str = "close") -> pl.Expr:
    r"""Compute rolling Hurst exponent via rescaled-range (R/S) analysis.

    $$H \approx \frac{\partial \ln(R/S)}{\partial \ln(n)}$$

    Interpretation:
        - $H \approx 0.5$ -- random walk (no memory).
        - $H > 0.5$ -- trending / persistent behaviour.
        - $H < 0.5$ -- mean-reverting / anti-persistent behaviour.

    Note:
        Uses ``rolling_map`` with a NumPy callback.  The default window
        of 100 bars yields acceptable latency for datasets up to ~50k
        rows.  For larger datasets, consider pre-computing in chunks.

    Args:
        window: Rolling window size (must be >= 20 for reliable R/S).
        close_col: Close price column name.

    Returns:
        Polars expression for the rolling Hurst exponent.
    """
    return pl.col(close_col).rolling_map(function=_hurst_rs, window_size=window, min_samples=window)


def return_zscore(
    window: int,
    *,
    close_col: str = "close",
) -> pl.Expr:
    r"""Compute rolling z-score of 1-bar log returns.

    .. math::

        z_t = \frac{r_t - \bar{r}_{\text{window}}}
                    {\sigma_{r,\text{window}} + \varepsilon}

    Args:
        window: Rolling window size.
        close_col: Close price column name.

    Returns:
        Polars expression for the return z-score.
    """
    logret: pl.Expr = log_return(pl.col(close_col), 1)
    return zscore_rolling(logret, window)


def bollinger_pct_b(
    window: int,
    num_std: float = 2.0,
    *,
    close_col: str = "close",
) -> pl.Expr:
    r"""Compute Bollinger Bands Percent B (%B).

    .. math::

        \%B = \frac{C_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t + \varepsilon}

    where Upper and Lower are the Bollinger Band boundaries at
    ``num_std`` standard deviations from the rolling mean.

    Args:
        window: Rolling window for the mean and standard deviation.
        num_std: Number of standard deviations for the bands.
        close_col: Close price column name.

    Returns:
        Polars expression for Bollinger %B.
    """
    close: pl.Expr = pl.col(close_col)
    middle: pl.Expr = close.rolling_mean(window_size=window, min_samples=window)
    std: pl.Expr = close.rolling_std(window_size=window, min_samples=window)
    upper: pl.Expr = middle + std * num_std
    lower: pl.Expr = middle - std * num_std
    return (close - lower) / (upper - lower + _EPS)


def bollinger_width(
    window: int,
    num_std: float = 2.0,
    *,
    close_col: str = "close",
) -> pl.Expr:
    r"""Compute Bollinger Bands Width.

    .. math::

        \text{Width} = \frac{\text{Upper}_t - \text{Lower}_t}
                             {\text{Middle}_t + \varepsilon}

    Args:
        window: Rolling window for the mean and standard deviation.
        num_std: Number of standard deviations for the bands.
        close_col: Close price column name.

    Returns:
        Polars expression for Bollinger Bands Width.
    """
    close: pl.Expr = pl.col(close_col)
    middle: pl.Expr = close.rolling_mean(window_size=window, min_samples=window)
    std: pl.Expr = close.rolling_std(window_size=window, min_samples=window)
    band_range: pl.Expr = std * num_std * 2.0
    return band_range / (middle + _EPS)


# ===================================================================
# 6. UTILITIES
# ===================================================================


def zscore_rolling(expr: pl.Expr, window: int) -> pl.Expr:
    r"""Compute rolling z-score.

    $$z_t = \frac{x_t - \bar{x}_{\text{window}}}
                {\sigma_{x,\text{window}} + \varepsilon}$$

    Args:
        expr: Input expression.
        window: Rolling window size.

    Returns:
        Polars expression for the rolling z-score.
    """
    mean: pl.Expr = expr.rolling_mean(window_size=window, min_samples=window)
    std: pl.Expr = expr.rolling_std(window_size=window, min_samples=window)
    return (expr - mean) / (std + _EPS)


def clip_expr(expr: pl.Expr, lo: float = -5.0, hi: float = 5.0) -> pl.Expr:
    """Clip expression values to a fixed range.

    Args:
        expr: Input expression.
        lo: Lower clipping bound.
        hi: Upper clipping bound.

    Returns:
        Clipped Polars expression.
    """
    return expr.clip(lo, hi)


# ===================================================================
# 7. ORCHESTRATOR
# ===================================================================


def _add_return_features(
    config: IndicatorConfig,
    close_col: str,
) -> list[pl.Expr]:
    """Build log-return expressions for all configured horizons.

    Args:
        config: Indicator configuration.
        close_col: Close price column name.

    Returns:
        List of aliased log-return expressions.
    """
    return [log_return(pl.col(close_col), h).alias(f"logret_{h}") for h in config.return_horizons]


def _add_volatility_features(
    config: IndicatorConfig,
    close_col: str,
) -> list[pl.Expr]:
    """Build volatility indicator expressions.

    Args:
        config: Indicator configuration.
        close_col: Close price column name.

    Returns:
        List of aliased volatility expressions.
    """
    logret_1: pl.Expr = log_return(pl.col(close_col), 1)
    exprs: list[pl.Expr] = [realized_vol(logret_1, w).alias(f"rv_{w}") for w in config.realized_vol_windows]
    exprs.append(garman_klass_vol(config.garman_klass_window).alias(f"gk_vol_{config.garman_klass_window}"))
    exprs.append(parkinson_vol(config.parkinson_window).alias(f"park_vol_{config.parkinson_window}"))
    exprs.append(atr(config.atr_period, wilder=config.atr_wilder).alias(f"atr_{config.atr_period}"))
    return exprs


def _add_momentum_features(
    config: IndicatorConfig,
    close_col: str,
) -> list[pl.Expr]:
    """Build momentum indicator expressions.

    Args:
        config: Indicator configuration.
        close_col: Close price column name.

    Returns:
        List of aliased momentum expressions.
    """
    exprs: list[pl.Expr] = [
        ema_crossover(
            config.ema_fast_span,
            config.ema_slow_span,
            config.atr_period,
            atr_wilder=config.atr_wilder,
        ).alias(f"ema_xover_{config.ema_fast_span}_{config.ema_slow_span}"),
        rsi(config.rsi_period).alias(f"rsi_{config.rsi_period}"),
    ]
    close: pl.Expr = pl.col(close_col)
    exprs.extend(roc(close, p).alias(f"roc_{p}") for p in config.roc_periods)
    return exprs


def _add_volume_features(
    config: IndicatorConfig,
) -> list[pl.Expr]:
    """Build volume indicator expressions.

    Args:
        config: Indicator configuration.

    Returns:
        List of aliased volume expressions.
    """
    return [
        volume_zscore(config.volume_zscore_window).alias(f"vol_zscore_{config.volume_zscore_window}"),
        amihud_illiquidity(config.amihud_window).alias(f"amihud_{config.amihud_window}"),
    ]


def _add_statistical_features(
    config: IndicatorConfig,
) -> list[pl.Expr]:
    """Build statistical indicator expressions (excluding rolling_map-based ones).

    Args:
        config: Indicator configuration.

    Returns:
        List of aliased statistical expressions.
    """
    return [
        return_zscore(config.return_zscore_window).alias(f"ret_zscore_{config.return_zscore_window}"),
        bollinger_pct_b(config.bollinger_window, config.bollinger_num_std).alias(
            f"bbpctb_{config.bollinger_window}_{config.bollinger_num_std}"
        ),
        bollinger_width(config.bollinger_window, config.bollinger_num_std).alias(
            f"bbwidth_{config.bollinger_window}_{config.bollinger_num_std}"
        ),
    ]


def _add_rolling_map_features(
    df: pl.DataFrame,
    config: IndicatorConfig,
) -> pl.DataFrame:
    """Add features that require ``rolling_map`` (NumPy callbacks).

    These are applied separately because ``rolling_map`` expressions
    cannot be batched in a single ``with_columns`` call alongside
    other ``rolling_map`` expressions without triggering sequential
    evaluation anyway.  Separating them makes the pipeline structure
    explicit.

    Args:
        df: DataFrame with OHLCV + previously computed features.
        config: Indicator configuration.

    Returns:
        DataFrame with rolling-map features appended.
    """
    result: pl.DataFrame = df.with_columns(
        rolling_slope(pl.col("close"), config.slope_window).alias(f"slope_{config.slope_window}"),
    )
    result = result.with_columns(
        obv_slope(config.obv_slope_window).alias(f"obv_slope_{config.obv_slope_window}"),
    )
    return result.with_columns(
        rolling_hurst(config.hurst_window).alias(f"hurst_{config.hurst_window}"),
    )


def compute_all_indicators(
    df: pl.DataFrame,
    config: IndicatorConfig,
) -> pl.DataFrame:
    """Compute all technical indicators and append them to the DataFrame.

    This is the main entry point for Phase 4A indicator computation.
    It applies every indicator group in order:

    1. **Returns** -- multi-horizon log returns.
    2. **Volatility** -- realized vol, Garman-Klass, Parkinson, ATR.
    3. **Momentum** -- EMA crossover, RSI, ROC.
    4. **Volume** -- volume z-score, Amihud illiquidity.
    5. **Statistical** -- return z-score, Bollinger %B and Width.
    6. **Rolling-map features** -- slope, OBV slope, Hurst exponent
       (applied separately due to NumPy callbacks).

    All feature columns are clipped to ``[clip_lower, clip_upper]``
    at the end to prevent extreme outliers from dominating downstream
    models.

    Expected input columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``.

    Args:
        df: Polars DataFrame with OHLCV columns.
        config: Indicator configuration controlling all parameters.

    Returns:
        DataFrame with all indicator columns appended.  The original
        OHLCV columns are preserved.
    """
    close_col: str = "close"
    input_cols: set[str] = set(df.columns)

    # --- Batch 1: expression-based indicators (vectorised, fast) ---
    exprs: list[pl.Expr] = []
    exprs.extend(_add_return_features(config, close_col))
    exprs.extend(_add_volatility_features(config, close_col))
    exprs.extend(_add_momentum_features(config, close_col))
    exprs.extend(_add_volume_features(config))
    exprs.extend(_add_statistical_features(config))

    result: pl.DataFrame = df.with_columns(exprs)

    # --- Batch 2: rolling_map-based indicators (NumPy callbacks) ---
    result = _add_rolling_map_features(result, config)

    # --- Batch 3: clip all feature columns ---
    feature_cols: list[str] = [c for c in result.columns if c not in input_cols]
    clip_exprs: list[pl.Expr] = [
        clip_expr(pl.col(col), config.clip_lower, config.clip_upper).alias(col) for col in feature_cols
    ]
    if clip_exprs:
        result = result.with_columns(clip_exprs)

    return result
