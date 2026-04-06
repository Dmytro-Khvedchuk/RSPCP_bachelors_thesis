"""Forward-looking regression targets -- training labels for ML models.

All target functions are stateless and produce Polars expressions that
use **negative shifts** (future data).  These columns must **never**
appear in live inference -- only during model training.

Column naming convention:
    ``fwd_logret_{h}`` -- forward log return at horizon *h*.
    ``fwd_vol_{h}``    -- forward realized volatility at horizon *h*.
    ``fwd_zret_{h}``   -- forward volatility-normalized return at horizon *h*.

The ``fwd_`` prefix distinguishes forward-looking targets from
backward-looking indicators (``logret_{h}``, ``rv_{w}``).
"""

from __future__ import annotations

from typing import Final

import polars as pl

from src.app.features.domain.value_objects import TargetConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: Final[float] = 1e-12
"""Epsilon for division-by-zero protection."""

TARGET_PREFIX: Final[str] = "fwd_"
"""Public prefix for downstream identification of target columns."""


# ===================================================================
# 1. FORWARD LOG RETURNS
# ===================================================================


def forward_log_return(close: pl.Expr, horizon: int) -> pl.Expr:
    r"""Compute forward log return at a given horizon.

    $$r^{\text{fwd}}_t = \ln\!\left(\frac{C_{t+h}}{C_t}\right)$$

    The last *horizon* rows will be null because no future data is
    available.

    Note:
        This is computed directly rather than reusing `log_return`
        because the shift direction is reversed (negative shift =
        future data) and reusing would invite sign-confusion bugs.

    Args:
        close: Close price expression (e.g. ``pl.col("close")``).
        horizon: Number of bars to look ahead (must be >= 1).

    Returns:
        Polars expression for the forward log return.
    """
    return (close.shift(-horizon) / close).log()


# ===================================================================
# 2. FORWARD REALIZED VOLATILITY
# ===================================================================


def forward_volatility(close: pl.Expr, horizon: int) -> pl.Expr:
    r"""Compute forward realized volatility at a given horizon.

    Methodology:
        1. Compute 1-bar log returns: $r_t = \ln(C_t / C_{t-1})$
        2. Rolling standard deviation over *horizon* bars
        3. Shift by ``-horizon`` so the value at row *t* reflects the
           volatility of returns from $[t+1, \ldots, t+h]$

    $$\sigma^{\text{fwd}}_t = \text{std}(r_{t+1}, \ldots, r_{t+h})$$

    The last *horizon* rows will be null.

    Args:
        close: Close price expression (e.g. ``pl.col("close")``).
        horizon: Number of bars to look ahead (must be >= 2).

    Returns:
        Polars expression for the forward realized volatility.
    """
    logret_1: pl.Expr = (close / close.shift(1)).log()
    rolling_std: pl.Expr = logret_1.rolling_std(window_size=horizon, min_samples=horizon)
    return rolling_std.shift(-horizon)


# ===================================================================
# 3. FORWARD VOLATILITY-NORMALIZED RETURNS (Z-RETURNS)
# ===================================================================


def forward_zreturn(close: pl.Expr, horizon: int, backward_vol_window: int) -> pl.Expr:
    r"""Compute forward volatility-normalized return (z-return).

    $$z^{\text{fwd}}_t = \frac{r^{\text{fwd}}_t}{\hat\sigma^{\text{bwd}}_t + \epsilon}$$

    where

    * $r^{\text{fwd}}_t = \ln(C_{t+h} / C_t)$ is the
      **forward** log return (uses future data -- this is a target).
    * $\hat\sigma^{\text{bwd}}_t = \text{std}(r_{t-w+1}, \ldots, r_t)$
      is the **backward-looking** realized volatility computed from
      *past* 1-bar log returns only.

    Leakage Safety:
        The denominator is purely backward-looking by construction:
        ``rolling_std`` with ``window_size=backward_vol_window`` operates
        on *past* log returns (positive ``.shift(1)``).  No negative shift
        is applied to the denominator, so it cannot see the future.

    Args:
        close: Close price expression (e.g. ``pl.col("close")``).
        horizon: Number of bars to look ahead for the return (must be >= 1).
        backward_vol_window: Rolling window for backward-looking realized
            volatility (must be >= 2).

    Returns:
        Polars expression for the forward z-return.
    """
    # Numerator: forward log return (uses future data -- target)
    fwd_ret: pl.Expr = forward_log_return(close, horizon)

    # Denominator: backward-looking realized volatility (NO future data)
    # 1-bar log returns: r_t = ln(C_t / C_{t-1})
    logret_1: pl.Expr = (close / close.shift(1)).log()
    # Rolling std of past log returns -- purely backward-looking
    backward_rv: pl.Expr = logret_1.rolling_std(
        window_size=backward_vol_window,
        min_samples=backward_vol_window,
    )

    return fwd_ret / (backward_rv + _EPS)


# ===================================================================
# 4. WINSORIZATION
# ===================================================================


def winsorize_series(
    df: pl.DataFrame,
    col_name: str,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pl.DataFrame:
    """Winsorize a column at given percentiles (clip to percentile bounds).

    Values below the ``lower_pct`` quantile are clipped up to that bound;
    values above the ``upper_pct`` quantile are clipped down to that bound.
    Null values are preserved.

    This is a DataFrame-level function (not a pure expression) because
    computing quantiles requires an aggregation over the full column.

    Args:
        df: Input DataFrame containing the column to winsorize.
        col_name: Name of the column to winsorize.
        lower_pct: Lower percentile (e.g. 0.01 for 1st percentile).
        upper_pct: Upper percentile (e.g. 0.99 for 99th percentile).

    Returns:
        DataFrame with the specified column winsorized in place.
    """
    lower_raw: int | float | list[int | float] | None = df[col_name].quantile(
        lower_pct,
        interpolation="linear",
    )
    upper_raw: int | float | list[int | float] | None = df[col_name].quantile(
        upper_pct,
        interpolation="linear",
    )

    # quantile() with a scalar percentile returns a scalar, not a list.
    lower_bound: float | None = float(lower_raw) if isinstance(lower_raw, (int, float)) else None
    upper_bound: float | None = float(upper_raw) if isinstance(upper_raw, (int, float)) else None

    if lower_bound is None or upper_bound is None:
        # All values are null -- nothing to winsorize
        return df

    return df.with_columns(
        pl.col(col_name).clip(lower_bound=lower_bound, upper_bound=upper_bound).alias(col_name),
    )


# ===================================================================
# 5. PRIVATE HELPERS
# ===================================================================


def _add_forward_return_targets(config: TargetConfig) -> list[pl.Expr]:
    """Build forward log-return expressions for all configured horizons.

    Args:
        config: Target configuration.

    Returns:
        List of aliased forward log-return expressions.
    """
    close: pl.Expr = pl.col(config.close_col)
    return [forward_log_return(close, h).alias(f"fwd_logret_{h}") for h in config.forward_return_horizons]


def _add_forward_vol_targets(config: TargetConfig) -> list[pl.Expr]:
    """Build forward volatility expressions for all configured horizons.

    Args:
        config: Target configuration.

    Returns:
        List of aliased forward volatility expressions.
    """
    close: pl.Expr = pl.col(config.close_col)
    return [forward_volatility(close, h).alias(f"fwd_vol_{h}") for h in config.forward_vol_horizons]


def _add_forward_zret_targets(config: TargetConfig) -> list[pl.Expr]:
    """Build forward z-return expressions for all configured horizons.

    Each z-return divides the forward log return by the backward-looking
    realized volatility, producing a volatility-normalized signal that is
    more stationary across market regimes than raw returns.

    Args:
        config: Target configuration.

    Returns:
        List of aliased forward z-return expressions.
    """
    close: pl.Expr = pl.col(config.close_col)
    return [
        forward_zreturn(close, h, config.backward_vol_window).alias(f"fwd_zret_{h}")
        for h in config.forward_zret_horizons
    ]


def _get_target_col_names(config: TargetConfig) -> list[str]:
    """Return target column names that will be winsorized.

    This helper centralizes column-name generation so both the
    winsorization loop and ``get_target_column_names`` stay in sync.

    Args:
        config: Target configuration.

    Returns:
        Flat list of all target column names (unsorted).
    """
    names: list[str] = [f"fwd_logret_{h}" for h in config.forward_return_horizons]
    names.extend(f"fwd_vol_{h}" for h in config.forward_vol_horizons)
    names.extend(f"fwd_zret_{h}" for h in config.forward_zret_horizons)
    return names


# ===================================================================
# 6. ORCHESTRATOR
# ===================================================================


def compute_all_targets(
    df: pl.DataFrame,
    config: TargetConfig,
) -> pl.DataFrame:
    """Compute all forward-looking regression targets and append to the DataFrame.

    This is the main entry point for target construction.  All targets are
    native Polars expressions computed in a single ``with_columns`` call,
    followed by optional winsorization at configured percentiles.

    Winsorization replaces hard-clip bounds with data-driven percentile
    clipping, preserving extreme events (COVID crash, FTX, Luna) while
    removing distributional outliers.

    **Important:** This function does NOT drop NaN rows -- that is
    Phase 4C's responsibility (``FeatureMatrixBuilder``).

    Expected input columns:
        At minimum, the column named by ``config.close_col`` (default
        ``"close"``).

    Args:
        df: Polars DataFrame with at least a close price column.
        config: Target configuration controlling horizons and column names.

    Returns:
        DataFrame with all target columns appended.  The original
        columns are preserved.
    """
    # Step 1: compute all target expressions in one pass
    exprs: list[pl.Expr] = []
    exprs.extend(_add_forward_return_targets(config))
    exprs.extend(_add_forward_vol_targets(config))
    exprs.extend(_add_forward_zret_targets(config))
    result: pl.DataFrame = df.with_columns(exprs)

    # Step 2: winsorize target columns at configured percentiles
    if config.winsorize:
        target_cols: list[str] = _get_target_col_names(config)
        for col_name in target_cols:
            result = winsorize_series(
                result,
                col_name,
                lower_pct=config.winsorize_lower_pct,
                upper_pct=config.winsorize_upper_pct,
            )

    return result


def get_target_column_names(config: TargetConfig) -> list[str]:
    """Return a sorted list of target column names for the given config.

    Enables downstream code to programmatically identify which columns
    are forward-looking targets.

    Args:
        config: Target configuration.

    Returns:
        Sorted list of target column names
        (e.g. ``["fwd_logret_1", ..., "fwd_vol_4", ..., "fwd_zret_1", ...]``).
    """
    return sorted(_get_target_col_names(config))
