"""Forward-looking regression targets -- training labels for ML models.

All target functions are stateless and produce Polars expressions that
use **negative shifts** (future data).  These columns must **never**
appear in live inference -- only during model training.

Column naming convention:
    ``fwd_logret_{h}`` -- forward log return at horizon *h*.
    ``fwd_vol_{h}``    -- forward realized volatility at horizon *h*.

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

    .. math::

        r^{\text{fwd}}_t = \ln\!\left(\frac{C_{t+h}}{C_t}\right)

    The last *horizon* rows will be null because no future data is
    available.

    Note:
        This is computed directly rather than reusing :func:`log_return`
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
        1. Compute 1-bar log returns: :math:`r_t = \ln(C_t / C_{t-1})`
        2. Rolling standard deviation over *horizon* bars
        3. Shift by ``-horizon`` so the value at row *t* reflects the
           volatility of returns from :math:`[t+1, \ldots, t+h]`

    .. math::

        \sigma^{\text{fwd}}_t = \text{std}(r_{t+1}, \ldots, r_{t+h})

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
# 3. PRIVATE HELPERS
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


# ===================================================================
# 4. ORCHESTRATOR
# ===================================================================


def compute_all_targets(
    df: pl.DataFrame,
    config: TargetConfig,
) -> pl.DataFrame:
    """Compute all forward-looking regression targets and append to the DataFrame.

    This is the main entry point for Phase 4B target construction.
    All targets are native Polars expressions computed in a single
    ``with_columns`` call.

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
    exprs: list[pl.Expr] = []
    exprs.extend(_add_forward_return_targets(config))
    exprs.extend(_add_forward_vol_targets(config))
    return df.with_columns(exprs)


def get_target_column_names(config: TargetConfig) -> list[str]:
    """Return a sorted list of target column names for the given config.

    Enables downstream code to programmatically identify which columns
    are forward-looking targets.

    Args:
        config: Target configuration.

    Returns:
        Sorted list of target column names
        (e.g. ``["fwd_logret_1", "fwd_logret_4", ..., "fwd_vol_4", ...]``).
    """
    names: list[str] = [f"fwd_logret_{h}" for h in config.forward_return_horizons]
    names.extend(f"fwd_vol_{h}" for h in config.forward_vol_horizons)
    return sorted(names)
