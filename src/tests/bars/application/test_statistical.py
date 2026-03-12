"""Statistical tests for bar construction quality.

Verifies the key properties of dollar bars vs tick bars:

1.  **Tick bars are insensitive to market activity** — they produce a constant
    number of bars per period (N / threshold) regardless of volatility or volume.
    This means tick bars have *zero* coefficient of variation (CV) across periods
    of equal row count.

2.  **Dollar bars are sensitive to dollar flow** — they produce more bars in
    volatile or high-volume periods because each row contributes more dollar value.
    This means dollar bars have a *higher* CV across periods of equal row count but
    differing activity.

3.  **Dollar bars normalise for price level changes** — across periods at
    *different price levels* (but same volume), dollar bars produce a similar bar
    count while tick bars produce identical counts regardless (since tick bars
    ignore price entirely).  The López de Prado uniformity claim applies to price
    LEVEL differences, not volatility/volume differences.

Reference: López de Prado, *Advances in Financial Machine Learning* (2018), §2.3.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import numpy as np
import polars as pl
import pytest

from src.app.bars.application.dollar_bars import DollarBarAggregator
from src.app.bars.application.tick_bars import TickBarAggregator
from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.tests.bars.conftest import BTC_ASSET


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RNG_SEED: int = 42
_EPOCH: datetime = datetime(2024, 1, 1, tzinfo=UTC)
_MINUTE: timedelta = timedelta(minutes=1)

# Period sizes for the simulation
_N_ROWS: int = 200

# Tick bar threshold (rows per bar)
_TICK_THRESHOLD: float = 20.0

# Number of synthetic periods per condition
_N_PERIODS: int = 3


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _make_period_df(
    n: int,
    *,
    base_ts: datetime,
    price_mean: float,
    price_std: float,
    volume_mean: float,
    volume_std: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Generate a synthetic OHLCV DataFrame representing one market period.

    Prices are sampled from a normal distribution clipped to a minimum
    of 10% of price_mean.  Volumes are sampled from a normal distribution
    clipped to a minimum of 0.01.

    Args:
        n: Number of rows.
        base_ts: Starting timestamp.
        price_mean: Mean price level.
        price_std: Price standard deviation.
        volume_mean: Mean volume per row.
        volume_std: Volume standard deviation.
        rng: NumPy random Generator for reproducibility.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    prices: np.ndarray = rng.normal(price_mean, price_std, n).clip(price_mean * 0.1)
    volumes: np.ndarray = rng.normal(volume_mean, volume_std, n).clip(0.01)

    timestamps: list[datetime] = [base_ts + i * _MINUTE for i in range(n)]
    opens: list[float] = prices.tolist()
    highs: list[float] = (prices * 1.01).tolist()
    lows: list[float] = (prices * 0.99).tolist()
    closes: list[float] = prices.tolist()
    vols: list[float] = volumes.tolist()

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )


def _coefficient_of_variation(counts: list[int]) -> float:
    """Compute coefficient of variation (std / mean) for a list of counts.

    Args:
        counts: Integer bar counts per period.

    Returns:
        CV = std / mean, or 0.0 if mean is zero or only one sample.
    """
    if len(counts) < 2 or sum(counts) == 0:
        return 0.0
    arr: np.ndarray = np.array(counts, dtype=np.float64)
    mean_val: float = float(np.mean(arr))
    if mean_val == 0.0:
        return 0.0
    return float(np.std(arr, ddof=1)) / mean_val


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


class TestDollarBarsUniformity:
    """Statistical tests for dollar bar vs tick bar sampling properties.

    Tick bars: insensitive to market activity (constant count per equal-row period).
    Dollar bars: sensitive to dollar flow (count rises with price × volume).

    The López de Prado uniformity claim is that dollar bars normalise for PRICE
    LEVEL differences — tested here by comparing bar counts across periods with
    different price levels but the same volume profile.
    """

    def test_tick_bars_produce_uniform_count_regardless_of_volatility(self) -> None:
        """Tick bar count per period must be constant regardless of price volatility.

        Tick bars depend only on row count, so their count per period is
        deterministic: ``floor(N / threshold)`` ± 1 for the partial last bar.
        This is both a feature (predictable) and a limitation (ignores activity).
        """
        rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
        tick_agg: TickBarAggregator = TickBarAggregator()
        tick_config: BarConfig = BarConfig(
            bar_type=BarType.TICK,
            threshold=_TICK_THRESHOLD,
            ewm_span=100,
            warmup_period=100,
        )

        counts: list[int] = []
        base_ts: datetime = _EPOCH

        for period_idx in range(_N_PERIODS * 2):
            # Alternate calm/volatile
            is_volatile: bool = period_idx % 2 == 1
            price_std: float = 1000.0 if is_volatile else 100.0
            volume_mean: float = 10.0 if is_volatile else 1.0

            df: pl.DataFrame = _make_period_df(
                _N_ROWS,
                base_ts=base_ts,
                price_mean=10_000.0,
                price_std=price_std,
                volume_mean=volume_mean,
                volume_std=volume_mean * 0.2,
                rng=rng,
            )
            base_ts += _N_ROWS * _MINUTE
            bars: list[AggregatedBar] = tick_agg.aggregate(df, asset=BTC_ASSET, config=tick_config)
            counts.append(len(bars))

        expected_count: int = _N_ROWS // int(_TICK_THRESHOLD)
        for count in counts:
            # Allow ±1 for the partial last bar
            assert abs(count - expected_count) <= 1, f"Tick bar count {count} deviates from expected {expected_count}"

    def test_dollar_bars_produce_more_bars_in_high_dollar_flow_period(self) -> None:
        """Dollar bar count increases when price × volume is higher.

        This verifies that dollar bars sample more frequently in active markets,
        which is the primary advantage over time/tick bars.
        """
        rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
        dollar_agg: DollarBarAggregator = DollarBarAggregator()

        # Calibrate threshold: calm period has price=10000, volume≈1 per row
        # → dollar_val ≈ 10000/row; threshold = 10000*20 = 200000 → ~10 bars in 200 rows
        dollar_threshold: float = 200_000.0
        dollar_config: BarConfig = BarConfig(
            bar_type=BarType.DOLLAR,
            threshold=dollar_threshold,
            ewm_span=100,
            warmup_period=100,
        )

        calm_df: pl.DataFrame = _make_period_df(
            _N_ROWS,
            base_ts=_EPOCH,
            price_mean=10_000.0,
            price_std=100.0,
            volume_mean=1.0,
            volume_std=0.1,
            rng=rng,
        )
        # High-flow period: 10x the volume
        high_flow_df: pl.DataFrame = _make_period_df(
            _N_ROWS,
            base_ts=_EPOCH + _N_ROWS * _MINUTE,
            price_mean=10_000.0,
            price_std=100.0,
            volume_mean=10.0,
            volume_std=1.0,
            rng=rng,
        )

        calm_bars: list[AggregatedBar] = dollar_agg.aggregate(calm_df, asset=BTC_ASSET, config=dollar_config)
        high_flow_bars: list[AggregatedBar] = dollar_agg.aggregate(high_flow_df, asset=BTC_ASSET, config=dollar_config)

        # High dollar flow → more bars
        assert len(high_flow_bars) > len(calm_bars), (
            f"Expected high-flow bars ({len(high_flow_bars)}) > calm bars ({len(calm_bars)})"
        )

    def test_dollar_bars_normalise_across_price_level_differences(self) -> None:
        """Dollar bars produce similar counts across periods at different price levels.

        This is the core López de Prado claim: at price 1000 vs 10000, if
        volume scales inversely (so dollar flow stays constant), dollar bars
        produce the same number of bars.  Tick bars also produce the same count
        (they ignore price entirely), but dollar bars preserve this property while
        also capturing total information content.

        We test a weaker but verifiable form: dollar bars at price P₁ with
        volume V₁ and at price P₂=2P₁ with volume V₂=0.5V₁ (same dollar flow)
        should produce a similar bar count (within a small relative tolerance).
        """
        rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
        dollar_agg: DollarBarAggregator = DollarBarAggregator()

        price_low: float = 5_000.0
        price_high: float = 10_000.0
        volume_at_low: float = 4.0  # dollar_val ≈ 20000/row
        volume_at_high: float = 2.0  # dollar_val ≈ 20000/row — same dollar flow

        threshold: float = 400_000.0  # ~20 rows/bar
        config: BarConfig = BarConfig(
            bar_type=BarType.DOLLAR,
            threshold=threshold,
            ewm_span=100,
            warmup_period=100,
        )

        df_low: pl.DataFrame = _make_period_df(
            _N_ROWS,
            base_ts=_EPOCH,
            price_mean=price_low,
            price_std=50.0,
            volume_mean=volume_at_low,
            volume_std=0.2,
            rng=rng,
        )
        df_high: pl.DataFrame = _make_period_df(
            _N_ROWS,
            base_ts=_EPOCH + _N_ROWS * _MINUTE,
            price_mean=price_high,
            price_std=100.0,
            volume_mean=volume_at_high,
            volume_std=0.1,
            rng=rng,
        )

        bars_low: list[AggregatedBar] = dollar_agg.aggregate(df_low, asset=BTC_ASSET, config=config)
        bars_high: list[AggregatedBar] = dollar_agg.aggregate(df_high, asset=BTC_ASSET, config=config)

        count_low: int = len(bars_low)
        count_high: int = len(bars_high)

        # Allow up to 50% relative difference (dollar flow is equal in expectation
        # but random noise adds variance)
        relative_diff: float = abs(count_low - count_high) / max(count_low, count_high)
        assert relative_diff < 0.5, (
            f"Dollar bar counts differ too much across price levels: "
            f"{count_low} bars at price={price_low}, {count_high} bars at price={price_high}"
        )

    def test_dollar_bar_count_higher_cv_than_tick_across_activity_regimes(self) -> None:
        """Dollar bar counts must show higher CV than tick bars across activity regimes.

        Tick bars always produce the same count regardless of activity (CV ≈ 0),
        while dollar bars respond to dollar flow (higher CV).  This demonstrates
        that dollar bars carry more information about market activity than tick bars.
        """
        rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
        tick_agg: TickBarAggregator = TickBarAggregator()
        dollar_agg: DollarBarAggregator = DollarBarAggregator()

        tick_config: BarConfig = BarConfig(
            bar_type=BarType.TICK,
            threshold=_TICK_THRESHOLD,
            ewm_span=100,
            warmup_period=100,
        )
        dollar_config: BarConfig = BarConfig(
            bar_type=BarType.DOLLAR,
            threshold=200_000.0,
            ewm_span=100,
            warmup_period=100,
        )

        tick_counts: list[int] = []
        dollar_counts: list[int] = []
        base_ts: datetime = _EPOCH

        for period_idx in range(_N_PERIODS * 2):
            is_high_activity: bool = period_idx % 2 == 1
            volume_mean: float = 10.0 if is_high_activity else 1.0

            df: pl.DataFrame = _make_period_df(
                _N_ROWS,
                base_ts=base_ts,
                price_mean=10_000.0,
                price_std=100.0,
                volume_mean=volume_mean,
                volume_std=volume_mean * 0.1,
                rng=rng,
            )
            base_ts += _N_ROWS * _MINUTE

            tick_bars: list[AggregatedBar] = tick_agg.aggregate(df, asset=BTC_ASSET, config=tick_config)
            dollar_bars: list[AggregatedBar] = dollar_agg.aggregate(df, asset=BTC_ASSET, config=dollar_config)

            tick_counts.append(len(tick_bars))
            dollar_counts.append(len(dollar_bars))

        tick_cv: float = _coefficient_of_variation(tick_counts)
        dollar_cv: float = _coefficient_of_variation(dollar_counts)

        # Dollar bars must show more variation across activity regimes than tick bars.
        # Tick bars have CV ≈ 0 (row count is fixed); dollar bars respond to dollar flow.
        assert dollar_cv > tick_cv, (
            f"Dollar bars CV ({dollar_cv:.4f}) must exceed tick bars CV ({tick_cv:.4f}).  "
            f"Tick counts: {tick_counts}.  Dollar counts: {dollar_counts}."
        )

    @pytest.mark.parametrize("n_rows", [100, 500])
    def test_tick_count_preserved_in_simulated_data(self, n_rows: int) -> None:
        """Total tick_count must equal n_rows for simulated realistic data."""
        rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
        dollar_agg: DollarBarAggregator = DollarBarAggregator()
        dollar_config: BarConfig = BarConfig(
            bar_type=BarType.DOLLAR,
            threshold=200_000.0,
            ewm_span=100,
            warmup_period=100,
        )

        df: pl.DataFrame = _make_period_df(
            n_rows,
            base_ts=_EPOCH,
            price_mean=10_000.0,
            price_std=200.0,
            volume_mean=2.0,
            volume_std=0.5,
            rng=rng,
        )
        bars: list[AggregatedBar] = dollar_agg.aggregate(df, asset=BTC_ASSET, config=dollar_config)
        total: int = sum(b.tick_count for b in bars)
        assert total == n_rows
