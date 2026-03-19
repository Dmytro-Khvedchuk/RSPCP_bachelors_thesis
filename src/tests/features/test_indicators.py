"""Unit tests for individual technical indicator functions in indicators.py.

Tests each indicator against known reference values and verifies edge-case
behaviour (all-gains RSI, all-losses RSI, constant series, sign correctness).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from src.app.features.application.indicators import (
    atr,
    bollinger_pct_b,
    bollinger_width,
    clip_expr,
    compute_all_indicators,
    ema,
    ema_crossover,
    log_return,
    obv_slope,
    parkinson_vol,
    realized_vol,
    roc,
    rolling_slope,
    rsi,
    true_range,
    volume_zscore,
    zscore_rolling,
)
from src.app.features.domain.value_objects import IndicatorConfig

from src.tests.features.conftest import (
    make_ohlcv_df,
    make_random_walk_df,
    make_small_indicator_config,
    make_trending_df,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-12
_RTOL: float = 1e-6


class TestLogReturn:
    """Tests for the log_return indicator."""

    def test_log_return_known_values(self) -> None:
        """Verify 1-period log returns on [100, 110, 121] equal ln(1.1)."""
        df: pl.DataFrame = pl.DataFrame({"close": [100.0, 110.0, 121.0]})
        result: pl.DataFrame = df.with_columns(log_return(pl.col("close"), 1).alias("lr"))
        lr_vals: list[float | None] = result["lr"].to_list()

        assert lr_vals[0] is None
        assert lr_vals[1] == pytest.approx(math.log(1.1), rel=_RTOL)
        assert lr_vals[2] == pytest.approx(math.log(1.1), rel=_RTOL)

    def test_log_return_multi_period(self) -> None:
        """Verify 4-period log return on [100, 110, 121, 133.1, 146.41]."""
        prices: list[float] = [100.0, 110.0, 121.0, 133.1, 146.41]
        df: pl.DataFrame = pl.DataFrame({"close": prices})
        result: pl.DataFrame = df.with_columns(log_return(pl.col("close"), 4).alias("lr4"))
        lr4_vals: list[float | None] = result["lr4"].to_list()

        # First 4 entries must be null (not enough prior data)
        for i in range(4):
            assert lr4_vals[i] is None

        # 146.41 / 100.0 = 1.4641 → ln(1.4641) ≈ 4 * ln(1.1)
        expected: float = math.log(146.41 / 100.0)
        assert lr4_vals[4] == pytest.approx(expected, rel=_RTOL)

    def test_log_return_negative_for_falling_prices(self) -> None:
        """Verify that falling prices produce negative log returns."""
        df: pl.DataFrame = pl.DataFrame({"close": [100.0, 90.0, 80.0]})
        result: pl.DataFrame = df.with_columns(log_return(pl.col("close"), 1).alias("lr"))
        lr_vals: list[float | None] = result["lr"].to_list()

        assert lr_vals[1] is not None
        assert lr_vals[1] < 0
        assert lr_vals[2] is not None
        assert lr_vals[2] < 0


class TestRealizedVol:
    """Tests for the realized_vol indicator."""

    def test_realized_vol_constant_returns(self) -> None:
        """Constant returns should yield near-zero realized volatility."""
        # Prices growing by exactly 10% each bar → all log returns = ln(1.1)
        prices: list[float] = [100.0 * (1.1**i) for i in range(30)]
        df: pl.DataFrame = pl.DataFrame({"close": prices})
        logret_expr: pl.Expr = log_return(pl.col("close"), 1)
        result: pl.DataFrame = df.with_columns(realized_vol(logret_expr, 10).alias("rv"))
        rv_vals: list[float | None] = result["rv"].to_list()
        # After warmup, every value must be effectively 0 (constant log returns)
        non_null: list[float] = [v for v in rv_vals if v is not None]
        assert len(non_null) > 0
        for v in non_null:
            assert v == pytest.approx(0.0, abs=1e-8)

    def test_realized_vol_window_correct_null_count(self) -> None:
        """Verify the number of null values at the start equals window size."""
        n: int = 30
        window: int = 8
        df: pl.DataFrame = make_ohlcv_df(n, price_step=1.0)
        logret_expr: pl.Expr = log_return(pl.col("close"), 1)
        # Need to first compute log_return and then apply realized_vol
        df2: pl.DataFrame = df.with_columns(logret_expr.alias("lr"))
        result: pl.DataFrame = df2.with_columns(realized_vol(pl.col("lr"), window).alias("rv"))
        rv_vals: list[float | None] = result["rv"].to_list()
        null_count: int = sum(1 for v in rv_vals if v is None)
        # log_return has 1 null, then rolling_std has window-1 additional nulls
        assert null_count == window

    def test_realized_vol_positive(self) -> None:
        """Realized volatility must be non-negative for all non-null values."""
        df: pl.DataFrame = make_random_walk_df(50, seed=1)
        lr_expr: pl.Expr = log_return(pl.col("close"), 1)
        df2: pl.DataFrame = df.with_columns(lr_expr.alias("lr"))
        result: pl.DataFrame = df2.with_columns(realized_vol(pl.col("lr"), 5).alias("rv"))
        non_null: list[float] = [v for v in result["rv"].to_list() if v is not None]
        assert all(v >= 0.0 for v in non_null)


class TestGarmanKlassVol:
    """Tests for the garman_klass_vol indicator."""

    def test_garman_klass_vol_known(self) -> None:
        """GK vol is zero when H=L=O=C (no intrabar variation).

        When high == low == open == close, both the HL and CO log terms are
        zero, so the GK variance is zero, and its square root is also zero.
        """
        n: int = 10
        df: pl.DataFrame = pl.DataFrame(
            {
                "open": [100.0] * n,
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
                "volume": [1.0] * n,
            }
        )
        from src.app.features.application.indicators import garman_klass_vol

        gk_result: pl.DataFrame = df.with_columns(garman_klass_vol(5).alias("gk"))
        non_null: list[float] = [v for v in gk_result["gk"].to_list() if v is not None]
        for v in non_null:
            assert v == pytest.approx(0.0, abs=1e-8)

    def test_garman_klass_vol_positive(self) -> None:
        """GK vol must be non-negative for all non-null values."""
        df: pl.DataFrame = make_random_walk_df(50, seed=2)
        from src.app.features.application.indicators import garman_klass_vol

        result: pl.DataFrame = df.with_columns(garman_klass_vol(5).alias("gk"))
        non_null: list[float] = [v for v in result["gk"].to_list() if v is not None]
        assert all(v >= 0.0 for v in non_null)


class TestParkinsonVol:
    """Tests for the parkinson_vol indicator."""

    def test_parkinson_vol_known(self) -> None:
        """Parkinson vol with H/L ratio of e^0.1 over window=1 gives known value.

        For a single bar: park_var = ln(H/L)^2 / (4*ln2)
        With H/L = exp(0.1): ln(H/L)^2 = 0.01
        sqrt(0.01 / (4*ln2)) = sqrt(0.01 / 2.7726) = sqrt(0.003607)
        """
        # Window of 1 so rolling mean is just the single value
        ratio: float = math.exp(0.1)
        h: float = 100.0 * ratio
        df: pl.DataFrame = pl.DataFrame(
            {
                "open": [100.0],
                "high": [h],
                "low": [100.0],
                "close": [100.0],
                "volume": [1.0],
            }
        )
        result: pl.DataFrame = df.with_columns(parkinson_vol(1).alias("pv"))
        pv: float | None = result["pv"].to_list()[0]
        assert pv is not None

        expected: float = math.sqrt(0.01 / (4.0 * math.log(2.0)))
        assert pv == pytest.approx(expected, rel=1e-5)

    def test_parkinson_vol_zero_for_equal_high_low(self) -> None:
        """When H == L, Parkinson vol is zero (no price range)."""
        n: int = 6
        df: pl.DataFrame = pl.DataFrame(
            {
                "open": [100.0] * n,
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
                "volume": [1.0] * n,
            }
        )
        result: pl.DataFrame = df.with_columns(parkinson_vol(5).alias("pv"))
        non_null: list[float] = [v for v in result["pv"].to_list() if v is not None]
        for v in non_null:
            assert v == pytest.approx(0.0, abs=1e-8)


class TestTrueRange:
    """Tests for the true_range indicator."""

    def test_true_range_basic(self) -> None:
        """Verify TR with explicit H=105, L=98, prev_close=103 → max(7, 2, 5) = 7."""
        # Row 0: no prev_close, TR = H - L = 5
        # Row 1: H=105, L=98, prev_close=100 → max(7, 5, 2) = 7
        df: pl.DataFrame = pl.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [105.0, 105.0],
                "low": [100.0, 98.0],
                "close": [100.0, 103.0],
                "volume": [1.0, 1.0],
            }
        )
        result: pl.DataFrame = df.with_columns(true_range(pl.col("high"), pl.col("low"), pl.col("close")).alias("tr"))
        tr_vals: list[float | None] = result["tr"].to_list()
        # Row 0: no prev close, tr = max(|H-L|, |H-None|, |L-None|) = |H-L|
        assert tr_vals[0] == pytest.approx(5.0)  # H=105, L=100
        # Row 1: H=105, L=98, prev_close=100 → max(7, 5, 2) = 7
        assert tr_vals[1] == pytest.approx(7.0)

    def test_true_range_nonnegative(self) -> None:
        """TR must always be non-negative."""
        df: pl.DataFrame = make_random_walk_df(30, seed=3)
        result: pl.DataFrame = df.with_columns(true_range(pl.col("high"), pl.col("low"), pl.col("close")).alias("tr"))
        non_null: list[float] = [v for v in result["tr"].to_list() if v is not None]
        assert all(v >= 0.0 for v in non_null)


class TestATR:
    """Tests for the atr indicator."""

    def test_atr_wilder_vs_simple_differ(self) -> None:
        """Wilder smoothing (EWM) and simple rolling ATR should differ on volatile data."""
        df: pl.DataFrame = make_random_walk_df(50, seed=4, volatility=10.0)
        result: pl.DataFrame = df.with_columns(
            atr(10, wilder=True).alias("atr_wilder"),
            atr(10, wilder=False).alias("atr_simple"),
        )
        atr_w: list[float | None] = result["atr_wilder"].to_list()
        atr_s: list[float | None] = result["atr_simple"].to_list()
        non_null_w: list[float] = [v for v in atr_w if v is not None]
        non_null_s: list[float] = [v for v in atr_s if v is not None]
        assert len(non_null_w) > 0
        assert len(non_null_s) > 0
        # At least some values must differ
        overlap_len: int = min(len(non_null_w), len(non_null_s))
        paired = zip(non_null_w[-overlap_len:], non_null_s[-overlap_len:], strict=True)
        diffs: list[float] = [abs(a - b) for a, b in paired]
        assert any(d > 1e-6 for d in diffs)

    def test_atr_nonnegative(self) -> None:
        """ATR must always be non-negative."""
        df: pl.DataFrame = make_random_walk_df(40, seed=5)
        result: pl.DataFrame = df.with_columns(atr(5).alias("atr_val"))
        non_null: list[float] = [v for v in result["atr_val"].to_list() if v is not None]
        assert all(v >= 0.0 for v in non_null)


class TestEMA:
    """Tests for the ema indicator."""

    def test_ema_known_values_span2(self) -> None:
        """EMA(span=2) on [1,2,3] with α=2/3 matches hand computation.

        α = 2/(2+1) = 2/3.
        EMA_0 = 1 (first value, no prior).
        EMA_1 = α*2 + (1-α)*1 = 4/3 + 1/3 = 5/3.
        EMA_2 = α*3 + (1-α)*5/3 = 2 + 5/9 = 23/9.
        """
        df: pl.DataFrame = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        alpha: float = 2.0 / 3.0
        result: pl.DataFrame = df.with_columns(ema(pl.col("x"), span=2).alias("ema"))
        ema_vals: list[float | None] = result["ema"].to_list()

        assert ema_vals[0] == pytest.approx(1.0, rel=_RTOL)
        assert ema_vals[1] == pytest.approx(alpha * 2.0 + (1.0 - alpha) * 1.0, rel=_RTOL)
        assert ema_vals[2] == pytest.approx(
            alpha * 3.0 + (1.0 - alpha) * (alpha * 2.0 + (1.0 - alpha) * 1.0),
            rel=_RTOL,
        )

    def test_ema_approaches_constant_input(self) -> None:
        """EMA of a constant series should equal the constant."""
        n: int = 50
        df: pl.DataFrame = pl.DataFrame({"x": [5.0] * n})
        result: pl.DataFrame = df.with_columns(ema(pl.col("x"), span=10).alias("ema"))
        ema_vals: list[float | None] = result["ema"].to_list()
        for v in ema_vals:
            assert v is not None
            assert v == pytest.approx(5.0, rel=_RTOL)


class TestEMACrossover:
    """Tests for the ema_crossover indicator."""

    def test_ema_crossover_positive_for_uptrend(self) -> None:
        """After sufficient warmup, fast EMA > slow EMA in uptrend → positive signal."""
        df: pl.DataFrame = make_trending_df(80, direction=1.0, step=1.0)
        result: pl.DataFrame = df.with_columns(ema_crossover(fast_span=5, slow_span=10, atr_period=5).alias("xover"))
        xover_vals: list[float | None] = result["xover"].to_list()
        non_null: list[float] = [v for v in xover_vals if v is not None]
        # Last 20 values should all be positive in a consistent uptrend
        assert all(v > 0 for v in non_null[-20:])

    def test_ema_crossover_negative_for_downtrend(self) -> None:
        """After warmup, fast EMA < slow EMA in downtrend → negative signal."""
        df: pl.DataFrame = make_trending_df(80, direction=-1.0, step=1.0, price_start=500.0)
        result: pl.DataFrame = df.with_columns(ema_crossover(fast_span=5, slow_span=10, atr_period=5).alias("xover"))
        xover_vals: list[float | None] = result["xover"].to_list()
        non_null: list[float] = [v for v in xover_vals if v is not None]
        assert all(v < 0 for v in non_null[-20:])


class TestRSI:
    """Tests for the rsi indicator."""

    def test_rsi_all_gains_approaches_100(self) -> None:
        """RSI should approach 100 when all price changes are positive."""
        prices: list[float] = [100.0 + i for i in range(50)]
        df: pl.DataFrame = make_ohlcv_df(50, price_start=100.0, price_step=1.0)
        result: pl.DataFrame = df.with_columns(rsi(7).alias("rsi_val"))
        rsi_vals: list[float | None] = result["rsi_val"].to_list()
        non_null: list[float] = [v for v in rsi_vals if v is not None]
        # After many consistent gains, RSI should be well above 90
        assert non_null[-1] > 90.0
        del prices

    def test_rsi_all_losses_approaches_0(self) -> None:
        """RSI should approach 0 when all price changes are negative."""
        df: pl.DataFrame = make_ohlcv_df(50, price_start=500.0, price_step=-1.0)
        result: pl.DataFrame = df.with_columns(rsi(7).alias("rsi_val"))
        rsi_vals: list[float | None] = result["rsi_val"].to_list()
        non_null: list[float] = [v for v in rsi_vals if v is not None]
        assert non_null[-1] < 10.0

    def test_rsi_range(self) -> None:
        """All RSI values must be in [0, 100]."""
        df: pl.DataFrame = make_random_walk_df(100, seed=6)
        result: pl.DataFrame = df.with_columns(rsi(14).alias("rsi_val"))
        non_null: list[float] = [v for v in result["rsi_val"].to_list() if v is not None]
        assert all(0.0 <= v <= 100.0 for v in non_null)

    def test_rsi_neutral_for_alternating(self) -> None:
        """Alternating gains and losses should produce RSI near 50."""
        # Alternating ±1 changes from a flat base
        closes: list[float] = [100.0 + (1 if i % 2 == 0 else -1) for i in range(100)]
        df: pl.DataFrame = pl.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000.0] * 100,
            }
        )
        result: pl.DataFrame = df.with_columns(rsi(14).alias("rsi_val"))
        non_null: list[float] = [v for v in result["rsi_val"].to_list() if v is not None]
        # Mean should be near 50 for alternating gains and losses
        mean_rsi: float = float(np.mean(non_null))
        assert 40.0 < mean_rsi < 60.0


class TestROC:
    """Tests for the roc indicator."""

    def test_roc_known_values(self) -> None:
        """ROC with period=1 on [100, 110] equals 0.1."""
        df: pl.DataFrame = pl.DataFrame({"close": [100.0, 110.0]})
        result: pl.DataFrame = df.with_columns(roc(pl.col("close"), 1).alias("roc_val"))
        roc_vals: list[float | None] = result["roc_val"].to_list()
        assert roc_vals[0] is None
        assert roc_vals[1] == pytest.approx(0.1, rel=_RTOL)

    def test_roc_zero_for_unchanged(self) -> None:
        """ROC is 0 when prices are unchanged."""
        df: pl.DataFrame = pl.DataFrame({"close": [50.0, 50.0, 50.0]})
        result: pl.DataFrame = df.with_columns(roc(pl.col("close"), 1).alias("roc_val"))
        non_null: list[float] = [v for v in result["roc_val"].to_list() if v is not None]
        for v in non_null:
            assert v == pytest.approx(0.0, abs=1e-10)


class TestRollingSlope:
    """Tests for the rolling_slope indicator."""

    def test_rolling_slope_linear_trend_positive(self) -> None:
        """Strictly increasing prices should yield a positive rolling slope."""
        n: int = 30
        df: pl.DataFrame = make_trending_df(n, direction=1.0, step=2.0)
        result: pl.DataFrame = df.with_columns(rolling_slope(pl.col("close"), 5).alias("slope"))
        non_null: list[float] = [v for v in result["slope"].to_list() if v is not None]
        assert len(non_null) > 0
        assert all(v > 0 for v in non_null)

    def test_rolling_slope_negative_for_downtrend(self) -> None:
        """Strictly decreasing prices should yield a negative rolling slope."""
        df: pl.DataFrame = make_trending_df(30, direction=-1.0, step=2.0, price_start=200.0)
        result: pl.DataFrame = df.with_columns(rolling_slope(pl.col("close"), 5).alias("slope"))
        non_null: list[float] = [v for v in result["slope"].to_list() if v is not None]
        assert len(non_null) > 0
        assert all(v < 0 for v in non_null)


class TestVolumeZscore:
    """Tests for the volume_zscore indicator."""

    def test_volume_zscore_constant_volume(self) -> None:
        """Constant volume produces z-score of zero (std ≈ 0, so z ≈ 0)."""
        n: int = 30
        df: pl.DataFrame = make_ohlcv_df(n, volume=500.0)
        result: pl.DataFrame = df.with_columns(volume_zscore(10).alias("vz"))
        non_null: list[float] = [v for v in result["vz"].to_list() if v is not None]
        for v in non_null:
            assert v == pytest.approx(0.0, abs=1e-6)

    def test_volume_zscore_shape(self) -> None:
        """Output length equals input length."""
        n: int = 40
        df: pl.DataFrame = make_random_walk_df(n, seed=7)
        result: pl.DataFrame = df.with_columns(volume_zscore(10).alias("vz"))
        assert len(result) == n


class TestOBVSlope:
    """Tests for the obv_slope indicator."""

    def test_obv_slope_bullish_positive(self) -> None:
        """All bullish candles (close > prev_close) → positive OBV → positive slope."""
        # Strictly monotonically increasing prices → all diffs positive
        df: pl.DataFrame = make_trending_df(30, direction=1.0, step=10.0)
        result: pl.DataFrame = df.with_columns(obv_slope(5).alias("obv_s"))
        non_null: list[float] = [v for v in result["obv_s"].to_list() if v is not None]
        assert len(non_null) > 0
        assert all(v > 0 for v in non_null)

    def test_obv_slope_bearish_negative(self) -> None:
        """All bearish candles → decreasing OBV → negative slope."""
        df: pl.DataFrame = make_trending_df(30, direction=-1.0, step=10.0, price_start=1000.0)
        result: pl.DataFrame = df.with_columns(obv_slope(5).alias("obv_s"))
        non_null: list[float] = [v for v in result["obv_s"].to_list() if v is not None]
        assert len(non_null) > 0
        assert all(v < 0 for v in non_null)


class TestAmihudIlliquidity:
    """Tests for the amihud_illiquidity indicator."""

    def test_amihud_nonnegative(self) -> None:
        """Amihud illiquidity must always be non-negative."""
        df: pl.DataFrame = make_random_walk_df(50, seed=8)
        from src.app.features.application.indicators import amihud_illiquidity

        result: pl.DataFrame = df.with_columns(amihud_illiquidity(10).alias("amihud"))
        non_null: list[float] = [v for v in result["amihud"].to_list() if v is not None]
        assert all(v >= 0.0 for v in non_null)

    def test_amihud_higher_volume_lower_illiquidity(self) -> None:
        """Higher volume (everything else equal) should lower Amihud illiquidity."""
        from src.app.features.application.indicators import amihud_illiquidity

        df_low: pl.DataFrame = make_ohlcv_df(30, price_step=1.0, volume=100.0)
        df_high: pl.DataFrame = make_ohlcv_df(30, price_step=1.0, volume=100_000.0)

        res_low: pl.DataFrame = df_low.with_columns(amihud_illiquidity(10).alias("a"))
        res_high: pl.DataFrame = df_high.with_columns(amihud_illiquidity(10).alias("a"))

        mean_low: float = float(pl.Series([v for v in res_low["a"].to_list() if v is not None]).mean())
        mean_high: float = float(pl.Series([v for v in res_high["a"].to_list() if v is not None]).mean())

        assert mean_high < mean_low


class TestReturnZscore:
    """Tests for the return_zscore indicator."""

    def test_return_zscore_shape(self) -> None:
        """Output length equals input length."""
        n: int = 40
        df: pl.DataFrame = make_random_walk_df(n, seed=9)
        from src.app.features.application.indicators import return_zscore

        result: pl.DataFrame = df.with_columns(return_zscore(10).alias("rz"))
        assert len(result) == n

    def test_return_zscore_clipped_for_constant(self) -> None:
        """Return z-score of constant prices (zero returns) is approximately 0."""
        n: int = 30
        df: pl.DataFrame = make_ohlcv_df(n, price_step=0.0)
        from src.app.features.application.indicators import return_zscore

        result: pl.DataFrame = df.with_columns(return_zscore(10).alias("rz"))
        non_null: list[float] = [v for v in result["rz"].to_list() if v is not None]
        for v in non_null:
            assert v == pytest.approx(0.0, abs=1e-6)


class TestBollingerPctB:
    """Tests for the bollinger_pct_b indicator."""

    def test_bollinger_pctb_at_mean_is_half(self) -> None:
        """When price == rolling mean, %B should equal ~0.5.

        We construct a symmetric window by alternating above and below the mean
        so that the final value equals exactly the mean, placing it at %B == 0.5.
        We verify this by checking that prices at the rolling mean yield %B near 0.5.
        """
        # Build a series: 9 values alternating ±1, then a value of exactly the
        # mean (0).  With a window of 10, the final position is at the mean.
        closes: list[float] = [100.0, 102.0, 98.0, 102.0, 98.0, 102.0, 98.0, 102.0, 98.0, 100.0]
        df: pl.DataFrame = pl.DataFrame(
            {
                "open": closes,
                "high": [c + 1.0 for c in closes],
                "low": [c - 1.0 for c in closes],
                "close": closes,
                "volume": [1000.0] * len(closes),
            }
        )
        result: pl.DataFrame = df.with_columns(bollinger_pct_b(10, 2.0).alias("pctb"))
        pctb_vals: list[float | None] = result["pctb"].to_list()
        # Last value should be at %B ≈ 0.5 (price at rolling mean)
        last_val: float | None = pctb_vals[-1]
        assert last_val is not None
        assert last_val == pytest.approx(0.5, abs=0.05)

    def test_bollinger_pctb_finite(self) -> None:
        """Bollinger %B values must be finite."""
        df: pl.DataFrame = make_random_walk_df(50, seed=10)
        result: pl.DataFrame = df.with_columns(bollinger_pct_b(10).alias("pctb"))
        non_null: list[float] = [v for v in result["pctb"].to_list() if v is not None]
        assert all(math.isfinite(v) for v in non_null)


class TestBollingerWidth:
    """Tests for the bollinger_width indicator."""

    def test_bollinger_width_zero_for_constant(self) -> None:
        """Constant prices have zero std → Bollinger width ≈ 0."""
        n: int = 30
        df: pl.DataFrame = make_ohlcv_df(n, price_step=0.0, price_start=100.0)
        result: pl.DataFrame = df.with_columns(bollinger_width(10).alias("bbw"))
        non_null: list[float] = [v for v in result["bbw"].to_list() if v is not None]
        for v in non_null:
            assert v == pytest.approx(0.0, abs=1e-6)

    def test_bollinger_width_higher_vol_wider(self) -> None:
        """Higher price volatility should produce wider Bollinger bands."""
        df_low: pl.DataFrame = make_random_walk_df(50, seed=11, volatility=0.1)
        df_high: pl.DataFrame = make_random_walk_df(50, seed=11, volatility=50.0)

        res_low: pl.DataFrame = df_low.with_columns(bollinger_width(10).alias("bbw"))
        res_high: pl.DataFrame = df_high.with_columns(bollinger_width(10).alias("bbw"))

        mean_low: float = float(pl.Series([v for v in res_low["bbw"].to_list() if v is not None]).mean())
        mean_high: float = float(pl.Series([v for v in res_high["bbw"].to_list() if v is not None]).mean())

        assert mean_high > mean_low


class TestZscoreRolling:
    """Tests for the zscore_rolling utility."""

    def test_zscore_known_values(self) -> None:
        """Z-score of a known series matches hand computation."""
        # For a window of 3 over [1, 2, 3]: mean=2, std=1, z for '3' = (3-2)/1 = 1.0
        df: pl.DataFrame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result: pl.DataFrame = df.with_columns(zscore_rolling(pl.col("x"), 3).alias("z"))
        z_vals: list[float | None] = result["z"].to_list()
        # At index 2: window=[1,2,3], mean=2, std=1, z=(3-2)/1=1
        assert z_vals[2] == pytest.approx(1.0, rel=1e-5)
        # At index 3: window=[2,3,4], mean=3, std=1, z=(4-3)/1=1
        assert z_vals[3] == pytest.approx(1.0, rel=1e-5)

    def test_zscore_mean_near_zero(self) -> None:
        """Rolling z-score of a stationary series should have mean near 0."""
        df: pl.DataFrame = make_random_walk_df(100, seed=12)
        result: pl.DataFrame = df.with_columns(zscore_rolling(pl.col("close"), 20).alias("z"))
        non_null: list[float] = [v for v in result["z"].to_list() if v is not None]
        mean_z: float = float(np.mean(non_null))
        assert abs(mean_z) < 1.0  # loose bound — just confirming centering


class TestClipExpr:
    """Tests for the clip_expr utility."""

    def test_clip_lower_bound(self) -> None:
        """Values below lo are clipped to lo."""
        df: pl.DataFrame = pl.DataFrame({"x": [-10.0, -5.0, 0.0, 5.0, 10.0]})
        result: pl.DataFrame = df.with_columns(clip_expr(pl.col("x"), lo=-3.0, hi=3.0).alias("c"))
        clipped: list[float | None] = result["c"].to_list()
        assert clipped[0] == pytest.approx(-3.0)
        assert clipped[1] == pytest.approx(-3.0)
        assert clipped[2] == pytest.approx(0.0)

    def test_clip_upper_bound(self) -> None:
        """Values above hi are clipped to hi."""
        df: pl.DataFrame = pl.DataFrame({"x": [1.0, 2.0, 5.0, 10.0]})
        result: pl.DataFrame = df.with_columns(clip_expr(pl.col("x"), lo=-5.0, hi=3.0).alias("c"))
        clipped: list[float | None] = result["c"].to_list()
        assert clipped[2] == pytest.approx(3.0)
        assert clipped[3] == pytest.approx(3.0)

    def test_clip_passthrough_within_range(self) -> None:
        """Values within range pass through unchanged."""
        df: pl.DataFrame = pl.DataFrame({"x": [-2.0, 0.0, 2.0]})
        result: pl.DataFrame = df.with_columns(clip_expr(pl.col("x"), lo=-5.0, hi=5.0).alias("c"))
        clipped: list[float | None] = result["c"].to_list()
        for original, clipped_val in zip([-2.0, 0.0, 2.0], clipped, strict=True):
            assert clipped_val == pytest.approx(original)


class TestComputeAllIndicators:
    """Tests for the compute_all_indicators orchestrator."""

    def test_compute_all_indicators_columns_present(self) -> None:
        """All expected feature column families should be present after compute."""
        df: pl.DataFrame = make_random_walk_df(150, seed=13)
        config: IndicatorConfig = make_small_indicator_config()
        result: pl.DataFrame = compute_all_indicators(df, config)
        cols: set[str] = set(result.columns)

        # Spot-check representative columns from each group
        expected_prefixes: list[str] = [
            "logret_",
            "rv_",
            "gk_vol_",
            "park_vol_",
            "atr_",
            "ema_xover_",
            "rsi_",
            "roc_",
            "vol_zscore_",
            "amihud_",
            "ret_zscore_",
            "bbpctb_",
            "bbwidth_",
            "slope_",
            "obv_slope_",
            "hurst_",
        ]
        for prefix in expected_prefixes:
            assert any(c.startswith(prefix) for c in cols), f"No column starting with '{prefix}' found"

    def test_compute_all_indicators_preserves_ohlcv(self) -> None:
        """Original OHLCV columns must be preserved in the output."""
        df: pl.DataFrame = make_random_walk_df(150, seed=14)
        config: IndicatorConfig = make_small_indicator_config()
        result: pl.DataFrame = compute_all_indicators(df, config)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_compute_all_indicators_same_row_count(self) -> None:
        """Row count must be preserved (indicators don't drop rows)."""
        n: int = 150
        df: pl.DataFrame = make_random_walk_df(n, seed=15)
        config: IndicatorConfig = make_small_indicator_config()
        result: pl.DataFrame = compute_all_indicators(df, config)
        assert len(result) == n

    def test_compute_all_indicators_clipped(self) -> None:
        """All feature columns must be within [clip_lower, clip_upper]."""
        df: pl.DataFrame = make_random_walk_df(150, seed=16)
        config: IndicatorConfig = make_small_indicator_config(clip_lower=-5.0, clip_upper=5.0)
        result: pl.DataFrame = compute_all_indicators(df, config)
        ohlcv_cols: set[str] = {"timestamp", "open", "high", "low", "close", "volume"}
        for col in result.columns:
            if col in ohlcv_cols:
                continue
            series: pl.Series = result[col].drop_nulls()
            if len(series) == 0:
                continue
            col_min: float = float(series.min())  # type: ignore[arg-type]
            col_max: float = float(series.max())  # type: ignore[arg-type]
            assert col_min >= config.clip_lower - 1e-6, f"{col} min {col_min} < {config.clip_lower}"
            assert col_max <= config.clip_upper + 1e-6, f"{col} max {col_max} > {config.clip_upper}"
