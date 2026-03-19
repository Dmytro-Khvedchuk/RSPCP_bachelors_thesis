"""Tests for the ProfilingService orchestrator, BH correction, and p-value extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.app.profiling.application.services import (
    ProfilingService,
    _apply_bh_correction,
    _extract_inferential_pvalues,
)
from src.app.profiling.domain.value_objects import (
    AssetBarProfile,
    AutocorrelationConfig,
    AutocorrelationProfile,
    BDSResult,
    CorrectedPValue,
    DistributionConfig,
    LjungBoxResult,
    PredictabilityConfig,
    ProfilingConfig,
    SampleTier,
    SignBiasResult,
    StatisticalReport,
    TierConfig,
    VolatilityConfig,
    VolatilityProfile,
)


# ---------------------------------------------------------------------------
# FakeDataLoader — duck-typed replacement for DataLoader
# ---------------------------------------------------------------------------


class FakeDataLoader:
    """Fake DataLoader returning synthetic data for testing without a database."""

    def __init__(
        self,
        ohlcv_data: dict[tuple[str, str], pd.DataFrame] | None = None,
        bar_data: dict[tuple[str, str, str], pd.DataFrame] | None = None,
    ) -> None:
        """Initialise with pre-built dictionaries of OHLCV and bar data.

        Args:
            ohlcv_data: Mapping from ``(asset, timeframe)`` to DataFrame.
            bar_data: Mapping from ``(asset, bar_type, config_hash)`` to DataFrame.
        """
        self._ohlcv: dict[tuple[str, str], pd.DataFrame] = ohlcv_data or {}
        self._bars: dict[tuple[str, str, str], pd.DataFrame] = bar_data or {}

    def load_ohlcv(
        self,
        asset: str,
        timeframe: str,
        date_range: tuple | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        """Return OHLCV data for the given asset and timeframe.

        Args:
            asset: Trading pair symbol.
            timeframe: Candlestick interval.
            date_range: Ignored in fake.

        Returns:
            DataFrame with OHLCV columns, or empty DataFrame.
        """
        key: tuple[str, str] = (asset, timeframe)
        if key in self._ohlcv:
            return self._ohlcv[key].copy()
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def load_bars(
        self,
        asset: str,
        bar_type: str,
        config_hash: str,
        date_range: tuple | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        """Return bar data for the given asset, bar type, and config hash.

        Args:
            asset: Trading pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Bar configuration hash.
            date_range: Ignored in fake.

        Returns:
            DataFrame with bar columns, or empty DataFrame.
        """
        key: tuple[str, str, str] = (asset, bar_type, config_hash)
        if key in self._bars:
            return self._bars[key].copy()
        return pd.DataFrame(
            columns=[
                "start_ts",
                "end_ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tick_count",
                "buy_volume",
                "sell_volume",
                "vwap",
            ]
        )

    def get_available_assets(self) -> list[str]:
        """Return sorted list of unique asset symbols across OHLCV and bar data.

        Returns:
            Sorted list of asset symbol strings.
        """
        assets: set[str] = set()
        for asset, _ in self._ohlcv:
            assets.add(asset)
        for asset, _, _ in self._bars:
            assets.add(asset)
        return sorted(assets)

    def get_available_bar_configs(self, asset: str) -> list[tuple[str, str]]:
        """Return bar configs for the given asset.

        Args:
            asset: Trading pair symbol.

        Returns:
            Sorted list of ``(bar_type, config_hash)`` tuples.
        """
        configs: list[tuple[str, str]] = []
        for a, bt, ch in self._bars:
            if a == asset:
                configs.append((bt, ch))
        return sorted(configs)


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------


def _make_ohlcv_df(
    n: int = 2500,
    seed: int = 42,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Args:
        n: Number of bars to generate.
        seed: Random seed for reproducibility.
        ts_col: Name for the timestamp column.

    Returns:
        DataFrame with timestamp, OHLCV columns.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    timestamps: pd.DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    close: np.ndarray = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))  # type: ignore[type-arg]
    return pd.DataFrame(
        {
            ts_col: timestamps,
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + abs(rng.normal(0, 0.002, n))),
            "low": close * (1 - abs(rng.normal(0, 0.002, n))),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        }
    )


def _make_bar_df(
    n: int = 2500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic bar data for testing.

    Args:
        n: Number of bars to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with start_ts, end_ts, OHLCV, and bar-specific columns.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    timestamps: pd.DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    close: np.ndarray = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))  # type: ignore[type-arg]
    return pd.DataFrame(
        {
            "start_ts": timestamps,
            "end_ts": timestamps + pd.Timedelta(hours=1),
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + abs(rng.normal(0, 0.002, n))),
            "low": close * (1 - abs(rng.normal(0, 0.002, n))),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
            "tick_count": rng.integers(10, 100, n),
            "buy_volume": rng.uniform(50, 500, n),
            "sell_volume": rng.uniform(50, 500, n),
            "vwap": close * (1 + rng.normal(0, 0.0005, n)),
        }
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_loader_single() -> FakeDataLoader:
    """Build a FakeDataLoader with one asset, one bar type, and one OHLCV timeframe.

    Returns:
        FakeDataLoader pre-loaded with BTCUSDT data.
    """
    ohlcv: dict[tuple[str, str], pd.DataFrame] = {("BTCUSDT", "1h"): _make_ohlcv_df(n=2500, seed=42)}
    bars: dict[tuple[str, str, str], pd.DataFrame] = {
        ("BTCUSDT", "dollar", "abc123"): _make_bar_df(n=2500, seed=43),
    }
    return FakeDataLoader(ohlcv_data=ohlcv, bar_data=bars)


@pytest.fixture
def fake_loader_multi() -> FakeDataLoader:
    """Build a FakeDataLoader with two assets, each with one bar type.

    Returns:
        FakeDataLoader with BTCUSDT and ETHUSDT data.
    """
    ohlcv: dict[tuple[str, str], pd.DataFrame] = {
        ("BTCUSDT", "1h"): _make_ohlcv_df(n=1500, seed=42),
        ("ETHUSDT", "1h"): _make_ohlcv_df(n=1500, seed=43),
    }
    bars: dict[tuple[str, str, str], pd.DataFrame] = {
        ("BTCUSDT", "dollar", "abc123"): _make_bar_df(n=1500, seed=44),
        ("ETHUSDT", "volume", "def456"): _make_bar_df(n=1500, seed=45),
    }
    return FakeDataLoader(ohlcv_data=ohlcv, bar_data=bars)


@pytest.fixture
def fake_loader_empty() -> FakeDataLoader:
    """Build a FakeDataLoader with no data at all.

    Returns:
        Empty FakeDataLoader.
    """
    return FakeDataLoader()


@pytest.fixture
def fake_loader_no_bars() -> FakeDataLoader:
    """Build a FakeDataLoader with OHLCV data but no aggregated bars.

    Returns:
        FakeDataLoader with only BTCUSDT OHLCV data.
    """
    ohlcv: dict[tuple[str, str], pd.DataFrame] = {("BTCUSDT", "1h"): _make_ohlcv_df(n=1000, seed=42)}
    return FakeDataLoader(ohlcv_data=ohlcv, bar_data={})


# ---------------------------------------------------------------------------
# Test: ProfilingConfig
# ---------------------------------------------------------------------------


class TestProfilingConfig:
    """Tests for the ProfilingConfig composite config model."""

    def test_defaults(self) -> None:
        """All sub-configs should use their default values."""
        cfg: ProfilingConfig = ProfilingConfig()
        assert isinstance(cfg.distribution, DistributionConfig)
        assert isinstance(cfg.autocorrelation, AutocorrelationConfig)
        assert isinstance(cfg.volatility, VolatilityConfig)
        assert isinstance(cfg.predictability, PredictabilityConfig)
        assert isinstance(cfg.tier, TierConfig)
        assert cfg.fdr_alpha == 0.05
        assert cfg.stationarity_alpha == 0.05

    def test_immutability(self) -> None:
        """Frozen model should reject attribute assignment."""
        cfg: ProfilingConfig = ProfilingConfig()
        with pytest.raises(Exception):  # noqa: B017, PT011
            cfg.fdr_alpha = 0.1  # type: ignore[misc]

    def test_custom_sub_configs(self) -> None:
        """Custom sub-config values should propagate correctly."""
        custom_dist: DistributionConfig = DistributionConfig(jb_alpha=0.01)
        custom_tier: TierConfig = TierConfig(tier_a_threshold=3000, tier_b_threshold=800)
        cfg: ProfilingConfig = ProfilingConfig(
            distribution=custom_dist,
            tier=custom_tier,
            fdr_alpha=0.10,
        )
        assert cfg.distribution.jb_alpha == 0.01
        assert cfg.tier.tier_a_threshold == 3000
        assert cfg.fdr_alpha == 0.10


# ---------------------------------------------------------------------------
# Test: CorrectedPValue
# ---------------------------------------------------------------------------


class TestCorrectedPValue:
    """Tests for the CorrectedPValue frozen model."""

    def test_creation(self) -> None:
        """Valid CorrectedPValue should be constructable with all fields."""
        pv: CorrectedPValue = CorrectedPValue(
            asset="BTCUSDT",
            bar_type="dollar",
            test_name="ljung_box_returns",
            parameter="lag=5",
            raw_pvalue=0.03,
            corrected_pvalue=0.06,
            significant_raw=True,
            significant_corrected=False,
        )
        assert pv.asset == "BTCUSDT"
        assert pv.raw_pvalue == 0.03

    def test_fields_accessible(self) -> None:
        """All fields should be accessible on a frozen instance."""
        pv: CorrectedPValue = CorrectedPValue(
            asset="ETHUSDT",
            bar_type="volume",
            test_name="bds",
            parameter="dim=3",
            raw_pvalue=0.001,
            corrected_pvalue=0.005,
            significant_raw=True,
            significant_corrected=True,
        )
        assert pv.test_name == "bds"
        assert pv.parameter == "dim=3"
        assert pv.corrected_pvalue == 0.005
        assert pv.significant_corrected is True


# ---------------------------------------------------------------------------
# Test: BH Correction
# ---------------------------------------------------------------------------


class TestBHCorrection:
    """Tests for the Benjamini-Hochberg FDR correction helper."""

    def _make_pv(self, raw: float, test: str = "test", param: str = "p") -> CorrectedPValue:
        """Build a CorrectedPValue with raw significance at alpha=0.05.

        Args:
            raw: Raw p-value.
            test: Test name.
            param: Parameter string.

        Returns:
            CorrectedPValue with corrected_pvalue initialised to raw.
        """
        return CorrectedPValue(
            asset="BTCUSDT",
            bar_type="dollar",
            test_name=test,
            parameter=param,
            raw_pvalue=raw,
            corrected_pvalue=raw,
            significant_raw=raw < 0.05,
            significant_corrected=raw < 0.05,
        )

    def test_single_pvalue(self) -> None:
        """Single test correction should equal the raw p-value."""
        raw: list[CorrectedPValue] = [self._make_pv(0.03)]
        corrected: tuple[CorrectedPValue, ...] = _apply_bh_correction(raw, alpha=0.05)
        assert len(corrected) == 1
        assert corrected[0].corrected_pvalue == pytest.approx(0.03, abs=1e-10)

    def test_multiple_pvalues_corrected_ge_raw(self) -> None:
        """Corrected p-values should always be >= raw p-values."""
        raw: list[CorrectedPValue] = [
            self._make_pv(0.01, param="a"),
            self._make_pv(0.03, param="b"),
            self._make_pv(0.04, param="c"),
            self._make_pv(0.50, param="d"),
        ]
        corrected: tuple[CorrectedPValue, ...] = _apply_bh_correction(raw, alpha=0.05)
        assert len(corrected) == 4
        for orig, corr in zip(raw, corrected, strict=True):
            assert corr.corrected_pvalue >= orig.raw_pvalue - 1e-10

    def test_monotonicity(self) -> None:
        """Corrected p-values sorted by raw should be monotonically non-decreasing."""
        raw: list[CorrectedPValue] = [
            self._make_pv(0.001, param="a"),
            self._make_pv(0.01, param="b"),
            self._make_pv(0.02, param="c"),
            self._make_pv(0.10, param="d"),
        ]
        corrected: tuple[CorrectedPValue, ...] = _apply_bh_correction(raw, alpha=0.05)
        sorted_corr: list[CorrectedPValue] = sorted(corrected, key=lambda x: x.raw_pvalue)
        for i in range(len(sorted_corr) - 1):
            assert sorted_corr[i].corrected_pvalue <= sorted_corr[i + 1].corrected_pvalue + 1e-10

    def test_empty_input(self) -> None:
        """Empty input should return an empty tuple."""
        corrected: tuple[CorrectedPValue, ...] = _apply_bh_correction([], alpha=0.05)
        assert corrected == ()


# ---------------------------------------------------------------------------
# Test: P-value extraction
# ---------------------------------------------------------------------------


class TestPValueExtraction:
    """Tests for _extract_inferential_pvalues."""

    def _make_lb_results(self) -> tuple[LjungBoxResult, ...]:
        """Build two Ljung-Box results (one significant, one not).

        Returns:
            Tuple of two LjungBoxResult objects.
        """
        return (
            LjungBoxResult(lag=5, q_statistic=10.0, p_value=0.02, significant=True),
            LjungBoxResult(lag=10, q_statistic=15.0, p_value=0.08, significant=False),
        )

    def test_extracts_ljung_box(self) -> None:
        """Ljung-Box p-values from both returns and squared returns should be extracted."""
        acf_profile: AutocorrelationProfile = AutocorrelationProfile(
            asset="BTCUSDT",
            bar_type="dollar",
            tier=SampleTier.A,
            n_observations=1000,
            acf_values=np.array([1.0, 0.1]),
            pacf_values=np.array([1.0, 0.1]),
            acf_squared_values=np.array([1.0, 0.2]),
            pacf_squared_values=np.array([1.0, 0.2]),
            ljung_box_returns=self._make_lb_results(),
            ljung_box_squared=self._make_lb_results(),
            has_serial_correlation=True,
            has_volatility_clustering=True,
        )
        profile: AssetBarProfile = AssetBarProfile(
            asset="BTCUSDT",
            bar_type="dollar",
            tier=SampleTier.A,
            n_observations=1000,
            autocorrelation=acf_profile,
        )
        pvals: list[CorrectedPValue] = _extract_inferential_pvalues(profile)
        lb_pvals: list[CorrectedPValue] = [p for p in pvals if p.test_name.startswith("ljung_box")]
        assert len(lb_pvals) == 4  # 2 returns + 2 squared

    def test_extracts_bds(self) -> None:
        """BDS test p-values should be extracted from the volatility sub-profile."""
        bds_results: tuple[BDSResult, ...] = (
            BDSResult(dimension=2, bds_statistic=3.0, p_value=0.001, significant=True),
            BDSResult(dimension=3, bds_statistic=2.5, p_value=0.01, significant=True),
        )
        vol_profile: VolatilityProfile = VolatilityProfile(
            asset="BTCUSDT",
            bar_type="time_1h",
            tier=SampleTier.A,
            n_observations=2000,
            is_time_bar=True,
            bds_results=bds_results,
        )
        profile: AssetBarProfile = AssetBarProfile(
            asset="BTCUSDT",
            bar_type="time_1h",
            tier=SampleTier.A,
            n_observations=2000,
            volatility=vol_profile,
        )
        pvals: list[CorrectedPValue] = _extract_inferential_pvalues(profile)
        bds_pvals: list[CorrectedPValue] = [p for p in pvals if p.test_name == "bds"]
        assert len(bds_pvals) == 2

    def test_extracts_sign_bias_and_arch_lm(self) -> None:
        """Sign bias and ARCH-LM p-values should be extracted from volatility sub-profile."""
        sign_bias: SignBiasResult = SignBiasResult(
            sign_bias_tstat=2.1,
            sign_bias_pvalue=0.03,
            neg_size_bias_tstat=1.5,
            neg_size_bias_pvalue=0.13,
            pos_size_bias_tstat=0.8,
            pos_size_bias_pvalue=0.42,
            joint_f_stat=3.5,
            joint_f_pvalue=0.01,
            has_leverage_effect=True,
        )
        vol_profile: VolatilityProfile = VolatilityProfile(
            asset="BTCUSDT",
            bar_type="time_1h",
            tier=SampleTier.A,
            n_observations=2000,
            is_time_bar=True,
            arch_lm_stat=15.0,
            arch_lm_pvalue=0.001,
            sign_bias=sign_bias,
        )
        profile: AssetBarProfile = AssetBarProfile(
            asset="BTCUSDT",
            bar_type="time_1h",
            tier=SampleTier.A,
            n_observations=2000,
            volatility=vol_profile,
        )
        pvals: list[CorrectedPValue] = _extract_inferential_pvalues(profile)
        arch_pvals: list[CorrectedPValue] = [p for p in pvals if p.test_name == "arch_lm"]
        sb_pvals: list[CorrectedPValue] = [p for p in pvals if p.test_name == "sign_bias_joint"]
        assert len(arch_pvals) == 1
        assert len(sb_pvals) == 1

    def test_none_profiles_skipped(self) -> None:
        """Profiles with no sub-profiles should produce zero p-values."""
        profile: AssetBarProfile = AssetBarProfile(
            asset="BTCUSDT",
            bar_type="dollar",
            tier=SampleTier.C,
            n_observations=100,
        )
        pvals: list[CorrectedPValue] = _extract_inferential_pvalues(profile)
        assert len(pvals) == 0


# ---------------------------------------------------------------------------
# Test: ProfilingService
# ---------------------------------------------------------------------------


class TestProfilingService:
    """Tests for the ProfilingService orchestrator."""

    def test_profile_single(self, fake_loader_single: FakeDataLoader) -> None:
        """Single asset-bar profile should have all sub-profiles populated."""
        service: ProfilingService = ProfilingService(fake_loader_single)  # type: ignore[arg-type]
        profile: AssetBarProfile = service.profile_single(
            asset="BTCUSDT",
            bar_type="dollar",
            config_hash="abc123",
        )
        assert profile.asset == "BTCUSDT"
        assert profile.bar_type == "dollar"
        assert profile.n_observations > 0
        assert profile.distribution is not None
        assert profile.autocorrelation is not None
        assert profile.volatility is not None
        assert profile.predictability is not None

    def test_profile_single_time_bar(self, fake_loader_single: FakeDataLoader) -> None:
        """Time_1h bar should be loaded from OHLCV data successfully."""
        service: ProfilingService = ProfilingService(fake_loader_single)  # type: ignore[arg-type]
        profile: AssetBarProfile = service.profile_single(
            asset="BTCUSDT",
            bar_type="time_1h",
        )
        assert profile.asset == "BTCUSDT"
        assert profile.bar_type == "time_1h"
        assert profile.n_observations > 0

    def test_profile_all_multiple_assets(self, fake_loader_multi: FakeDataLoader) -> None:
        """Multiple assets should produce correct number of profiles."""
        service: ProfilingService = ProfilingService(fake_loader_multi)  # type: ignore[arg-type]
        report: StatisticalReport = service.profile_all()
        # 2 assets: BTCUSDT (dollar + time_1h), ETHUSDT (volume + time_1h) = 4 profiles
        assert len(report.profiles) == 4
        assert report.n_assets == 2

    def test_tier_classification(self, fake_loader_single: FakeDataLoader) -> None:
        """2500 bars minus 1 for log returns should classify as Tier A."""
        service: ProfilingService = ProfilingService(fake_loader_single)  # type: ignore[arg-type]
        profile: AssetBarProfile = service.profile_single(
            asset="BTCUSDT",
            bar_type="dollar",
            config_hash="abc123",
        )
        assert profile.tier == SampleTier.A

    def test_statistical_report_summary(self, fake_loader_single: FakeDataLoader) -> None:
        """Summary counts should be internally consistent after FDR correction."""
        service: ProfilingService = ProfilingService(fake_loader_single)  # type: ignore[arg-type]
        report: StatisticalReport = service.profile_all(assets=["BTCUSDT"])
        # 1 asset: dollar + time_1h = 2 profiles
        assert len(report.profiles) == 2
        assert report.n_total_tests >= 0
        assert report.n_significant_raw >= report.n_significant_corrected

    def test_fdr_correction_applied(self, fake_loader_single: FakeDataLoader) -> None:
        """Corrected p-values should exist and be >= raw p-values."""
        service: ProfilingService = ProfilingService(fake_loader_single)  # type: ignore[arg-type]
        report: StatisticalReport = service.profile_all(assets=["BTCUSDT"])
        assert len(report.corrected_pvalues) > 0
        for cpv in report.corrected_pvalues:
            assert cpv.corrected_pvalue >= cpv.raw_pvalue - 1e-10


# ---------------------------------------------------------------------------
# Test: StatisticalReport
# ---------------------------------------------------------------------------


class TestStatisticalReport:
    """Tests for the StatisticalReport frozen model."""

    def test_creation(self) -> None:
        """Empty StatisticalReport should be constructable."""
        report: StatisticalReport = StatisticalReport(
            profiles=(),
            corrected_pvalues=(),
            n_assets=0,
            n_bar_types=0,
            n_total_tests=0,
            n_significant_raw=0,
            n_significant_corrected=0,
        )
        assert report.n_total_tests == 0
        assert report.profiles == ()

    def test_summary_counts_constraint(self, fake_loader_single: FakeDataLoader) -> None:
        """BH correction should not increase the number of significant tests."""
        service: ProfilingService = ProfilingService(fake_loader_single)  # type: ignore[arg-type]
        report: StatisticalReport = service.profile_all(assets=["BTCUSDT"])
        assert report.n_significant_raw >= report.n_significant_corrected


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases: empty data, no bar configs, small samples."""

    def test_empty_data(self, fake_loader_empty: FakeDataLoader) -> None:
        """Empty DataFrame should produce a Tier C profile with zero observations."""
        service: ProfilingService = ProfilingService(fake_loader_empty)  # type: ignore[arg-type]
        profile: AssetBarProfile = service.profile_single(
            asset="BTCUSDT",
            bar_type="dollar",
            config_hash="abc123",
        )
        assert profile.n_observations == 0
        assert profile.tier == SampleTier.C

    def test_no_bar_configs(self, fake_loader_no_bars: FakeDataLoader) -> None:
        """Asset with no aggregated bars should only produce a time_1h profile."""
        service: ProfilingService = ProfilingService(fake_loader_no_bars)  # type: ignore[arg-type]
        report: StatisticalReport = service.profile_all(assets=["BTCUSDT"])
        assert len(report.profiles) == 1
        assert report.profiles[0].bar_type == "time_1h"

    def test_single_row_data(self) -> None:
        """Single-row DataFrame should produce zero observations (no returns after diff)."""
        ohlcv: dict[tuple[str, str], pd.DataFrame] = {("BTCUSDT", "1h"): _make_ohlcv_df(n=1, seed=42)}
        loader: FakeDataLoader = FakeDataLoader(ohlcv_data=ohlcv)
        service: ProfilingService = ProfilingService(loader)  # type: ignore[arg-type]
        profile: AssetBarProfile = service.profile_single(asset="BTCUSDT", bar_type="time_1h")
        assert profile.n_observations == 0

    def test_small_sample_tier_c(self) -> None:
        """50 rows minus 1 for returns should classify as Tier C."""
        ohlcv: dict[tuple[str, str], pd.DataFrame] = {("BTCUSDT", "1h"): _make_ohlcv_df(n=50, seed=42)}
        loader: FakeDataLoader = FakeDataLoader(ohlcv_data=ohlcv)
        service: ProfilingService = ProfilingService(loader)  # type: ignore[arg-type]
        profile: AssetBarProfile = service.profile_single(asset="BTCUSDT", bar_type="time_1h")
        assert profile.tier == SampleTier.C
