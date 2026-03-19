"""Orchestrator service for statistical profiling across all asset-bar combinations."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from loguru import logger
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]

from src.app.profiling.application.distribution import DistributionAnalyzer
from src.app.profiling.application.predictability import PredictabilityAnalyzer
from src.app.profiling.application.serial_dependence import SerialDependenceAnalyzer
from src.app.profiling.application.stationarity import StationarityScreener
from src.app.profiling.application.volatility import VolatilityAnalyzer
from src.app.profiling.domain.value_objects import (
    AssetBarProfile,
    AutocorrelationProfile,
    CorrectedPValue,
    DataPartition,
    DistributionProfile,
    PredictabilityProfile,
    ProfilingConfig,
    SampleTier,
    StationarityReport,
    StatisticalReport,
    TierClassifier,
    VolatilityProfile,
)
from src.app.research.application.data_loader import DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SECONDS_PER_DAY: float = 86400.0
"""Number of seconds in a calendar day."""

_DEFAULT_SIGNIFICANCE_ALPHA: float = 0.05
"""Default significance level used when the sub-profile does not carry its own alpha."""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_inferential_pvalues(profile: AssetBarProfile) -> list[CorrectedPValue]:
    """Extract all inferential test p-values from a single asset-bar profile.

    Scans autocorrelation (Ljung-Box, variance ratio, Granger) and
    volatility (BDS, ARCH-LM, sign bias joint F-test) sub-profiles
    for raw p-values.  Non-applicable fields (``None``) are skipped.

    Args:
        profile: A frozen ``AssetBarProfile`` containing sub-profile results.

    Returns:
        List of ``CorrectedPValue`` objects with ``raw_pvalue`` set and
        ``corrected_pvalue`` initialised to ``raw_pvalue`` (to be updated
        by BH correction).
    """
    results: list[CorrectedPValue] = []
    asset: str = profile.asset
    bar_type: str = profile.bar_type

    # --- Autocorrelation sub-profile ---
    acf_profile: AutocorrelationProfile | None = profile.autocorrelation
    if acf_profile is not None:
        # Ljung-Box on raw returns
        results.extend(
            CorrectedPValue(
                asset=asset,
                bar_type=bar_type,
                test_name="ljung_box_returns",
                parameter=f"lag={lb.lag}",
                raw_pvalue=lb.p_value,
                corrected_pvalue=lb.p_value,
                significant_raw=lb.significant,
                significant_corrected=lb.significant,
            )
            for lb in acf_profile.ljung_box_returns
        )

        # Ljung-Box on squared returns
        results.extend(
            CorrectedPValue(
                asset=asset,
                bar_type=bar_type,
                test_name="ljung_box_squared",
                parameter=f"lag={lb.lag}",
                raw_pvalue=lb.p_value,
                corrected_pvalue=lb.p_value,
                significant_raw=lb.significant,
                significant_corrected=lb.significant,
            )
            for lb in acf_profile.ljung_box_squared
        )

        # Variance ratio tests
        if acf_profile.vr_results is not None:
            results.extend(
                CorrectedPValue(
                    asset=asset,
                    bar_type=bar_type,
                    test_name="variance_ratio",
                    parameter=f"horizon={vr.calendar_horizon_days}d",
                    raw_pvalue=vr.p_value,
                    corrected_pvalue=vr.p_value,
                    significant_raw=vr.significant,
                    significant_corrected=vr.significant,
                )
                for vr in acf_profile.vr_results
            )

        # Granger causality tests
        if acf_profile.granger_results is not None:
            results.extend(
                CorrectedPValue(
                    asset=asset,
                    bar_type=bar_type,
                    test_name="granger",
                    parameter=f"{gr.source_name}->{gr.target_name},lag={gr.lag}",
                    raw_pvalue=gr.p_value,
                    corrected_pvalue=gr.p_value,
                    significant_raw=gr.significant,
                    significant_corrected=gr.significant,
                )
                for gr in acf_profile.granger_results
            )

    # --- Volatility sub-profile ---
    vol_profile: VolatilityProfile | None = profile.volatility
    if vol_profile is not None:
        # BDS test results
        if vol_profile.bds_results is not None:
            results.extend(
                CorrectedPValue(
                    asset=asset,
                    bar_type=bar_type,
                    test_name="bds",
                    parameter=f"dim={bds_r.dimension}",
                    raw_pvalue=bds_r.p_value,
                    corrected_pvalue=bds_r.p_value,
                    significant_raw=bds_r.significant,
                    significant_corrected=bds_r.significant,
                )
                for bds_r in vol_profile.bds_results
            )

        # ARCH-LM test
        if vol_profile.arch_lm_pvalue is not None:
            arch_pval: float = vol_profile.arch_lm_pvalue
            results.append(
                CorrectedPValue(
                    asset=asset,
                    bar_type=bar_type,
                    test_name="arch_lm",
                    parameter="residuals",
                    raw_pvalue=arch_pval,
                    corrected_pvalue=arch_pval,
                    significant_raw=arch_pval < _DEFAULT_SIGNIFICANCE_ALPHA,
                    significant_corrected=arch_pval < _DEFAULT_SIGNIFICANCE_ALPHA,
                )
            )

        # Sign bias joint F-test
        if vol_profile.sign_bias is not None:
            joint_pval: float = vol_profile.sign_bias.joint_f_pvalue
            results.append(
                CorrectedPValue(
                    asset=asset,
                    bar_type=bar_type,
                    test_name="sign_bias_joint",
                    parameter="joint_f",
                    raw_pvalue=joint_pval,
                    corrected_pvalue=joint_pval,
                    significant_raw=joint_pval < _DEFAULT_SIGNIFICANCE_ALPHA,
                    significant_corrected=joint_pval < _DEFAULT_SIGNIFICANCE_ALPHA,
                )
            )

    return results


def _apply_bh_correction(
    raw_pvalues: list[CorrectedPValue],
    alpha: float,
) -> tuple[CorrectedPValue, ...]:
    """Apply Benjamini-Hochberg FDR correction to a collection of p-values.

    Uses ``statsmodels.stats.multitest.multipletests`` with ``method='fdr_bh'``
    for correctness.  When the input list is empty, returns an empty tuple.

    Args:
        raw_pvalues: List of ``CorrectedPValue`` objects with ``raw_pvalue``
            already set.
        alpha: Family-wise significance level for the FDR procedure.

    Returns:
        Tuple of new ``CorrectedPValue`` objects with ``corrected_pvalue``
        and ``significant_corrected`` updated.
    """
    if not raw_pvalues:
        return ()

    pvals: np.ndarray = np.array(  # type: ignore[type-arg]
        [p.raw_pvalue for p in raw_pvalues],
        dtype=np.float64,
    )

    reject: np.ndarray  # type: ignore[type-arg]
    corrected: np.ndarray  # type: ignore[type-arg]
    reject, corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")  # type: ignore[no-untyped-call]

    corrected_results: list[CorrectedPValue] = []
    for i, orig in enumerate(raw_pvalues):
        corrected_pval: float = float(corrected[i])
        is_significant: bool = bool(reject[i])
        corrected_results.append(
            CorrectedPValue(
                asset=orig.asset,
                bar_type=orig.bar_type,
                test_name=orig.test_name,
                parameter=orig.parameter,
                raw_pvalue=orig.raw_pvalue,
                corrected_pvalue=corrected_pval,
                significant_raw=orig.significant_raw,
                significant_corrected=is_significant,
            )
        )

    return tuple(corrected_results)


def _compute_bars_per_day(df: pd.DataFrame, ts_col: str) -> float:
    """Compute the average number of bars per calendar day from timestamps.

    Args:
        df: DataFrame containing at least one timestamp column.
        ts_col: Name of the timestamp column.

    Returns:
        Bars per day (>= 1.0).  Returns 1.0 for single-row DataFrames.
    """
    n: int = len(df)
    if n <= 1:
        return 1.0

    first_ts: pd.Timestamp = pd.Timestamp(df[ts_col].iloc[0])  # type: ignore[arg-type]
    last_ts: pd.Timestamp = pd.Timestamp(df[ts_col].iloc[-1])  # type: ignore[arg-type]
    total_seconds: float = (last_ts - first_ts).total_seconds()

    if total_seconds <= 0:
        return 1.0

    total_days: float = total_seconds / _SECONDS_PER_DAY
    bars_per_day: float = float(n) / total_days
    return max(1.0, bars_per_day)


# ---------------------------------------------------------------------------
# Public service class
# ---------------------------------------------------------------------------


class ProfilingService:
    """Orchestrates Phase 5 statistical profiling across all asset-bar combinations.

    Iterates over every (asset, bar_type) pair available in the database,
    classifies each into a sample-size tier, dispatches to the appropriate
    Phase 5 analyzers, and aggregates the results into a ``StatisticalReport``
    with Benjamini-Hochberg FDR-corrected p-values.

    The ``DataLoader`` dependency is duck-typed: any object implementing the
    same public API (``load_ohlcv``, ``load_bars``, ``get_available_assets``,
    ``get_available_bar_configs``) can be injected for testing.
    """

    def __init__(self, data_loader: DataLoader) -> None:
        """Initialise the service with a data loader dependency.

        Args:
            data_loader: DataLoader (or duck-typed equivalent) for reading
                OHLCV and aggregated bar data from the database.
        """
        self._data_loader: DataLoader = data_loader
        self._tier_classifier: TierClassifier = TierClassifier()
        self._distribution_analyzer: DistributionAnalyzer = DistributionAnalyzer()
        self._serial_analyzer: SerialDependenceAnalyzer = SerialDependenceAnalyzer()
        self._volatility_analyzer: VolatilityAnalyzer = VolatilityAnalyzer()
        self._predictability_analyzer: PredictabilityAnalyzer = PredictabilityAnalyzer()
        self._stationarity_screener: StationarityScreener = StationarityScreener()

    def profile_all(  # noqa: PLR0912, PLR0914
        self,
        assets: list[str] | None = None,
        config: ProfilingConfig | None = None,
        partition: DataPartition | None = None,
    ) -> StatisticalReport:
        """Profile all asset-bar combinations and return an FDR-corrected report.

        For each asset, loads all available bar types plus time_1h (from OHLCV),
        computes log returns, and dispatches to each Phase 5 analyzer.
        Inferential p-values are extracted from all profiles and corrected
        via Benjamini-Hochberg FDR.

        Args:
            assets: Asset symbols to profile.  When ``None``, queries the
                database for all available assets.
            config: Composite profiling configuration.  Uses defaults when ``None``.
            partition: Temporal partition for filtering data.  Uses
                ``DataPartition.default()`` when ``None``.

        Returns:
            Frozen ``StatisticalReport`` with all profiles and corrected p-values.
        """
        if config is None:
            config = ProfilingConfig()
        if partition is None:
            partition = DataPartition.default()

        if assets is None:
            assets = self._data_loader.get_available_assets()

        logger.info("Starting full profiling for {} asset(s)", len(assets))

        all_profiles: list[AssetBarProfile] = []

        for asset in assets:
            # Aggregated bar types
            bar_configs: list[tuple[str, str]] = self._data_loader.get_available_bar_configs(asset)
            for bar_type, config_hash in bar_configs:
                profile: AssetBarProfile = self.profile_single(
                    asset=asset,
                    bar_type=bar_type,
                    config_hash=config_hash,
                    config=config,
                    partition=partition,
                )
                all_profiles.append(profile)

            # Time bars (time_1h) from OHLCV
            profile_time: AssetBarProfile = self.profile_single(
                asset=asset,
                bar_type="time_1h",
                config_hash=None,
                config=config,
                partition=partition,
            )
            all_profiles.append(profile_time)

        # Extract all inferential p-values across all profiles
        all_raw_pvalues: list[CorrectedPValue] = []
        for prof in all_profiles:
            all_raw_pvalues.extend(_extract_inferential_pvalues(prof))

        # Apply Benjamini-Hochberg FDR correction
        corrected_pvalues: tuple[CorrectedPValue, ...] = _apply_bh_correction(all_raw_pvalues, config.fdr_alpha)

        # Compute summary statistics
        n_assets: int = len({p.asset for p in all_profiles})
        n_bar_types: int = len({p.bar_type for p in all_profiles})
        n_total_tests: int = len(corrected_pvalues)
        n_significant_raw: int = sum(1 for p in corrected_pvalues if p.significant_raw)
        n_significant_corrected: int = sum(1 for p in corrected_pvalues if p.significant_corrected)

        logger.info(
            "Profiling complete: {} profiles, {} tests, {} sig raw, {} sig corrected",
            len(all_profiles),
            n_total_tests,
            n_significant_raw,
            n_significant_corrected,
        )

        return StatisticalReport(
            profiles=tuple(all_profiles),
            corrected_pvalues=corrected_pvalues,
            n_assets=n_assets,
            n_bar_types=n_bar_types,
            n_total_tests=n_total_tests,
            n_significant_raw=n_significant_raw,
            n_significant_corrected=n_significant_corrected,
        )

    def profile_single(  # noqa: PLR0913, PLR0917, PLR0914
        self,
        asset: str,
        bar_type: str,
        config_hash: str | None = None,
        config: ProfilingConfig | None = None,
        partition: DataPartition | None = None,
        feature_columns: list[str] | None = None,
    ) -> AssetBarProfile:
        """Profile a single (asset, bar_type) combination.

        Loads bar or OHLCV data, computes log returns, classifies the
        sample tier, and runs all applicable Phase 5 analyzers.

        Args:
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"dollar"``, ``"time_1h"``).
            config_hash: Bar configuration hash (required for non-time bars).
                ``None`` for time bars which are loaded from OHLCV.
            config: Composite profiling configuration.  Uses defaults when ``None``.
            partition: Temporal partition for filtering data.  Uses
                ``DataPartition.default()`` when ``None``.
            feature_columns: Optional list of feature column names to run
                stationarity screening on.  When ``None``, stationarity is skipped.

        Returns:
            Frozen ``AssetBarProfile`` value object.
        """
        if config is None:
            config = ProfilingConfig()
        if partition is None:
            partition = DataPartition.default()

        # Load data
        df: pd.DataFrame = self._load_data(asset, bar_type, config_hash, partition)

        if df.empty:
            logger.warning("No data for asset={}, bar_type={}, returning empty profile", asset, bar_type)
            return AssetBarProfile(
                asset=asset,
                bar_type=bar_type,
                tier=SampleTier.C,
                n_observations=0,
            )

        # Determine timestamp column
        ts_col: str = "timestamp" if "timestamp" in df.columns else "start_ts"

        # Compute log returns
        returns: pd.Series = np.log(df["close"] / df["close"].shift(1)).dropna()  # type: ignore[type-arg]

        n_returns: int = len(returns)
        tier: SampleTier = self._tier_classifier.classify(n_returns, config.tier)

        logger.debug(
            "Profiling asset={}, bar_type={}, n_returns={}, tier={}",
            asset,
            bar_type,
            n_returns,
            tier.value,
        )

        if n_returns == 0:
            return AssetBarProfile(
                asset=asset,
                bar_type=bar_type,
                tier=tier,
                n_observations=0,
            )

        # Compute bars_per_day for serial dependence
        bars_per_day: float = _compute_bars_per_day(df, ts_col)

        # Run all analyzers
        distribution: DistributionProfile | None = self._run_distribution(returns, asset, bar_type, tier, config)
        autocorrelation: AutocorrelationProfile | None = self._run_serial_dependence(
            returns, asset, bar_type, tier, bars_per_day, config
        )
        volatility: VolatilityProfile | None = self._run_volatility(returns, asset, bar_type, tier, config)
        predictability: PredictabilityProfile | None = self._run_predictability(returns, asset, bar_type, tier, config)
        stationarity: StationarityReport | None = self._run_stationarity(df, asset, bar_type, feature_columns, config)

        return AssetBarProfile(
            asset=asset,
            bar_type=bar_type,
            tier=tier,
            n_observations=n_returns,
            distribution=distribution,
            autocorrelation=autocorrelation,
            volatility=volatility,
            predictability=predictability,
            stationarity=stationarity,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_data(
        self,
        asset: str,
        bar_type: str,
        config_hash: str | None,
        partition: DataPartition,
    ) -> pd.DataFrame:
        """Load and temporally filter bar or OHLCV data.

        Args:
            asset: Trading pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Bar configuration hash (``None`` for time bars).
            partition: Temporal partition for date filtering.

        Returns:
            Filtered Pandas DataFrame ordered by timestamp.
        """
        date_range: tuple[pd.Timestamp, pd.Timestamp] = (  # type: ignore[assignment]
            partition.feature_selection_start,
            partition.feature_selection_end,
        )

        if bar_type == "time_1h":
            df: pd.DataFrame = self._data_loader.load_ohlcv(asset, "1h", date_range=date_range)  # type: ignore[arg-type]
        else:
            if config_hash is None:
                logger.warning("config_hash is None for non-time bar_type={}, returning empty", bar_type)
                return pd.DataFrame()
            df = self._data_loader.load_bars(asset, bar_type, config_hash, date_range=date_range)  # type: ignore[arg-type]

        return df

    def _run_distribution(  # noqa: PLR6301
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        tier: SampleTier,
        config: ProfilingConfig,
    ) -> DistributionProfile | None:
        """Run distribution analysis, catching unexpected exceptions.

        Args:
            returns: Log return series.
            asset: Trading pair symbol.
            bar_type: Bar aggregation type.
            tier: Sample tier.
            config: Composite profiling configuration.

        Returns:
            Distribution profile, or ``None`` on failure.
        """
        try:
            return self._distribution_analyzer.analyze(returns, asset, bar_type, tier, config.distribution)
        except Exception:
            logger.exception("Distribution analysis failed for asset={}, bar_type={}", asset, bar_type)
            return None

    def _run_serial_dependence(  # noqa: PLR6301, PLR0913, PLR0917
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        tier: SampleTier,
        bars_per_day: float,
        config: ProfilingConfig,
    ) -> AutocorrelationProfile | None:
        """Run serial dependence analysis, catching unexpected exceptions.

        Args:
            returns: Log return series.
            asset: Trading pair symbol.
            bar_type: Bar aggregation type.
            tier: Sample tier.
            bars_per_day: Average bars per calendar day.
            config: Composite profiling configuration.

        Returns:
            Autocorrelation profile, or ``None`` on failure.
        """
        try:
            return self._serial_analyzer.analyze(returns, asset, bar_type, tier, bars_per_day, config.autocorrelation)
        except Exception:
            logger.exception("Serial dependence analysis failed for asset={}, bar_type={}", asset, bar_type)
            return None

    def _run_volatility(  # noqa: PLR6301
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        tier: SampleTier,
        config: ProfilingConfig,
    ) -> VolatilityProfile | None:
        """Run volatility analysis, catching unexpected exceptions.

        Args:
            returns: Log return series.
            asset: Trading pair symbol.
            bar_type: Bar aggregation type.
            tier: Sample tier.
            config: Composite profiling configuration.

        Returns:
            Volatility profile, or ``None`` on failure.
        """
        try:
            return self._volatility_analyzer.analyze(returns, asset, bar_type, tier, config.volatility)
        except Exception:
            logger.exception("Volatility analysis failed for asset={}, bar_type={}", asset, bar_type)
            return None

    def _run_predictability(  # noqa: PLR6301
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        tier: SampleTier,
        config: ProfilingConfig,
    ) -> PredictabilityProfile | None:
        """Run predictability assessment, catching unexpected exceptions.

        Args:
            returns: Log return series.
            asset: Trading pair symbol.
            bar_type: Bar aggregation type.
            tier: Sample tier.
            config: Composite profiling configuration.

        Returns:
            Predictability profile, or ``None`` on failure.
        """
        try:
            return self._predictability_analyzer.analyze(returns, asset, bar_type, tier, config.predictability)
        except Exception:
            logger.exception("Predictability analysis failed for asset={}, bar_type={}", asset, bar_type)
            return None

    def _run_stationarity(  # noqa: PLR6301, PLR0913, PLR0917
        self,
        df: pd.DataFrame,
        asset: str,
        bar_type: str,
        feature_columns: list[str] | None,
        config: ProfilingConfig,
    ) -> StationarityReport | None:
        """Run stationarity screening on feature columns if available.

        Args:
            df: Full DataFrame (may include feature columns).
            asset: Trading pair symbol.
            bar_type: Bar aggregation type.
            feature_columns: Feature column names to screen.  When ``None``,
                stationarity screening is skipped entirely.
            config: Composite profiling configuration.

        Returns:
            Stationarity report, or ``None`` when no features or on failure.
        """
        if feature_columns is None:
            return None

        # Filter to feature columns that actually exist in the DataFrame
        available: list[str] = [c for c in feature_columns if c in df.columns]
        if not available:
            return None

        try:
            return self._stationarity_screener.screen(df, available, asset, bar_type, config.stationarity_alpha)
        except Exception:
            logger.exception("Stationarity screening failed for asset={}, bar_type={}", asset, bar_type)
            return None
