"""Profiling domain value objects -- data partitions, sample tiers, distribution profiles, and stationarity results."""

from __future__ import annotations

from datetime import datetime, UTC
from enum import Enum
from typing import Annotated, Self

import polars as pl
from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import model_validator


class DataPartition(BaseModel, frozen=True):
    """Project-level authoritative temporal partition for data usage.

    Defines non-overlapping time boundaries that control which data
    is used for feature selection, model development, and final holdout
    evaluation.  This prevents information leakage across research phases.

    Invariants:
        * ``feature_selection_start < feature_selection_end``
        * ``model_dev_start < model_dev_end``
        * ``holdout_start >= model_dev_end`` (no overlap between dev and holdout)
        * Feature selection period must be contained within model development period
    """

    feature_selection_start: datetime
    """Start of the feature selection period (inclusive)."""

    feature_selection_end: datetime
    """End of the feature selection period (exclusive)."""

    model_dev_start: datetime
    """Start of the model development period (inclusive)."""

    model_dev_end: datetime
    """End of the model development period (exclusive)."""

    holdout_start: datetime
    """Start of the final holdout period (inclusive). Extends to end of data."""

    @model_validator(mode="after")
    def _validate_partition_order(self) -> Self:
        """Ensure partitions are properly ordered and non-overlapping.

        Returns:
            Validated instance.

        Raises:
            ValueError: If partition boundaries violate ordering constraints.
        """
        if self.feature_selection_start >= self.feature_selection_end:
            msg: str = (
                f"feature_selection_start ({self.feature_selection_start}) "
                f"must be < feature_selection_end ({self.feature_selection_end})"
            )
            raise ValueError(msg)
        if self.model_dev_start >= self.model_dev_end:
            msg = f"model_dev_start ({self.model_dev_start}) must be < model_dev_end ({self.model_dev_end})"
            raise ValueError(msg)
        if self.holdout_start < self.model_dev_end:
            msg = f"holdout_start ({self.holdout_start}) must be >= model_dev_end ({self.model_dev_end})"
            raise ValueError(msg)
        if self.feature_selection_start < self.model_dev_start:
            msg = (
                f"feature_selection_start ({self.feature_selection_start}) "
                f"must be >= model_dev_start ({self.model_dev_start})"
            )
            raise ValueError(msg)
        if self.feature_selection_end > self.model_dev_end:
            msg = (
                f"feature_selection_end ({self.feature_selection_end}) must be <= model_dev_end ({self.model_dev_end})"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def default(cls) -> DataPartition:
        """Return the standard project-level temporal partition.

        Returns:
            DataPartition with the following boundaries:
                - Feature selection: 2020-01-01 to 2022-12-31
                - Model development: 2020-01-01 to 2023-12-31
                - Final holdout: 2024-01-01 onwards
        """
        return cls(
            feature_selection_start=datetime(2020, 1, 1, tzinfo=UTC),
            feature_selection_end=datetime(2023, 1, 1, tzinfo=UTC),
            model_dev_start=datetime(2020, 1, 1, tzinfo=UTC),
            model_dev_end=datetime(2024, 1, 1, tzinfo=UTC),
            holdout_start=datetime(2024, 1, 1, tzinfo=UTC),
        )

    def filter_feature_selection(self, df: pl.DataFrame, timestamp_col: str) -> pl.DataFrame:
        """Filter a DataFrame to the feature selection period.

        Args:
            df: Input Polars DataFrame.
            timestamp_col: Name of the timestamp column to filter on.

        Returns:
            Filtered DataFrame with rows where ``start <= timestamp < end``.
        """
        return df.filter(
            (pl.col(timestamp_col) >= self.feature_selection_start)
            & (pl.col(timestamp_col) < self.feature_selection_end)
        )

    def filter_model_dev(self, df: pl.DataFrame, timestamp_col: str) -> pl.DataFrame:
        """Filter a DataFrame to the model development period.

        Args:
            df: Input Polars DataFrame.
            timestamp_col: Name of the timestamp column to filter on.

        Returns:
            Filtered DataFrame with rows where ``start <= timestamp < end``.
        """
        return df.filter(
            (pl.col(timestamp_col) >= self.model_dev_start) & (pl.col(timestamp_col) < self.model_dev_end)
        )

    def filter_holdout(self, df: pl.DataFrame, timestamp_col: str) -> pl.DataFrame:
        """Filter a DataFrame to the holdout period.

        Args:
            df: Input Polars DataFrame.
            timestamp_col: Name of the timestamp column to filter on.

        Returns:
            Filtered DataFrame with rows where ``timestamp >= holdout_start``.
        """
        return df.filter(pl.col(timestamp_col) >= self.holdout_start)


class SampleTier(Enum):
    """Sample-size tier classification for bar types.

    Determines which modelling techniques are appropriate given the
    number of available samples.

    Attributes:
        A: >= tier_a_threshold samples. Full ML pipeline available.
        B: Between tier_b_threshold and tier_a_threshold. Restricted to
           simpler models with stronger regularisation.
        C: < tier_b_threshold samples. Statistical profiling only;
           ML modelling is unreliable.
    """

    A = "A"
    B = "B"
    C = "C"


class TierConfig(BaseModel, frozen=True):
    """Configuration for sample-size tier thresholds.

    Attributes:
        tier_a_threshold: Minimum sample count for Tier A (full ML pipeline).
        tier_b_threshold: Minimum sample count for Tier B (restricted models).
    """

    tier_a_threshold: Annotated[
        int,
        PydanticField(default=2000, ge=1, description="Minimum samples for Tier A"),
    ]

    tier_b_threshold: Annotated[
        int,
        PydanticField(default=500, ge=1, description="Minimum samples for Tier B"),
    ]

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> Self:
        """Ensure tier_a_threshold > tier_b_threshold.

        Returns:
            Validated instance.

        Raises:
            ValueError: If ``tier_a_threshold`` is not greater than ``tier_b_threshold``.
        """
        if self.tier_a_threshold <= self.tier_b_threshold:
            msg: str = (
                f"tier_a_threshold ({self.tier_a_threshold}) must be > tier_b_threshold ({self.tier_b_threshold})"
            )
            raise ValueError(msg)
        return self


class TierClassifier:
    """Stateless classifier that assigns a SampleTier based on sample count.

    Example:
        >>> classifier = TierClassifier()
        >>> classifier.classify(3000, TierConfig())
        <SampleTier.A: 'A'>
    """

    def classify(self, n_samples: int, config: TierConfig) -> SampleTier:  # noqa: PLR6301
        """Classify a sample count into a tier.

        Args:
            n_samples: Number of available samples.
            config: Tier threshold configuration.

        Returns:
            SampleTier.A if n_samples > tier_a_threshold,
            SampleTier.B if tier_b_threshold <= n_samples <= tier_a_threshold,
            SampleTier.C otherwise.
        """
        if n_samples > config.tier_a_threshold:
            return SampleTier.A
        if n_samples >= config.tier_b_threshold:
            return SampleTier.B
        return SampleTier.C


class DistributionConfig(BaseModel, frozen=True):
    """Configuration for return distribution analysis.

    Attributes:
        jb_alpha: Significance level for the Jarque-Bera normality test.
        price_col: Name of the price column used to compute log returns.
        min_samples_jb: Minimum number of samples required for Jarque-Bera test.
        min_samples_fit: Minimum number of samples required for MLE fitting.
    """

    jb_alpha: Annotated[float, PydanticField(gt=0, lt=1)] = 0.05
    price_col: str = "close"
    min_samples_jb: Annotated[int, PydanticField(ge=3)] = 3
    min_samples_fit: Annotated[int, PydanticField(ge=10)] = 30


class DistributionProfile(BaseModel, frozen=True):
    """Per-asset, per-bar-type return distribution profile.

    Contains Jarque-Bera normality test results, effect sizes,
    Student-t MLE fit parameters, AIC/BIC model comparison,
    and KS distance measure. Tier-gated: Tier C gets descriptive
    stats only (Student-t fields are None).

    Attributes:
        asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
        bar_type: Bar aggregation type (e.g. ``"dollar"``).
        tier: Sample-size tier that controls analysis depth.
        n_observations: Number of return observations analysed.
        mean_return: Mean of log returns.
        std_return: Standard deviation of log returns.
        skewness: Fisher skewness of log returns.
        excess_kurtosis: Fisher excess kurtosis of log returns.
        jb_stat: Jarque-Bera test statistic.
        jb_pvalue: Jarque-Bera p-value.
        is_normal: Whether normality was NOT rejected at the configured alpha.
        student_t_nu: Fitted Student-t degrees of freedom (Tier A/B only).
        student_t_loc: Fitted Student-t location parameter (Tier A/B only).
        student_t_scale: Fitted Student-t scale parameter (Tier A/B only).
        aic_normal: AIC for the fitted Normal distribution (Tier A/B only).
        aic_student_t: AIC for the fitted Student-t distribution (Tier A/B only).
        bic_normal: BIC for the fitted Normal distribution (Tier A/B only).
        bic_student_t: BIC for the fitted Student-t distribution (Tier A/B only).
        best_fit: Best-fitting distribution by AIC (Tier A/B only).
        ks_statistic: KS D_n statistic vs fitted Normal (Tier A/B only).
    """

    asset: str
    bar_type: str
    tier: SampleTier

    # Sample info
    n_observations: Annotated[int, PydanticField(ge=0)]

    # Descriptive statistics (all tiers)
    mean_return: float
    std_return: Annotated[float, PydanticField(ge=0)]
    skewness: float
    excess_kurtosis: float

    # Jarque-Bera normality test (all tiers)
    jb_stat: Annotated[float, PydanticField(ge=0)]
    jb_pvalue: Annotated[float, PydanticField(ge=0, le=1)]
    is_normal: bool

    # Student-t MLE fit (Tier A/B only -- None for Tier C)
    student_t_nu: float | None = None
    student_t_loc: float | None = None
    student_t_scale: float | None = None

    # Model comparison (Tier A/B only -- None for Tier C)
    aic_normal: float | None = None
    aic_student_t: float | None = None
    bic_normal: float | None = None
    bic_student_t: float | None = None
    best_fit: str | None = None

    # KS distance measure (Tier A/B only -- None for Tier C)
    ks_statistic: float | None = None


class StationarityTestResult(BaseModel, frozen=True):
    """Per-feature stationarity test result from ADF and KPSS tests.

    The joint interpretation of ADF (null: unit root) and KPSS
    (null: stationary) determines the classification:

    - **stationary**: ADF rejects AND KPSS fails to reject.
    - **trend_stationary**: ADF rejects AND KPSS rejects.
    - **unit_root**: ADF fails to reject AND KPSS rejects.
    - **inconclusive**: Neither test rejects its null.

    Attributes:
        feature_name: Column name of the tested feature.
        adf_statistic: ADF test statistic.
        adf_pvalue: ADF test p-value.
        kpss_statistic: KPSS test statistic.
        kpss_pvalue: KPSS test p-value.
        is_stationary: True when ADF rejects AND KPSS fails to reject.
        classification: One of "stationary", "trend_stationary", "unit_root", "inconclusive".
        suggested_transformation: Recommended transformation for non-stationary features.
    """

    feature_name: str
    adf_statistic: float
    adf_pvalue: Annotated[float, PydanticField(ge=0, le=1)]
    kpss_statistic: float
    kpss_pvalue: Annotated[float, PydanticField(ge=0, le=1)]
    is_stationary: bool
    classification: str
    suggested_transformation: str | None


class StationarityReport(BaseModel, frozen=True):
    """Aggregate stationarity screening report for a single asset-bar combination.

    Attributes:
        results: Per-feature stationarity test results.
        n_stationary: Count of features classified as stationary.
        n_non_stationary: Count of features not classified as stationary.
        asset: Asset symbol (e.g. "BTCUSDT").
        bar_type: Bar type identifier (e.g. "dollar", "volume").
    """

    results: tuple[StationarityTestResult, ...]
    n_stationary: int
    n_non_stationary: int
    asset: str
    bar_type: str
