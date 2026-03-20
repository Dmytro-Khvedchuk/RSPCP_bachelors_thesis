"""Stationarity screening via joint ADF + KPSS testing.

Uses the ML-research path (Pandas / NumPy / statsmodels) per CLAUDE.md.
The conversion boundary is at the ``screen()`` method entry point.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from loguru import logger
from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore[import-untyped]

from src.app.profiling.domain.value_objects import (
    StationarityReport,
    StationarityTestResult,
)


_KNOWN_TRANSFORMATIONS: dict[str, str] = {
    "atr_": "pct_atr",
    "amihud_": "rolling_zscore",
    "hurst_": "first_difference",
    "bbwidth_": "first_difference",
}
"""Prefix-to-transformation mapping for known non-stationary feature families."""


def _suggest_transformation(feature_name: str) -> str | None:
    """Suggest a transformation for a non-stationary feature based on its prefix.

    Args:
        feature_name: Column name of the feature.

    Returns:
        Suggested transformation string, or ``None`` if no known transformation applies.
    """
    for prefix, transform in _KNOWN_TRANSFORMATIONS.items():
        if feature_name.startswith(prefix):
            return transform
    return None


def _classify_stationarity(
    adf_rejects: bool,
    kpss_rejects: bool,
) -> str:
    """Classify feature stationarity from joint ADF + KPSS test outcomes.

    Args:
        adf_rejects: Whether ADF rejects its null (unit root).
        kpss_rejects: Whether KPSS rejects its null (stationarity).

    Returns:
        One of "stationary", "trend_stationary", "unit_root", "inconclusive".
    """
    if adf_rejects and not kpss_rejects:
        return "stationary"
    if adf_rejects and kpss_rejects:
        return "trend_stationary"
    if not adf_rejects and kpss_rejects:
        return "unit_root"
    return "inconclusive"


def _run_adf(series: np.ndarray[tuple[int], np.dtype[np.float64]]) -> tuple[float, float]:
    """Run the Augmented Dickey-Fuller test.

    Args:
        series: 1-D array of values (must not contain NaN/inf).

    Returns:
        Tuple of (test_statistic, p_value).
    """
    # adfuller returns a heterogeneous tuple; statsmodels stubs are incomplete
    result: tuple[object, ...] = adfuller(series, autolag="AIC")  # type: ignore[assignment]
    statistic: float = float(result[0])  # type: ignore[arg-type]
    pvalue: float = float(result[1])  # type: ignore[arg-type]
    return statistic, pvalue


def _run_kpss(series: np.ndarray[tuple[int], np.dtype[np.float64]]) -> tuple[float, float]:
    """Run the KPSS test for level stationarity.

    KPSS issues ``InterpolationWarning`` when the p-value is outside
    the tabulated range.  We suppress the warning and clamp the p-value
    to [0, 1].

    Args:
        series: 1-D array of values (must not contain NaN/inf).

    Returns:
        Tuple of (test_statistic, p_value).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # kpss returns a heterogeneous tuple; statsmodels stubs are incomplete
        result: tuple[object, ...] = kpss(series, regression="c", nlags="auto")  # type: ignore[assignment]
    statistic: float = float(result[0])  # type: ignore[arg-type]
    pvalue: float = float(np.clip(result[1], 0.0, 1.0))  # type: ignore[arg-type]
    return statistic, pvalue


class StationarityScreener:
    """Screen features for stationarity using joint ADF + KPSS testing.

    This screener uses Pandas DataFrames as input because the underlying
    ``statsmodels`` functions expect NumPy arrays.  The conversion boundary
    is at the ``screen()`` method.

    Classification logic (joint hypothesis):
        - ADF rejects (p < alpha) AND KPSS doesn't reject (p >= alpha) -> "stationary"
        - ADF rejects AND KPSS rejects -> "trend_stationary"
        - ADF doesn't reject AND KPSS rejects -> "unit_root"
        - Neither rejects -> "inconclusive"
    """

    def screen(  # noqa: PLR6301
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        asset: str,
        bar_type: str,
        alpha: float = 0.05,
    ) -> StationarityReport:
        """Screen all features for stationarity.

        Args:
            df: Pandas DataFrame containing the feature columns.
            feature_names: List of feature column names to test.
            asset: Asset symbol for the report metadata.
            bar_type: Bar type identifier for the report metadata.
            alpha: Significance level for both ADF and KPSS tests.

        Returns:
            StationarityReport with per-feature results.

        Raises:
            ValueError: If any feature column contains NaN or inf values.
        """
        logger.info(
            "Screening {} features for stationarity (asset={}, bar_type={}, alpha={})",
            len(feature_names),
            asset,
            bar_type,
            alpha,
        )

        results: list[StationarityTestResult] = []
        n_stationary: int = 0

        for fname in feature_names:
            series: np.ndarray[tuple[int], np.dtype[np.float64]] = df[fname].to_numpy(dtype=np.float64)

            if np.any(np.isnan(series)) or np.any(np.isinf(series)):
                msg: str = f"Feature '{fname}' contains NaN or inf values"
                raise ValueError(msg)

            # Constant series cannot be tested — classify as "inconclusive"
            if series.max() == series.min():
                logger.warning("Feature '{}' is constant — skipping ADF/KPSS, marking inconclusive", fname)
                results.append(
                    StationarityTestResult(
                        feature_name=fname,
                        adf_statistic=0.0,
                        adf_pvalue=1.0,
                        kpss_statistic=0.0,
                        kpss_pvalue=1.0,
                        is_stationary=False,
                        classification="inconclusive",
                        suggested_transformation=_suggest_transformation(fname),
                    ),
                )
                continue

            adf_stat: float
            adf_pval: float
            adf_stat, adf_pval = _run_adf(series)

            kpss_stat: float
            kpss_pval: float
            kpss_stat, kpss_pval = _run_kpss(series)

            adf_rejects: bool = adf_pval < alpha
            kpss_rejects: bool = kpss_pval < alpha

            classification: str = _classify_stationarity(adf_rejects, kpss_rejects)
            is_stationary: bool = classification == "stationary"

            suggested: str | None = None
            if not is_stationary:
                suggested = _suggest_transformation(fname)

            if is_stationary:
                n_stationary += 1

            results.append(
                StationarityTestResult(
                    feature_name=fname,
                    adf_statistic=adf_stat,
                    adf_pvalue=adf_pval,
                    kpss_statistic=kpss_stat,
                    kpss_pvalue=kpss_pval,
                    is_stationary=is_stationary,
                    classification=classification,
                    suggested_transformation=suggested,
                ),
            )

        n_non_stationary: int = len(results) - n_stationary
        logger.info(
            "Stationarity screening done: {}/{} stationary, {}/{} non-stationary",
            n_stationary,
            len(feature_names),
            n_non_stationary,
            len(feature_names),
        )

        return StationarityReport(
            results=tuple(results),
            n_stationary=n_stationary,
            n_non_stationary=n_non_stationary,
            asset=asset,
            bar_type=bar_type,
        )
