"""RC2 Section 3 feature exploration -- VIF, correlation, distributions, and rationale table.

Provides the ``RC2FeatureAnalyzer`` service for computing multicollinearity diagnostics
(VIF via Belsley et al. 1980), Pearson correlation matrices, distribution summaries,
and the feature rationale table that maps each feature to its economic justification.

Uses the ML-research path (Pandas / NumPy) per CLAUDE.md.
"""

from __future__ import annotations

import warnings
from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from loguru import logger
from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore[import-untyped]
from statsmodels.tools.tools import add_constant  # type: ignore[import-untyped]

from src.app.research.domain.rc2_value_objects import (
    FeatureRationale,
    build_default_feature_rationales,
)

_VIF_INF_THRESHOLD: Final[float] = 1e10
"""VIF values above this are treated as infinity (near-singular design matrix)."""

_EPS: Final[float] = 1e-12
"""Epsilon for division-by-zero protection."""

_NEAR_ZERO_STD: Final[float] = 1e-15
"""Standard deviation below this is treated as constant (zero variance)."""


class RC2FeatureAnalyzer:
    """Feature exploration analysis for RC2 Section 3.

    All methods are stateless and operate on Pandas DataFrames.
    The analyzer computes diagnostics for ALL 23 features (kept and dropped)
    to prevent survivorship bias per the pre-registration commitment.
    """

    def compute_vif(
        self,
        feature_df: pd.DataFrame,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Compute Variance Inflation Factor for each feature.

        Uses ``statsmodels.stats.outliers_influence.variance_inflation_factor``
        on the design matrix with an added constant column.  Features with
        near-zero variance or singular columns are assigned VIF = inf.

        Args:
            feature_df: DataFrame containing the feature columns.
            feature_names: Column names to compute VIF for.

        Returns:
            DataFrame with columns ``feature``, ``vif``, sorted by VIF descending.

        Raises:
            ValueError: If ``feature_names`` is empty or not found in ``feature_df``.
        """
        if not feature_names:
            msg: str = "feature_names must not be empty"
            raise ValueError(msg)

        missing: set[str] = set(feature_names) - set(feature_df.columns)
        if missing:
            msg = f"Features not found in DataFrame: {sorted(missing)}"
            raise ValueError(msg)

        # Extract feature matrix and add constant for intercept
        x_raw: np.ndarray = feature_df[feature_names].to_numpy(dtype=np.float64)

        # Drop rows with NaN/inf — replacing with 0 corrupts the correlation structure
        nan_mask: np.ndarray = np.isnan(x_raw).any(axis=1) | np.isinf(x_raw).any(axis=1)
        n_dropped: int = int(nan_mask.sum())
        if n_dropped > 0:
            logger.warning("Dropped {}/{} rows with NaN/inf for VIF computation", n_dropped, len(x_raw))
        x_clean: np.ndarray = x_raw[~nan_mask]
        if len(x_clean) < len(feature_names) + 2:
            logger.warning("Too few complete rows ({}) for VIF; returning inf for all", len(x_clean))
            return pd.DataFrame({"feature": feature_names, "vif": [float("inf")] * len(feature_names)})

        x_const: np.ndarray = add_constant(x_clean)

        # Determine column offset: add_constant may not add a column if one already exists
        const_added: bool = x_const.shape[1] > x_clean.shape[1]
        col_offset: int = 1 if const_added else 0

        vif_values: list[float] = []
        for i in range(len(feature_names)):
            # Zero-variance (constant) columns get inf VIF -- perfectly collinear with intercept
            if np.std(x_clean[:, i]) < _NEAR_ZERO_STD:
                vif_values.append(float("inf"))
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vif_val: float = float(variance_inflation_factor(x_const, i + col_offset))
                # Cap extreme values
                if np.isnan(vif_val) or np.isinf(vif_val) or vif_val > _VIF_INF_THRESHOLD:
                    vif_val = float("inf")
            except (np.linalg.LinAlgError, ValueError, IndexError):  # fmt: skip
                logger.warning("VIF computation failed for feature '{}', assigning inf", feature_names[i])
                vif_val = float("inf")
            vif_values.append(vif_val)

        result: pd.DataFrame = pd.DataFrame(
            {
                "feature": feature_names,
                "vif": vif_values,
            }
        )
        return result.sort_values("vif", ascending=False).reset_index(drop=True)

    def compute_correlation_matrix(
        self,
        feature_df: pd.DataFrame,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Compute Pearson correlation matrix for the given features.

        Args:
            feature_df: DataFrame containing the feature columns.
            feature_names: Column names to include in the correlation matrix.

        Returns:
            Square DataFrame with features as both index and columns,
            values are Pearson correlation coefficients.

        Raises:
            ValueError: If ``feature_names`` is empty.
        """
        if not feature_names:
            msg: str = "feature_names must not be empty"
            raise ValueError(msg)

        corr_matrix: pd.DataFrame = feature_df[feature_names].corr(method="pearson")
        return corr_matrix

    def build_feature_rationale_table(self) -> pd.DataFrame:
        """Build the feature rationale table from pre-defined rationales.

        Each row contains the feature name, group, formula, economic rationale,
        stationarity expectation, literature reference, and expected sign.

        Returns:
            DataFrame with one row per feature (23 rows for the default set).
        """
        rationales: tuple[FeatureRationale, ...] = build_default_feature_rationales()

        rows: list[dict[str, object]] = [
            {
                "Feature": r.feature_name,
                "Group": r.group,
                "Formula": r.formula_summary,
                "Economic Rationale": r.economic_rationale,
                "Stationarity Exp.": r.stationarity_expectation.value,
                "Transformation": r.transformation_if_nonstationary or "--",
                "Reference": r.literature_reference or "--",
            }
            for r in rationales
        ]

        return pd.DataFrame(rows)

    def compute_feature_distributions(
        self,
        feature_df: pd.DataFrame,
        feature_names: list[str],
        kept_features: list[str],
    ) -> pd.DataFrame:
        """Compute summary statistics per feature with kept/dropped classification.

        Args:
            feature_df: DataFrame containing the feature columns.
            feature_names: All feature column names.
            kept_features: Subset of feature_names that passed validation.

        Returns:
            DataFrame with columns: feature, status (kept/dropped), mean, std,
            median, skew, kurtosis, min, max, q25, q75.

        Raises:
            ValueError: If ``feature_names`` is empty.
        """
        if not feature_names:
            msg: str = "feature_names must not be empty"
            raise ValueError(msg)

        kept_set: set[str] = set(kept_features)
        rows: list[dict[str, object]] = []

        for fname in feature_names:
            if fname not in feature_df.columns:
                continue

            series: pd.Series = feature_df[fname].dropna()  # type: ignore[type-arg]
            status: str = "kept" if fname in kept_set else "dropped"

            rows.append(
                {
                    "feature": fname,
                    "status": status,
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "median": float(series.median()),
                    "skew": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "n_obs": len(series),
                }
            )

        return pd.DataFrame(rows)

    def compute_feature_target_correlations(
        self,
        feature_df: pd.DataFrame,
        feature_names: list[str],
        target_col: str,
    ) -> pd.DataFrame:
        """Compute Pearson and Spearman correlation of each feature with the target.

        Args:
            feature_df: DataFrame containing feature and target columns.
            feature_names: Feature column names.
            target_col: Target column name.

        Returns:
            DataFrame with columns: feature, pearson_r, spearman_r, abs_pearson_r.
            Sorted by absolute Pearson correlation descending.

        Raises:
            ValueError: If ``target_col`` is not in ``feature_df``.
        """
        if target_col not in feature_df.columns:
            msg: str = f"Target column '{target_col}' not found in DataFrame"
            raise ValueError(msg)

        rows: list[dict[str, object]] = []
        target_series: pd.Series = feature_df[target_col]  # type: ignore[type-arg]

        for fname in feature_names:
            if fname not in feature_df.columns:
                continue
            feature_series: pd.Series = feature_df[fname]  # type: ignore[type-arg]
            # Drop rows where either is NaN
            valid_mask: pd.Series = feature_series.notna() & target_series.notna()  # type: ignore[type-arg]
            if valid_mask.sum() < 3:  # noqa: PLR2004
                rows.append(
                    {
                        "feature": fname,
                        "pearson_r": float("nan"),
                        "spearman_r": float("nan"),
                        "abs_pearson_r": float("nan"),
                    }
                )
                continue

            pearson_r: float = float(feature_series[valid_mask].corr(target_series[valid_mask]))
            spearman_r: float = float(feature_series[valid_mask].corr(target_series[valid_mask], method="spearman"))
            rows.append(
                {
                    "feature": fname,
                    "pearson_r": pearson_r,
                    "spearman_r": spearman_r,
                    "abs_pearson_r": abs(pearson_r),
                }
            )

        result: pd.DataFrame = pd.DataFrame(rows)
        return result.sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
