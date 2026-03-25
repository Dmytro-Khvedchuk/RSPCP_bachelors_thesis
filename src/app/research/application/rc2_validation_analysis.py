"""RC2 Section 3 validation analysis -- MI tables, DA tables, stability, cross-bar, holdout.

Wraps the Phase 4D ``ValidationReport`` to produce thesis-grade tables and
comparison data for the RC2 notebook.  Every method returns a Pandas DataFrame
suitable for styled rendering in Jupyter.

Key tables:
    - **MI results table**: feature, MI (nats), raw p, BH p, MI/H(target) %.
    - **Ridge DA table**: feature, DA_observed, DA_null, DA excess (pp), p.
    - **Stability heatmap data**: feature x temporal window matrix.
    - **Cross-bar-type comparison**: MI scores across bar types per feature.
    - **Multi-horizon comparison**: MI and DA across forecast horizons.
    - **Holdout retention**: significance retention from train to holdout period.

Uses the ML-research path (Pandas / NumPy) per CLAUDE.md.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from loguru import logger

from src.app.features.domain.entities import FeatureValidationResult, ValidationReport


_EPS: Final[float] = 1e-12
"""Epsilon for division-by-zero protection."""


class RC2ValidationAnalyzer:
    """Validation result analysis for RC2 Section 3.

    All methods are pure transformations from ``ValidationReport`` (or dicts
    thereof) to Pandas DataFrames.  No database or file I/O is performed.
    """

    def build_mi_table(
        self,
        validation_report: ValidationReport,
        target_entropy: float,
    ) -> pd.DataFrame:
        """Build MI results table with effect size (MI/H(target) %).

        The MI/H(target) column provides a normalised effect size that is
        comparable across targets with different entropy levels.  For a
        Gaussian target, H(target) = 0.5 * log(2*pi*e*var).

        Args:
            validation_report: Output from ``FeatureValidator.validate()``.
            target_entropy: Shannon entropy of the target in nats.
                Use ``compute_target_entropy()`` to estimate this.

        Returns:
            DataFrame with columns: Feature, Group, MI (nats), Raw p,
            BH p, Significant, MI/H(target) %, Keep.  Sorted by MI
            descending.
        """
        rows: list[dict[str, object]] = []
        for r in validation_report.feature_results:
            mi_pct: float = (r.mi_score / max(target_entropy, _EPS)) * 100.0
            rows.append(
                {
                    "Feature": r.feature_name,
                    "Group": r.group,
                    "MI (nats)": round(r.mi_score, 6),
                    "Raw p": round(r.mi_pvalue, 4),
                    "BH p": round(r.fdr_corrected_p, 4),
                    "Significant": r.mi_significant,
                    "MI/H(target) %": round(mi_pct, 3),
                    "Keep": r.keep,
                }
            )

        df: pd.DataFrame = pd.DataFrame(rows)
        df = df.sort_values("MI (nats)", ascending=False).reset_index(drop=True)
        logger.debug(
            "MI table built: {}/{} significant",
            int(df["Significant"].sum()),
            len(df),
        )
        return df

    def build_da_table(
        self,
        validation_report: ValidationReport,
        breakeven_da: float,
    ) -> pd.DataFrame:
        """Build Ridge DA results table with economic significance markers.

        Computes two excess metrics:
            - **DA excess (pp)**: ``(DA_observed - 0.50) * 100`` -- raw edge
              over coin flip.
            - **DA vs break-even (pp)**: ``(DA_observed - breakeven_da) * 100``
              -- edge over the economic viability threshold.

        Args:
            validation_report: Output from ``FeatureValidator.validate()``.
            breakeven_da: Break-even directional accuracy from
                ``compute_breakeven_da()``.

        Returns:
            DataFrame with columns: Feature, Group, DA observed, DA null,
            DA excess (pp), DA vs break-even (pp), p, Beats null, Keep.
            Sorted by DA excess descending.
        """
        rows: list[dict[str, object]] = []
        for r in validation_report.feature_results:
            da_excess_pp: float = (r.directional_accuracy - 0.50) * 100.0
            da_vs_breakeven_pp: float = (r.directional_accuracy - breakeven_da) * 100.0
            rows.append(
                {
                    "Feature": r.feature_name,
                    "Group": r.group,
                    "DA observed": round(r.directional_accuracy, 4),
                    "DA null": round(r.da_null_mean, 4),
                    "DA excess (pp)": round(da_excess_pp, 2),
                    "DA vs break-even (pp)": round(da_vs_breakeven_pp, 2),
                    "p": round(r.da_pvalue, 4),
                    "Beats null": r.da_beats_null,
                    "Keep": r.keep,
                }
            )

        df: pd.DataFrame = pd.DataFrame(rows)
        df = df.sort_values("DA excess (pp)", ascending=False).reset_index(drop=True)
        logger.debug(
            "DA table built: {}/{} beat null",
            int(df["Beats null"].sum()),
            len(df),
        )
        return df

    def build_stability_heatmap_data(
        self,
        validation_reports_by_window: dict[str, ValidationReport],
    ) -> pd.DataFrame:
        """Build feature x temporal-window matrix of MI significance for heatmap.

        Each cell is 1 (MI significant at alpha) or 0 (not significant).
        Rows are features, columns are window labels.

        Args:
            validation_reports_by_window: Mapping of window label (e.g.
                ``"2020-2021"``) to the ``ValidationReport`` for that window.

        Returns:
            DataFrame with feature names as index, window labels as columns,
            and 0/1 integer values.  Returns empty DataFrame if input is empty.
        """
        if not validation_reports_by_window:
            return pd.DataFrame()

        # Use the first report to get feature names in order
        first_report: ValidationReport = next(iter(validation_reports_by_window.values()))
        feature_names: list[str] = [r.feature_name for r in first_report.feature_results]
        window_labels: list[str] = list(validation_reports_by_window.keys())

        # Build lookup: (window_label, feature_name) -> mi_significant
        data: dict[str, list[int]] = {}
        for label, report in validation_reports_by_window.items():
            sig_lookup: dict[str, bool] = {r.feature_name: r.mi_significant for r in report.feature_results}
            col: list[int] = [int(sig_lookup.get(fname, False)) for fname in feature_names]
            data[label] = col

        df: pd.DataFrame = pd.DataFrame(data, index=feature_names)
        df.index.name = "Feature"
        logger.debug(
            "Stability heatmap built: {} features x {} windows",
            len(feature_names),
            len(window_labels),
        )
        return df

    def build_cross_bar_comparison(
        self,
        reports: dict[str, ValidationReport],
    ) -> pd.DataFrame:
        """Build MI scores across bar types for each feature.

        Args:
            reports: Mapping of bar type label (e.g. ``"dollar"``) to
                ``ValidationReport``.

        Returns:
            DataFrame with Feature as index, bar types as columns,
            MI scores as values.  Missing features in a bar type get NaN.
        """
        if not reports:
            return pd.DataFrame()

        # Collect all unique feature names across bar types
        all_features: list[str] = []
        seen: set[str] = set()
        for report in reports.values():
            for r in report.feature_results:
                if r.feature_name not in seen:
                    all_features.append(r.feature_name)
                    seen.add(r.feature_name)

        bar_types: list[str] = list(reports.keys())
        data: dict[str, list[float]] = {}
        for bt, report in reports.items():
            mi_lookup: dict[str, float] = {r.feature_name: r.mi_score for r in report.feature_results}
            col: list[float] = [mi_lookup.get(fname, float("nan")) for fname in all_features]
            data[bt] = col

        df: pd.DataFrame = pd.DataFrame(data, index=all_features)
        df.index.name = "Feature"
        logger.debug(
            "Cross-bar comparison built: {} features x {} bar types",
            len(all_features),
            len(bar_types),
        )
        return df

    def build_multi_horizon_comparison(
        self,
        reports: dict[str, ValidationReport],
    ) -> pd.DataFrame:
        """Build MI and DA comparison across forecast horizons.

        For each feature, shows MI (nats), MI significant, DA observed,
        DA beats null across all tested horizons side by side.

        Args:
            reports: Mapping of horizon label (e.g. ``"fwd_logret_1"``)
                to ``ValidationReport``.

        Returns:
            DataFrame with multi-level columns: (horizon, metric).
            Features as rows.
        """
        if not reports:
            return pd.DataFrame()

        # Get feature names from the first report
        first_report: ValidationReport = next(iter(reports.values()))
        feature_names: list[str] = [r.feature_name for r in first_report.feature_results]

        rows: list[dict[str, object]] = []
        for fname in feature_names:
            row: dict[str, object] = {"Feature": fname}
            for horizon, report in reports.items():
                result: FeatureValidationResult | None = next(
                    (r for r in report.feature_results if r.feature_name == fname),
                    None,
                )
                if result is not None:
                    row[f"{horizon}|MI (nats)"] = round(result.mi_score, 6)
                    row[f"{horizon}|MI sig"] = result.mi_significant
                    row[f"{horizon}|DA"] = round(result.directional_accuracy, 4)
                    row[f"{horizon}|DA beats null"] = result.da_beats_null
                    row[f"{horizon}|Keep"] = result.keep
                else:
                    row[f"{horizon}|MI (nats)"] = float("nan")
                    row[f"{horizon}|MI sig"] = False
                    row[f"{horizon}|DA"] = float("nan")
                    row[f"{horizon}|DA beats null"] = False
                    row[f"{horizon}|Keep"] = False
            rows.append(row)

        df: pd.DataFrame = pd.DataFrame(rows)
        df = df.set_index("Feature")

        # Convert flat columns to MultiIndex for nicer display
        flat_cols: list[str] = list(df.columns)
        tuples: list[tuple[str, str]] = []
        for col in flat_cols:
            parts: list[str] = col.split("|", maxsplit=1)
            if len(parts) == 2:  # noqa: PLR2004
                tuples.append((parts[0], parts[1]))
            else:
                tuples.append((col, ""))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=["Horizon", "Metric"])

        logger.debug(
            "Multi-horizon comparison built: {} features x {} horizons",
            len(feature_names),
            len(reports),
        )
        return df

    def compute_holdout_retention(
        self,
        train_report: ValidationReport,
        holdout_report: ValidationReport,
    ) -> pd.DataFrame:
        """Compare significance between train and holdout periods.

        For each feature, reports whether it was kept on train data and
        whether it retains significance on holdout data (2023+).

        Args:
            train_report: ``ValidationReport`` from the training period.
            holdout_report: ``ValidationReport`` from the holdout period.

        Returns:
            DataFrame with columns: Feature, Train MI sig, Holdout MI sig,
            Train DA beats, Holdout DA beats, Train Keep, Holdout Keep,
            Retained.
        """
        # Build lookups
        train_lookup: dict[str, FeatureValidationResult] = {r.feature_name: r for r in train_report.feature_results}
        holdout_lookup: dict[str, FeatureValidationResult] = {
            r.feature_name: r for r in holdout_report.feature_results
        }

        all_features: list[str] = [r.feature_name for r in train_report.feature_results]
        rows: list[dict[str, object]] = []
        for fname in all_features:
            t: FeatureValidationResult | None = train_lookup.get(fname)
            h: FeatureValidationResult | None = holdout_lookup.get(fname)
            rows.append(
                {
                    "Feature": fname,
                    "Train MI sig": t.mi_significant if t else False,
                    "Holdout MI sig": h.mi_significant if h else False,
                    "Train DA beats": t.da_beats_null if t else False,
                    "Holdout DA beats": h.da_beats_null if h else False,
                    "Train Keep": t.keep if t else False,
                    "Holdout Keep": h.keep if h else False,
                    "Retained": (t.keep if t else False) and (h.keep if h else False),
                }
            )

        df: pd.DataFrame = pd.DataFrame(rows)
        n_train_kept: int = int(df["Train Keep"].sum())
        n_retained: int = int(df["Retained"].sum())
        retention_rate: float = n_retained / max(n_train_kept, 1) * 100.0
        logger.info(
            "Holdout retention: {}/{} features retained ({:.1f}%)",
            n_retained,
            n_train_kept,
            retention_rate,
        )
        return df

    def compute_horizon_summary(
        self,
        reports: dict[str, ValidationReport],
    ) -> pd.DataFrame:
        """Summarise each horizon: N features kept, N significant, mean DA.

        Args:
            reports: Mapping of horizon label to ``ValidationReport``.

        Returns:
            DataFrame with one row per horizon and summary statistics.
        """
        rows: list[dict[str, object]] = []
        for horizon, report in reports.items():
            n_mi_sig: int = sum(1 for r in report.feature_results if r.mi_significant)
            n_da_beats: int = sum(1 for r in report.feature_results if r.da_beats_null)
            n_kept: int = report.n_features_kept
            mean_da: float = float(np.mean([r.directional_accuracy for r in report.feature_results]))
            max_da: float = float(np.max([r.directional_accuracy for r in report.feature_results]))
            rows.append(
                {
                    "Horizon": horizon,
                    "N features": report.n_features_total,
                    "MI significant": n_mi_sig,
                    "DA beats null": n_da_beats,
                    "Kept": n_kept,
                    "Mean DA": round(mean_da, 4),
                    "Max DA": round(max_da, 4),
                    "Fallback": report.fallback_triggered,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------


def compute_target_entropy_gaussian(target: np.ndarray) -> float:  # type: ignore[type-arg]
    """Estimate target entropy assuming a Gaussian distribution.

    Uses the closed-form Gaussian entropy:
        H(X) = 0.5 * log(2 * pi * e * var(X))

    This is an upper bound on the true entropy for any distribution
    with the same variance (maximum entropy principle).

    Args:
        target: 1-D array of target values.

    Returns:
        Estimated entropy in nats.  Returns 0.0 if variance is zero.
    """
    var: float = float(np.var(target))
    if var <= 0.0:
        return 0.0
    return 0.5 * math.log(2.0 * math.pi * math.e * var)
