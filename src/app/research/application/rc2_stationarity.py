"""RC2 Section 2 stationarity analysis -- cross-asset aggregation and reporting.

Wraps the Phase 5 ``StationarityScreener`` to produce cross-asset stationarity
summaries for the RC2 notebook.  The analyzer runs ADF + KPSS on every feature
across all (asset, bar_type) combinations and classifies features as:

- **Universally stationary**: stationary in ALL combinations.
- **Universally non-stationary**: non-stationary in ALL combinations.
- **Mixed**: stationary in some combinations but not others.

The ``generate_therefore`` method produces a thesis-grade "Therefore" paragraph
that can be inserted directly into the RC2 notebook.

Uses the ML-research path (Pandas) per CLAUDE.md.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel

from src.app.profiling.application.stationarity import StationarityScreener
from src.app.profiling.domain.value_objects import StationarityReport


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class StationaritySummary(BaseModel, frozen=True):
    """Cross-asset stationarity summary for RC2.

    Aggregates per-(asset, bar_type) stationarity reports into a single
    summary that identifies universal and mixed stationarity features.

    Attributes:
        per_asset_bar: Individual stationarity reports per (asset, bar_type).
        universally_stationary: Features stationary in ALL (asset, bar_type) combos.
        universally_non_stationary: Features non-stationary in ALL combos.
        mixed_features: Features that are stationary in some combos but not all.
        n_total_features: Total number of distinct features tested.
        n_stationary_pct: Overall percentage of (feature, combo) pairs that are stationary.
        recommended_transformations: Mapping of non-stationary features to suggested transforms.
    """

    per_asset_bar: tuple[StationarityReport, ...]
    universally_stationary: tuple[str, ...]
    universally_non_stationary: tuple[str, ...]
    mixed_features: tuple[str, ...]
    n_total_features: int
    n_stationary_pct: float
    recommended_transformations: dict[str, str]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class RC2StationarityAnalyzer:
    """Generate stationarity analysis for RC2 notebook.

    Delegates individual (asset, bar_type) screening to ``StationarityScreener``
    and aggregates results across the full matrix of combinations.
    """

    def __init__(self, screener: StationarityScreener | None = None) -> None:
        """Initialize with an optional custom screener.

        Args:
            screener: Screener instance. Defaults to a new ``StationarityScreener``.
        """
        self._screener: StationarityScreener = screener or StationarityScreener()

    def analyze_features(  # noqa: PLR0912
        self,
        feature_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
        alpha: float = 0.05,
    ) -> StationaritySummary:
        """Run stationarity analysis across all (asset, bar_type) combinations.

        Args:
            feature_dfs: Mapping of ``(asset, bar_type)`` to Pandas DataFrames
                containing the feature columns.
            feature_names: Column names to test for stationarity.
            alpha: Significance level for ADF and KPSS tests.

        Returns:
            ``StationaritySummary`` with per-combination reports and cross-asset aggregation.

        Raises:
            ValueError: If ``feature_dfs`` is empty or ``feature_names`` is empty.
        """
        if not feature_dfs:
            msg: str = "feature_dfs must not be empty"
            raise ValueError(msg)
        if not feature_names:
            msg = "feature_names must not be empty"
            raise ValueError(msg)

        logger.info(
            "RC2 stationarity analysis: {} features across {} (asset, bar_type) combinations",
            len(feature_names),
            len(feature_dfs),
        )

        # Run per-(asset, bar_type) screening
        reports: list[StationarityReport] = []
        for (asset, bar_type), df in feature_dfs.items():
            report: StationarityReport = self._screener.screen(
                df=df,
                feature_names=feature_names,
                asset=asset,
                bar_type=bar_type,
                alpha=alpha,
            )
            reports.append(report)

        # Build cross-asset feature classification
        n_combos: int = len(reports)
        stationary_counts: dict[str, int] = defaultdict(int)
        total_tests: int = 0
        total_stationary: int = 0

        for report in reports:
            for result in report.results:
                if result.is_stationary:
                    stationary_counts[result.feature_name] += 1
                    total_stationary += 1
                total_tests += 1

        universally_stationary: list[str] = []
        universally_non_stationary: list[str] = []
        mixed_features: list[str] = []

        for fname in feature_names:
            count: int = stationary_counts.get(fname, 0)
            if count == n_combos:
                universally_stationary.append(fname)
            elif count == 0:
                universally_non_stationary.append(fname)
            else:
                mixed_features.append(fname)

        # Collect recommended transformations from non-stationary features
        recommended_transformations: dict[str, str] = {}
        for report in reports:
            for result in report.results:
                if (
                    not result.is_stationary
                    and result.suggested_transformation is not None
                    and result.feature_name not in recommended_transformations
                ):
                    recommended_transformations[result.feature_name] = result.suggested_transformation

        n_stationary_pct: float = (total_stationary / total_tests * 100.0) if total_tests > 0 else 0.0

        logger.info(
            "RC2 stationarity complete: {}/{} universally stationary, "
            "{} universally non-stationary, {} mixed, {:.1f}% overall",
            len(universally_stationary),
            len(feature_names),
            len(universally_non_stationary),
            len(mixed_features),
            n_stationary_pct,
        )

        return StationaritySummary(
            per_asset_bar=tuple(reports),
            universally_stationary=tuple(sorted(universally_stationary)),
            universally_non_stationary=tuple(sorted(universally_non_stationary)),
            mixed_features=tuple(sorted(mixed_features)),
            n_total_features=len(feature_names),
            n_stationary_pct=n_stationary_pct,
            recommended_transformations=recommended_transformations,
        )

    def render_summary_table(self, summary: StationaritySummary) -> pd.DataFrame:  # noqa: PLR6301
        """Render a per-feature summary table across all (asset, bar_type) combos.

        Columns: feature_name, n_stationary, n_combos, pct_stationary, classification.

        Args:
            summary: The stationarity summary to render.

        Returns:
            Pandas DataFrame with one row per feature.
        """
        n_combos: int = len(summary.per_asset_bar)
        rows: list[dict[str, object]] = []

        # Count stationary per feature
        stationary_counts: dict[str, int] = defaultdict(int)
        for report in summary.per_asset_bar:
            for result in report.results:
                if result.is_stationary:
                    stationary_counts[result.feature_name] += 1

        # Collect all feature names in original order
        feature_names: list[str] = _extract_ordered_feature_names(summary)

        for fname in feature_names:
            count: int = stationary_counts.get(fname, 0)
            pct: float = count / n_combos * 100.0 if n_combos > 0 else 0.0

            if fname in summary.universally_stationary:
                classification: str = "universally_stationary"
            elif fname in summary.universally_non_stationary:
                classification = "universally_non_stationary"
            else:
                classification = "mixed"

            transform: str = summary.recommended_transformations.get(fname, "")

            rows.append(
                {
                    "feature_name": fname,
                    "n_stationary": count,
                    "n_combos": n_combos,
                    "pct_stationary": round(pct, 1),
                    "classification": classification,
                    "suggested_transformation": transform,
                }
            )

        return pd.DataFrame(rows)

    def render_cross_asset_table(self, summary: StationaritySummary) -> pd.DataFrame:  # noqa: PLR6301
        """Render a feature-vs-(asset, bar_type) matrix of stationarity classifications.

        Rows are features; columns are ``"asset|bar_type"`` strings.
        Cell values are the joint ADF+KPSS classification
        (``"stationary"``, ``"trend_stationary"``, ``"unit_root"``, ``"inconclusive"``).

        Args:
            summary: The stationarity summary to render.

        Returns:
            Pandas DataFrame with features as rows, combos as columns.
        """
        feature_names: list[str] = _extract_ordered_feature_names(summary)
        combo_labels: list[str] = [f"{r.asset}|{r.bar_type}" for r in summary.per_asset_bar]

        # Build a lookup: (combo_label, feature) -> classification
        lookup: dict[tuple[str, str], str] = {}
        for report in summary.per_asset_bar:
            label: str = f"{report.asset}|{report.bar_type}"
            for result in report.results:
                lookup[(label, result.feature_name)] = result.classification

        data: dict[str, list[str]] = {}
        for combo in combo_labels:
            col_values: list[str] = []
            for fname in feature_names:
                classification: str = lookup.get((combo, fname), "missing")
                col_values.append(classification)
            data[combo] = col_values

        df: pd.DataFrame = pd.DataFrame(data, index=feature_names)
        df.index.name = "feature"
        return df

    def generate_therefore(self, summary: StationaritySummary) -> str:  # noqa: PLR6301
        """Generate the programmatic "Therefore" conclusion for RC2 Section 2.

        The conclusion follows a mechanical structure:
        1. State overall stationarity rate.
        2. List universally stationary features (safe for modeling).
        3. List universally non-stationary features with suggested transforms.
        4. List mixed features with a caution note.
        5. Conclude with the implication for modeling.

        Args:
            summary: The stationarity summary to base the conclusion on.

        Returns:
            Multi-line markdown string for the "Therefore" paragraph.
        """
        lines: list[str] = ["**Therefore:**\n"]

        # Overall rate
        n_univ_stat: int = len(summary.universally_stationary)
        n_univ_nonstat: int = len(summary.universally_non_stationary)
        n_mixed: int = len(summary.mixed_features)
        n_total: int = summary.n_total_features
        n_combos: int = len(summary.per_asset_bar)

        lines.append(
            f"Across {n_combos} (asset, bar_type) combinations, "
            f"{summary.n_stationary_pct:.1f}% of (feature, combination) pairs "
            f"are stationary by joint ADF+KPSS at alpha=0.05.\n"
        )

        # Universally stationary
        if n_univ_stat > 0:
            stat_list: str = ", ".join(f"`{f}`" for f in summary.universally_stationary)
            lines.append(
                f"- **{n_univ_stat}/{n_total} features are universally stationary** "
                f"(stationary in all {n_combos} combinations): {stat_list}. "
                f"These features are safe for direct use in ML models.\n"
            )
        else:
            lines.append(
                f"- **No features are universally stationary** across all {n_combos} combinations. "
                f"This indicates that transformations or careful feature-by-bar-type "
                f"selection is required.\n"
            )

        # Universally non-stationary
        if n_univ_nonstat > 0:
            nonstat_parts: list[str] = []
            for fname in summary.universally_non_stationary:
                transform: str = summary.recommended_transformations.get(fname, "investigate")
                nonstat_parts.append(f"`{fname}` (suggested: {transform})")
            nonstat_list: str = ", ".join(nonstat_parts)
            lines.append(
                f"- **{n_univ_nonstat}/{n_total} features are universally non-stationary**: "
                f"{nonstat_list}. "
                f"These MUST be transformed before modeling or excluded.\n"
            )

        # Mixed
        if n_mixed > 0:
            mixed_list: str = ", ".join(f"`{f}`" for f in summary.mixed_features)
            lines.append(
                f"- **{n_mixed}/{n_total} features have mixed stationarity** "
                f"(stationary in some combinations but not all): {mixed_list}. "
                f"These require per-combination treatment or robust transforms.\n"
            )

        # Implication
        high_stationarity_threshold: float = 80.0
        low_stationarity_threshold: float = 50.0

        if summary.n_stationary_pct >= high_stationarity_threshold:
            lines.append(
                "**Implication:** The majority of features are stationary. "
                "The feature set is suitable for direct use in ML modeling "
                "with only targeted transformations for non-stationary outliers."
            )
        elif summary.n_stationary_pct >= low_stationarity_threshold:
            lines.append(
                "**Implication:** A moderate fraction of features require transformation. "
                "Fractional differentiation or returns-based transforms should be applied "
                "to non-stationary features before modeling. The modeling pipeline should "
                "include stationarity as a pre-processing gate."
            )
        else:
            lines.append(
                "**Implication:** The majority of features are non-stationary. "
                "This is a significant concern for ML modeling. Aggressive "
                "transformation (fractional differentiation, first-differencing, "
                "z-scoring) is required before any model training."
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_ordered_feature_names(summary: StationaritySummary) -> list[str]:
    """Extract feature names in a stable order from the summary.

    Uses the first report's result ordering as the canonical order.
    Falls back to sorted universally_stationary + mixed + non_stationary.

    Args:
        summary: The stationarity summary.

    Returns:
        Ordered list of feature names.
    """
    if summary.per_asset_bar:
        first_report: StationarityReport = summary.per_asset_bar[0]
        names: list[str] = [r.feature_name for r in first_report.results]
        if names:
            return names

    # Fallback: combine all categories in a stable order
    all_names: list[str] = (
        list(summary.universally_stationary) + list(summary.mixed_features) + list(summary.universally_non_stationary)
    )
    return all_names
