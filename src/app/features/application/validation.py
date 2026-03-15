"""Feature validation via permutation testing, BH correction, and temporal stability.

Uses the ML-research path (Pandas / NumPy / scikit-learn) per CLAUDE.md.
Polars-to-Pandas conversion happens at the ``validate()`` entry point.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from loguru import logger
from sklearn.feature_selection import mutual_info_regression  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]

from src.app.features.domain.entities import (
    FeatureValidationResult,
    InteractionTestResult,
    ValidationReport,
)
from src.app.features.domain.value_objects import FeatureSet, ValidationConfig


_EPS: Final[float] = 1e-12
"""Epsilon for division-by-zero protection."""


# ===================================================================
# Pure helper functions
# ===================================================================


def compute_mi_score(
    feature: np.ndarray[tuple[int], np.dtype[np.float64]],
    target: np.ndarray[tuple[int], np.dtype[np.float64]],
    random_seed: int,
) -> float:
    """Compute mutual information between a single feature and target.

    Args:
        feature: 1-D feature array.
        target: 1-D target array.
        random_seed: Random seed for MI estimation.

    Returns:
        Mutual information in nats (non-negative).
    """
    mi_values: np.ndarray[tuple[int], np.dtype[np.float64]] = mutual_info_regression(
        feature.reshape(-1, 1),
        target,
        random_state=random_seed,
    )
    return float(max(mi_values[0], 0.0))


def compute_mi_null_distribution(
    feature: np.ndarray[tuple[int], np.dtype[np.float64]],
    target: np.ndarray[tuple[int], np.dtype[np.float64]],
    n_permutations: int,
    random_seed: int,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Build a null distribution for MI by permuting the target.

    Args:
        feature: 1-D feature array.
        target: 1-D target array.
        n_permutations: Number of shuffles.
        random_seed: Base random seed.

    Returns:
        Array of MI scores under the null hypothesis.
    """
    rng: np.random.Generator = np.random.default_rng(random_seed)
    null_mis: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        shuffled_target: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.permutation(target)
        null_mis[i] = compute_mi_score(feature, shuffled_target, random_seed=random_seed + i + 1)
    return null_mis


def compute_empirical_pvalue(
    observed: float,
    null_distribution: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    """Compute empirical p-value using Phipson & Smyth (2010) formula.

    ``p = (count(null >= observed) + 1) / (n_permutations + 1)``

    Args:
        observed: Observed test statistic.
        null_distribution: Array of null test statistics.

    Returns:
        Empirical p-value in [0, 1].
    """
    n_permutations: int = len(null_distribution)
    count_ge: int = int(np.sum(null_distribution >= observed))
    return (count_ge + 1) / (n_permutations + 1)


def apply_bh_correction(
    pvalues: np.ndarray[tuple[int], np.dtype[np.float64]],
    alpha: float,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.bool_]], np.ndarray[tuple[int], np.dtype[np.float64]]]:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        pvalues: Array of raw p-values.
        alpha: Significance level.

    Returns:
        Tuple of (reject mask, corrected p-values).
    """
    reject: np.ndarray[tuple[int], np.dtype[np.bool_]]
    corrected: np.ndarray[tuple[int], np.dtype[np.float64]]
    reject, corrected, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    return reject, corrected


def compute_directional_accuracy(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    """Compute directional accuracy -- fraction with matching sign.

    Zero values are excluded from evaluation.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Directional accuracy in [0, 1], or 0.5 if no non-zero pairs.
    """
    nonzero_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = (y_true != 0) & (y_pred != 0)
    if not np.any(nonzero_mask):
        return 0.5
    return float(np.mean(np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask])))


def compute_dc_mae(
    y_true: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    """Compute direction-conditional MAE (only correct-sign predictions).

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        DC-MAE, or ``inf`` if no correct-direction predictions exist.
    """
    nonzero_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = (y_true != 0) & (y_pred != 0)
    correct_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = nonzero_mask & (np.sign(y_true) == np.sign(y_pred))
    if not np.any(correct_mask):
        return float("inf")
    return float(np.mean(np.abs(y_true[correct_mask] - y_pred[correct_mask])))


def evaluate_single_feature_ridge(
    feature: np.ndarray[tuple[int], np.dtype[np.float64]],
    target: np.ndarray[tuple[int], np.dtype[np.float64]],
    ridge_alpha: float,
    train_fraction: float = 0.7,
) -> tuple[float, float]:
    """Fit single-feature Ridge on temporal train split, evaluate on test split.

    Args:
        feature: 1-D feature array.
        target: 1-D target array.
        ridge_alpha: Ridge regularisation parameter.
        train_fraction: Fraction of data for training (temporal, no shuffle).

    Returns:
        Tuple of (directional_accuracy, dc_mae) on the test split.
    """
    n: int = len(feature)
    split_idx: int = int(n * train_fraction)
    model: Ridge = Ridge(alpha=ridge_alpha)
    x_2d: np.ndarray[tuple[int, int], np.dtype[np.float64]] = feature.reshape(-1, 1)
    model.fit(x_2d[:split_idx], target[:split_idx])
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = model.predict(x_2d[split_idx:])
    da: float = compute_directional_accuracy(target[split_idx:], y_pred)
    dc: float = compute_dc_mae(target[split_idx:], y_pred)
    return da, dc


def compute_ridge_null_distribution(  # noqa: PLR0913, PLR0917
    feature: np.ndarray[tuple[int], np.dtype[np.float64]],
    target: np.ndarray[tuple[int], np.dtype[np.float64]],
    n_permutations: int,
    ridge_alpha: float,
    random_seed: int,
    train_fraction: float = 0.7,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]:
    """Build null distributions for DA and DC-MAE by permuting the target.

    Permutation shuffles the full target BEFORE the temporal split,
    which is correct — both observed and null use the same split boundary.

    Args:
        feature: 1-D feature array.
        target: 1-D target array.
        n_permutations: Number of shuffles.
        ridge_alpha: Ridge regularisation parameter.
        random_seed: Base random seed.
        train_fraction: Fraction of data for training (forwarded to Ridge).

    Returns:
        Tuple of (DA null array, DC-MAE null array).
    """
    rng: np.random.Generator = np.random.default_rng(random_seed)
    null_da: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(n_permutations, dtype=np.float64)
    null_dc_mae: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        shuffled_target: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.permutation(target)
        da_i: float
        dc_i: float
        da_i, dc_i = evaluate_single_feature_ridge(feature, shuffled_target, ridge_alpha, train_fraction)
        null_da[i] = da_i
        null_dc_mae[i] = dc_i
    return null_da, null_dc_mae


def classify_feature_group(feature_name: str, feature_groups: dict[str, tuple[str, ...]]) -> str:
    """Classify a feature into its group based on prefix matching.

    Args:
        feature_name: Column name to classify.
        feature_groups: Mapping of group name to prefix tuples.

    Returns:
        Group name, or ``"other"`` if no prefix matches.
    """
    for group_name, prefixes in feature_groups.items():
        for prefix in prefixes:
            if feature_name.startswith(prefix):
                return group_name
    return "other"


def _assemble_report(  # noqa: PLR0913, PLR0914, PLR0917
    feature_names: list[str],
    mi_results: tuple[list[float], list[float]],
    reject_mask: np.ndarray[tuple[int], np.dtype[np.bool_]],
    corrected_pvalues: np.ndarray[tuple[int], np.dtype[np.float64]],
    ridge_results: tuple[list[float], list[float], list[bool], list[float], list[float], list[float]],
    stability_results: tuple[list[float], list[bool]],
    interaction_results: list[InteractionTestResult],
    config: ValidationConfig,
    *,
    stability_skipped: bool,
) -> ValidationReport:
    """Assemble per-feature results into a ValidationReport.

    All intermediate test results are bundled into tuples by their
    respective test batteries.  This function unpacks them and creates
    one :class:`FeatureValidationResult` per feature.

    Args:
        feature_names: Feature column names.
        mi_results: Tuple of (MI scores, raw p-values).
        reject_mask: BH rejection mask.
        corrected_pvalues: BH-corrected p-values.
        ridge_results: Tuple of (DA scores, DA null means, DA beats,
            DC-MAEs, DC-MAE null means, DA p-values).
        stability_results: Tuple of (stability scores, is_stable flags).
        interaction_results: Group interaction test results.
        config: Validation config.
        stability_skipped: Whether temporal stability was skipped.

    Returns:
        Complete validation report.
    """
    mi_scores: list[float] = mi_results[0]
    raw_pvalues: list[float] = mi_results[1]
    da_scores, da_null_means, da_beats, dc_maes, dc_mae_null_means, da_pvalues = ridge_results
    stability_scores: list[float] = stability_results[0]
    is_stable_flags: list[bool] = stability_results[1]

    feature_results: list[FeatureValidationResult] = []
    for i, fname in enumerate(feature_names):
        mi_sig: bool = bool(reject_mask[i])
        keep: bool = mi_sig and da_beats[i] and is_stable_flags[i]
        group: str = classify_feature_group(fname, config.feature_groups)
        feature_results.append(
            FeatureValidationResult(
                feature_name=fname,
                mi_score=mi_scores[i],
                mi_pvalue=raw_pvalues[i],
                fdr_corrected_p=float(corrected_pvalues[i]),
                mi_significant=mi_sig,
                directional_accuracy=da_scores[i],
                da_null_mean=da_null_means[i],
                da_pvalue=da_pvalues[i],
                da_beats_null=da_beats[i],
                dc_mae=dc_maes[i],
                dc_mae_null_mean=dc_mae_null_means[i],
                stability_score=stability_scores[i],
                is_stable=is_stable_flags[i],
                group=group,
                keep=keep,
            ),
        )

    kept: list[str] = sorted(r.feature_name for r in feature_results if r.keep)
    dropped: list[str] = sorted(r.feature_name for r in feature_results if not r.keep)

    # --- Fallback: ensure minimum features kept (C3) ---
    fallback_triggered: bool = False
    if len(kept) < config.min_features_kept and len(feature_names) >= config.min_features_kept:
        fallback_triggered = True
        logger.warning(
            "Only {}/{} features kept — triggering fallback to keep top {}",
            len(kept),
            len(feature_names),
            config.min_features_kept,
        )
        # Rank by composite score: (1 - mi_pvalue) + (DA - DA_null_mean) + stability_score
        scored: list[tuple[float, int]] = []
        for i, result in enumerate(feature_results):
            composite: float = (
                (1.0 - result.mi_pvalue) + (result.directional_accuracy - result.da_null_mean) + result.stability_score
            )
            scored.append((composite, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_indices: set[int] = {idx for _, idx in scored[: config.min_features_kept]}

        rebuilt: list[FeatureValidationResult] = []
        for i, result in enumerate(feature_results):
            if i in top_indices and not result.keep:
                rebuilt.append(
                    FeatureValidationResult(**{**result.model_dump(), "keep": True}),
                )
            else:
                rebuilt.append(result)
        feature_results = rebuilt
        kept = sorted(r.feature_name for r in feature_results if r.keep)
        dropped = sorted(r.feature_name for r in feature_results if not r.keep)

    logger.info(
        "Validation complete: {}/{} features kept, {}/{} dropped{}",
        len(kept),
        len(feature_names),
        len(dropped),
        len(feature_names),
        " (fallback)" if fallback_triggered else "",
    )

    return ValidationReport(
        feature_results=tuple(feature_results),
        interaction_results=tuple(interaction_results),
        n_features_total=len(feature_names),
        n_features_kept=len(kept),
        n_features_dropped=len(dropped),
        kept_feature_names=tuple(kept),
        dropped_feature_names=tuple(dropped),
        fallback_triggered=fallback_triggered,
        stability_skipped=stability_skipped,
    )


def _evaluate_group_interaction(  # noqa: PLR0913, PLR0917
    group_features: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    target: np.ndarray[tuple[int], np.dtype[np.float64]],
    group_name: str,
    group_feature_names: tuple[str, ...],
    ridge_alpha: float,
    redundancy_tolerance: float,
) -> InteractionTestResult:
    """Evaluate interaction within a single feature group.

    Args:
        group_features: 2-D array with only this group's columns.
        target: 1-D target array.
        group_name: Name of this feature group.
        group_feature_names: Column names in this group.
        ridge_alpha: Ridge regularisation parameter.
        redundancy_tolerance: Multiplier for max individual R² when testing redundancy.

    Returns:
        Interaction test result for the group.
    """
    model_combined: Ridge = Ridge(alpha=ridge_alpha)
    model_combined.fit(group_features, target)
    combined_r2: float = float(model_combined.score(group_features, target))

    individual_r2s: list[float] = []
    for j in range(group_features.shape[1]):
        feat_col: np.ndarray[tuple[int, int], np.dtype[np.float64]] = group_features[:, j].reshape(-1, 1)
        model_single: Ridge = Ridge(alpha=ridge_alpha)
        model_single.fit(feat_col, target)
        individual_r2s.append(float(model_single.score(feat_col, target)))

    sum_individual: float = sum(individual_r2s)
    max_individual: float = max(individual_r2s)
    return InteractionTestResult(
        group_name=group_name,
        features_in_group=group_feature_names,
        combined_r2=combined_r2,
        sum_individual_r2=sum_individual,
        interaction_ratio=combined_r2 / (sum_individual + _EPS),
        has_interaction=combined_r2 > sum_individual,
        is_redundant=combined_r2 <= max_individual * redundancy_tolerance,
    )


# ===================================================================
# FeatureValidator
# ===================================================================


class FeatureValidator:
    """Stateless validator that tests features for genuine predictive power.

    The single public method :meth:`validate` runs four independent test
    batteries and assembles a :class:`ValidationReport`:

    1. **MI permutation test** -- mutual information with BH correction.
    2. **Ridge DA / DC-MAE test** -- directional accuracy and direction-
       conditional MAE from a single-feature Ridge regression.
    3. **Temporal stability** -- MI significance across year-based windows.
    4. **Group interaction test** (informational) -- Ridge R-squared
       synergy / redundancy within feature groups.
    """

    def validate(  # noqa: PLR6301
        self,
        feature_set: FeatureSet,
        config: ValidationConfig,
    ) -> ValidationReport:
        """Run full validation pipeline on a feature set.

        Args:
            feature_set: Output from FeatureMatrixBuilder.
            config: Validation parameters.

        Returns:
            Complete validation report with per-feature keep/drop decisions.

        Raises:
            ValueError: If feature matrix or target contains NaN or inf values.
        """
        logger.info("Starting feature validation ({} features)", len(feature_set.feature_columns))

        df_pd: pd.DataFrame = feature_set.df.to_pandas()
        feature_names: list[str] = list(feature_set.feature_columns)
        target: np.ndarray[tuple[int], np.dtype[np.float64]] = df_pd[config.target_col].to_numpy(
            dtype=np.float64,
        )
        feature_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]] = df_pd[feature_names].to_numpy(
            dtype=np.float64,
        )

        # --- NaN/inf guard (I6) ---
        if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
            msg: str = "Feature matrix contains NaN or inf values"
            raise ValueError(msg)
        if np.any(np.isnan(target)) or np.any(np.isinf(target)):
            msg = "Target array contains NaN or inf values"
            raise ValueError(msg)

        stability_skipped: bool = config.timestamp_col not in df_pd.columns

        # --- 1. MI test ---
        mi_results: tuple[list[float], list[float]] = FeatureValidator._run_mi_test(
            feature_matrix,
            target,
            config,
        )
        reject_mask: np.ndarray[tuple[int], np.dtype[np.bool_]]
        corrected_pvalues: np.ndarray[tuple[int], np.dtype[np.float64]]
        reject_mask, corrected_pvalues = apply_bh_correction(
            np.array(mi_results[1], dtype=np.float64),
            config.alpha,
        )

        # --- 2. Ridge test ---
        ridge_results: tuple[list[float], list[float], list[bool], list[float], list[float], list[float]] = (
            FeatureValidator._run_ridge_test(feature_matrix, target, config)
        )

        # --- 3. Temporal stability ---
        stability_results: tuple[list[float], list[bool]] = FeatureValidator._run_temporal_stability_test(
            df_pd,
            feature_names,
            config,
        )

        # --- 4. Interaction test ---
        interaction_results: list[InteractionTestResult] = FeatureValidator._run_interaction_test(
            feature_matrix,
            target,
            feature_names,
            config,
        )

        return _assemble_report(
            feature_names,
            mi_results,
            reject_mask,
            corrected_pvalues,
            ridge_results,
            stability_results,
            interaction_results,
            config,
            stability_skipped=stability_skipped,
        )

    # ------------------------------------------------------------------
    # Private test batteries
    # ------------------------------------------------------------------

    @staticmethod
    def _run_mi_test(
        feature_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        target: np.ndarray[tuple[int], np.dtype[np.float64]],
        config: ValidationConfig,
    ) -> tuple[list[float], list[float]]:
        """Run MI permutation test for each feature.

        Args:
            feature_matrix: 2-D array (n_samples, n_features).
            target: 1-D target array.
            config: Validation config.

        Returns:
            Tuple of (MI scores, raw p-values).
        """
        n_features: int = feature_matrix.shape[1]
        logger.info("Running MI permutation test ({} permutations, {} features)", config.n_permutations_mi, n_features)

        mi_scores: list[float] = []
        raw_pvalues: list[float] = []

        for j in range(n_features):
            seed_j: int = config.random_seed + j * (config.n_permutations_mi + 1)
            feat_col: np.ndarray[tuple[int], np.dtype[np.float64]] = feature_matrix[:, j]
            observed_mi: float = compute_mi_score(feat_col, target, seed_j)
            null_dist: np.ndarray[tuple[int], np.dtype[np.float64]] = compute_mi_null_distribution(
                feat_col,
                target,
                config.n_permutations_mi,
                seed_j,
            )
            p_val: float = compute_empirical_pvalue(observed_mi, null_dist)
            mi_scores.append(observed_mi)
            raw_pvalues.append(p_val)

        n_sig: int = sum(1 for p in raw_pvalues if p < config.alpha)
        logger.info("MI test done: {}/{} features with raw p < {}", n_sig, n_features, config.alpha)
        return mi_scores, raw_pvalues

    @staticmethod
    def _run_ridge_test(  # noqa: PLR0914
        feature_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        target: np.ndarray[tuple[int], np.dtype[np.float64]],
        config: ValidationConfig,
    ) -> tuple[list[float], list[float], list[bool], list[float], list[float], list[float]]:
        """Run single-feature Ridge evaluation with null distributions.

        Args:
            feature_matrix: 2-D array (n_samples, n_features).
            target: 1-D target array.
            config: Validation config.

        Returns:
            Tuple of (DA scores, DA null means, DA beats null, DC-MAEs,
            DC-MAE null means, DA p-values).
        """
        n_features: int = feature_matrix.shape[1]
        logger.info(
            "Running Ridge DA/DC-MAE test ({} permutations, {} features)",
            config.n_permutations_ridge,
            n_features,
        )

        da_scores: list[float] = []
        da_null_means: list[float] = []
        da_beats: list[bool] = []
        da_pvalues: list[float] = []
        dc_maes: list[float] = []
        dc_mae_null_means: list[float] = []

        for j in range(n_features):
            seed_j: int = config.random_seed + j * (config.n_permutations_ridge + 1)
            feat_col: np.ndarray[tuple[int], np.dtype[np.float64]] = feature_matrix[:, j]
            da: float
            dc: float
            da, dc = evaluate_single_feature_ridge(feat_col, target, config.ridge_alpha)

            null_da: np.ndarray[tuple[int], np.dtype[np.float64]]
            null_dc: np.ndarray[tuple[int], np.dtype[np.float64]]
            null_da, null_dc = compute_ridge_null_distribution(
                feat_col,
                target,
                config.n_permutations_ridge,
                config.ridge_alpha,
                seed_j,
            )
            da_null_mean: float = float(np.mean(null_da))
            dc_mae_null_mean: float = (
                float(np.mean(null_dc[np.isfinite(null_dc)])) if np.any(np.isfinite(null_dc)) else float("inf")
            )
            da_pval: float = compute_empirical_pvalue(da, null_da)

            da_scores.append(da)
            da_null_means.append(da_null_mean)
            da_pvalues.append(da_pval)
            da_beats.append(da_pval < config.alpha)
            dc_maes.append(dc)
            dc_mae_null_means.append(dc_mae_null_mean)

        n_beats: int = sum(da_beats)
        logger.info("Ridge test done: {}/{} features beat DA null", n_beats, n_features)
        return da_scores, da_null_means, da_beats, dc_maes, dc_mae_null_means, da_pvalues

    @staticmethod
    def _run_temporal_stability_test(
        df_pd: pd.DataFrame,
        feature_names: list[str],
        config: ValidationConfig,
    ) -> tuple[list[float], list[bool]]:
        """Test MI significance across temporal windows.

        Args:
            df_pd: Full Pandas DataFrame.
            feature_names: List of feature column names.
            config: Validation config.

        Returns:
            Tuple of (stability scores, is_stable flags).
        """
        if config.timestamp_col not in df_pd.columns:
            logger.warning(
                "Timestamp column '{}' not found -- skipping temporal stability (all stable)",
                config.timestamp_col,
            )
            return [1.0] * len(feature_names), [True] * len(feature_names)

        n_features: int = len(feature_names)
        logger.info(
            "Running temporal stability test ({} windows, {} features)",
            len(config.temporal_windows),
            n_features,
        )

        sig_counts: list[int] = [0] * n_features
        valid_windows: int = 0
        timestamps: pd.Series = pd.to_datetime(df_pd[config.timestamp_col])  # type: ignore[call-overload]

        for window_idx, (start_year, end_year) in enumerate(config.temporal_windows):
            mask: pd.Series = (timestamps.dt.year >= start_year) & (timestamps.dt.year < end_year)  # type: ignore[union-attr]
            window_df: pd.DataFrame = df_pd.loc[mask]

            if len(window_df) < config.min_window_rows:
                logger.debug(
                    "Skipping window [{}, {}) -- only {} rows (need {})",
                    start_year,
                    end_year,
                    len(window_df),
                    config.min_window_rows,
                )
                continue

            valid_windows += 1
            window_target: np.ndarray[tuple[int], np.dtype[np.float64]] = window_df[config.target_col].to_numpy(
                dtype=np.float64,
            )

            for j, fname in enumerate(feature_names):
                seed_wf: int = config.random_seed + window_idx * 10000 + j * (config.n_permutations_stability + 1)
                feat_col: np.ndarray[tuple[int], np.dtype[np.float64]] = window_df[fname].to_numpy(dtype=np.float64)
                observed_mi: float = compute_mi_score(feat_col, window_target, seed_wf)
                null_dist: np.ndarray[tuple[int], np.dtype[np.float64]] = compute_mi_null_distribution(
                    feat_col,
                    window_target,
                    config.n_permutations_stability,
                    seed_wf,
                )
                p_val: float = compute_empirical_pvalue(observed_mi, null_dist)
                if p_val < config.alpha:
                    sig_counts[j] += 1

        if valid_windows < config.min_valid_windows:
            logger.warning(
                "Only {} valid temporal windows (need {}) -- defaulting all features to stable",
                valid_windows,
                config.min_valid_windows,
            )
            return [1.0] * n_features, [True] * n_features

        stability_scores: list[float] = [c / valid_windows for c in sig_counts]
        is_stable: list[bool] = [s >= config.stability_threshold for s in stability_scores]

        n_stable: int = sum(is_stable)
        logger.info("Temporal stability done: {}/{} features stable", n_stable, n_features)
        return stability_scores, is_stable

    @staticmethod
    def _run_interaction_test(
        feature_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        target: np.ndarray[tuple[int], np.dtype[np.float64]],
        feature_names: list[str],
        config: ValidationConfig,
    ) -> list[InteractionTestResult]:
        """Test group-level Ridge R-squared for synergy / redundancy.

        Args:
            feature_matrix: 2-D array (n_samples, n_features).
            target: 1-D target array.
            feature_names: Feature column names.
            config: Validation config.

        Returns:
            List of interaction test results per group.
        """
        logger.info("Running group interaction test ({} groups)", len(config.feature_groups))

        group_indices: dict[str, list[int]] = {}
        for j, fname in enumerate(feature_names):
            group: str = classify_feature_group(fname, config.feature_groups)
            if group not in group_indices:
                group_indices[group] = []
            group_indices[group].append(j)

        results: list[InteractionTestResult] = [
            _evaluate_group_interaction(
                group_features=feature_matrix[:, indices],
                target=target,
                group_name=group_name,
                group_feature_names=tuple(feature_names[i] for i in indices),
                ridge_alpha=config.ridge_alpha,
                redundancy_tolerance=config.redundancy_tolerance,
            )
            for group_name, indices in group_indices.items()
            if len(indices) >= config.min_group_features
        ]

        logger.info("Interaction test done: {} groups with {}+ features", len(results), config.min_group_features)
        return results
