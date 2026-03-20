"""RC2 pre-registration specification and rendering for thesis-grade confirmatory analysis.

Generates an immutable pre-registration document from project configs, encoding
all decision rules, trial counting categories, and negative-result protocols
BEFORE any RC2 analysis is executed.  This converts exploratory analysis into
confirmatory analysis per Nosek et al. (2018) and reduces researcher degrees
of freedom for the Phase 14 Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

The narrative arc:
    1. Pre-Registration (Section 1) -- defines mechanical rules
    2. Sections 2-7 -- apply rules mechanically, no human judgment
    3. Section 8 Go/No-Go -- DETERMINED by rules, not by post-hoc interpretation
    4. Phase 14 DSR -- uses trial count FROM pre-registration
    5. Any post-hoc deviation -- documented, increments trial count
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.features.domain.value_objects import FeatureConfig, ValidationConfig
from src.app.profiling.domain.value_objects import (
    DataPartition,
    PredictabilityConfig,
    ProfilingConfig,
    TierConfig,
)


# ---------------------------------------------------------------------------
# Domain value objects
# ---------------------------------------------------------------------------


class TrialCountCategory(BaseModel, frozen=True):
    """A single category of researcher decisions counted for DSR correction.

    Each category represents a class of choices that inflate the effective
    number of independent trials.  The counting rule documents HOW the
    count is determined so the Phase 14 DSR computation is reproducible.

    Attributes:
        name: Human-readable category name (e.g. "Feature selection decisions").
        counting_rule: Description of what counts as one trial in this category.
        initial_count: Number of trials counted at pre-registration time.
    """

    name: str
    counting_rule: str
    initial_count: Annotated[int, PydanticField(ge=0)]


class GoNoGoCriterion(BaseModel, frozen=True):
    """A single row in the Section 8 Go/No-Go decision table.

    Each criterion maps a quantitative threshold to a binary decision.
    The ``result`` and ``decision`` fields are left empty at pre-registration
    time and filled mechanically in Section 8 after analysis.

    Attributes:
        criterion: What is being evaluated.
        threshold: Mechanical threshold for passing.
        rationale: Why this threshold was chosen (literature or economic argument).
    """

    criterion: str
    threshold: str
    rationale: str


class NegativeResultAction(BaseModel, frozen=True):
    """An action to take if results are negative, with thesis framing.

    Pre-registering negative-result actions prevents post-hoc rationalization
    and demonstrates intellectual honesty to the examiner.

    Attributes:
        condition: The specific negative outcome triggering this action.
        action: What the thesis does in response.
        thesis_value: Why the negative result is still a valid contribution.
    """

    condition: str
    action: str
    thesis_value: str


class ExaminerDefense(BaseModel, frozen=True):
    """An anticipated examiner attack with a pre-planned defense.

    Documenting these in advance shows the thesis is aware of its own
    weaknesses and has structural defenses, not ad-hoc excuses.

    Attributes:
        attack: The criticism an examiner might raise.
        defense: The structural defense the thesis provides.
        evidence_section: Which RC2 section provides the evidence for the defense.
    """

    attack: str
    defense: str
    evidence_section: str


class PreRegistrationSpec(BaseModel, frozen=True):
    """Immutable pre-registration specification for the RC2 research checkpoint.

    Encodes ALL mechanical decision rules before any RC2 analysis is executed.
    Every field is derived from existing project configs (ProfilingConfig,
    ValidationConfig, FeatureConfig, DataPartition) to ensure consistency
    between the pre-registration and the code that implements it.

    The spec serves three purposes:
        1. Converts exploratory analysis to confirmatory (Nosek et al., 2018).
        2. Documents researcher degrees of freedom for DSR (Bailey & LdP, 2014).
        3. Provides the examiner with a roadmap that makes the Go/No-Go
           decision appear inevitable rather than cherry-picked.

    Attributes:
        generated_at: Timestamp when this spec was generated.
        feature_gate_rules: Mechanical rules for feature keep/drop decisions.
        asset_universe_rules: Rules for which assets proceed to modeling.
        bar_type_rules: Rules for which bar types proceed to modeling.
        horizon_selection_rules: Rules for selecting forecast horizons.
        minimum_viable_da: Break-even DA formula and threshold.
        negative_result_actions: Protocol for each negative-result scenario.
        model_complexity_rules: Rules linking BDS test results to model choice.
        trial_count_categories: DSR trial categories with counting rules.
        go_no_go_criteria: The Section 8 decision table (thresholds only).
        examiner_defenses: Anticipated attacks and structural defenses.
        partition: Temporal partition used for all RC2 analysis.
        profiling_config: Profiling configuration snapshot for reproducibility.
        validation_config: Validation configuration snapshot for reproducibility.
        feature_config: Feature engineering configuration snapshot.
    """

    generated_at: datetime

    # Decision rules
    feature_gate_rules: tuple[str, ...]
    asset_universe_rules: tuple[str, ...]
    bar_type_rules: tuple[str, ...]
    horizon_selection_rules: tuple[str, ...]
    minimum_viable_da: str

    # Negative result handling
    negative_result_actions: tuple[NegativeResultAction, ...]

    # Model complexity
    model_complexity_rules: tuple[str, ...]

    # DSR trial counting
    trial_count_categories: tuple[TrialCountCategory, ...]

    # Go/No-Go
    go_no_go_criteria: tuple[GoNoGoCriterion, ...]

    # Examiner defenses
    examiner_defenses: tuple[ExaminerDefense, ...]

    # Config snapshots for reproducibility
    partition: DataPartition
    profiling_config: ProfilingConfig
    validation_config: ValidationConfig
    feature_config: FeatureConfig


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_feature_gate_rules(validation_config: ValidationConfig) -> tuple[str, ...]:
    """Derive feature gating rules from the validation config.

    Args:
        validation_config: The three-gate validation config from Phase 4D.

    Returns:
        Tuple of human-readable rule strings.
    """
    alpha: float = validation_config.alpha
    stability_threshold: float = validation_config.stability_threshold
    min_features: int = validation_config.min_features_kept
    n_windows: int = len(validation_config.temporal_windows)

    return (
        (f"Keep features passing all three gates at alpha={alpha} after Benjamini-Hochberg FDR correction:"),
        f"  Gate 1 (MI): BH-corrected MI permutation p-value < {alpha}",
        f"  Gate 2 (DA): Ridge directional accuracy exceeds null at p < {alpha}",
        (f"  Gate 3 (Stability): MI significant in >= {stability_threshold:.0%} of {n_windows} temporal windows"),
        (
            f"Fallback: if fewer than {min_features} features pass, "
            f"relax to top-{min_features} by MI score (flagged as post-hoc, +1 trial)"
        ),
        "All features shown in Section 3 (kept AND dropped) to prevent survivorship bias",
    )


def _build_asset_universe_rules(
    tier_config: TierConfig,
) -> tuple[str, ...]:
    """Derive asset universe rules from tier thresholds.

    Args:
        tier_config: Sample-size tier configuration.

    Returns:
        Tuple of human-readable rule strings.
    """
    return (
        (
            f"Keep assets with Tier A (>{tier_config.tier_a_threshold} bars) "
            f"on the primary bar type (dollar) for full ML pipeline"
        ),
        (
            f"Keep assets with Tier B ({tier_config.tier_b_threshold}-"
            f"{tier_config.tier_a_threshold} bars) for restricted models"
        ),
        f"Drop assets with Tier C (<{tier_config.tier_b_threshold} bars) from modeling",
        "Asset dropping is mechanical based on bar count, not on feature performance",
    )


def _build_bar_type_rules(tier_config: TierConfig) -> tuple[str, ...]:
    """Derive bar type rules from tier config.

    Args:
        tier_config: Sample-size tier configuration.

    Returns:
        Tuple of human-readable rule strings.
    """
    return (
        (f"Tier A bar types (>{tier_config.tier_a_threshold} bars): proceed to full modeling pipeline"),
        (
            f"Tier B bar types ({tier_config.tier_b_threshold}-"
            f"{tier_config.tier_a_threshold} bars): restricted to "
            f"linear models with stronger regularization"
        ),
        (
            f"Tier C bar types (<{tier_config.tier_b_threshold} bars): "
            f"statistical profiling only, flagged as exploratory"
        ),
        "time_1h included as baseline for all comparisons",
    )


def _build_horizon_selection_rules(validation_config: ValidationConfig) -> tuple[str, ...]:
    """Derive horizon selection rules.

    Args:
        validation_config: Validation config containing alpha and target info.

    Returns:
        Tuple of human-readable rule strings.
    """
    alpha: float = validation_config.alpha
    return (
        (f"Keep horizons where at least 1 feature achieves DA > null at p < {alpha} after BH correction"),
        "Compare fwd_logret_1, fwd_logret_4, fwd_logret_24 side-by-side",
        "Horizon with highest count of significant features is primary",
        "Horizons with zero significant features are dropped from modeling",
    )


def _build_minimum_viable_da(predictability_config: PredictabilityConfig) -> str:
    """Derive the minimum viable DA statement from transaction cost config.

    Args:
        predictability_config: Config containing round-trip cost.

    Returns:
        Human-readable DA threshold statement.
    """
    cost_bps: float = predictability_config.round_trip_cost * 10000
    return (
        f"Minimum viable DA = break-even DA from Phase 5D predictability analysis. "
        f"At {cost_bps:.0f} bps round-trip cost, the break-even DA is computed per "
        f"(asset, bar_type) from mean absolute return. Features whose DA excess over "
        f"50% is less than (break-even_DA - 50%) are flagged as economically "
        f"insignificant regardless of statistical significance (Ziliak & McCloskey, 2008)."
    )


def _build_negative_result_actions() -> tuple[NegativeResultAction, ...]:
    """Define the complete negative-result protocol.

    Returns:
        Tuple of pre-registered negative result actions.
    """
    return (
        NegativeResultAction(
            condition="No features pass the three-gate validation",
            action=(
                "Report honestly. Discuss: (a) longer forecast horizons, "
                "(b) alternative target variables (e.g., volatility), "
                "(c) the recommender's value as a pure NO-TRADE filter"
            ),
            thesis_value=(
                "Consistent with R5 (crypto ~ Brownian noise). "
                "A negative result is itself a finding: individual features "
                "lack predictive power for short-horizon returns in crypto"
            ),
        ),
        NegativeResultAction(
            condition="DA excess over baseline < break-even DA for ALL combinations",
            action=(
                "Document the gap between statistical and economic significance. "
                "Proceed to modeling with explicit caveat that profitability "
                "after costs is unlikely without model combination"
            ),
            thesis_value=(
                "Demonstrates the Ziliak-McCloskey distinction between "
                "statistical and economic significance in crypto markets. "
                "The recommender becomes a risk management tool (abstain signal) "
                "rather than an alpha generator"
            ),
        ),
        NegativeResultAction(
            condition="Permutation entropy H_norm >= 0.98 for all bar types",
            action=(
                "Confirm alignment with R5. Restrict modeling to "
                "regime-conditional approaches only (trade only in low-entropy windows)"
            ),
            thesis_value=(
                "Validates the thesis's novel contribution: permutation entropy "
                "as a real-time feature for the recommender. High unconditional "
                "entropy with occasional low-entropy windows = conditional predictability"
            ),
        ),
        NegativeResultAction(
            condition="Cross-asset feature consistency (Kendall tau) is insignificant",
            action=(
                "Switch from shared model to per-asset feature selection. "
                "Document this as a post-hoc decision (+1 trial for DSR)"
            ),
            thesis_value=(
                "Reveals heterogeneity in crypto market microstructure across "
                "assets, supporting asset-specific modeling"
            ),
        ),
        NegativeResultAction(
            condition="Feature selection instability: holdout preview loses >50% of features",
            action=(
                "Flag feature selection as unstable. Add feature stability as "
                "a constraint in modeling (ensemble over multiple feature subsets)"
            ),
            thesis_value=(
                "Identifies non-stationarity in feature informativeness, "
                "motivating adaptive feature selection in production"
            ),
        ),
    )


def _build_model_complexity_rules() -> tuple[str, ...]:
    """Define rules linking profiling results to model complexity.

    Returns:
        Tuple of human-readable model complexity rules.
    """
    return (
        "If BDS test rejects i.i.d. on GARCH residuals: nonlinear models justified",
        "If BDS test does NOT reject: restrict to linear/tree models only",
        "If GARCH persistence > 0.99 (IGARCH): include volatility regime as feature",
        "If sign bias test significant: include asymmetric volatility features",
        "Model complexity ceiling for Tier B bar types: Ridge, Lasso, Random Forest",
        "Model complexity ceiling for Tier A bar types: add XGBoost, LightGBM, TFT",
    )


def _build_trial_count_categories(
    validation_config: ValidationConfig,
    feature_config: FeatureConfig,
) -> tuple[TrialCountCategory, ...]:
    """Define DSR trial counting categories.

    Each category documents a class of researcher decisions that inflate
    the number of effective independent trials for the Deflated Sharpe Ratio.

    Args:
        validation_config: Validation config for parameter counts.
        feature_config: Feature config for indicator/target counts.

    Returns:
        Tuple of trial count categories with initial counts.
    """
    n_return_horizons: int = len(feature_config.target_config.forward_return_horizons)
    n_vol_horizons: int = len(feature_config.target_config.forward_vol_horizons)
    n_windows: int = len(validation_config.temporal_windows)

    return (
        TrialCountCategory(
            name="Bar type selection",
            counting_rule=(
                "Count each bar type evaluated in RC2. "
                "Pre-registered set: dollar, volume, volume_imbalance, "
                "dollar_imbalance, time_1h"
            ),
            initial_count=5,
        ),
        TrialCountCategory(
            name="Forecast horizon selection",
            counting_rule=(
                "Count each forward return/volatility horizon evaluated. "
                f"Return horizons: {n_return_horizons}, "
                f"volatility horizons: {n_vol_horizons}"
            ),
            initial_count=n_return_horizons + n_vol_horizons,
        ),
        TrialCountCategory(
            name="Feature validation configuration",
            counting_rule=(
                "Count as 1 trial (pre-registered config). "
                "Any post-hoc change to alpha, stability threshold, "
                "or window boundaries adds +1 per change"
            ),
            initial_count=1,
        ),
        TrialCountCategory(
            name="Temporal window boundaries",
            counting_rule=(
                f"Pre-registered: {n_windows} windows. Count as 1 trial. Changing window boundaries = +1 per change"
            ),
            initial_count=1,
        ),
        TrialCountCategory(
            name="Post-hoc deviations",
            counting_rule=(
                "Any decision that deviates from pre-registered rules. "
                "Each deviation adds +1. Document reason and alternative "
                "considered. Starts at 0"
            ),
            initial_count=0,
        ),
    )


def _build_go_no_go_criteria(
    tier_config: TierConfig,
    validation_config: ValidationConfig,
) -> tuple[GoNoGoCriterion, ...]:
    """Define the Section 8 Go/No-Go decision table criteria.

    Args:
        tier_config: Tier config for sample size thresholds.
        validation_config: Validation config for alpha and min features.

    Returns:
        Tuple of go/no-go criteria with thresholds and rationale.
    """
    min_features: int = validation_config.min_features_kept
    return (
        GoNoGoCriterion(
            criterion="Features passing three-gate validation",
            threshold=f">= {min_features}",
            rationale=(
                f"Minimum {min_features} features needed for Ridge/ensemble "
                f"to have enough dimensionality for directional prediction"
            ),
        ),
        GoNoGoCriterion(
            criterion="DA excess over baseline",
            threshold=">= break-even DA for >= 1 (asset, bar_type)",
            rationale=(
                "Economic significance threshold (Ziliak & McCloskey, 2008). "
                "At least one combination must show DA above transaction cost floor"
            ),
        ),
        GoNoGoCriterion(
            criterion="Permutation entropy H_norm",
            threshold="< 0.98 for >= 1 (asset, bar_type)",
            rationale=(
                "Entropy below 0.98 indicates detectable structure beyond "
                "Brownian noise (R5). Threshold from Bandt & Pompe (2002)"
            ),
        ),
        GoNoGoCriterion(
            criterion="Effective sample size N_eff",
            threshold=f">= {tier_config.tier_a_threshold} for primary bar type",
            rationale=("Kish N_eff must meet Tier A threshold for reliable asymptotic inference in ML models"),
        ),
        GoNoGoCriterion(
            criterion="Cross-asset feature consistency (Kendall tau)",
            threshold="tau > 0, significant at alpha=0.05",
            rationale=(
                "Positive rank correlation of MI scores across assets "
                "supports shared model; otherwise asset-specific selection"
            ),
        ),
        GoNoGoCriterion(
            criterion="BDS on GARCH residuals",
            threshold="Rejects i.i.d. for >= 1 asset",
            rationale=(
                "Rejection indicates nonlinear structure beyond GARCH, "
                "justifying nonlinear ML models over linear baselines"
            ),
        ),
    )


def _build_examiner_defenses() -> tuple[ExaminerDefense, ...]:
    """Define anticipated examiner attacks and structural defenses.

    Returns:
        Tuple of examiner defense pairs.
    """
    return (
        ExaminerDefense(
            attack="Your features are data-mined",
            defense=(
                "Pre-registration + feature rationale table (Section 3). "
                "Every feature has a priori economic justification documented "
                "BEFORE seeing results. Three-gate validation with BH correction "
                "controls false discovery rate"
            ),
            evidence_section="Section 1 (pre-registration) + Section 3 (rationale table)",
        ),
        ExaminerDefense(
            attack="Crypto is unpredictable (R5 shows Brownian noise)",
            defense=(
                "Section 4 directly confronts R5. Permutation entropy on "
                "information-driven bars shows lower entropy than time bars, "
                "suggesting sampling method extracts structure. The recommender's "
                "value is conditional (WHEN to trade), not unconditional"
            ),
            evidence_section="Section 4 (Confronting R5)",
        ),
        ExaminerDefense(
            attack="Your results could be random (multiple testing)",
            defense=(
                "DSR trial counting starts HERE (pre-registration). "
                "BH correction on all MI and DA p-values. Monte Carlo null "
                "in Phase 14. Every post-hoc deviation documented and counted"
            ),
            evidence_section="Section 1 (trial counting) + Section 3 (BH tables)",
        ),
        ExaminerDefense(
            attack="You only show good results (survivorship bias)",
            defense=(
                "ALL features shown in Section 3 with color-coded "
                "kept/dropped overlay. Negative result protocol pre-registered. "
                "Dropped features documented with reasons"
            ),
            evidence_section="Section 3 (all-features overlay)",
        ),
        ExaminerDefense(
            attack="Your feature selection is unstable",
            defense=(
                "Stability heatmap (feature x temporal window) in Section 3. "
                "Holdout preview on 2023 data shows which features retain "
                "significance. Instability itself is documented as a finding"
            ),
            evidence_section="Section 3 (stability heatmap + holdout preview)",
        ),
        ExaminerDefense(
            attack="Statistical significance != economic significance",
            defense=(
                "Break-even DA from transaction costs bridges the gap. "
                "Every DA result framed against economic threshold. "
                "Harvey et al. (2016) t > 3.0 criterion applied"
            ),
            evidence_section="Section 7 (baselines & economic significance)",
        ),
        ExaminerDefense(
            attack="Your Go/No-Go decision is subjective",
            defense=(
                "Decision table is MECHANICAL: 6 criteria, each with "
                "quantitative threshold, evaluated by code. The Go/No-Go "
                "is determined by the pre-registered rules, not by human judgment"
            ),
            evidence_section="Section 8 (Go/No-Go decision table)",
        ),
    )


def build_preregistration_spec(
    profiling_config: ProfilingConfig | None = None,
    validation_config: ValidationConfig | None = None,
    feature_config: FeatureConfig | None = None,
    partition: DataPartition | None = None,
    generated_at: datetime | None = None,
) -> PreRegistrationSpec:
    """Build a complete pre-registration specification from project configs.

    All arguments are optional and default to the project's standard configs.
    This ensures that the pre-registration is always consistent with the code
    that will execute the analysis.

    Args:
        profiling_config: Phase 5 profiling configuration.
        validation_config: Phase 4D validation configuration.
        feature_config: Feature engineering configuration.
        partition: Temporal data partition.
        generated_at: Override timestamp for reproducible tests.

    Returns:
        Frozen ``PreRegistrationSpec`` ready for rendering.
    """
    if profiling_config is None:
        profiling_config = ProfilingConfig()
    if validation_config is None:
        validation_config = ValidationConfig()
    if feature_config is None:
        feature_config = FeatureConfig()
    if partition is None:
        partition = DataPartition.default()
    if generated_at is None:
        generated_at = datetime.now()  # noqa: DTZ005

    return PreRegistrationSpec(
        generated_at=generated_at,
        feature_gate_rules=_build_feature_gate_rules(validation_config),
        asset_universe_rules=_build_asset_universe_rules(profiling_config.tier),
        bar_type_rules=_build_bar_type_rules(profiling_config.tier),
        horizon_selection_rules=_build_horizon_selection_rules(validation_config),
        minimum_viable_da=_build_minimum_viable_da(profiling_config.predictability),
        negative_result_actions=_build_negative_result_actions(),
        model_complexity_rules=_build_model_complexity_rules(),
        trial_count_categories=_build_trial_count_categories(validation_config, feature_config),
        go_no_go_criteria=_build_go_no_go_criteria(profiling_config.tier, validation_config),
        examiner_defenses=_build_examiner_defenses(),
        partition=partition,
        profiling_config=profiling_config,
        validation_config=validation_config,
        feature_config=feature_config,
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class PreRegistrationRenderer:
    """Renders a PreRegistrationSpec as markdown for notebook insertion.

    The rendered markdown is designed to be pasted into the first cell of
    the RC2 notebook, serving as both documentation and a contract that
    the subsequent analysis must honor.
    """

    def render_markdown(self, spec: PreRegistrationSpec) -> str:  # noqa: PLR6301
        """Render the complete pre-registration document as markdown.

        Args:
            spec: The frozen pre-registration specification.

        Returns:
            Multi-line markdown string suitable for a Jupyter notebook cell.
        """
        sections: list[str] = [
            _render_header(spec),
            _render_feature_gates(spec),
            _render_asset_universe(spec),
            _render_bar_types(spec),
            _render_horizon_selection(spec),
            _render_minimum_viable_da(spec),
            _render_negative_result_protocol(spec),
            _render_model_complexity(spec),
            _render_trial_counting(spec),
            _render_examiner_defenses(spec),
            _render_go_no_go_template(spec),
            _render_config_snapshot(spec),
        ]
        return "\n\n".join(sections)

    def render_decision_table(self, spec: PreRegistrationSpec) -> str:  # noqa: PLR6301
        """Render the Go/No-Go decision table template for Section 8.

        The ``Result`` and ``Decision`` columns are left as ``?`` placeholders
        to be filled after mechanical evaluation.

        Args:
            spec: The frozen pre-registration specification.

        Returns:
            Markdown table string for the Go/No-Go decision.
        """
        return _render_go_no_go_template(spec)

    def compute_initial_trial_count(self, spec: PreRegistrationSpec) -> int:  # noqa: PLR6301
        """Compute the total initial trial count for DSR.

        Sums all ``initial_count`` values across trial count categories.
        This is the N_trials input for the Phase 14 Deflated Sharpe Ratio.

        Args:
            spec: The frozen pre-registration specification.

        Returns:
            Total number of pre-registered trials.
        """
        total: int = sum(cat.initial_count for cat in spec.trial_count_categories)
        return total


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------


def _render_header(spec: PreRegistrationSpec) -> str:
    """Render the document header with timestamp and partition info.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown header string.
    """
    fs_start: str = spec.partition.feature_selection_start.strftime("%Y-%m-%d")
    fs_end: str = spec.partition.feature_selection_end.strftime("%Y-%m-%d")
    md_start: str = spec.partition.model_dev_start.strftime("%Y-%m-%d")
    md_end: str = spec.partition.model_dev_end.strftime("%Y-%m-%d")
    ho_start: str = spec.partition.holdout_start.strftime("%Y-%m-%d")

    return (
        "# RC2 Pre-Registration: Decision Rules & Analysis Protocol\n\n"
        f"**Generated:** {spec.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "**Purpose:** This document defines ALL mechanical decision rules for RC2 "
        "analysis BEFORE any results are observed. Per Nosek et al. (2018), this "
        "converts exploratory analysis into confirmatory analysis and reduces "
        "researcher degrees of freedom for the Phase 14 Deflated Sharpe Ratio.\n\n"
        "**Temporal partition:**\n"
        f"- Feature selection: {fs_start} to {fs_end}\n"
        f"- Model development: {md_start} to {md_end}\n"
        f"- Final holdout: {ho_start} onwards\n\n"
        "**Contract:** Any deviation from these rules is documented as a post-hoc "
        "decision and adds +1 to the trial count for DSR correction."
    )


def _render_feature_gates(spec: PreRegistrationSpec) -> str:
    """Render the feature gate rules section.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section string.
    """
    rules: str = "\n".join(f"- {rule}" for rule in spec.feature_gate_rules)
    return f"## 1. Feature Selection Rules\n\n{rules}"


def _render_asset_universe(spec: PreRegistrationSpec) -> str:
    """Render the asset universe rules section.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section string.
    """
    rules: str = "\n".join(f"- {rule}" for rule in spec.asset_universe_rules)
    return f"## 2. Asset Universe Rules\n\n{rules}"


def _render_bar_types(spec: PreRegistrationSpec) -> str:
    """Render the bar type rules section.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section string.
    """
    rules: str = "\n".join(f"- {rule}" for rule in spec.bar_type_rules)
    return f"## 3. Bar Type Rules\n\n{rules}"


def _render_horizon_selection(spec: PreRegistrationSpec) -> str:
    """Render the horizon selection rules section.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section string.
    """
    rules: str = "\n".join(f"- {rule}" for rule in spec.horizon_selection_rules)
    return f"## 4. Forecast Horizon Selection Rules\n\n{rules}"


def _render_minimum_viable_da(spec: PreRegistrationSpec) -> str:
    """Render the minimum viable DA section.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section string.
    """
    return f"## 5. Minimum Viable Directional Accuracy\n\n{spec.minimum_viable_da}"


def _render_negative_result_protocol(spec: PreRegistrationSpec) -> str:
    """Render the negative result protocol section.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section string.
    """
    lines: list[str] = [
        "## 6. Negative Result Protocol\n",
        "If results are negative, the following pre-registered actions apply. "
        "Negative results are valid findings and strengthen the thesis by "
        "demonstrating intellectual honesty.\n",
    ]
    for i, action in enumerate(spec.negative_result_actions, start=1):
        lines.append(f"### 6.{i}. {action.condition}\n")
        lines.append(f"**Action:** {action.action}\n")
        lines.append(f"**Thesis value:** {action.thesis_value}\n")

    return "\n".join(lines)


def _render_model_complexity(spec: PreRegistrationSpec) -> str:
    """Render the model complexity rules section.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section string.
    """
    rules: str = "\n".join(f"- {rule}" for rule in spec.model_complexity_rules)
    return f"## 7. Model Complexity Rules\n\n{rules}"


def _render_trial_counting(spec: PreRegistrationSpec) -> str:
    """Render the DSR trial counting section.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section with table of trial categories.
    """
    total: int = sum(cat.initial_count for cat in spec.trial_count_categories)

    header: str = (
        "## 8. DSR Trial Counting (Bailey & Lopez de Prado, 2014)\n\n"
        "Every independent decision that could have gone differently counts "
        "as a trial for the Deflated Sharpe Ratio. Pre-registering these "
        "categories ensures honest N_trials in Phase 14.\n\n"
        "| Category | Counting Rule | Initial Count |\n"
        "|----------|---------------|---------------|\n"
    )
    rows: list[str] = [
        f"| {cat.name} | {cat.counting_rule} | {cat.initial_count} |" for cat in spec.trial_count_categories
    ]

    rows.append(f"| **Total** | | **{total}** |")

    return header + "\n".join(rows)


def _render_examiner_defenses(spec: PreRegistrationSpec) -> str:
    """Render the examiner defense section.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section with attack/defense table.
    """
    header: str = (
        "## 9. Anticipated Examiner Questions & Structural Defenses\n\n"
        "| Attack | Defense | Evidence |\n"
        "|--------|---------|----------|\n"
    )
    rows: list[str] = [
        f"| {defense.attack} | {defense.defense} | {defense.evidence_section} |" for defense in spec.examiner_defenses
    ]

    return header + "\n".join(rows)


def _render_go_no_go_template(spec: PreRegistrationSpec) -> str:
    """Render the Go/No-Go decision table template.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section with decision table (Result/Decision as placeholders).
    """
    header: str = (
        "## 10. Go/No-Go Decision Table (Section 8)\n\n"
        "This table is filled MECHANICALLY after Sections 2-7 complete. "
        "The result column is computed by code; the decision column follows "
        "deterministically from the threshold.\n\n"
        "| Criterion | Threshold | Rationale | Result | Decision |\n"
        "|-----------|-----------|-----------|--------|----------|\n"
    )
    rows: list[str] = [
        f"| {criterion.criterion} | {criterion.threshold} | {criterion.rationale} | ? | ? |"
        for criterion in spec.go_no_go_criteria
    ]

    return header + "\n".join(rows)


def _render_config_snapshot(spec: PreRegistrationSpec) -> str:
    """Render config snapshot for reproducibility.

    Args:
        spec: Pre-registration specification.

    Returns:
        Markdown section with JSON config dumps.
    """
    return (
        "## 11. Configuration Snapshot (for reproducibility)\n\n"
        "These configurations are frozen at pre-registration time. "
        "Any change constitutes a post-hoc deviation.\n\n"
        "### Profiling Config\n"
        f"```json\n{spec.profiling_config.model_dump_json(indent=2)}\n```\n\n"
        "### Validation Config\n"
        f"```json\n{spec.validation_config.model_dump_json(indent=2)}\n```\n\n"
        "### Feature Config\n"
        f"```json\n{spec.feature_config.model_dump_json(indent=2)}\n```"
    )
