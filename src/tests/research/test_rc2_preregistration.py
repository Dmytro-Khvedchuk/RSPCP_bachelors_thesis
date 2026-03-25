"""Unit tests for RC2 pre-registration specification builder and renderer.

Tests cover spec construction from default and custom configs, immutability
of the frozen spec, trial count arithmetic, markdown rendering completeness,
and the Go/No-Go decision table template.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.app.features.domain.value_objects import (
    FeatureConfig,
    TargetConfig,
    ValidationConfig,
)
from src.app.profiling.domain.value_objects import (
    DataPartition,
    PredictabilityConfig,
    ProfilingConfig,
    TierConfig,
)
from src.app.research.application.rc2_preregistration import (
    ExaminerDefense,
    GoNoGoCriterion,
    NegativeResultAction,
    PreRegistrationRenderer,
    PreRegistrationSpec,
    TrialCountCategory,
    build_preregistration_spec,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIXED_TIMESTAMP: datetime = datetime(2026, 3, 19, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_spec() -> PreRegistrationSpec:
    """Build a spec with all default configs and a fixed timestamp.

    Returns:
        PreRegistrationSpec with project default configuration.
    """
    return build_preregistration_spec(generated_at=_FIXED_TIMESTAMP)


@pytest.fixture
def custom_spec() -> PreRegistrationSpec:
    """Build a spec with custom configs to test parameterization.

    Returns:
        PreRegistrationSpec with non-default configuration.
    """
    profiling = ProfilingConfig(
        tier=TierConfig(tier_a_threshold=3000, tier_b_threshold=800),
        predictability=PredictabilityConfig(round_trip_cost=0.003),
    )
    validation = ValidationConfig(
        alpha=0.01,
        stability_threshold=0.6,
        min_features_kept=8,
        temporal_windows=((2020, 2021), (2021, 2022), (2022, 2023)),
    )
    feature = FeatureConfig(
        target_config=TargetConfig(
            forward_return_horizons=(1, 4, 12, 24),
            forward_vol_horizons=(4, 12, 24),
        ),
    )
    partition = DataPartition(
        feature_selection_start=datetime(2020, 1, 1, tzinfo=UTC),
        feature_selection_end=datetime(2022, 1, 1, tzinfo=UTC),
        model_dev_start=datetime(2020, 1, 1, tzinfo=UTC),
        model_dev_end=datetime(2023, 1, 1, tzinfo=UTC),
        holdout_start=datetime(2023, 1, 1, tzinfo=UTC),
    )
    return build_preregistration_spec(
        profiling_config=profiling,
        validation_config=validation,
        feature_config=feature,
        partition=partition,
        generated_at=_FIXED_TIMESTAMP,
    )


@pytest.fixture
def renderer() -> PreRegistrationRenderer:
    """Return a fresh renderer instance.

    Returns:
        PreRegistrationRenderer.
    """
    return PreRegistrationRenderer()


# ---------------------------------------------------------------------------
# TestPreRegistrationSpec -- construction and immutability
# ---------------------------------------------------------------------------


class TestPreRegistrationSpec:
    """Tests for building and validating the PreRegistrationSpec."""

    def test_default_spec_builds_without_error(self, default_spec: PreRegistrationSpec) -> None:
        """Default config produces a valid, non-empty spec."""
        assert default_spec.generated_at == _FIXED_TIMESTAMP
        assert len(default_spec.feature_gate_rules) > 0
        assert len(default_spec.asset_universe_rules) > 0
        assert len(default_spec.bar_type_rules) > 0
        assert len(default_spec.horizon_selection_rules) > 0
        assert len(default_spec.minimum_viable_da) > 0
        assert len(default_spec.negative_result_actions) > 0
        assert len(default_spec.model_complexity_rules) > 0
        assert len(default_spec.trial_count_categories) > 0
        assert len(default_spec.go_no_go_criteria) > 0
        assert len(default_spec.examiner_defenses) > 0

    def test_spec_is_frozen(self, default_spec: PreRegistrationSpec) -> None:
        """Spec must be immutable (frozen=True)."""
        with pytest.raises(Exception):  # noqa: B017, PT011
            default_spec.generated_at = datetime.now(tz=UTC)  # type: ignore[misc]

    def test_partition_snapshot_matches_input(self, default_spec: PreRegistrationSpec) -> None:
        """Partition in spec must match the project default."""
        expected = DataPartition.default()
        assert default_spec.partition == expected

    def test_profiling_config_snapshot(self, default_spec: PreRegistrationSpec) -> None:
        """Profiling config in spec must match the default."""
        expected = ProfilingConfig()
        assert default_spec.profiling_config == expected

    def test_validation_config_snapshot(self, default_spec: PreRegistrationSpec) -> None:
        """Validation config in spec must match the default."""
        expected = ValidationConfig()
        assert default_spec.validation_config == expected

    def test_feature_config_snapshot(self, default_spec: PreRegistrationSpec) -> None:
        """Feature config in spec must match the default."""
        expected = FeatureConfig()
        assert default_spec.feature_config == expected

    def test_custom_configs_propagate(self, custom_spec: PreRegistrationSpec) -> None:
        """Custom configs must appear in the spec snapshots."""
        assert custom_spec.profiling_config.tier.tier_a_threshold == 3000
        assert custom_spec.profiling_config.tier.tier_b_threshold == 800
        assert custom_spec.validation_config.alpha == 0.01
        assert custom_spec.validation_config.min_features_kept == 8
        assert len(custom_spec.validation_config.temporal_windows) == 3


# ---------------------------------------------------------------------------
# TestFeatureGateRules
# ---------------------------------------------------------------------------


class TestFeatureGateRules:
    """Tests for feature gate rule derivation from ValidationConfig."""

    def test_alpha_appears_in_rules(self, default_spec: PreRegistrationSpec) -> None:
        """The configured alpha must appear in the gate rule text."""
        all_rules = " ".join(default_spec.feature_gate_rules)
        assert "0.05" in all_rules

    def test_custom_alpha_appears(self, custom_spec: PreRegistrationSpec) -> None:
        """Custom alpha=0.01 must appear in the gate rule text."""
        all_rules = " ".join(custom_spec.feature_gate_rules)
        assert "0.01" in all_rules

    def test_three_gates_mentioned(self, default_spec: PreRegistrationSpec) -> None:
        """All three gates (MI, DA, Stability) must be mentioned."""
        all_rules = " ".join(default_spec.feature_gate_rules)
        assert "MI" in all_rules
        assert "DA" in all_rules or "directional accuracy" in all_rules.lower()
        assert "Stability" in all_rules or "stability" in all_rules.lower()

    def test_fallback_rule_mentions_min_features(self, default_spec: PreRegistrationSpec) -> None:
        """The fallback rule must mention the min_features_kept threshold."""
        all_rules = " ".join(default_spec.feature_gate_rules)
        min_features = default_spec.validation_config.min_features_kept
        assert str(min_features) in all_rules

    def test_survivorship_bias_rule_present(self, default_spec: PreRegistrationSpec) -> None:
        """A rule about showing all features (not just kept) must exist."""
        all_rules = " ".join(default_spec.feature_gate_rules).lower()
        assert "survivorship" in all_rules or "all features" in all_rules


# ---------------------------------------------------------------------------
# TestAssetUniverseRules
# ---------------------------------------------------------------------------


class TestAssetUniverseRules:
    """Tests for asset universe rule derivation from TierConfig."""

    def test_tier_thresholds_appear(self, default_spec: PreRegistrationSpec) -> None:
        """Tier A and B thresholds must appear in the rules."""
        all_rules = " ".join(default_spec.asset_universe_rules)
        assert "2000" in all_rules  # default tier_a
        assert "500" in all_rules  # default tier_b

    def test_custom_thresholds(self, custom_spec: PreRegistrationSpec) -> None:
        """Custom tier thresholds must appear."""
        all_rules = " ".join(custom_spec.asset_universe_rules)
        assert "3000" in all_rules
        assert "800" in all_rules

    def test_mechanical_rule_stated(self, default_spec: PreRegistrationSpec) -> None:
        """Asset dropping must be described as mechanical."""
        all_rules = " ".join(default_spec.asset_universe_rules).lower()
        assert "mechanical" in all_rules


# ---------------------------------------------------------------------------
# TestTrialCounting
# ---------------------------------------------------------------------------


class TestTrialCounting:
    """Tests for DSR trial count categories."""

    def test_categories_are_non_empty(self, default_spec: PreRegistrationSpec) -> None:
        """At least 3 trial count categories must exist."""
        assert len(default_spec.trial_count_categories) >= 3

    def test_post_hoc_category_starts_at_zero(self, default_spec: PreRegistrationSpec) -> None:
        """The post-hoc deviations category must start at 0."""
        post_hoc = [
            c
            for c in default_spec.trial_count_categories
            if "post-hoc" in c.name.lower() or "post-hoc" in c.counting_rule.lower()
        ]
        assert len(post_hoc) >= 1
        post_hoc_zero = [c for c in post_hoc if c.initial_count == 0]
        assert len(post_hoc_zero) >= 1

    def test_bar_type_count_is_five(self, default_spec: PreRegistrationSpec) -> None:
        """Bar type selection category must count 5 bar types."""
        bar_cat = [c for c in default_spec.trial_count_categories if "bar type" in c.name.lower()]
        assert len(bar_cat) == 1
        assert bar_cat[0].initial_count == 5

    def test_horizon_count_matches_config(self, default_spec: PreRegistrationSpec) -> None:
        """Horizon trial count must equal return_horizons + vol_horizons."""
        target_config = default_spec.feature_config.target_config
        expected = len(target_config.forward_return_horizons) + len(target_config.forward_vol_horizons)
        horizon_cat = [c for c in default_spec.trial_count_categories if "horizon" in c.name.lower()]
        assert len(horizon_cat) == 1
        assert horizon_cat[0].initial_count == expected

    def test_custom_horizon_count(self, custom_spec: PreRegistrationSpec) -> None:
        """Custom config with 4 return + 3 vol horizons = 7."""
        horizon_cat = [c for c in custom_spec.trial_count_categories if "horizon" in c.name.lower()]
        assert len(horizon_cat) == 1
        assert horizon_cat[0].initial_count == 7  # 4 + 3

    def test_total_trial_count_is_sum(self, default_spec: PreRegistrationSpec) -> None:
        """Total initial trial count must equal sum of all categories."""
        expected = sum(c.initial_count for c in default_spec.trial_count_categories)
        renderer = PreRegistrationRenderer()
        assert renderer.compute_initial_trial_count(default_spec) == expected


# ---------------------------------------------------------------------------
# TestGoNoGoCriteria
# ---------------------------------------------------------------------------


class TestGoNoGoCriteria:
    """Tests for the Go/No-Go decision table criteria."""

    def test_six_criteria_exist(self, default_spec: PreRegistrationSpec) -> None:
        """Exactly 6 Go/No-Go criteria must be defined."""
        assert len(default_spec.go_no_go_criteria) == 6

    def test_all_criteria_have_thresholds(self, default_spec: PreRegistrationSpec) -> None:
        """Every criterion must have a non-empty threshold."""
        for criterion in default_spec.go_no_go_criteria:
            assert len(criterion.threshold) > 0

    def test_all_criteria_have_rationale(self, default_spec: PreRegistrationSpec) -> None:
        """Every criterion must have a non-empty rationale."""
        for criterion in default_spec.go_no_go_criteria:
            assert len(criterion.rationale) > 0

    def test_features_criterion_uses_min_features(self, default_spec: PreRegistrationSpec) -> None:
        """Features criterion threshold must reference min_features_kept."""
        features_criteria = [c for c in default_spec.go_no_go_criteria if "feature" in c.criterion.lower()]
        assert len(features_criteria) >= 1
        min_features = default_spec.validation_config.min_features_kept
        assert str(min_features) in features_criteria[0].threshold

    def test_neff_criterion_uses_tier_a(self, default_spec: PreRegistrationSpec) -> None:
        """N_eff criterion must reference the Tier A threshold."""
        neff_criteria = [c for c in default_spec.go_no_go_criteria if "n_eff" in c.criterion.lower()]
        assert len(neff_criteria) == 1
        tier_a = default_spec.profiling_config.tier.tier_a_threshold
        assert str(tier_a) in neff_criteria[0].threshold

    def test_entropy_criterion_present(self, default_spec: PreRegistrationSpec) -> None:
        """Permutation entropy criterion must be present with 0.98 threshold."""
        entropy_criteria = [c for c in default_spec.go_no_go_criteria if "entropy" in c.criterion.lower()]
        assert len(entropy_criteria) == 1
        assert "0.98" in entropy_criteria[0].threshold


# ---------------------------------------------------------------------------
# TestNegativeResultProtocol
# ---------------------------------------------------------------------------


class TestNegativeResultProtocol:
    """Tests for the negative result protocol."""

    def test_at_least_three_actions(self, default_spec: PreRegistrationSpec) -> None:
        """At least 3 negative result actions must be pre-registered."""
        assert len(default_spec.negative_result_actions) >= 3

    def test_all_actions_have_thesis_value(self, default_spec: PreRegistrationSpec) -> None:
        """Every negative result must document its thesis value."""
        for action in default_spec.negative_result_actions:
            assert len(action.thesis_value) > 0

    def test_r5_mentioned_in_no_features_scenario(self, default_spec: PreRegistrationSpec) -> None:
        """The 'no features pass' scenario must reference R5."""
        no_features = [a for a in default_spec.negative_result_actions if "no features" in a.condition.lower()]
        assert len(no_features) == 1
        assert "R5" in no_features[0].thesis_value

    def test_no_trade_filter_mentioned(self, default_spec: PreRegistrationSpec) -> None:
        """At least one negative result action must mention the NO-TRADE filter value."""
        all_text = " ".join(f"{a.action} {a.thesis_value}" for a in default_spec.negative_result_actions)
        assert "no-trade" in all_text.lower() or "NO-TRADE" in all_text

    def test_economic_significance_scenario_exists(self, default_spec: PreRegistrationSpec) -> None:
        """A scenario for DA below break-even must exist."""
        breakeven_scenarios = [
            a
            for a in default_spec.negative_result_actions
            if "break-even" in a.condition.lower() or "da excess" in a.condition.lower()
        ]
        assert len(breakeven_scenarios) >= 1


# ---------------------------------------------------------------------------
# TestExaminerDefenses
# ---------------------------------------------------------------------------


class TestExaminerDefenses:
    """Tests for examiner attack/defense pairs."""

    def test_at_least_five_defenses(self, default_spec: PreRegistrationSpec) -> None:
        """At least 5 examiner defenses must be pre-registered."""
        assert len(default_spec.examiner_defenses) >= 5

    def test_all_defenses_have_evidence_section(self, default_spec: PreRegistrationSpec) -> None:
        """Every defense must point to a specific RC2 section."""
        for defense in default_spec.examiner_defenses:
            assert len(defense.evidence_section) > 0
            assert "section" in defense.evidence_section.lower()

    def test_data_mining_attack_present(self, default_spec: PreRegistrationSpec) -> None:
        """The 'data-mined features' attack must be addressed."""
        attacks = [d.attack.lower() for d in default_spec.examiner_defenses]
        assert any("data-min" in a for a in attacks)

    def test_r5_attack_present(self, default_spec: PreRegistrationSpec) -> None:
        """The 'crypto is unpredictable (R5)' attack must be addressed."""
        attacks = [d.attack.lower() for d in default_spec.examiner_defenses]
        assert any("r5" in a or "unpredictable" in a for a in attacks)

    def test_multiple_testing_attack_present(self, default_spec: PreRegistrationSpec) -> None:
        """The 'results could be random' attack must be addressed."""
        attacks = [d.attack.lower() for d in default_spec.examiner_defenses]
        assert any("random" in a or "multiple testing" in a for a in attacks)

    def test_go_no_go_subjective_attack_present(self, default_spec: PreRegistrationSpec) -> None:
        """The 'Go/No-Go is subjective' attack must be addressed."""
        attacks = [d.attack.lower() for d in default_spec.examiner_defenses]
        assert any("subjective" in a or "go/no-go" in a for a in attacks)


# ---------------------------------------------------------------------------
# TestPreRegistrationRenderer
# ---------------------------------------------------------------------------


class TestPreRegistrationRenderer:
    """Tests for markdown rendering."""

    def test_render_produces_non_empty_markdown(
        self,
        renderer: PreRegistrationRenderer,
        default_spec: PreRegistrationSpec,
    ) -> None:
        """Full render must produce substantial markdown."""
        md = renderer.render_markdown(default_spec)
        assert len(md) > 1000  # non-trivial document
        assert md.startswith("# RC2 Pre-Registration")

    def test_render_contains_all_sections(
        self,
        renderer: PreRegistrationRenderer,
        default_spec: PreRegistrationSpec,
    ) -> None:
        """All numbered sections must appear in rendered output."""
        md = renderer.render_markdown(default_spec)
        assert "## 1. Feature Selection Rules" in md
        assert "## 2. Asset Universe Rules" in md
        assert "## 3. Bar Type Rules" in md
        assert "## 4. Forecast Horizon Selection Rules" in md
        assert "## 5. Minimum Viable Directional Accuracy" in md
        assert "## 6. Negative Result Protocol" in md
        assert "## 7. Model Complexity Rules" in md
        assert "## 8. DSR Trial Counting" in md
        assert "## 9. Anticipated Examiner Questions" in md
        assert "## 10. Go/No-Go Decision Table" in md
        assert "## 11. Configuration Snapshot" in md

    def test_render_contains_nosek_reference(
        self,
        renderer: PreRegistrationRenderer,
        default_spec: PreRegistrationSpec,
    ) -> None:
        """The Nosek et al. (2018) reference must appear."""
        md = renderer.render_markdown(default_spec)
        assert "Nosek" in md

    def test_render_contains_harvey_reference(
        self,
        renderer: PreRegistrationRenderer,
        default_spec: PreRegistrationSpec,
    ) -> None:
        """The Harvey et al. (2016) reference must appear in the DA section."""
        md = renderer.render_markdown(default_spec)
        assert "Harvey" in md

    def test_render_contains_config_json(
        self,
        renderer: PreRegistrationRenderer,
        default_spec: PreRegistrationSpec,
    ) -> None:
        """Config snapshot section must contain JSON."""
        md = renderer.render_markdown(default_spec)
        assert "```json" in md

    def test_render_decision_table_has_placeholder_columns(
        self,
        renderer: PreRegistrationRenderer,
        default_spec: PreRegistrationSpec,
    ) -> None:
        """Decision table must have Result and Decision placeholder columns."""
        table = renderer.render_decision_table(default_spec)
        assert "| Result |" in table or "Result" in table
        assert "| Decision |" in table or "Decision" in table
        assert "| ? | ? |" in table  # placeholders

    def test_decision_table_row_count(
        self,
        renderer: PreRegistrationRenderer,
        default_spec: PreRegistrationSpec,
    ) -> None:
        """Decision table must have one row per criterion plus header."""
        table = renderer.render_decision_table(default_spec)
        # Count data rows (lines with | ? | ? |)
        data_rows = [line for line in table.split("\n") if "| ? | ? |" in line]
        assert len(data_rows) == len(default_spec.go_no_go_criteria)

    def test_trial_count_computation(
        self,
        renderer: PreRegistrationRenderer,
        default_spec: PreRegistrationSpec,
    ) -> None:
        """compute_initial_trial_count must return sum of all categories."""
        expected = sum(c.initial_count for c in default_spec.trial_count_categories)
        assert renderer.compute_initial_trial_count(default_spec) == expected
        assert expected > 0  # sanity check

    def test_custom_spec_renders_different_thresholds(
        self,
        renderer: PreRegistrationRenderer,
        custom_spec: PreRegistrationSpec,
    ) -> None:
        """Custom config thresholds must appear in rendered markdown."""
        md = renderer.render_markdown(custom_spec)
        assert "3000" in md  # custom tier_a
        assert "800" in md  # custom tier_b
        assert "0.01" in md  # custom alpha
        assert "30" in md  # 30 bps cost

    def test_render_contains_timestamp(
        self,
        renderer: PreRegistrationRenderer,
        default_spec: PreRegistrationSpec,
    ) -> None:
        """Rendered document must contain the generation timestamp."""
        md = renderer.render_markdown(default_spec)
        assert "2026-03-19" in md


# ---------------------------------------------------------------------------
# TestValueObjectImmutability
# ---------------------------------------------------------------------------


class TestValueObjectImmutability:
    """Tests that all pre-registration value objects are frozen."""

    def test_trial_count_category_frozen(self) -> None:
        """TrialCountCategory must be immutable."""
        cat = TrialCountCategory(
            name="test",
            counting_rule="test rule",
            initial_count=1,
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            cat.name = "changed"  # type: ignore[misc]

    def test_go_no_go_criterion_frozen(self) -> None:
        """GoNoGoCriterion must be immutable."""
        criterion = GoNoGoCriterion(
            criterion="test",
            threshold="test",
            rationale="test",
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            criterion.criterion = "changed"  # type: ignore[misc]

    def test_negative_result_action_frozen(self) -> None:
        """NegativeResultAction must be immutable."""
        action = NegativeResultAction(
            condition="test",
            action="test",
            thesis_value="test",
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            action.condition = "changed"  # type: ignore[misc]

    def test_examiner_defense_frozen(self) -> None:
        """ExaminerDefense must be immutable."""
        defense = ExaminerDefense(
            attack="test",
            defense="test",
            evidence_section="Section 1",
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            defense.attack = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_none_args_use_defaults(self) -> None:
        """Calling build_preregistration_spec() with no args must succeed."""
        spec = build_preregistration_spec(generated_at=_FIXED_TIMESTAMP)
        assert spec.generated_at == _FIXED_TIMESTAMP
        assert spec.partition == DataPartition.default()

    def test_empty_temporal_windows_does_not_crash(self) -> None:
        """ValidationConfig with minimal windows still builds a valid spec."""
        validation = ValidationConfig(
            temporal_windows=((2020, 2021),),
            min_valid_windows=1,
        )
        spec = build_preregistration_spec(
            validation_config=validation,
            generated_at=_FIXED_TIMESTAMP,
        )
        all_rules = " ".join(spec.feature_gate_rules)
        assert "1 temporal window" in all_rules or "1" in all_rules

    def test_renderer_on_custom_spec(
        self,
        renderer: PreRegistrationRenderer,
        custom_spec: PreRegistrationSpec,
    ) -> None:
        """Renderer must handle custom specs without error."""
        md = renderer.render_markdown(custom_spec)
        assert len(md) > 500

    def test_trial_count_with_no_post_hoc(self, default_spec: PreRegistrationSpec) -> None:
        """Initial trial count must not include post-hoc deviations."""
        renderer = PreRegistrationRenderer()
        total = renderer.compute_initial_trial_count(default_spec)
        # total should be > 0 but post-hoc should be 0
        post_hoc_count = sum(
            c.initial_count for c in default_spec.trial_count_categories if "post-hoc" in c.name.lower()
        )
        assert post_hoc_count == 0
        assert total > 0

    def test_minimum_viable_da_mentions_cost(self, default_spec: PreRegistrationSpec) -> None:
        """Minimum viable DA statement must mention the configured cost."""
        cost_bps = default_spec.profiling_config.predictability.round_trip_cost * 10000
        assert str(int(cost_bps)) in default_spec.minimum_viable_da

    def test_model_complexity_rules_cover_both_directions(self, default_spec: PreRegistrationSpec) -> None:
        """Model complexity rules must cover both BDS-reject and BDS-not-reject."""
        all_rules = " ".join(default_spec.model_complexity_rules).lower()
        assert "rejects" in all_rules
        assert "not reject" in all_rules
