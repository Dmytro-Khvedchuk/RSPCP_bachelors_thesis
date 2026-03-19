"""Tests for SampleTier enum, TierConfig, and TierClassifier."""

from __future__ import annotations

import pytest

from src.app.profiling.domain.value_objects import SampleTier, TierClassifier, TierConfig


class TestSampleTier:
    """Tests for the SampleTier enum."""

    def test_tier_values(self) -> None:
        """SampleTier must have exactly three members: A, B, C."""
        assert SampleTier.A.value == "A"
        assert SampleTier.B.value == "B"
        assert SampleTier.C.value == "C"
        assert len(SampleTier) == 3


class TestTierConfig:
    """Tests for TierConfig validation."""

    def test_default_config(self) -> None:
        """Default TierConfig must have tier_a=2000, tier_b=500."""
        config = TierConfig()
        assert config.tier_a_threshold == 2000
        assert config.tier_b_threshold == 500

    def test_custom_config(self) -> None:
        """Custom thresholds should be accepted when a > b."""
        config = TierConfig(tier_a_threshold=5000, tier_b_threshold=1000)
        assert config.tier_a_threshold == 5000
        assert config.tier_b_threshold == 1000

    def test_tier_a_le_tier_b_raises(self) -> None:
        """tier_a_threshold <= tier_b_threshold must raise ValueError."""
        with pytest.raises(ValueError, match="tier_a_threshold"):
            TierConfig(tier_a_threshold=500, tier_b_threshold=500)

    def test_tier_a_lt_tier_b_raises(self) -> None:
        """tier_a_threshold < tier_b_threshold must raise ValueError."""
        with pytest.raises(ValueError, match="tier_a_threshold"):
            TierConfig(tier_a_threshold=100, tier_b_threshold=500)

    def test_config_is_frozen(self) -> None:
        """TierConfig must be immutable."""
        from pydantic import ValidationError

        config = TierConfig()
        with pytest.raises(ValidationError):
            config.tier_a_threshold = 3000  # type: ignore[misc]


class TestTierClassifier:
    """Tests for TierClassifier.classify."""

    def test_tier_a_above_threshold(self) -> None:
        """Samples above tier_a_threshold should classify as Tier A."""
        classifier = TierClassifier()
        config = TierConfig()
        assert classifier.classify(5000, config) == SampleTier.A

    def test_tier_a_at_boundary(self) -> None:
        """Exactly tier_a_threshold should classify as Tier B (not >)."""
        classifier = TierClassifier()
        config = TierConfig()
        assert classifier.classify(2000, config) == SampleTier.B

    def test_tier_a_just_above(self) -> None:
        """One above tier_a_threshold should classify as Tier A."""
        classifier = TierClassifier()
        config = TierConfig()
        assert classifier.classify(2001, config) == SampleTier.A

    def test_tier_b_middle(self) -> None:
        """Samples between thresholds should classify as Tier B."""
        classifier = TierClassifier()
        config = TierConfig()
        assert classifier.classify(1000, config) == SampleTier.B

    def test_tier_b_at_lower_boundary(self) -> None:
        """Exactly tier_b_threshold should classify as Tier B."""
        classifier = TierClassifier()
        config = TierConfig()
        assert classifier.classify(500, config) == SampleTier.B

    def test_tier_c_below_b_threshold(self) -> None:
        """Samples below tier_b_threshold should classify as Tier C."""
        classifier = TierClassifier()
        config = TierConfig()
        assert classifier.classify(499, config) == SampleTier.C

    def test_tier_c_zero_samples(self) -> None:
        """Zero samples should classify as Tier C."""
        classifier = TierClassifier()
        config = TierConfig()
        assert classifier.classify(0, config) == SampleTier.C

    def test_custom_thresholds(self) -> None:
        """Classification with custom thresholds should respect new boundaries."""
        classifier = TierClassifier()
        config = TierConfig(tier_a_threshold=100, tier_b_threshold=10)
        assert classifier.classify(200, config) == SampleTier.A
        assert classifier.classify(50, config) == SampleTier.B
        assert classifier.classify(5, config) == SampleTier.C
