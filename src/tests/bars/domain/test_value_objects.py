"""Unit tests for bars domain value objects.

Tests cover ``BarType`` enum, ``BarConfig`` validation,
``is_information_driven`` property, and ``config_hash`` determinism.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.app.bars.domain.value_objects import BarConfig, BarType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STANDARD_TYPES: list[BarType] = [BarType.TIME, BarType.TICK, BarType.VOLUME, BarType.DOLLAR]
_IMBALANCE_TYPES: list[BarType] = [
    BarType.TICK_IMBALANCE,
    BarType.VOLUME_IMBALANCE,
    BarType.DOLLAR_IMBALANCE,
]
_RUN_TYPES: list[BarType] = [BarType.TICK_RUN, BarType.VOLUME_RUN, BarType.DOLLAR_RUN]
_INFO_TYPES: list[BarType] = _IMBALANCE_TYPES + _RUN_TYPES

_DEFAULT_THRESHOLD: float = 100.0
_DEFAULT_EWM_SPAN: int = 100
_DEFAULT_WARMUP: int = 100


# ---------------------------------------------------------------------------
# BarType
# ---------------------------------------------------------------------------


class TestBarType:
    """Tests for the ``BarType`` StrEnum."""

    def test_time_value(self) -> None:
        """TIME bar type value must equal the string 'time'."""
        assert BarType.TIME == "time"

    def test_tick_value(self) -> None:
        """TICK bar type value must equal the string 'tick'."""
        assert BarType.TICK == "tick"

    def test_volume_value(self) -> None:
        """VOLUME bar type value must equal the string 'volume'."""
        assert BarType.VOLUME == "volume"

    def test_dollar_value(self) -> None:
        """DOLLAR bar type value must equal the string 'dollar'."""
        assert BarType.DOLLAR == "dollar"

    def test_tick_imbalance_value(self) -> None:
        """TICK_IMBALANCE bar type value must equal the string 'tick_imbalance'."""
        assert BarType.TICK_IMBALANCE == "tick_imbalance"

    def test_volume_imbalance_value(self) -> None:
        """VOLUME_IMBALANCE bar type value must equal the string 'volume_imbalance'."""
        assert BarType.VOLUME_IMBALANCE == "volume_imbalance"

    def test_dollar_imbalance_value(self) -> None:
        """DOLLAR_IMBALANCE bar type value must equal the string 'dollar_imbalance'."""
        assert BarType.DOLLAR_IMBALANCE == "dollar_imbalance"

    def test_tick_run_value(self) -> None:
        """TICK_RUN bar type value must equal the string 'tick_run'."""
        assert BarType.TICK_RUN == "tick_run"

    def test_volume_run_value(self) -> None:
        """VOLUME_RUN bar type value must equal the string 'volume_run'."""
        assert BarType.VOLUME_RUN == "volume_run"

    def test_dollar_run_value(self) -> None:
        """DOLLAR_RUN bar type value must equal the string 'dollar_run'."""
        assert BarType.DOLLAR_RUN == "dollar_run"

    def test_total_member_count(self) -> None:
        """BarType must have exactly 10 members."""
        expected_count: int = 10
        assert len(BarType) == expected_count

    def test_from_string_roundtrip(self) -> None:
        """Every BarType member must round-trip through its string value."""
        for bt in BarType:
            assert BarType(bt.value) == bt

    def test_is_strable(self) -> None:
        """BarType members must be usable as plain strings (StrEnum contract)."""
        assert str(BarType.TICK) == "tick"


# ---------------------------------------------------------------------------
# BarConfig — construction
# ---------------------------------------------------------------------------


class TestBarConfigConstruction:
    """Tests for valid and invalid BarConfig construction."""

    def test_minimal_valid_config(self) -> None:
        """BarConfig must be constructable with only bar_type and threshold."""
        cfg: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD)
        assert cfg.bar_type == BarType.TICK
        assert cfg.threshold == _DEFAULT_THRESHOLD

    def test_default_ewm_span(self) -> None:
        """Default ewm_span must be 100."""
        cfg: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD)
        assert cfg.ewm_span == _DEFAULT_EWM_SPAN

    def test_default_warmup_period(self) -> None:
        """Default warmup_period must be 100."""
        cfg: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD)
        assert cfg.warmup_period == _DEFAULT_WARMUP

    def test_explicit_ewm_span(self) -> None:
        """Custom ewm_span must be stored as provided.

        warmup_period must be <= ewm_span, so we also pass a compatible warmup_period.
        """
        cfg: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD, ewm_span=50, warmup_period=25)
        assert cfg.ewm_span == 50

    def test_explicit_warmup_period(self) -> None:
        """Custom warmup_period must be stored as provided."""
        cfg: BarConfig = BarConfig(
            bar_type=BarType.TICK,
            threshold=_DEFAULT_THRESHOLD,
            ewm_span=50,
            warmup_period=25,
        )
        assert cfg.warmup_period == 25

    def test_warmup_equal_ewm_span_is_valid(self) -> None:
        """warmup_period == ewm_span must be accepted (boundary condition)."""
        cfg: BarConfig = BarConfig(
            bar_type=BarType.TICK,
            threshold=_DEFAULT_THRESHOLD,
            ewm_span=10,
            warmup_period=10,
        )
        assert cfg.warmup_period == cfg.ewm_span

    def test_warmup_one_is_valid(self) -> None:
        """warmup_period of 1 is at the lower bound and must be accepted."""
        cfg: BarConfig = BarConfig(
            bar_type=BarType.TICK,
            threshold=_DEFAULT_THRESHOLD,
            warmup_period=1,
        )
        assert cfg.warmup_period == 1

    def test_threshold_small_positive_is_valid(self) -> None:
        """A very small positive threshold (0.001) must be accepted."""
        cfg: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=0.001)
        assert cfg.threshold == pytest.approx(0.001)

    @pytest.mark.parametrize("bar_type", list(BarType))
    def test_all_bar_types_accepted(self, bar_type: BarType) -> None:
        """Every BarType member must be accepted by BarConfig."""
        cfg: BarConfig = BarConfig(bar_type=bar_type, threshold=_DEFAULT_THRESHOLD)
        assert cfg.bar_type == bar_type

    def test_frozen_prevents_mutation(self) -> None:
        """Attempting to mutate a frozen BarConfig field must raise ValidationError."""
        cfg: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD)
        with pytest.raises(ValidationError):
            cfg.threshold = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BarConfig — invalid inputs
# ---------------------------------------------------------------------------


class TestBarConfigValidationErrors:
    """Tests for BarConfig rejection of invalid inputs."""

    def test_zero_threshold_raises(self) -> None:
        """threshold=0 must raise ValidationError (gt=0 constraint)."""
        with pytest.raises(ValidationError):
            BarConfig(bar_type=BarType.TICK, threshold=0.0)

    def test_negative_threshold_raises(self) -> None:
        """Negative threshold must raise ValidationError."""
        with pytest.raises(ValidationError):
            BarConfig(bar_type=BarType.TICK, threshold=-1.0)

    def test_ewm_span_below_ten_raises(self) -> None:
        """ewm_span < 10 must raise ValidationError (ge=10 constraint)."""
        with pytest.raises(ValidationError):
            BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD, ewm_span=9)

    def test_ewm_span_zero_raises(self) -> None:
        """ewm_span=0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD, ewm_span=0)

    def test_warmup_period_zero_raises(self) -> None:
        """warmup_period=0 must raise ValidationError (ge=1 constraint)."""
        with pytest.raises(ValidationError):
            BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD, warmup_period=0)

    def test_warmup_exceeds_ewm_span_raises(self) -> None:
        """warmup_period > ewm_span must raise ValueError from model_validator."""
        with pytest.raises(ValidationError, match="warmup_period"):
            BarConfig(
                bar_type=BarType.TICK,
                threshold=_DEFAULT_THRESHOLD,
                ewm_span=10,
                warmup_period=11,
            )

    def test_missing_threshold_raises(self) -> None:
        """Omitting threshold must raise ValidationError."""
        with pytest.raises(ValidationError):
            BarConfig(bar_type=BarType.TICK)  # type: ignore[call-arg]

    def test_missing_bar_type_raises(self) -> None:
        """Omitting bar_type must raise ValidationError."""
        with pytest.raises(ValidationError):
            BarConfig(threshold=_DEFAULT_THRESHOLD)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# BarConfig.is_information_driven
# ---------------------------------------------------------------------------


class TestBarConfigIsInformationDriven:
    """Tests for the ``is_information_driven`` property."""

    @pytest.mark.parametrize("bar_type", _STANDARD_TYPES)
    def test_standard_types_are_not_information_driven(self, bar_type: BarType) -> None:
        """Standard bar types (TIME, TICK, VOLUME, DOLLAR) must return False."""
        cfg: BarConfig = BarConfig(bar_type=bar_type, threshold=_DEFAULT_THRESHOLD)
        assert cfg.is_information_driven is False

    @pytest.mark.parametrize("bar_type", _INFO_TYPES)
    def test_information_driven_types_return_true(self, bar_type: BarType) -> None:
        """Imbalance and run bar types must return True."""
        cfg: BarConfig = BarConfig(bar_type=bar_type, threshold=_DEFAULT_THRESHOLD)
        assert cfg.is_information_driven is True

    def test_all_bar_types_covered_by_property(self) -> None:
        """is_information_driven must return a bool for every BarType member."""
        for bt in BarType:
            cfg: BarConfig = BarConfig(bar_type=bt, threshold=_DEFAULT_THRESHOLD)
            result: bool = cfg.is_information_driven
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# BarConfig.config_hash
# ---------------------------------------------------------------------------


class TestBarConfigHash:
    """Tests for the ``config_hash`` property."""

    def test_hash_is_16_chars(self) -> None:
        """config_hash must be exactly 16 hex characters."""
        cfg: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD)
        assert len(cfg.config_hash) == 16

    def test_same_config_produces_same_hash(self) -> None:
        """Two identical BarConfig instances must produce the same hash."""
        cfg_a: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD)
        cfg_b: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD)
        assert cfg_a.config_hash == cfg_b.config_hash

    def test_different_threshold_produces_different_hash(self) -> None:
        """Changing threshold must produce a different hash."""
        cfg_a: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=100.0)
        cfg_b: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=200.0)
        assert cfg_a.config_hash != cfg_b.config_hash

    def test_different_bar_type_produces_different_hash(self) -> None:
        """Changing bar_type must produce a different hash."""
        cfg_a: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD)
        cfg_b: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=_DEFAULT_THRESHOLD)
        assert cfg_a.config_hash != cfg_b.config_hash

    def test_different_ewm_span_produces_different_hash(self) -> None:
        """Changing ewm_span must produce a different hash.

        Both configs must be valid (warmup_period <= ewm_span), so we adjust
        warmup_period accordingly.
        """
        cfg_a: BarConfig = BarConfig(
            bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD, ewm_span=100, warmup_period=50
        )
        cfg_b: BarConfig = BarConfig(
            bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD, ewm_span=50, warmup_period=25
        )
        assert cfg_a.config_hash != cfg_b.config_hash

    def test_different_warmup_period_produces_different_hash(self) -> None:
        """Changing warmup_period must produce a different hash."""
        cfg_a: BarConfig = BarConfig(
            bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD, ewm_span=100, warmup_period=50
        )
        cfg_b: BarConfig = BarConfig(
            bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD, ewm_span=100, warmup_period=10
        )
        assert cfg_a.config_hash != cfg_b.config_hash

    def test_hash_is_hex_string(self) -> None:
        """config_hash must contain only hexadecimal characters."""
        cfg: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_DEFAULT_THRESHOLD)
        assert all(c in "0123456789abcdef" for c in cfg.config_hash)
