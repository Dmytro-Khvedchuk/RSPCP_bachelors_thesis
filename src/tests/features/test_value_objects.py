"""Unit tests for features domain value objects.

Tests construction, validation invariants, and error conditions for
IndicatorConfig, TargetConfig, ValidationConfig, FeatureConfig, and FeatureSet.
"""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from src.app.features.domain.value_objects import (
    FeatureConfig,
    FeatureSet,
    IndicatorConfig,
    TargetConfig,
    ValidationConfig,
)


class TestIndicatorConfig:
    """Tests for IndicatorConfig value object."""

    def test_indicator_config_defaults_construct_successfully(self) -> None:
        """Default IndicatorConfig must construct without errors."""
        config: IndicatorConfig = IndicatorConfig()
        assert config.ema_fast_span < config.ema_slow_span
        assert config.clip_lower < config.clip_upper

    def test_indicator_config_custom_values(self) -> None:
        """Custom valid values should be accepted."""
        config: IndicatorConfig = IndicatorConfig(
            ema_fast_span=5,
            ema_slow_span=20,
            clip_lower=-3.0,
            clip_upper=3.0,
        )
        assert config.ema_fast_span == 5
        assert config.ema_slow_span == 20

    def test_indicator_config_fast_gte_slow_raises(self) -> None:
        """ema_fast_span >= ema_slow_span must raise ValidationError."""
        with pytest.raises(ValidationError):
            IndicatorConfig(ema_fast_span=21, ema_slow_span=21)

    def test_indicator_config_fast_gt_slow_raises(self) -> None:
        """ema_fast_span > ema_slow_span must raise ValidationError."""
        with pytest.raises(ValidationError):
            IndicatorConfig(ema_fast_span=30, ema_slow_span=10)

    def test_indicator_config_clip_bounds_equal_raises(self) -> None:
        """clip_lower == clip_upper must raise ValidationError."""
        with pytest.raises(ValidationError):
            IndicatorConfig(clip_lower=3.0, clip_upper=3.0)

    def test_indicator_config_clip_lower_gt_upper_raises(self) -> None:
        """clip_lower > clip_upper must raise ValidationError."""
        with pytest.raises(ValidationError):
            IndicatorConfig(clip_lower=5.0, clip_upper=-5.0)

    def test_indicator_config_atr_period_too_small_raises(self) -> None:
        """atr_period < 2 must raise ValidationError."""
        with pytest.raises(ValidationError):
            IndicatorConfig(atr_period=1)

    def test_indicator_config_is_frozen(self) -> None:
        """IndicatorConfig is frozen — mutation must raise."""
        config: IndicatorConfig = IndicatorConfig()
        with pytest.raises(ValidationError):
            config.rsi_period = 99  # type: ignore[misc]

    def test_indicator_config_bollinger_num_std_positive(self) -> None:
        """bollinger_num_std must be > 0."""
        with pytest.raises(ValidationError):
            IndicatorConfig(bollinger_num_std=0.0)

    def test_indicator_config_hurst_window_minimum(self) -> None:
        """hurst_window must be >= 20."""
        with pytest.raises(ValidationError):
            IndicatorConfig(hurst_window=10)


class TestTargetConfig:
    """Tests for TargetConfig value object."""

    def test_target_config_defaults(self) -> None:
        """Default TargetConfig should construct without errors."""
        config: TargetConfig = TargetConfig()
        assert len(config.forward_return_horizons) > 0
        assert len(config.forward_vol_horizons) > 0

    def test_target_config_empty_return_horizons_raises(self) -> None:
        """Empty forward_return_horizons must raise ValidationError."""
        with pytest.raises(ValidationError):
            TargetConfig(forward_return_horizons=())

    def test_target_config_return_horizon_less_than_1_raises(self) -> None:
        """forward_return_horizons with value < 1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            TargetConfig(forward_return_horizons=(0,))

    def test_target_config_vol_horizon_lt2_raises(self) -> None:
        """forward_vol_horizons with value < 2 must raise ValidationError."""
        with pytest.raises(ValidationError):
            TargetConfig(forward_vol_horizons=(1,))

    def test_target_config_duplicate_return_horizons_raises(self) -> None:
        """Duplicate values in forward_return_horizons must raise ValidationError."""
        with pytest.raises(ValidationError):
            TargetConfig(forward_return_horizons=(1, 4, 1))

    def test_target_config_duplicate_vol_horizons_raises(self) -> None:
        """Duplicate values in forward_vol_horizons must raise ValidationError."""
        with pytest.raises(ValidationError):
            TargetConfig(forward_vol_horizons=(4, 4))

    def test_target_config_is_frozen(self) -> None:
        """TargetConfig is frozen — mutation must raise."""
        config: TargetConfig = TargetConfig()
        with pytest.raises(ValidationError):
            config.close_col = "adjusted_close"  # type: ignore[misc]

    def test_target_config_custom_horizons(self) -> None:
        """Custom valid horizons should be accepted."""
        config: TargetConfig = TargetConfig(
            forward_return_horizons=(2, 5, 10),
            forward_vol_horizons=(3, 7),
        )
        assert config.forward_return_horizons == (2, 5, 10)


class TestValidationConfig:
    """Tests for ValidationConfig value object."""

    def test_validation_config_defaults(self) -> None:
        """Default ValidationConfig should construct without errors."""
        config: ValidationConfig = ValidationConfig()
        assert config.n_permutations_mi >= 100
        assert config.alpha > 0

    def test_validation_config_window_start_gte_end_raises(self) -> None:
        """Temporal window with start >= end must raise ValidationError."""
        with pytest.raises(ValidationError):
            ValidationConfig(temporal_windows=((2022, 2020),))

    def test_validation_config_window_start_equal_end_raises(self) -> None:
        """Temporal window with start == end must raise ValidationError."""
        with pytest.raises(ValidationError):
            ValidationConfig(temporal_windows=((2022, 2022),))

    def test_validation_config_n_permutations_too_small_raises(self) -> None:
        """n_permutations_mi < 100 must raise ValidationError."""
        with pytest.raises(ValidationError):
            ValidationConfig(n_permutations_mi=50)

    def test_validation_config_alpha_out_of_range_raises(self) -> None:
        """Alpha >= 1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            ValidationConfig(alpha=1.0)

    def test_validation_config_is_frozen(self) -> None:
        """ValidationConfig is frozen — mutation must raise."""
        config: ValidationConfig = ValidationConfig()
        with pytest.raises(ValidationError):
            config.alpha = 0.1  # type: ignore[misc]

    def test_validation_config_multiple_windows_all_ordered(self) -> None:
        """All windows in a multi-window config must be ordered."""
        config: ValidationConfig = ValidationConfig(
            temporal_windows=((2020, 2021), (2021, 2022), (2022, 2023)),
            n_permutations_mi=100,
            n_permutations_ridge=50,
        )
        for start, end in config.temporal_windows:
            assert start < end


class TestFeatureConfig:
    """Tests for FeatureConfig composite value object."""

    def test_feature_config_defaults(self) -> None:
        """Default FeatureConfig should construct without errors."""
        config: FeatureConfig = FeatureConfig()
        assert config.drop_na is True
        assert config.compute_targets is True
        assert isinstance(config.indicator_config, IndicatorConfig)
        assert isinstance(config.target_config, TargetConfig)

    def test_feature_config_custom_subconfigs(self) -> None:
        """Custom indicator and target configs should be accepted."""
        ind: IndicatorConfig = IndicatorConfig(ema_fast_span=5, ema_slow_span=10)
        tgt: TargetConfig = TargetConfig(forward_return_horizons=(1, 2))
        config: FeatureConfig = FeatureConfig(
            indicator_config=ind,
            target_config=tgt,
            drop_na=False,
            compute_targets=False,
        )
        assert config.indicator_config.ema_fast_span == 5
        assert config.compute_targets is False

    def test_feature_config_is_frozen(self) -> None:
        """FeatureConfig is frozen — mutation must raise."""
        config: FeatureConfig = FeatureConfig()
        with pytest.raises(ValidationError):
            config.drop_na = False  # type: ignore[misc]


class TestFeatureSet:
    """Tests for FeatureSet value object."""

    def _make_valid_df(self, n: int = 10) -> pl.DataFrame:
        """Build a minimal DataFrame with feature and target columns.

        Args:
            n: Number of rows.

        Returns:
            DataFrame suitable for constructing a FeatureSet.
        """
        return pl.DataFrame(
            {
                "close": [float(i + 1) for i in range(n)],
                "feat_a": [float(i) for i in range(n)],
                "fwd_logret_1": [0.1] * n,
            }
        )

    def test_feature_set_valid_construction(self) -> None:
        """FeatureSet should construct successfully with valid inputs."""
        df: pl.DataFrame = self._make_valid_df(n=10)
        fs: FeatureSet = FeatureSet(
            df=df,
            feature_columns=("feat_a",),
            target_columns=("fwd_logret_1",),
            n_rows_raw=10,
            n_rows_clean=10,
        )
        assert fs.n_rows_clean == 10
        assert "feat_a" in fs.feature_columns

    def test_feature_set_clean_exceeds_raw_raises(self) -> None:
        """n_rows_clean > n_rows_raw must raise ValidationError."""
        df: pl.DataFrame = self._make_valid_df(n=10)
        with pytest.raises(ValidationError):
            FeatureSet(
                df=df,
                feature_columns=("feat_a",),
                target_columns=("fwd_logret_1",),
                n_rows_raw=8,  # less than clean
                n_rows_clean=10,
            )

    def test_feature_set_clean_mismatch_df_len_raises(self) -> None:
        """n_rows_clean != len(df) must raise ValidationError."""
        df: pl.DataFrame = self._make_valid_df(n=10)
        with pytest.raises(ValidationError):
            FeatureSet(
                df=df,
                feature_columns=("feat_a",),
                target_columns=("fwd_logret_1",),
                n_rows_raw=10,
                n_rows_clean=5,  # wrong: df has 10 rows
            )

    def test_feature_set_missing_feature_column_raises(self) -> None:
        """Declaring a feature column not in df must raise ValidationError."""
        df: pl.DataFrame = self._make_valid_df(n=10)
        with pytest.raises(ValidationError):
            FeatureSet(
                df=df,
                feature_columns=("feat_a", "feat_nonexistent"),
                target_columns=("fwd_logret_1",),
                n_rows_raw=10,
                n_rows_clean=10,
            )

    def test_feature_set_missing_target_column_raises(self) -> None:
        """Declaring a target column not in df must raise ValidationError."""
        df: pl.DataFrame = self._make_valid_df(n=10)
        with pytest.raises(ValidationError):
            FeatureSet(
                df=df,
                feature_columns=("feat_a",),
                target_columns=("fwd_logret_99",),  # does not exist
                n_rows_raw=10,
                n_rows_clean=10,
            )

    def test_feature_set_is_frozen(self) -> None:
        """FeatureSet is frozen — mutation must raise."""
        df: pl.DataFrame = self._make_valid_df(n=5)
        fs: FeatureSet = FeatureSet(
            df=df,
            feature_columns=("feat_a",),
            target_columns=("fwd_logret_1",),
            n_rows_raw=5,
            n_rows_clean=5,
        )
        with pytest.raises(ValidationError):
            fs.n_rows_clean = 99  # type: ignore[misc]

    def test_feature_set_empty_target_columns_allowed(self) -> None:
        """FeatureSet with empty target_columns (inference mode) is valid."""
        df: pl.DataFrame = self._make_valid_df(n=5)
        fs: FeatureSet = FeatureSet(
            df=df,
            feature_columns=("feat_a",),
            target_columns=(),
            n_rows_raw=5,
            n_rows_clean=5,
        )
        assert fs.target_columns == ()
