---
name: RC2 S3 Agent 2 MI + DA + Stability
description: Validation analysis insights -- MI effect sizes, DA economic significance, stability patterns, cross-bar, holdout, multi-horizon
type: reference
---

## RC2 Section 3 Part 2: Validation Analysis Architecture

### Service Created
- `src/app/research/application/rc2_validation_analysis.py` -- `RC2ValidationAnalyzer` class
- Methods: `build_mi_table`, `build_da_table`, `build_stability_heatmap_data`, `build_cross_bar_comparison`, `build_multi_horizon_comparison`, `compute_holdout_retention`, `compute_horizon_summary`
- Standalone: `compute_target_entropy_gaussian()` -- Gaussian entropy upper bound

### Key Design Decisions
- **MI/H(target) % effect size**: Normalizes MI by target entropy for cross-target comparability. For Gaussian target, H = 0.5*log(2*pi*e*var). Expect MI/H(target) well below 5% for financial features.
- **DA excess computed two ways**: (1) vs 50% coin flip, (2) vs break-even DA from transaction costs. The second is the economically meaningful one (Ziliak-McCloskey distinction).
- **Stability heatmap**: Binary matrix (0/1) of MI significance per window. Window reports run with reduced permutations (stability config: 500 MI, 200 Ridge) for speed.
- **Cross-bar comparison**: Uses even fewer permutations (200 MI, 100 Ridge) since it's directional, not definitive. Full permutations only on the primary (BTCUSDT, dollar) validation.
- **Holdout preview**: Uses 2023 data with temporal windows restricted to ((2023, 2024),) and min_valid_windows=1. This is NOT the final holdout (2024 in Phase 14).
- **Multi-horizon**: Primary horizon (fwd_logret_1) gets full 1000/500 permutations; others get 500/200. Rule F4 robustness requires keep in >= 2/3 horizons.

### Notebook Cells Added
- Appended after cell `kngmvwfx7sn` (end of Section 2 "Therefore")
- 13 cells total: 1 section header (md), 1 setup+validation (code), 2 MI table+interp (code+md), 2 DA table+interp (code+md), 1 stability heatmap (code), 2 cross-bar+interp (code+md), 1 holdout preview (code), 1 multi-horizon (code), 2 multi-horizon interp + section summary (md+md)

### Tests Created
- `src/tests/research/test_rc2_validation_analysis.py` -- 25 tests across 8 test classes
- Covers: MI table columns/sorting/effect-size/zero-entropy/single-feature, DA table columns/excess/negative-breakeven/sorting, stability shape/binary/empty/values, cross-bar shape/NaN/empty, multi-horizon shape/empty/labels, holdout retention/loss/columns, horizon summary, target entropy Gaussian/zero-variance

### Integration Points
- Uses `FeatureValidator.validate()` from `src/app/features/application/validation.py`
- Uses `compute_breakeven_da()` from `src/app/research/application/rc2_thresholds.py`
- References `ValidationReport`, `FeatureValidationResult` from `src/app/features/domain/entities.py`
- References `FeatureSet`, `ValidationConfig` from `src/app/features/domain/value_objects.py`
- Notebook depends on variables from Section 1 setup cell (`prereg`, `validation_config`, `feature_config`) and Section 2 data loading cell (`_load_bar_data_as_polars`, `builder`, `stationarity_reports`)
