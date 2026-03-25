---
name: RC2 Section 2 Stationarity Service
description: Analysis service for cross-asset stationarity reporting in RC2
type: reference
---

## RC2 Section 2: Stationarity Analysis Service

**Files created:**
- `src/app/research/application/rc2_stationarity.py` -- `RC2StationarityAnalyzer` + `StationaritySummary`
- `src/tests/research/test_rc2_stationarity.py` -- ~15 tests covering aggregation, rendering, edge cases

**Design:**
- `RC2StationarityAnalyzer` delegates per-(asset, bar_type) ADF+KPSS screening to the existing `StationarityScreener` from Phase 5
- Aggregates results into three buckets: universally_stationary, universally_non_stationary, mixed
- `render_summary_table()` produces a per-feature Pandas DataFrame with counts and classifications
- `render_cross_asset_table()` produces a feature-vs-combo matrix of ADF+KPSS joint classifications
- `generate_therefore()` builds the RC2 Section 2 conclusion paragraph programmatically with stationarity thresholds at 80% (high), 50% (moderate), <50% (aggressive transforms needed)
- Recommended transformations propagated from `_KNOWN_TRANSFORMATIONS` in the screener (prefix-matched: atr_ -> pct_atr, amihud_ -> rolling_zscore, hurst_ -> first_difference, bbwidth_ -> first_difference)

**Key insight:** The stationarity screener uses joint ADF+KPSS classification (4 outcomes: stationary, trend_stationary, unit_root, inconclusive) but the RC2 summary reduces this to binary is_stationary for aggregation, preserving the full classification in the cross-asset matrix.
