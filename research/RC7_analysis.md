# RC7 Analysis: Phase 7 Research Checkpoint Summary

> **Phase 7 — Profiling & Audit Closure** | Pre-registration deviations: 0 | Trial count: 60 (unchanged)
>
> Phase 7 resolves six audit gaps from RC2, establishes formal policies, and confirms/extends the asset universe for downstream modeling.

---

## Table of Contents

1. [Phase 7.1 — Cost Sensitivity (C3, GH #71)](#phase-71--cost-sensitivity)
2. [Phase 7.2 — Feature Degeneracy (C4, GH #72)](#phase-72--feature-degeneracy)
3. [Phase 7.3 — Stationarity Transformation Policy (B3, GH #73)](#phase-73--stationarity-transformation-policy)
4. [Phase 7.4 — MI Normalization Fix (Audit Gap 5, GH #74)](#phase-74--mi-normalization-fix)
5. [Phase 7.5 — LTCUSDT Volume-Bar Profiling (A2, GH #75)](#phase-75--ltcusdt-volume-bar-profiling)
6. [Phase 7.6 — Conditional Break-Even DA (B1, GH #76)](#phase-76--conditional-break-even-da)
7. [Phase 7.7 — SOLUSDT Tier B Protocol (Audit Gap 4, GH #77)](#phase-77--solusdt-tier-b-protocol)
8. [Cross-Cutting Decisions](#cross-cutting-decisions)
9. [Updated Asset Universe](#updated-asset-universe)
10. [Implications for Downstream Phases](#implications-for-downstream-phases)

---

## Phase 7.1 — Cost Sensitivity

**Notebook:** `RC7_profiling_closure.ipynb` (Section C3) | **Issue:** GH #71

### Goal

Sweep break-even DA across {10, 15, 20, 25, 30} bps for all 16 canonical (asset, bar_type) combinations. RC2 only tested 20 bps.

### Key Results

| Cost tier | Combos viable (BE_DA <= 55%) | Best gap vs 51.81% DA |
|-----------|------------------------------|----------------------|
| 10 bps   | 9/16                         | +1.17 pp             |
| 15 bps   | 6/16                         | +0.85 pp             |
| 20 bps   | 7/16                         | +0.53 pp             |
| 25 bps   | 1/16                         | +0.21 pp             |
| 30 bps   | 0/16                         | -0.11 pp             |

**Maximum viable cost by bar type:**
- **Imbalance bars** (vol_imb, dol_imb): viable at all cost tiers (max viable 47–78 bps)
- **Dollar bars**: viable at VIP tier (~10–14 bps), marginal at standard (20 bps)
- **Time bars**: not viable at any cost tier (BE_DA > 60% even at 10 bps)
- **Volume bars** (ETH/SOL/LTC): not viable at any tier

### Decision

**RESOLVED (C3).** Cost sensitivity is substantial. Imbalance bars are most cost-tolerant; time bars are non-viable for directional trading. No single feature exceeds BE_DA on dollar bars — multi-feature ensemble is required.

---

## Phase 7.2 — Feature Degeneracy

**Notebook:** `RC7_profiling_closure.ipynb` (Section C4) | **Issue:** GH #72

### Goal

Investigate `atr_14` and `rsi_14` constant-value degeneracy flagged during RC2 stationarity screening.

### Key Results

| Feature  | Degenerate combos | Root cause                                                                 |
|----------|-------------------|----------------------------------------------------------------------------|
| `atr_14` | 7/16              | High bar thresholds → `high ≈ low ≈ close` → True Range ≈ 0 → constant   |
| `rsi_14` | **16/16**          | Near-zero close-to-close changes → avg_gain/avg_loss → 0/0 → RSI = 50.0  |

### Decision

**Both `atr_14` and `rsi_14` DROPPED from ALL bar types.** Feature count reduced from 23 → **21 features** universally.

**RESOLVED (C4).**

---

## Phase 7.3 — Stationarity Transformation Policy

**Notebook:** `RC7_stationarity_policy.ipynb` | **Issue:** GH #73

### Goal

Establish explicit transformation policy for all features across all (asset, bar_type) combinations. Resolve 40 unit-root and 108 trend-stationary cases from RC2.

### Formal Policy (Rules ST1–ST4)

| Rule | Statement |
|------|-----------|
| **ST1** | Dollar-bar stationarity governs. Unit root on dollar bars → transform globally |
| **ST2** | Unit root on secondary bars only → flag, don't transform |
| **ST3** | Constant features (`atr_14`, `rsi_14`) → exclude from modeling |
| **ST4** | Trend-stationary features → accepted for tree-based models without transformation |

### Transformation Mapping (7 features)

| Feature          | Transformation    | Formula                                    |
|------------------|-------------------|--------------------------------------------|
| `amihud_24`      | rolling_zscore    | `(x - rolling_mean(24)) / rolling_std(24)` |
| `bbwidth_20_2.0` | first_difference  | `bbwidth.diff()`                           |
| `gk_vol_24`      | rolling_zscore    | `(x - rolling_mean(24)) / rolling_std(24)` |
| `park_vol_24`    | first_difference  | `park_vol.diff()`                          |
| `rv_12`          | first_difference  | `rv_12.diff()`                             |
| `rv_24`          | first_difference  | `rv_24.diff()`                             |
| `rv_48`          | first_difference  | `rv_48.diff()`                             |

### Effectiveness

- **Before:** 40 unit roots (10.2% of 391 tests)
- **After:** 6 unit roots (1.5%) — all on secondary bar types (BTCUSDT/volume_imbalance, N=430)
- **Success rate:** 95.8% of transformations produce stationary results

### Decision

**GO.** Policy established. Unit roots reduced 85% (40 → 6). Remaining 6 are secondary-bar-type artifacts.

---

## Phase 7.4 — MI Normalization Fix

**Notebook:** `RC7_mi_normalization.ipynb` | **Issue:** GH #74

### Goal

Fix nonsensical MI/H(target) percentages in RC2 tables caused by negative Gaussian differential entropy for small-variance continuous targets.

### Root Cause

Crypto bar-level log returns have variance ~1e-4, far below the critical threshold of 0.0586 where Gaussian differential entropy turns negative. Dividing positive MI by negative H(target) produced negative or trillion-scale "percentages."

### Key Finding

**Zero keep/drop decisions were affected.** The `validation.py` pipeline uses MI permutation p-values with BH correction — entropy normalization was never referenced in the decision logic. The bug was purely cosmetic (display layer only).

### Corrected Reporting Metric

| Metric | Description |
|--------|-------------|
| **Primary:** Raw MI (nats) | From sklearn's Kraskov k-NN estimator — well-defined, non-negative |
| **Secondary:** MI / H_disc(feature) % | Discrete Shannon entropy (always positive) |
| **Qualitative scale** | Strong (>0.05), Moderate (0.01–0.05), Weak (0.001–0.01), Negligible (<0.001) |

### MI Effect-Size Distribution (BTCUSDT/dollar)

| Effect Size   | Count | Top features                                            |
|---------------|-------|---------------------------------------------------------|
| **Strong**    | 6/23  | amihud_24, gk_vol_24, rv_24, park_vol_24, rv_48, rv_12 |
| **Moderate**  | 6/23  | bbwidth, logret_24, roc_4, logret_4, ema_xover, bbpctb  |
| **Weak**      | 3/23  | roc_12, logret_12, slope_14                             |
| **Negligible** | 8/23 | hurst_100, atr_14, obv_slope_14, rsi_14, etc.          |

### Decision

**GO.** Zero deviations from pre-registration. Fix is cosmetic only. Raw MI (nats) adopted as reporting metric.

---

## Phase 7.5 — LTCUSDT Volume-Bar Profiling

**Notebook:** `RC7_ltcusdt_profiling.ipynb` | **Issue:** GH #75

### Goal

Confirm viability of LTCUSDT volume bars (N=26,987) for Phases 10–11 modeling pipeline. LTCUSDT was excluded from dollar-bar modeling (only 199 bars).

### Profiling Results

| Metric                  | LTCUSDT/volume |
|-------------------------|----------------|
| N_bars                  | 26,987         |
| N_clean (after warmup)  | 26,864         |
| Tier                    | **A** (DL-eligible) |
| Stationary features     | 14/23 (tied best with BTC) |
| MI-significant features | **16/23** (69.6%, most of all assets) |
| Kept features (3-gate)  | 6: gk_vol_24, logret_24, park_vol_24, rv_12, rv_24, rv_48 |
| Best single DA          | 52.77% (logret_1) |
| Break-even DA (20 bps)  | 60.19% |
| Gap vs BE               | -7.42 pp |
| Stable features (>= 50% windows) | 3/23 (park_vol_24, rv_24, rv_48) |

### Cross-Asset MI Rank Correlation (Kendall tau)

| Pair                 | tau   | p-value  |
|----------------------|-------|----------|
| LTCUSDT vs BTCUSDT   | 0.500 | 0.0009   |
| LTCUSDT vs ETHUSDT   | 0.493 | 0.0010   |
| LTCUSDT vs SOLUSDT   | 0.713 | < 0.0001 |

All correlations significant → **pooled cross-asset training supported (Rule A2).**

### Decision

**VIABLE.** LTCUSDT/volume bars included in volume-bar modeling pipeline. Viability checklist 6/6 pass.

---

## Phase 7.6 — Conditional Break-Even DA

**Notebook:** `RC7_conditional_breakeven.ipynb` | **Issue:** GH #76

### Goal

Demonstrate that regime-conditional deployment (trading only HIGH-volatility bars) lowers break-even DA, directly validating the recommender's value proposition.

### Key Results

**Amplification ratio (HIGH-regime mean |r| / unconditional mean |r|):**
- Mean across 16 combos: **1.722x**
- Range: 1.469x (ETHUSDT/dollar) to 2.067x (ETHUSDT/vol_imb)
- **All 16 combos > 1.0** — universal structural property

**Break-even DA reduction at 20 bps:**

| Statistic           | Value     |
|---------------------|-----------|
| Mean BE_DA (uncond)  | 58.20%   |
| Mean BE_DA (HIGH)    | **54.80%** |
| Mean reduction       | **+3.40 pp** |
| Max reduction        | +10.95 pp (BTCUSDT/time_1h) |

**Feasibility improvement:**
- Unconditionally feasible combos (20 bps): 4/16
- Conditionally feasible combos (20 bps): **6/16** (+2 newly feasible)
- At 30 bps: 0/16 unconditional → **5/16 conditional** (selective trading is the only path to viability)

**Bar type benefit pattern:**
- Time bars: largest absolute reduction (+7.80 pp mean) but still highest BE_DA
- Imbalance bars: smallest reduction (+0.7–0.8 pp) but already have lowest BE_DA (<51%)
- Dollar bars: moderate reduction (+1.55 pp)

### Decision

**RC2 GO strengthened.** Regime-conditional deployment fundamentally changes the economics. The recommender doesn't need to beat unconditional BE_DA — only the lower conditional BE_DA on selected bars.

---

## Phase 7.7 — SOLUSDT Tier B Protocol

**Notebook:** `RC7_solusdt_tier_b_protocol.ipynb` | **Issue:** GH #77

### Goal

Define concrete protocol for SOLUSDT dollar bars (Tier B, N_eff = 808). RC2 said "MARGINAL — included but flagged" without specifying what "flagged" means.

### Statistical Power Analysis

| Asset    | N_eff | Tier | MDE DA | CI width (pp) | Power at 53% DA |
|----------|-------|------|--------|---------------|-----------------|
| BTCUSDT  | 5,286 | A    | 51.13% | 2.70          | 99.7%           |
| ETHUSDT  | 2,454 | A    | 51.66% | 3.96          | 90.8%           |
| SOLUSDT  | 808   | B    | **52.89%** | **6.90**  | **52.4%**       |

SOLUSDT is blind to effects below +2.89 pp and reaches 80% power only at ~54% true DA.

### Formal Protocol (P1–P8)

| Rule | Summary |
|------|---------|
| **P1** | Every SOLUSDT table/chart carries Tier B label (dagger + orange / hatched bars) |
| **P2** | Regularisation 2x: Ridge alpha x 2; LightGBM min_child_samples x 2, num_leaves / 2, max_depth - 1; CPCV 3 folds |
| **P3** | Bootstrapped 95% CIs (10,000 resamples, percentile method) |
| **P4** | Tier B results are NOT primary evidence — robustness check only |
| **P5** | If SOLUSDT is the only positive asset, RC4 criterion does NOT pass |
| **P6** | Kelly fraction x 0.5 for Tier B assets |
| **P7** | All tables include N_eff and Tier columns; thesis text uses explicit caveats |
| **P8** | Same thresholds apply to any future (asset, bar_type) combination |

### Tier Classification

| Tier | N_eff       | Treatment                                    |
|------|-------------|----------------------------------------------|
| **A** | >= 2,000   | Standard pipeline, primary evidence          |
| **B** | 500–1,999  | Modified pipeline, robustness check only     |
| **C** | < 500      | Excluded from modeling                       |

### Decision

**RC2 GO unchanged.** Protocol established. SOLUSDT included with explicit constraints across all downstream phases.

---

## Cross-Cutting Decisions

| # | Decision | Source |
|---|----------|--------|
| 1 | Feature count: **23 → 21** (drop `atr_14`, `rsi_14`) | Phase 7.2 |
| 2 | 7 features require stationarity transformation before modeling | Phase 7.3 |
| 3 | MI reported as raw nats + qualitative scale (not % of entropy) | Phase 7.4 |
| 4 | LTCUSDT enters pipeline on volume bars only (Tier A, N=26,987) | Phase 7.5 |
| 5 | Regime-conditional BE_DA replaces unconditional for recommender evaluation | Phase 7.6 |
| 6 | Tier B protocol (P1–P8) governs SOLUSDT dollar bars | Phase 7.7 |
| 7 | Imbalance bars most cost-tolerant; time bars non-viable for directional trading | Phase 7.1 |

---

## Updated Asset Universe

### Dollar Bars

| Asset    | N_eff | Tier | Status     |
|----------|-------|------|------------|
| BTCUSDT  | 5,164 | A    | CONFIRMED  |
| ETHUSDT  | 2,636 | A    | CONFIRMED  |
| SOLUSDT  | 686   | B    | CONFIRMED (Tier B protocol) |
| LTCUSDT  | 199   | C    | **EXCLUDED** |

### Volume Bars

| Asset    | N_eff  | Tier | Status     |
|----------|--------|------|------------|
| BTCUSDT  | 3,141  | A    | CONFIRMED  |
| ETHUSDT  | 23,915 | A    | CONFIRMED  |
| SOLUSDT  | 47,055 | A    | CONFIRMED  |
| LTCUSDT  | 26,864 | A    | **CONFIRMED (Phase 7.5)** |

### Imbalance Bars

| Asset    | Bar Type           | N_eff | Tier | Status    |
|----------|--------------------|-------|------|-----------|
| BTCUSDT  | volume_imbalance   | 431   | C    | Profiling only |
| BTCUSDT  | dollar_imbalance   | 470   | C    | Profiling only |
| ETHUSDT  | volume_imbalance   | 599   | B    | Tier B    |
| SOLUSDT  | volume_imbalance   | 772   | B    | Tier B    |
| LTCUSDT  | volume_imbalance   | 639   | B    | Tier B    |

---

## Implications for Downstream Phases

| Phase | Impact |
|-------|--------|
| **Phase 9 (Features)** | 21 features (not 23); apply 7 stationarity transformations; propagate tier metadata |
| **Phase 10 (Classification)** | Tier B: Ridge alpha x 2, LightGBM regularised, CPCV 3 folds; use raw MI nats |
| **Phase 11 (Regression)** | Same regularisation; bootstrapped CIs for Tier B; LTCUSDT on volume bars only |
| **Phase 12 (Recommendation)** | Regime-conditional deployment; output `tier` field; Kelly x 0.5 for Tier B |
| **Phase 13 (Backtest)** | Cost sensitivity at {10, 20, 30} bps; Tier B flagging; imbalance bars prioritised |
| **Phase 14 (Evaluation)** | Tier B = robustness check only; conditional BE_DA as primary feasibility metric |
| **RC4 (Go/No-Go)** | SOLUSDT counts with flag; cannot be sole positive; LTCUSDT on volume bars |
| **Thesis** | MI in nats; Tier B caveats; conditional BE_DA; 21-feature count |
