# RC2 Section 8: Go/No-Go Decision

## Methodology

The Go/No-Go decision is **fully mechanical** -- computed by code from the
pre-registered rules defined in Section 1. Every criterion is evaluated
programmatically against its threshold. No subjective judgment is applied.

### Decision Structure

- **G1, G2, G4 are BLOCKERS:** All three must pass for GO.
- **G3, G5, G6, G7 are INFORMATIONAL:** They shape model design but do not block.
- **Overall GO:** G1 AND G2 AND G4
- **Overall NO-GO:** G1 OR G2 OR G4 fails -> pivot to negative result protocols

## Results: Criterion-by-Criterion

### G1: Features Passing Validation (BLOCKER)

**Threshold:** >= 5 features (or fallback triggered)
**Result:** 5 features kept (via fallback)
**Decision: GO**

The three-gate validation pipeline found 0/23 features passing all three gates
(MI + DA + Stability). The F2 fallback mechanism kept the top 5 by composite score:
amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48.

**Nuance:** This is technically a PASS because the fallback was pre-registered as an
acceptable mechanism. However, the fact that 0 features passed organically is a strong
signal that directional prediction on dollar bars is extremely challenging.

### G2: DA Excess Over Baseline (BLOCKER)

**Threshold:** >= 0.5 pp for >= 1 (asset, bar_type)
**Result:** Best excess = +4.47 pp (BTCUSDT/volume_imbalance)
**Decision: GO**

Critically, this pass comes from volume_imbalance bars, not the primary dollar bars.
On dollar bars alone, the best DA excess is +1.81 pp (ret_zscore_24), which also
passes the 0.5 pp threshold. The +4.47 pp on vol_imbalance is the strongest single
finding, but it is on a Tier B bar type with only 529 observations.

**Cross-bar results:** The G2 criterion checked across all bar types via the
cross_bar_reports dictionary, finding the maximum DA excess across all available
(asset, bar_type) combinations.

### G3: Permutation Entropy (INFORMATIONAL)

**Threshold:** H_norm < 0.98 at d=5 for >= 1 bar type
**Result:** 2 bar types below 0.98 (BTC vol_imbalance: 0.9740, BTC dollar_imbalance: 0.9796)
**Decision: structure** (not near-random-walk)

The threshold is met, but only on imbalance bars. Dollar and volume bars are above
0.98 (near-random-walk). This means:
- Directional strategies are theoretically viable (structure exists)
- But the structure is concentrated in information-driven bars with small samples
- The recommender should use PE as a gating feature: trade only when PE < 0.98

### G4: N_eff on Primary Bar Type (BLOCKER)

**Threshold:** >= 1,000
**Result:** N_eff = 5,286 (BTCUSDT/dollar)
**Decision: GO**

BTCUSDT dollar bars have abundant effective observations. Statistical tests have
adequate power (MDE DA = 0.5171, meaning effects as small as 1.71 pp above 50%
are detectable at 80% power).

### G5: Cross-Asset MI Consistency (INFORMATIONAL)

**Threshold:** Kendall tau > 0 (p < 0.05)
**Result:** tau = 0.571, p = 0.0000 (3 assets: BTC, ETH, SOL)
**Decision: shared** (pooled training justified)

Feature MI rankings generalize across assets. This is positive for the recommendation
system: a single feature engineering pipeline serves all assets, and cross-asset
training increases effective sample size.

### G6: BDS on GARCH Residuals (INFORMATIONAL)

**Threshold:** Rejects i.i.d. for >= 1 asset
**Result:** All 4 assets reject (BTCUSDT, ETHUSDT, LTCUSDT, SOLUSDT)
**Decision: nonlinear** (tree ensembles + DL justified)

This is a strong, universal finding. GARCH(1,1) does not fully explain the conditional
dependence structure in any of the four assets. Nonlinear models are justified per
Rule M1, and the deep learning gate (Rule M2) is open for Tier A combinations.

### G7: Break-Even DA Feasibility (INFORMATIONAL)

**Threshold:** Break-even DA < 55% for >= 1 (asset, bar_type, horizon)
**Result:** 7 combinations have break-even DA < 55%
**Decision: feasible**

Imbalance bars and some dollar-bar combinations have break-even DA below 55%,
meaning that economically meaningful signals are within reach. However, no single
feature currently achieves this level.

## Overall Decision

### OVERALL: GO

| Gate | Criterion | Pass? |
|------|-----------|-------|
| G1 (BLOCKER) | Features >= 5 | YES (5, via fallback) |
| G2 (BLOCKER) | DA excess >= 0.5 pp | YES (+4.47 pp, vol_imbalance) |
| G4 (BLOCKER) | N_eff >= 1,000 | YES (5,286) |
| G3 (INFO) | PE < 0.98 | YES (2 bar types) |
| G5 (INFO) | Kendall tau > 0 | YES (0.571) |
| G6 (INFO) | BDS rejects i.i.d. | YES (4/4 assets) |
| G7 (INFO) | Break-even DA < 55% | YES (7 combos) |

**All seven criteria produce positive results.** This is the best possible outcome
for the GO decision, though the strength of each finding varies considerably.

## Final Outputs for Phase 6+ Modeling

### 1. Final Feature Set

**Primary (BTCUSDT/dollar, fwd_logret_1):** amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48

**Per-horizon:**
- fwd_logret_1: amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48
- fwd_logret_4: amihud_24, gk_vol_24, logret_12, park_vol_24, roc_12
- fwd_logret_24: amihud_24, bbwidth_20_2.0, park_vol_24, rv_24, rv_48

**Robustly informative (>= 2/3 horizons):** amihud_24, bbwidth_20_2.0, park_vol_24, rv_24, rv_48

### 2. Asset Universe

| Asset | Status | N_eff (dollar) |
|-------|--------|---------------|
| BTCUSDT | CONFIRMED | 5,286 |
| ETHUSDT | CONFIRMED | 2,454 |
| SOLUSDT | MARGINAL (included, flagged) | 808 |
| LTCUSDT | EXCLUDED (insufficient dollar bars) | N/A |

### 3. Confirmed Bar Types

All 5 bar types confirmed for at least 2 assets:
- dollar (primary), volume, volume_imbalance, dollar_imbalance, time_1h (baseline)

### 4. Confirmed Forecast Horizons

All 3 confirmed: fwd_logret_1, fwd_logret_4, fwd_logret_24

### 5. Model Complexity Recommendation

**Deep Learning Gate: OPEN**
- N_eff >= 2,000: PASS (5,286)
- Features >= 3: PASS (5)
- BDS rejects i.i.d.: PASS (all assets)

**Recommendation:** Nonlinear (tree ensembles + DL candidates)

### 6. Regression Feasibility

**FEASIBLE** -- both classification (SIDE) and regression (SIZE) tracks proceed.

### 7. Pre-Registration Integrity

- Post-hoc deviations: **0**
- Total trial count: **60** (60 pre-registered + 0 post-hoc)
- Pre-registration followed exactly.

## Honest Assessment

While the overall decision is GO, several caveats must be stated clearly:

1. **The GO is conditional on multi-feature combination working.** Single features
   fail the economic test. If ensemble DA also fails to exceed break-even, the thesis
   reports an honest N2 (partial failure) result.

2. **G2 passes on a Tier B bar type.** The +4.47 pp DA excess is on vol_imbalance
   with only 529 observations. On dollar bars (Tier A), the best excess is +1.81 pp --
   still above the 0.5 pp threshold but much weaker.

3. **G1 passes only via fallback.** Zero features pass the three-gate validation
   organically. The "5 features kept" is an artifact of the safety net, not genuine
   multi-gate validation success.

4. **LTCUSDT is effectively excluded** from the primary modeling pipeline despite
   being a pre-registered asset. The dollar bar threshold needs recalibration for
   lower-cap assets.

5. **The signal, where it exists, is regime-conditional.** MI significance concentrates
   in the 2022-2023 window. Models trained on the full period will be diluted.

These caveats do not change the GO decision (the rules are mechanical), but they
must be prominently documented in the thesis.

## Connection to Lopez de Prado

The entire Go/No-Go framework embodies Lopez de Prado's (2018) principle that
research should follow pre-committed rules. By defining thresholds before seeing data,
the researcher cannot cherry-pick outcomes. The 0 post-hoc deviation count proves
that no rules were bent after seeing results.

The Deflated Sharpe Ratio trial count of 60 is well-documented and honest. When
combined with the expected ~21 model/configuration trials in Phases 9-12, the total
will be ~81 trials. At crypto kurtosis of ~5-15 and a strategy SR_obs of ~2.0, the
DSR would be approximately 0.90-0.95 -- a credible but not overwhelming result.
This realistic framing is more valuable than an inflated SR that collapses under DSR
scrutiny.
