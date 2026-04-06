# RC2 Plan Audit Report

**Auditor:** Quant-Crypto-Architect Agent
**Date:** 2026-03-25
**Scope:** IMPLEMENTATION_PLAN.md vs RC2 findings (RC2_analysis.md, RC2_plan_updates.md, all rc2_sections/)
**Verdict:** PASS WITH CONCERNS

---

## 1. Executive Summary

The implementation plan is thorough, methodologically rigorous, and well-grounded in
the Lopez de Prado framework. RC2 validated the core architecture but revealed a
fundamental constraint: no single feature exceeds the break-even DA on the primary bar
type (best = 51.81% vs break-even 57.23%). The plan update proposal (RC2_plan_updates.md)
correctly identifies the major implications, but several gaps remain between what RC2
found and what the plan commits to doing about it.

**Key Findings:**

1. **CRITICAL: Trial count inconsistency.** The plan says "200-500+" trials for DSR
   (Phase 14E, line 2321) while pre-registration says 60 trials (Section 1) and the
   plan update says "~81 trials." These must be reconciled. The DSR is only as honest as
   the trial count is accurate.

2. **MAJOR: LTCUSDT exclusion creates a silent asset universe reduction.** The plan
   still references "4 assets" in multiple places (Phase 9F: "Pool 4 assets...~14000+
   samples"; Phase 13A: ">= 3 out of 4 assets"). RC2 effectively reduces this to 3
   assets for dollar bars. The pooled sample size estimate is inflated.

3. **MAJOR: Phase reordering (10 before 9) has an unresolved dependency.** Phase 10
   regression metrics (DC-MAE, DC-RMSE) are defined as conditional on Phase 9 classifier
   correctness. If Phase 10 runs first, what is the conditioning event?

4. **MAJOR: The ensemble DA justification is hand-waved.** The plan assumes "multi-feature
   combination may push ensemble DA above break-even" without specifying a test for this
   claim or a fallback if it fails.

5. **MINOR: Non-stationary features entering the modeling pipeline.** RC2 Section 2
   identified 19 features as non-stationary in at least one bar type, with suggested
   transformations. The plan (Phase 5pre.4) specifies transformations for 4 features
   (atr_14, amihud_24, hurst_100, bbwidth_20_2.0) but does not address the other 15.

---

## 2. Section-by-Section Findings

### A. Consistency Check

**A1. [CRITICAL] Trial count contradiction across documents.**

Three different numbers appear in the corpus:

| Source | Trial Count | Context |
|--------|-------------|---------|
| RC2 Section 1 pre-registration | 60 | 4 assets x 5 bar_types x 3 horizons |
| RC2 Section 8 / Plan Updates | ~81 | 60 pre-registered + ~21 model configs |
| IMPLEMENTATION_PLAN Phase 14E | 200-500+ | "honestly exhaustive" count |

The 60 is the pre-registered combinatorial count. The "~21 model configs" is mentioned
in the plan update and Section 8 (S8 line: "combined with the expected ~21
model/configuration trials in Phases 9-12, the total will be ~81 trials"). But Phase 14E
says "Likely 200-500+ trials" and instructs to "count ALL configs including RC decision
points, hyperparameter searches, model variants."

**Issue:** The plan was written before RC2 established the pre-registration. Phase 14E's
"200-500+" instruction directly contradicts the pre-registration's commitment to 60+
deviations. If the DSR uses 500 trials, a strategy needs SR > 3.0 to be significant. If
it uses 60, SR > 2.0 suffices. This is a 50% difference in the significance bar.

**Recommendation:** Resolve by defining two trial tiers: (a) pre-registered exploration
space = 60, (b) post-hoc hyperparameter/architecture choices = counted individually.
Update Phase 14E to reference the pre-registration framework and compute DSR at BOTH
the pre-registered count and the exhaustive count. Report both. The honest approach is
to use the larger count as the primary result and the smaller count as a sensitivity
analysis.

---

**A2. [MAJOR] Asset universe inconsistency: "4 assets" vs "3 assets".**

RC2 Section 8 states: LTCUSDT EXCLUDED from dollar-bar modeling (199 bars). SOLUSDT is
MARGINAL (N_eff = 808, Tier B, not DL-eligible).

However, the plan still references 4 assets in:
- Phase 9F: "Pool 4 assets with asset_id categorical (~14000+ samples)"
- Phase 12A: "4 assets x 3 strategies x ~150 weeks = 1800 labels"
- Phase 13A: ">= 3 out of 4 assets show positive contribution"

The pooled sample size of "~14000+" should be ~12000+ (3 assets x ~4000 bars avg for
dollar bars after warmup), or ~15000+ if including volume bars. The recommendation system
label count drops from 1800 to 1350 (3 assets x 3 strategies x 150 weeks).

**Recommendation:** Update all forward-looking references to specify "3 assets (BTC, ETH,
SOL) for dollar bars; 4 assets including LTC for volume bars." Re-derive pooled sample
sizes. Adjust the RC4 criterion from "3 out of 4" to "2 out of 3" (or define it as
"majority of included assets").

---

**A3. [MINOR] Bar type tier classification discrepancy.**

RC2 Section 6 classifies SOLUSDT/dollar as Tier B (N_eff = 808). The plan update
correctly notes this but does not update Phase 9's DL gate assessment, which states
"BTCUSDT: 5,286" as the only cited example. ETH dollar (N_eff = 2,454) also passes
the DL gate but is not explicitly called out.

RC2 Section 6 also shows BTCUSDT vol_imbalance N_eff = 451, not the 529 raw bars.
The plan update table (Section 4.5) lists N_eff = 451 for vol_imbalance but uses N = 529
elsewhere. Consistency needed.

**Recommendation:** Add a canonical table to the plan: (asset, bar_type, N_raw, N_eff,
tier, DL_eligible) that becomes the single source of truth for all downstream phases.

---

**A4. [MINOR] Robustly informative feature set differs between sources.**

- RC2 Section 3 multi-horizon analysis: amihud_24, bbwidth_20_2.0, park_vol_24, rv_24, rv_48
- RC2 Section 8 primary feature set (fwd_logret_1): amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48
- Plan update Section 3 "Robustly informative": amihud_24, bbwidth_20_2.0, rv_24, rv_48, park_vol_24

Note: rv_12 is in the primary set but NOT in the robustly informative set (it only
appears for fwd_logret_1). park_vol_24 is robustly informative but NOT in the primary
set. These are both correct within their definitions but could confuse downstream phases.

**Recommendation:** Explicitly define the feature set hierarchy: (a) primary = fallback-5
for fwd_logret_1, (b) robust = intersection across >= 2/3 horizons, (c) full = all 23
for regularized models. Reference these by name in Phases 9-10.

---

### B. Statistical Rigor

**B1. [MAJOR] Break-even DA formula verification.**

The formula used is:
```
break_even_DA = 0.5 + round_trip_cost / (2 * mean(|r_t|))
             = 0.5 + 0.002 / (2 * 0.013829)
             = 0.5 + 0.0723
             = 0.5723 (57.23%)
```

Verified: 0.002 / (2 * 0.013829) = 0.002 / 0.027658 = 0.07231. Correct.

However, this formula assumes: (a) symmetric bet sizing (equal $ on longs and shorts),
(b) no slippage, (c) cost is per-trade not per-bar, (d) the strategy trades every bar.

RC2 does not discuss what happens when the strategy does NOT trade every bar (which is
the recommendation system's primary value). If the strategy only trades on 30% of bars
(those with high confidence), the mean |r_t| should be computed only on traded bars. If
the recommender selects high-volatility periods (which RC2 suggests), the conditional
mean |r_t| would be larger, LOWERING the break-even DA.

**Recommendation:** Add a conditional break-even DA calculation: `break_even_DA_conditional
= 0.5 + cost / (2 * mean(|r_t| | traded))`. Estimate the conditional mean from the
high-volatility regime subset (available from RC2 Section 5.6). This directly
demonstrates the recommender's value proposition.

---

**B2. [NOTE] Pre-registration integrity is strong.**

0 post-hoc deviations, 60 pre-registered trials, all thresholds defined before data
examination. This is the strongest methodological element. The frozen Pydantic model
for `RC2PreRegistration` prevents mutation. Exemplary.

---

**B3. [MINOR] Stationarity testing is joint ADF+KPSS but the plan mentions fractional
differentiation as optional.**

RC2 Section 2 identifies 40 unit-root cases (10.2%) and 108 trend-stationary cases
(27.6%) across 391 tests. The stationarity section (rc2_s2_stationarity.md) recommends
transformations for 4 features but notes that "Phase 5+ could explore fractional
differentiation for the unit_root features" as an enhancement.

The plan (Phase 5pre.4) also lists fractional differentiation as optional ("or
first-difference"). RC2 Section 2 suggests simpler transformations (z-scores, percentage
ratios, first differences) are "more interpretable but may discard useful information."

**Issue:** 40 features classified as unit_root are entering the pipeline without
mandatory transformation. The plan says "features that fail stationarity" get transformed,
but the RC2 results show that some features are non-stationary on some bar types and
stationary on others. Is the transformation applied per (feature, bar_type) or globally?

**Recommendation:** Clarify transformation policy: if a feature is unit_root on the
PRIMARY bar type (dollar), apply the transformation globally. If it is only non-stationary
on secondary bar types, flag but do not transform (the primary-bar-type stationarity is
what matters for the modeling pipeline).

---

**B4. [NOTE] Walk-forward purging and embargo are well-specified.**

Phase 9A defines CPCV with cross-asset temporal purging (purge ALL assets in
[t - embargo, t + h + embargo]). This is the correct implementation for correlated crypto
assets. The plan correctly identifies BTC-ETH correlation (~0.85) as the leakage vector.

---

**B5. [NOTE] Monte Carlo validation methodology is sound.**

Three-tier null (GBM, GARCH-bootstrap, Politis-Romano stationary bootstrap) is the
gold standard. The plan correctly identifies that a volatility-aware strategy will "fail"
on GBM for the "wrong reason" and requires GARCH-bootstrapped paths as the proper null.

---

### C. Methodological Gaps

**C1. [MAJOR] No specified test for ensemble DA exceeding break-even.**

The entire plan's viability rests on the claim that multi-feature combination will push
DA above the 57.23% break-even threshold. But:

- No specific test is pre-registered for this claim.
- No fallback is specified if multi-feature Ridge DA (or any model DA) also fails to
  exceed break-even.
- The plan update Section 4 mentions "Multi-feature DA still below break-even" as a
  HIGH probability risk with mitigation "Frame as information boundary finding; recommender
  pivots to risk filter." But this pivot is not operationalized in any phase.

**What is missing:** A pre-registered test at the START of Phase 9:
"Run 23-feature Ridge + 23-feature LightGBM on BTCUSDT/dollar, fwd_logret_1. If neither
exceeds break-even DA (57.23%), downgrade the directional arm to exploratory and
re-focus the thesis on volatility forecasting + risk management." This is the single
most important gating decision after RC2.

**Recommendation:** Add a "Phase 9-pre" gating test. Define it now, before seeing the
results. If the gate fails, the thesis narrative shifts from "profitable directional
trading" to "information-theoretic limits on crypto prediction + a risk management
recommendation system." Both are valid thesis contributions.

---

**C2. [MAJOR] Volatility forecasting target is mentioned but not formalized.**

The plan update (Section 2, Phase 10) recommends: "Focus on volatility forecasting as the
regression target." RC2 confirms that volatility features dominate (all 5 kept features
are vol/liquidity). But:

- Phase 10A only defines `fwd_zret_h` (volatility-normalized return) as the target.
- `forward_volatility(horizon)` is listed in Phase 4B as a "secondary target" but no
  regression model in Phase 10B explicitly targets it.
- No evaluation metric specific to volatility forecasting is defined (e.g., QLIKE loss,
  Mincer-Zarnowitz regression, realized vol R-squared).

**Recommendation:** Add explicit volatility forecasting to Phase 10:
- Target: `forward_rv_h = realized_vol(close, h)` (forward-looking, used only as label)
- Models: GARCH(1,1) baseline, HAR-RV (Corsi 2009), LightGBM regressor
- Metrics: QLIKE, R-squared (Mincer-Zarnowitz), MAE on log-vol
- This is the lowest-hanging fruit given RC2's findings and should be prioritized.

---

**C3. [MINOR] Cost sensitivity analysis is recommended but not in a specific phase.**

The plan update recommends "sensitivity analysis on the round-trip cost parameter (10, 20,
30 bps)" for Phase 5 and "cost-sensitivity parameter" for Phase 7. Phase 7B includes
`run_with_cost_sweep`. But the RC2 analysis only used 20 bps.

The plan update (Section 7.3) notes that VIP Binance tiers reduce costs to ~10 bps,
lowering break-even DA from 57.23% to ~53.6%. This is material: at 10 bps, the best
feature (51.81%) is only 1.79 pp below break-even instead of 5.42 pp.

**Recommendation:** Run the break-even DA calculation at {10, 15, 20, 25, 30} bps as
part of Phase 5 closure (before Phase 9). Include in the thesis as a sensitivity
table. This takes ~10 lines of code and dramatically strengthens the feasibility argument.

---

**C4. [MINOR] atr_14 and rsi_14 constant-feature issue is flagged but not resolved.**

RC2 Section 2 notes: "atr_14 and rsi_14 are flagged as constant in multiple (asset,
bar_type) combinations." This is attributed to "how indicators are computed on alternative
bars with limited variation."

The plan (Phase 5pre.4) lists atr_14 -> pct_atr as a transformation. But if atr_14 is
constant (zero variance), pct_atr will also be constant (0/close = 0). The root cause
is likely that dollar bars with normalized dollar volume produce OHLCV bars where
high-low-close relationships are degenerate.

**Recommendation:** Investigate whether atr_14 and rsi_14 are truly constant or whether
this is a computation bug. If constant on dollar bars, drop both features from the
dollar-bar feature set (they carry no information). This reduces the feature count from
23 to 21 for dollar bars. Document the reason.

---

**C5. [NOTE] Temporal instability finding is acknowledged but not operationalized.**

RC2 Section 3 shows MI significance concentrated in 2022-2023 (8/23 features significant)
with 0/23 in all other windows. The plan update recommends "regime-conditional deployment"
but no phase specifies how to detect the "MI-significant regime" in real time.

**Recommendation:** In Phase 12 (recommendation system), add a feature: "rolling MI
significance proxy" -- e.g., rolling volatility above the 2022-2023 average. Or simpler:
use the volatility regime labels (HIGH regime correlates with the 2022-2023 window).
This is already partially addressed by the regime-conditional activation recommendation,
but should be made explicit.

---

### D. Practical Feasibility

**D1. [MAJOR] Phase reordering (10 before 9) creates a dependency problem.**

The plan update recommends: Phase 10 (regression) before Phase 9 (classification).
Rationale: "leverages the strongest signal first (volatility features)."

However, Phase 10D explicitly states: "Regression metrics (MAE, RMSE, R-squared) are only
meaningful WHEN the direction classifier is correct." The entire DC-MAE / DC-RMSE
framework depends on Phase 9 output.

If Phase 10 runs first, the "standalone regression evaluation" mode (Phase 10D) can
produce raw MAE/RMSE, but the PIPELINE evaluation requires a classifier. The
"Economic Sharpe" metric (10D) also requires a direction decision.

**Possible resolution:** Run Phase 10 in "standalone" mode (volatility forecasting, raw
return regression without direction conditioning). Then run Phase 9 (classification).
Then run the combined pipeline evaluation as part of RC3. This is what the plan update
seems to intend but does not explicitly state.

**Recommendation:** Clarify that Phase 10 (reordered) runs ONLY standalone regression
and volatility forecasting. The direction-conditional evaluation and Economic Sharpe
are computed AFTER Phase 9 completes, in Phase 11 (RC3). Update Phase 10D to split
metrics into "standalone" (available immediately) and "pipeline" (requires Phase 9).

---

**D2. [MINOR] Sample sizes for the recommendation system labels.**

Phase 12A estimates: "4 assets x 3 strategies x ~150 weeks = 1800 labels." With LTCUSDT
excluded from dollar bars, this becomes 3 x 3 x 150 = 1350 labels (or fewer if SOL is
dropped from some strategies due to Tier B status).

For a LightGBM regressor with ~40 features (market state + classifier + regressor +
regime + cross-asset + historical), 1350 samples is borderline. The rule of thumb for
gradient boosting on tabular data is N >= 10 * p (features). At p = 40, N_min = 400.
So 1350 is adequate but not comfortable, especially with walk-forward splitting reducing
the effective training set.

**Recommendation:** Consider strategies on volume bars (which have larger N for all
assets including LTC) alongside dollar bars. This increases the label count. Also consider
weekly non-overlapping windows (the current plan) vs overlapping with appropriate
purging (which increases N but introduces correlation).

---

**D3. [NOTE] Computational requirements are realistic.**

The primary models (LightGBM, Ridge, Logistic, GRU with 2 layers/64 hidden) are all
lightweight. The most expensive operation is Monte Carlo simulation (1000 paths x full
pipeline), which is embarrassingly parallel. No GPU required for the primary workload.
CPCV with C(6,2) = 15 combinations is manageable.

---

### E. Thesis Defense Readiness

**E1. [MAJOR] The R5 confrontation is adequate but needs a sharper conclusion.**

RC2 Section 4 thoroughly addresses R5 (crypto near Brownian noise) with PE analysis,
VR tests, and the complexity-entropy plane. The findings are honest: dollar bar PE =
0.9977 (above 0.98 threshold), imbalance bar PE = 0.9740 (below threshold).

However, the plan does not specify a "thesis-ready" paragraph that converts this into a
positive contribution. The rc2_s4_predictability.md recommends: "Do not claim that
crypto is 'unpredictable' -- claim it is 'near-unpredictable with transient structure.'"

**What is missing from the plan:** A Phase 14 deliverable that presents the PE analysis
as a standalone contribution: "We confirm R5's finding on standard bars but demonstrate
that information-driven sampling (imbalance bars) extracts structure below the Brownian
noise threshold. This is a novel empirical finding." This paragraph writes itself but
needs to be committed to in the plan.

**Recommendation:** Add to Phase 14 a "Contribution Synthesis" section that frames each
major finding (R5 confrontation, imbalance bar paradox, volatility dominance, signal
regime-conditionality) as a thesis contribution. Negative results ARE contributions if
framed correctly.

---

**E2. [NOTE] Pre-registration is the strongest defense against p-hacking accusations.**

0 deviations, frozen Pydantic config, mechanically computed Go/No-Go. An examiner
asking "how do we know you didn't cherry-pick?" gets the immediate answer: "Section 1
defines all rules before data examination; Section 8 computes the decision
programmatically." This is above the standard for bachelor's theses and approaching
registered-report quality.

---

**E3. [MINOR] Negative results framing could be more explicit in the plan.**

The plan correctly states "Negative results are valid" (CLAUDE.md) and "Document all
negative results explicitly" (Phase 13A). But the plan does not list WHICH negative
results are expected, given RC2.

Expected negative results from RC2:
- No single feature exceeds break-even DA (already found)
- GRU classifier will underperform LightGBM (expected, documented as intentional)
- LTCUSDT dollar bars are unusable (already found)
- Time bars have the worst properties for modeling (already found, RC1)

Possible additional negative results:
- Multi-feature ensemble DA still below break-even (HIGH probability per plan update)
- Directional accuracy not significantly above majority class (51.14%) after BH
  correction
- Recommendation system alpha disappears at realistic transaction costs

**Recommendation:** Pre-register the expected negative results alongside the expected
positive results. This converts potential failures into confirmatory findings.

---

### F. Lopez de Prado Methodology

**F1. [NOTE] Alternative bars are properly used, not just mentioned.**

RC1 constructed 9 bar types; RC2 validated 5. Dollar bars are the primary type with
strong justification (no serial correlation, preserves vol clustering, kurtosis 6.7).
The imbalance bar paradox (most structure but insufficient sample) is a genuine
Lopez de Prado-inspired finding.

---

**F2. [MINOR] Triple barrier labeling is deferred but the simplification rationale is
well-documented.**

Phase 9 (line 1758-1762) explicitly justifies binary sign target over triple barrier:
"The 'time expiry' class typically captures 40-60% of samples, halving per-class counts
to ~1200. For a bachelor's thesis, binary sign(fwd_logret_h) is cleaner, easier to
defend, and sufficient." This is sound reasoning given the small sample sizes.

---

**F3. [NOTE] Meta-labeling generalization (binary to regression) is well-motivated.**

Phase 12 operationalizes the generalization: instead of bet/no-bet, the recommender
predicts expected strategy return (continuous). The theoretical foundation cites
Lopez de Prado (2018 Ch. 3) and extends it. The position sizing formula
`size proportional to max(r_hat - threshold, 0) / sigma` is Kelly-adjacent and appropriate.

---

**F4. [NOTE] CPCV, purging, and embargo are correctly described.**

Phase 9A provides the most detailed CPCV specification I have seen in a bachelor's
thesis plan. Cross-asset temporal purging is explicitly addressed. The embargo window
is linked to ACF decay length from Phase 5. Correct.

---

**F5. [MINOR] Sequential bootstrapping is not mentioned.**

Lopez de Prado (2018, Ch. 4) introduces sequential bootstrapping for label uniqueness
in triple barrier labeling. Since triple barrier is deferred, this is not critical. But
for the binary sign target, label uniqueness is relevant when adjacent bars have
overlapping forward windows (fwd_logret_24 uses 24 bars forward, creating significant
label overlap for adjacent observations).

**Recommendation:** For fwd_logret_24, compute the label uniqueness matrix and use it
to weight training samples or subsample. This prevents the model from counting the same
price move 24 times. Alternatively, use non-overlapping forward windows (every 24th bar)
at the cost of reduced sample size. Document the tradeoff.

---

## 3. Gap Analysis: What RC2 Found That the Plan Does Not Address

### Gap 1: Conditional Break-Even DA

RC2 shows the recommender's value is in trading SELECTIVELY (regime-conditional
deployment). The plan computes break-even DA only unconditionally (all bars). A
conditional break-even DA (on traded bars only) would be lower and more favorable.
This is a free, analytically simple improvement to the thesis argument.

### Gap 2: HAR-RV Model for Volatility Forecasting

RC2 demonstrates that volatility is the strongest predictable component. The plan
includes GARCH(1,1) and ARIMA-GARCH (Phase 10B) but not the HAR-RV model
(Corsi 2009), which is the standard academic benchmark for realized volatility
forecasting. HAR-RV is trivially simple (3 regressors: daily, weekly, monthly RV)
and would strengthen the volatility forecasting narrative.

### Gap 3: Granger Causality Operationalization

RC2 Section 5.4 confirms BTC Granger-causes all altcoins at lag 1. The plan update
recommends "BTC-lagged features" for altcoin models. But no specific feature is defined
in Phase 4A or Phase 12A's feature builder. The plan says "BTC return at t-1 as input
to ETH model at t" is justified, but the feature engineering code does not implement it.

### Gap 4: SOLUSDT Tier B Handling

SOLUSDT dollar bars are Tier B (N_eff = 808). RC2 Section 6 recommends "SOL results
should be flagged with wider confidence intervals." But no phase specifies what "flagged"
means operationally. Are SOL models trained separately with stronger regularization?
Are SOL results reported with bootstrapped CIs while BTC/ETH get asymptotic CIs?

### Gap 5: MI Normalization Fix

RC2 Section 3.3 reports an MI/H(target) normalization issue due to negative differential
entropy (H(target) = -2.5371 nats). RC2 Risk 5 recommends "alternative normalizations
(e.g., MI as fraction of feature entropy, or normalized MI bounds)." No phase addresses
this fix.

---

## 4. Recommendations (Prioritized by Impact)

### Priority 1 (Must-do before Phase 9)

1. **Add Phase 9-pre gating test:** 23-feature Ridge/LightGBM DA on BTCUSDT/dollar.
   If below break-even, formally pivot thesis narrative to volatility forecasting +
   risk management. Define this test now, pre-register it.

2. **Reconcile trial counts:** Define exactly what counts as a "trial" for DSR.
   Pre-registered exploration space (60) vs total (including hyperparameters). Update
   Phase 14E to compute DSR at both counts.

3. **Run cost sensitivity analysis:** Break-even DA at {10, 15, 20, 25, 30} bps.
   ~10 lines of code. Include in Phase 5 closure.

4. **Update all "4 assets" references** to reflect RC2's effective universe of 3 assets
   for dollar bars.

### Priority 2 (Should-do before Phase 10)

5. **Add explicit volatility forecasting target** (forward_rv_h) and HAR-RV model to
   Phase 10. Define QLIKE and Mincer-Zarnowitz metrics.

6. **Clarify Phase 10 standalone vs pipeline evaluation split** given the Phase 10-before-9
   reordering.

7. **Define conditional break-even DA** for the recommender's value proposition.

8. **Investigate atr_14 and rsi_14 constant-feature issue.** If confirmed as degenerate
   on dollar bars, remove from the dollar-bar feature set.

### Priority 3 (Should-do before RC3)

9. **Add BTC-lagged return features** to the feature engineering pipeline (Phase 4A
   extension) for altcoin models.

10. **Define SOLUSDT Tier B handling protocol** -- stronger regularization, wider CIs,
    explicit flagging in all results tables.

11. **Address label overlap for fwd_logret_24** via sequential bootstrapping or
    non-overlapping subsampling.

12. **Pre-register expected negative results** alongside positive ones.

### Priority 4 (Nice-to-have)

13. **Fix MI normalization** -- use NMI (normalized mutual information) or MI / H(feature)
    instead of MI / H(target).

14. **Add a "Contribution Synthesis" section** to Phase 14 that frames each finding as
    a thesis contribution.

15. **Document the information-sample-size tradeoff** (imbalance bar paradox) as a
    standalone thesis contribution.

---

## 5. Risk Matrix (Updated with RC1 + RC2 Knowledge)

| # | Risk | Prob | Impact | Evidence | Mitigation | Owner Phase |
|---|------|------|--------|----------|------------|-------------|
| R1 | Multi-feature DA below break-even on dollar bars | HIGH | HIGH | Best single-feature DA = 51.81% vs break-even 57.23%; gap = -5.42 pp | Phase 9-pre gating test; pivot to vol forecasting + risk management; conditional break-even on traded bars | 9 |
| R2 | Overfitting to 2022-2023 MI spike | MEDIUM | HIGH | MI significant in 1/4 windows only; 8/23 features in that window, 0/23 in others | CPCV with embargo; regime-conditional evaluation; rolling stability check | 9, 11 |
| R3 | Recommendation system has insufficient training data | MEDIUM | MEDIUM | ~1350 labels (3 assets x 3 strategies x 150 weeks); 40+ features | Pool across volume bars; simplify feature set; stronger regularization; weekly overlap with purging | 12 |
| R4 | SOLUSDT results unreliable due to Tier B status | MEDIUM | LOW | N_eff = 808; not DL-eligible; MDE DA = 54.37% (very coarse) | Flag all SOL results; wider CIs; use as robustness check not primary evidence | 9-14 |
| R5 | LTCUSDT exclusion weakens universality claim | LOW | MEDIUM | 199 dollar bars; threshold calibration issue | Show LTC works on volume bars; document as known limitation; recommend threshold recalibration as future work | 5, thesis |
| R6 | GRU negative-result experiment is uninformative | LOW | LOW | N = 5,286 is well below the ~10,000 threshold where DL starts to compete | Already framed as intentional negative result; Grinsztajn et al. 2022 provides prior | 9 |
| R7 | Non-stationary features contaminate models | LOW | MEDIUM | 40 unit-root cases, 108 trend-stationary; suggested transformations not all applied | Apply transformations per recommendation; verify stationarity after transformation; document residual non-stationarity | 5, 9 |
| R8 | Holdout period (2024+) is non-representative regime | LOW | HIGH | Post-halving bull market; unlike 2022-2023 where MI was significant | Regime characterization of holdout; MBL computation; honest power statement | 14 |
| R9 | Strategy profitable on GBM synthetic paths (overfitting) | LOW | CRITICAL | PE = 0.9977 suggests near-random; strategy fitting noise patterns | Three-tier MC validation (GBM, GARCH-boot, Politis-Romano); automated alarm | 14 |
| R10 | Trial count understated, inflating DSR | MEDIUM | HIGH | Ambiguity between 60 and 200-500+ counts | Reconcile now; compute DSR at both counts; report the conservative (larger) number | 14 |
| R11 | Imbalance bar signal vanishes with threshold recalibration | MEDIUM | LOW | PE = 0.9740 on 529 bars; more bars would increase PE | Treat as exploratory; do not base primary conclusions on imbalance bars | 5, thesis |
| R12 | Label overlap inflates apparent predictive power at long horizons | LOW | MEDIUM | fwd_logret_24 creates 24-bar label overlap; N_eff < N_raw | Sequential bootstrapping or non-overlapping subsampling; compute effective information ratio | 9, 10 |

---

## 6. Conclusion

The RSPCP implementation plan is among the most rigorous bachelor's thesis plans I have
reviewed. The pre-registration framework, the Lopez de Prado methodology adherence, and
the honest confrontation of R5 are all above the expected standard. The RC2 findings are
correctly diagnosed: the signal is weak, concentrated in volatility features, and
regime-conditional.

The primary concern is the gap between "no single feature exceeds break-even" and "the
ensemble will exceed break-even." This is the thesis's central bet, and it is not yet
operationalized with a pre-registered test. Adding the Phase 9-pre gating test and
defining the volatility forecasting fallback would convert this from a hope into a
research design with a guaranteed valid outcome (either the ensemble works, or the
thesis contributes an information-theoretic limit finding).

The trial count inconsistency (60 vs 200-500+) is the single most critical item to
resolve before proceeding, because the DSR is the final arbiter of the thesis's
quantitative claim.

**Overall verdict: PASS WITH CONCERNS. Address Priority 1 items before starting Phase 9.**
