# RC2 Notebook — Consolidated Review Report

> 6 reviewers: 3 quant-crypto-architect (Q1-Q3) + 3 fintech-python-engineer (F1-F3)
> Date: 2026-03-20

---

## Root Cause of 33-Minute Runtime

**F2 identified it:** Section 3 Part 2 and Part 1 each independently load data from DuckDB, build the feature matrix, and run the FULL validation pipeline (1000 MI permutations + 500 Ridge permutations). Since both use the same data and seed=42, this is a pure waste — the validation runs **twice** on identical data. Fix: single shared loading cell at the top of Section 3.

---

## CRITICAL Issues (Must Fix)

| # | Source | Issue | Fix |
|---|--------|-------|-----|
| C1 | F1,F3 | **Python 2 except syntax** in `rc2_features.py:96` — `except LinAlgError, ValueError, IndexError:` only catches LinAlgError | `except (LinAlgError, ValueError, IndexError):` |
| C2 | F1 | **VIF NaN→0 replacement** corrupts correlation structure | Drop NaN rows instead of replacing with 0 |
| C3 | F1,Q2 | **Duplicate `FeatureRationale` class** — different schemas in `rc2_value_objects.py` and `rc2_feature_rationale.py` | Consolidate into one class |
| C4 | Q1 | **DA p-values NOT BH-corrected** — Gate 2 raw while Gate 1 uses BH | Either BH-correct DA p-values or document hierarchical testing justification |
| C5 | Q1 | **Temporal stability MI uncorrected** across 23 features × 4 windows | Apply BH within each window |
| C6 | Q2 | **Sections 5-8 missing** — Go/No-Go can't be computed | Implement or restructure scope |
| C7 | Q2 | **Non-stationary features enter MI/DA untransformed** | Apply transformations or document why not |
| C8 | Q2,F2 | **Section 3 ordering reversed** — Part 2 before Part 1 | Reorder + merge loading cells |
| C9 | Q3 | **Cross-bar comparison uses reduced permutations** (200 MI) — weakest evidence for central claim | Run full-strength (1000+) |
| C10 | Q3 | **No sample-size correction** for cross-bar MI (dollar N=5286 vs time N=26000) | Subsample time bars to match |
| C11 | Q3 | **No regime-conditional analysis** — central thesis argument unsupported | Add volatility-regime-split DA/MI |
| C12 | F2 | **Redundant double validation run** — the 33-minute bottleneck | Single shared loading cell |
| C13 | Q1 | **PE JS complexity** uses simplified LMC normalization, not Rosso et al. (2007) | Document which variant or implement full Rosso |

## IMPORTANT Issues (Should Fix)

| # | Source | Issue |
|---|--------|-------|
| I1 | Q1 | Target entropy Gaussian upper bound underestimates MI/H% by 10-30% |
| I2 | Q1 | Holdout preview 2023 data overlaps stability gate window |
| I3 | Q2 | Trial counting inconsistent — notebook says 60, code computes ~13 |
| I4 | Q2 | Section 2 claims 53.7% stationarity but "Therefore" says "confirms status" |
| I5 | F1 | Naive `datetime.now()` without UTC |
| I6 | F1 | Mutable dict in frozen StationaritySummary |
| I7 | F1 | `build_multi_horizon_comparison` assumes same features across reports |
| I8 | F2 | ConnectionManager never closed (DuckDB lock risk) |
| I9 | F2 | Cross-cell `# noqa: F821` suppressions masking dependency problems |
| I10 | F2 | Deprecated `applymap()` used in 3 cells (Pandas 2.1+) |
| I11 | F2 | No global matplotlib style — inconsistent chart appearance |
| I12 | F2 | `_row_color` helper duplicated 5 times |
| I13 | F2 | 3 identical DataFrame copies (`df_pd_primary`, `df_pd_full`, `df_features_pd`) |
| I14 | F3 | No end-to-end integration test for RC2 pipeline |
| I15 | F3 | No cross-module consistency test for 23-feature name set |

## ENHANCEMENT — Novel Contribution Opportunities

| # | Source | What | Why It Matters |
|---|--------|------|----------------|
| E1 | Q3 | **"Economic Feasibility Dashboard" chart** | The visual the examiner remembers |
| E2 | Q3 | **Regime-conditional DA/MI** (volatility split) | Tests the central thesis claim |
| E3 | Q3 | **GBM synthetic benchmark** (500 paths) | Defense against snooping |
| E4 | Q3 | **Bootstrap CI for PE** (dollar vs time) | Makes R5 confrontation rigorous |
| E5 | Q3 | **Formal Wilcoxon test** for "dollar > time" MI | Transforms observation into finding |
| E6 | Q3 | **Temporal MI decay curves** | Novel finding on alpha persistence |
| E7 | Q3 | **Break-even DA per regime** | Quantifies conditional feasibility |
| E8 | Q1 | Condition number alongside VIF | Better multicollinearity diagnostic |
| E9 | Q1 | Pearson-Spearman gap for nonlinearity | Motivates nonlinear models |
| E10 | Q1 | N_eff from Newey-West bandwidth | More robust to long memory |
| E11 | Q3 | Cross-asset Granger causality | Lead-lag structure for features |

## 3 STRONGEST Parts (per Q2)

1. **Pre-registration specification** — specific, falsifiable, mechanically evaluable rules with negative result protocol
2. **Feature rationale table architecture** — dual-level a priori documentation counters "data-mined features" attack
3. **Economic significance framing** — DA excess over break-even throughout, Ziliak-McCloskey distinction

## 3 WEAKEST Parts (per Q2)

1. **Missing Sections 5-8** — pre-registration looks performative without the Go/No-Go
2. **Untransformed non-stationary features** — Section 2 screening serves no functional purpose
3. **R5 confrontation is assertion-based** — conditional predictability claimed but not demonstrated

## Priority Action Plan

1. **Fix C1** (except syntax bug) — 2 minutes, prevents crashes
2. **Fix C8+C12** (reorder Section 3, merge loading cells) — eliminates 33-min bottleneck
3. **Fix C2** (VIF NaN handling) — 10 minutes, prevents misleading results
4. **Fix C3** (consolidate FeatureRationale) — 30 minutes
5. **Add E2** (regime-conditional DA/MI) — the "big result" that validates the thesis
6. **Add E5** (Wilcoxon test for dollar>time MI) — transforms the key finding into rigorous evidence
7. **Implement Sections 5-8** — completes the pre-registration promise
