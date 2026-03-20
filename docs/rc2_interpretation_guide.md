# RC2 Results Interpretation Guide

> Run the notebook top-to-bottom (`Kernel → Restart & Run All`). This guide tells you what to expect from each section and how to interpret the results.

---

## Section 1: Pre-Registration

**What to see:** A formatted table of all 19 rules + the Go/No-Go matrix template.
**Action:** Verify the timestamp is correct. This is your audit trail.

---

## Section 2: Stationarity Report

**Expected results:**
- ~80-90% of features should be stationary (returns, z-scores, RSI, Bollinger %B, Hurst)
- **Non-stationary candidates:** `atr_14` (absolute price units), `slope_14` ($/bar), possibly `amihud_24` on time bars
- Classification distribution: mostly "stationary", a few "unit_root" or "trend_stationary"

**What to watch for:**
- If >50% of features are non-stationary → something is wrong with the feature engineering
- If `logret_*` features are non-stationary → BUG (log returns should always be stationary)
- The cross-bar heatmap should show consistent stationarity across bar types for most features

**"Therefore" should say:** "19-20/23 features are stationary. The 2-3 non-stationary features (atr_14, slope_14) have documented transformation paths."

---

## Section 3: Feature Exploration

### Part 1 — Rationale, VIF, Distributions

**Feature Rationale Table:** All 23 rows should render with economic intuition and literature refs. This is documentation, not a result.

**Correlation Heatmap:**
- Expect HIGH correlation (>0.7) between: `rv_5/rv_10/rv_20` (all measure volatility), `gk_vol_14/park_vol_14` (alternative vol), `roc_3/roc_6/roc_12` (same indicator, different windows)
- Expect LOW correlation between: returns and volume features, volatility and momentum

**VIF:**
- Volatility features will have VIF > 10 (they're redundant with each other) — this is EXPECTED
- Returns features may have moderate VIF (5-10)
- VIF > 10 does NOT mean we drop the feature (Ridge handles collinearity)

**Distributions:** Violin plots should show most features centered near 0 with heavy tails. Kept features (green) should look similar to dropped features (red) — no visual selection bias.

### Part 2 — MI, DA, Stability

**MI Table:**
- Most MI values will be very small (0.001-0.05 nats)
- MI/H(target)% will be 0.1-5% — this is NORMAL for financial data
- Expect 5-15 features to be significant after BH correction
- If 0 features are significant → negative result, trigger Rule N1

**Ridge DA Table:**
- DA values will be 50.1-53% for individual features
- DA excess: 0.1-3.0 percentage points above 50%
- DA vs break-even: almost all features will be BELOW break-even (negative margin)
- This is the KEY INSIGHT: no single feature is economically sufficient

**Stability Heatmap:**
- Green cells = feature significant in that window
- Features that are green across all windows = robust
- Features that flip green/red = regime-specific
- If <50% of windows are green for a feature → flag as unstable

**Cross-Bar Comparison:**
- If dollar bars consistently show higher MI than time bars → thesis-worthy finding
- If no consistent pattern → bar type doesn't matter for features

**Multi-Horizon:**
- Signal typically concentrates at horizon 1 (strongest autocorrelation)
- Horizon 24 (daily) may show LESS signal due to noise accumulation
- Dead horizons (0 features) get dropped per Rule H1

---

## Section 4: Confronting R5

**PE Table:**
- H_norm values will be 0.95-0.99 (close to 1.0 = Brownian noise)
- R5 reports BTC=0.985, ETH=0.987 at d=5 on hourly data
- If our dollar bar PE is 0.96-0.97 → "2-3% lower than R5's time bars"
- If our PE matches R5 (~0.985) → R5 confirmed, but conditional predictability still possible

**Complexity-Entropy Plane:**
- All points will cluster near (H=1.0, C=0) — the Brownian noise corner
- If dollar bars are slightly left (lower H) and higher (higher C) than time bars → information-driven bars extract structure

**VR Profile:**
- VR ≈ 1.0 at most horizons → random walk
- VR < 1.0 at 1-day horizon → mean reversion
- VR > 1.0 at 7-14 day horizons → momentum
- If ALL VR ≈ 1.0 → no horizon-dependent structure

**Feasibility Gap:**
- MDE DA (what we can detect): ~52.3% for dollar bars
- Break-even DA (what we need): ~56-63%
- The gap is FAVORABLE: we have plenty of power, the question is whether signal exists

---

## Section 5: Statistical Profiling

**Return Distributions:**
- Expect Student-t with ν ≈ 3-8 (fat tails, not Gaussian)
- JB test should reject normality for ALL assets (p ≈ 0)
- AIC(Student-t) < AIC(Normal) for all → Student-t is better fit

**ACF/PACF:**
- Raw returns: near-zero ACF (efficient market)
- Squared returns: significant ACF (volatility clustering)
- This justifies GARCH and volatility features

**Ljung-Box:**
- Raw returns: may not reject at short lags (market efficiency)
- Squared returns: should reject at all lags (ARCH effects)

**GARCH Parameters:**
- α + β (persistence) should be 0.90-0.99 (high persistence, typical for crypto)
- If persistence > 0.99 → IGARCH, volatility shocks are permanent
- Student-t or Skewed-t should beat Normal as innovation distribution

**BDS on GARCH residuals:**
- If BDS rejects → nonlinear structure remains after GARCH → DL justified
- If BDS does not reject → GARCH captures everything → linear models sufficient

---

## Section 6: Data Adequacy

**Sample Size Table:**
- Dollar bars: N ≈ 5,000-5,500, Tier A
- Volume bars: N ≈ 3,000-3,500, Tier A
- Imbalance bars: N ≈ 400-600, Tier B/C
- Time 1h: N ≈ 26,000+, Tier A

**MDE vs Break-Even:**
- All Tier A bar types should be "feasible" (MDE << break-even)
- Imbalance bars may be "marginal" or "underpowered"
- This determines which bar types proceed to modeling

**Cross-Asset Consistency:**
- Kendall τ > 0 + significant → same features work across assets → shared model OK
- Kendall τ ≈ 0 → asset-specific feature selection needed

---

## Section 7: Baselines

**Buy-and-Hold:**
- Sharpe depends on the period (2020-2022 was mixed: bull run + crash)
- Expect Sharpe 0.5-1.5 for BTC, lower for alts

**Random Walk DA:** Should be exactly 50.0% (predict 0, measure sign mismatch)

**Coin-Flip MC:** DA ≈ 50.0% ± 1.3% (95% CI) for N ≈ 5000

**Economic Feasibility Dashboard:**
- The chart showing all features sorted by DA with break-even line
- Expect ALL or nearly all features to be BELOW the break-even line
- This is the visual argument for WHY the recommendation system is needed
- If any feature exceeds break-even → that's a positive result

---

## Section 8: Go/No-Go

**Expected outcomes for each criterion:**

| Criterion | Expected Result | Expected Decision |
|-----------|----------------|-------------------|
| G1: Features ≥ 5 | 8-15 features pass | GO |
| G2: DA excess exists | Yes (statistically, not economically) | GO (marginal) |
| G3: H_norm < 0.98 | Possibly for dollar bars | GO or MARGINAL |
| G4: N_eff ≥ 1000 | Yes for dollar/volume/time | GO |
| G5: Cross-asset τ > 0 | Likely yes | SHARED model |
| G6: BDS rejects | Depends on data | NONLINEAR or LINEAR |
| G7: MDE < break-even | Yes for Tier A | GO |

**Overall decision:** Most likely GO, but the economic margin is thin. The thesis argument is that ML COMBINATION can bridge the gap that individual features cannot.

---

## Red Flags (Stop and Investigate)

1. **0 features pass validation** → check if feature engineering ran correctly
2. **All PE values > 0.995** → data may be too short or bar computation is wrong
3. **GARCH fails to converge for all assets** → data quality issue
4. **N_eff < 100 for dollar bars** → extreme autocorrelation, check data
5. **Stationarity < 50%** → feature engineering bug
6. **Break-even DA > 75%** → transaction cost assumption is wrong (check `round_trip_cost`)
