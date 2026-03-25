---
name: RC2 Agent 2 Statistical Thresholds
description: Quantitative threshold analysis for RC2 pre-registration — break-even DA, MDE, feasibility gaps
type: reference
---

# RC2 Statistical Thresholds: What the Numbers Actually Say

## 1. Break-Even Directional Accuracy

Formula: `p = 0.5 + c / (2 * E[|r_t|])`  where c = 0.002 (Binance spot round-trip)

| Scenario | E[|r_t|] | Break-even DA | Required Edge |
|----------|----------|---------------|---------------|
| BTC dollar bars | ~0.008 | 0.625 | 12.5pp |
| ETH dollar bars | ~0.010 | 0.600 | 10.0pp |
| SOL dollar bars | ~0.012 | 0.583 | 8.3pp |
| LTC dollar bars | ~0.009 | 0.611 | 11.1pp |
| Imbalance bars (higher |r|) | ~0.015 | 0.567 | 6.7pp |
| Time 1h bars (lower |r|) | ~0.005 | 0.700 | 20.0pp |

**Key insight**: Time bars have the worst break-even DA (0.70) because their smaller per-bar returns are dominated by fixed transaction costs. Dollar bars are more favorable (0.60-0.63). Imbalance bars, if they produce larger absolute returns, have the most favorable break-even.

## 2. Minimum Detectable Effect (MDE DA)

Formula: `MDE_DA = 0.5 + (z_alpha + z_beta) / (2 * sqrt(N_eff))`  at alpha=0.05, power=0.80

| N_eff | MDE DA | Detectable Edge |
|-------|--------|-----------------|
| 350 (imbalance) | 0.566 | 6.6pp |
| 500 (vol_imb/dol_imb) | 0.556 | 5.6pp |
| 2000 | 0.528 | 2.8pp |
| 3000 (dollar bars) | 0.523 | 2.3pp |
| 5000 | 0.518 | 1.8pp |

## 3. The Feasibility Gap — The Critical Insight

The gap between MDE_DA and break-even DA tells us whether we are powered to detect profitability.

**Dollar bars (the best case):**
- N_eff ~ 3000, MDE_DA ~ 0.523 (can detect 2.3pp edge)
- Break-even DA ~ 0.625 (need 12.5pp edge for BTC)
- Gap = +10.2pp **WELL-POWERED**
- Required N_eff for break-even = ~25 samples (!). We have 100x more than needed.
- This means: *If a profitable signal exists in dollar bars, we will almost certainly detect it. The question is not power — it's whether the signal exists at all.*

**Imbalance bars (the marginal case):**
- N_eff ~ 350, MDE_DA ~ 0.566 (can detect 6.6pp edge)
- Break-even DA ~ 0.567 (need 6.7pp for mean|r|=0.015)
- Gap ~ -0.1pp **MARGINAL**
- Required N_eff ~ 36, we have ~350. We are powered, but just barely.
- With lower mean|r| (say 0.010), break-even rises to 0.60 and gap becomes +3.4pp (OK).

**Time 1h bars (the worst case):**
- N_eff likely high (~30,000+), so MDE_DA ~ 0.507
- Break-even DA ~ 0.700 (need 20pp edge!)
- Gap = +19.3pp **WELL-POWERED but needs huge edge**
- We can detect tiny effects, but need an enormous 20pp edge to be profitable. This is nearly impossible in crypto.

## 4. Harvey Multiple-Testing Threshold

With 23 features x 5 bar types x 3 horizons = 345 tests:
- Bonferroni alpha = 0.05 / 345 = 0.000145 -> t_threshold ~ 3.63
- Harvey empirical: t > 3.0
- Holm-Bonferroni (rank-1): same as Bonferroni = 3.63

For single-asset analysis (23 x 3 = 69 tests):
- Bonferroni t ~ 3.25

**Recommendation**: Use Harvey t > 3.0 as a pragmatic middle ground. It is less conservative than Bonferroni but accounts for publication-level data snooping. For this thesis (not published literature), BH-FDR at q=0.05 is more appropriate and already implemented in Phase 4 validation.

## 5. Deflated Sharpe Ratio

The DSR penalizes for the number of strategy configurations tried:
- SR = 1.5 with 50 trials over 3000 bars: expected_max_SR ~ 0.65, making observed 1.5 still significant
- SR = 1.0 with 100 trials over 1000 bars: expected_max_SR ~ 0.80, marginal significance
- SR = 2.0 with 200 trials over 5000 bars: still significant (large T compensates)

Rule of thumb: keep trial count < 50 where possible, or use SR > 2.0 as the hurdle.

## 6. VIF and Stability Thresholds

- **VIF**: Use VIF > 5 (conservative) for all scenarios. With 23 features and N ~ 500-5000, the N/p ratio ranges from 22 to 217. Even for dollar bars (N/p ~ 230), the conservative threshold is safer given potential non-stationarity.
- **Feature stability**: Require significance in >= 2 out of 4 temporal windows (50%). This balances false positive risk with the reality that feature relevance shifts across regimes.

## 7. MI Effect Size Threshold

- Binary direction target (balanced): H(Y) = ln(2) ~ 0.693 nats
- 1% of entropy = 0.007 nats (minimum meaningful MI)
- 5% of entropy = 0.035 nats (strongly informative)
- Empirically, most crypto features have MI/H(Y) < 3%, so the 1% threshold is appropriate.

## 8. Overall Feasibility Assessment

The numbers are *paradoxically optimistic* for statistical power: we have far more effective samples than needed to detect the break-even edge. The challenge is NOT detection power — it is whether any genuine predictive signal exists at all (R5: crypto unpredictability ~ Brownian noise).

This motivates the RC2 pre-registration: we commit to thresholds BEFORE looking at results, because we have enough power to detect effects that might be artifacts of data snooping.

## Implementation

Module: `src/app/research/application/rc2_thresholds.py`
Tests: `src/tests/research/test_rc2_thresholds.py`

All functions are pure, stateless, and return frozen Pydantic BaseModel results. The composite function `compute_rc2_thresholds()` runs the full analysis for one asset-bar combination.
