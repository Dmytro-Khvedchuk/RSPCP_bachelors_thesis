# Implementation Plan: Probabilistic Crypto Strategy Recommender

> **Core design:** Research-interleaved workflow with two forecasting tracks:
> classification (predict direction → SIDE) and regression (predict magnitude → SIZE).
> Both feed into an ML-trained recommendation system with train/test splits and hypothesis
> testing — not a hand-crafted scoring formula. Regression metrics are evaluated conditionally
> on correct direction classification.

---

## Research Foundation & Novel Combinations

### Key References

| # | Paper / Source | Key Idea | How We Use It |
|---|---------------|----------|---------------|
| R1 | López de Prado, *Advances in Financial Machine Learning* (2018) | Information-driven bars (tick, volume, dollar, imbalance, run), triple barrier labeling, meta-labeling, CUSUM filter | Bar construction (Phase 2). But we diverge: **regression targets instead of triple-barrier classification**. |
| R2 | [Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning](https://link.springer.com/article/10.1186/s40854-025-00866-w) (Financial Innovation, 2025) | Combines dollar/volume bars + triple barrier + deep learning for BTC/ETH. First rigorous application of López de Prado's full pipeline to crypto. | Validates our bar choice. We extend beyond classification to **return magnitude regression**. |
| R3 | [Adaptive TFT for Cryptocurrency Price Prediction](https://arxiv.org/abs/2509.10542) (2025) | Dynamic subseries segmentation + pattern-conditioned TFT models. Outperforms fixed-length TFT and LSTM. | Inspiration for our transformer regressor (Phase 9C). We adapt the pattern categorization for **regime-conditional forecasting**. |
| R4 | [CryptoPulse: Dual-Prediction with Cross-Correlated Indicators](https://arxiv.org/html/2502.19349v3) (2025) | Dual-prediction (macro environment + individual dynamics) fused with sentiment. Cross-crypto attention. MAE improvement 10-64%. | We adopt the **cross-asset correlation features** idea for the recommendation system (Phase 11A) — but without sentiment, using strategy performance instead. |
| R5 | [Quantifying Cryptocurrency Unpredictability](https://arxiv.org/html/2502.09079v1) (2025) | Permutation entropy + complexity-entropy plane show crypto returns are near Brownian noise. Naive models outperform complex ML. | Critical sanity check. We use **permutation entropy as a feature** for the recommender: high entropy → don't deploy strategy. Also motivates honest evaluation in RC3. |
| R6 | [LSTM-Conformal Forecasting for Bitcoin](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0319008) (2025) | LSTM + conformal prediction with Adaptive Coverage Interval (ACI) for BTC forecasting. | Direct precedent for our calibration approach (Phase 9D). We extend to **multi-asset conformal intervals**. |
| R7 | [Adaptive Conformal Inference for 4000+ Crypto-Assets](https://www.mdpi.com/1911-8074/17/6/248) (2024) | ACI for VaR estimation across massive crypto universe. ACI handles non-stationarity. | Validates conformal prediction at scale for crypto. We use ACI for **return forecast intervals**, not just VaR. |
| R8 | [Conformal Prediction for Time-series with Change Points](https://arxiv.org/abs/2509.02844) (2025) | CPTC algorithm integrates change-point detection with online conformal prediction for non-stationary series. | Regime changes in crypto → we integrate **change-point awareness** into conformal intervals. |
| R9 | [Learning-to-Rank for Momentum Portfolio Selection](https://link.springer.com/article/10.1007/s10489-024-05377-2) (2024) | Knowledge graph embeddings + LTR for stock ranking. Cross-domain knowledge transfer. | We adapt the **learning-to-rank framing** for strategy deployment recommendation (Phase 11). Instead of "which stocks to buy", it's "which assets to deploy strategy on". |
| R10 | [Optimizing Portfolio via Stock Ranking and Matching with RL](https://www.sciencedirect.com/science/article/abs/pii/S0957417425000521) (2025) | RL agent learns to rank stocks, then matches positions. Combines LSTM + XGBoost + Deep RankNet. | Alternative recommender architecture: **RankNet loss** for training the recommender to rank assets by predicted strategy performance. |
| R11 | [Meta-Labeling](https://en.wikipedia.org/wiki/Meta-Labeling) (López de Prado, 2017) | Secondary ML model that learns when and how much to bet on a primary model's signal. Binary classification for bet/no-bet. | We **generalize meta-labeling from classification to regression**: instead of bet/no-bet, predict the expected strategy return. The recommender IS a regression meta-label model. |

### Novel Combinations (Not Yet Explored Together)

These are method combinations that individually exist in the literature but haven't been
combined in a single system — this is where the thesis contributes:

1. **Information-driven bars + return regression (not classification)**
   Most papers (R1, R2) combine bars with triple barrier labeling → classification (+1/-1/0).
   We use dollar/imbalance bars but predict **actual log return magnitude**. This is richer
   than direction alone and feeds directly into risk-adjusted recommendation.

2. **Meta-labeling generalized to regression for strategy recommendation**
   López de Prado's meta-labeling (R11) is binary: bet or don't bet. We extend it to:
   "Given market state + features + return forecast, predict the **expected strategy return**
   (continuous value)." This turns the recommendation into a supervised regression problem
   with proper train/test/hypothesis testing.

3. **Permutation entropy as a predictability feature for the recommender**
   Paper R5 shows crypto is near-Brownian noise. Instead of just being discouraged by this,
   we use **rolling permutation entropy as a feature** for the recommender. High entropy → asset
   is currently unpredictable → recommender learns to avoid deploying strategy on it.
   This is a principled, data-driven "skip" signal.

4. **Cross-asset attention + strategy performance correlation**
   Paper R4 uses cross-crypto correlations for price prediction. We adapt this for
   **strategy recommendation**: if Strategy X is performing well on BTC and ETH is correlated
   with BTC, boost ETH's recommendation score. The recommender's feature set includes
   cross-asset beta, correlation rank, and relative strength.

5. **Conformal prediction on strategy return forecasts (not just price)**
   Papers R6/R7/R8 apply conformal prediction to price forecasting. We apply it one level
   higher: conformal intervals on the **predicted strategy return** from the recommendation
   model. This gives guaranteed-coverage intervals on "how much will the strategy make if
   deployed here" — directly actionable for risk management.

6. **Learning-to-rank loss for strategy deployment recommendation**
   Papers R9/R10 use LTR for portfolio construction. We use it for a different problem:
   **ranking assets by predicted strategy performance**, not by predicted price return.
   The recommender can be trained with pairwise ranking loss (RankNet/LambdaRank) in addition
   to pointwise regression loss.

---

### Statistical Testing Framework

Every claim in the thesis is backed by a formal statistical test. The framework below
is organized by *what question each test answers*, with the intuition for how it works.

#### A. "Is there any exploitable structure?" — Market Structure Tests

**A1. Lo-MacKinlay Variance Ratio Test**
([Lo & MacKinlay 1988](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=396681))

Under a random walk, the variance of q-period returns should scale linearly: Var(r_q) = q × Var(r_1).
The variance ratio VR(q) = Var(r_q) / (q × Var(r_1)) should equal 1.

```
VR(q) > 1 → positive autocorrelation → momentum (trending markets)
VR(q) < 1 → negative autocorrelation → mean reversion
VR(q) ≈ 1 → random walk → no exploitable structure at this horizon
```

The test uses two variants: one assuming homoscedasticity (Z₁) and one robust to
heteroscedasticity (Z₂, more appropriate for crypto). Test at multiple horizons
q = {2, 5, 10, 20} to build a profile of predictability across timescales.

*Used in:* RC1 (per asset — is this asset predictable?), Phase 4 (rolling VR as a feature
for the recommender — high VR deviation = currently predictable).

*Python:* `arch.unitroot.VarianceRatio`

**A2. BDS Test for Nonlinear Dependence**
([Brock, Dechert & Scheinkman 1987](https://www.jstor.org/stable/2938174))

After removing linear structure (e.g., ARMA fit), tests if residuals are i.i.d. Detects
nonlinear dependence that autocorrelation-based tests miss: GARCH effects, threshold
effects, chaos.

```
H₀: residuals are i.i.d.
H₁: residuals contain nonlinear dependence
Rejection → there is structure beyond what linear models capture → DL models justified
```

*How it works:* Computes the correlation integral C(ε, m) which measures how often
pairs of m-dimensional embedded points fall within distance ε. Under i.i.d., C(ε, m) =
C(ε, 1)^m. The test statistic measures the deviation from this relationship.

*Used in:* Phase 5 (profiling — applied to GARCH residuals to justify deep learning),
Phase 11 (RC3 — applied to model residuals to check if remaining structure exists).

*Python:* `statsmodels.stats.diagnostic.acorr_bds`

---

#### B. "Is this model better than that one?" — Model Comparison Tests

**B1. Diebold-Mariano Test** (already in plan, expanded here)
([Diebold & Mariano 1995](https://www.nber.org/papers/t0169))

Tests if the difference in forecast loss between two models is significantly different
from zero. Works with any loss function (MSE, MAE, CRPS, economic loss).

```
d_t = L(e_{1,t}) - L(e_{2,t})    # loss difference at time t
DM = mean(d_t) / SE(d_t)         # studentized mean difference
H₀: E[d_t] = 0 (equal accuracy)
```

The standard error uses Newey-West HAC estimator to handle autocorrelated loss
differences. This is crucial for financial data where errors are serially correlated.

*Key subtlety:* For nested models (e.g., model with 5 features vs model with same 5 + 2 more),
standard DM is oversized. Use Clark-West adjustment in that case.

*Used in:* Phase 11 (RC3 — all pairwise comparisons), Phase 13 (RC4 — recommender vs baselines).

*Python:* custom implementation or `arch.bootstrap.SPA` (includes DM internally)

**B2. Model Confidence Set (MCS)**
([Hansen, Lunde & Nason 2011](https://economics.brown.edu/sites/default/files/papers/2003-5_paper.pdf))

Instead of testing pairs, MCS finds the *smallest set of models* that contains the true best model
with (1-α) confidence. It works by iteratively eliminating the worst model until the remaining
set passes an equivalence test.

```
Step 1: Start with all K models
Step 2: Test H₀: "all models in set are equally good"
Step 3: If rejected → remove the worst model → go to Step 2
Step 4: When H₀ not rejected → remaining models = Model Confidence Set
```

Output: for each model, a p-value. Models with p > α are in the MCS. This naturally
handles multiple comparisons — no Bonferroni needed.

*Why better than pairwise DM:* If you have 10 models and do 45 pairwise DM tests, you
inflate Type I error. MCS controls the familywise error rate and gives you one clean answer.

*Used in:* Phase 11 (RC3 — which classifiers/regressors to keep), Phase 13 (RC4 — which
recommender configurations to keep).

*Python:* `arch.bootstrap.MCS`

**B3. Hansen's Superior Predictive Ability (SPA) Test**
([Hansen 2005](https://www.tandfonline.com/doi/abs/10.1198/073500105000000063))

Tests if the best model in a set significantly outperforms a **specific benchmark**,
after controlling for the number of models tried.

```
H₀: No model is better than the benchmark
H₁: At least one model significantly outperforms the benchmark
```

Improvement over White's Reality Check: uses studentized test statistics that reduce
the influence of poor/irrelevant models in the set. More powerful when some alternatives
are clearly bad (which is common — not all models work well).

*Used in:* Phase 13 (RC4). Benchmark = AllAssetsRecommender (unfiltered). Alternative set =
{ML recommender, classifier-only, regressor-only, volume filter, momentum filter}.
Answer: "Does ANY of our approaches beat unfiltered deployment?"

*Python:* `arch.bootstrap.SPA`

**B4. Giacomini-White Conditional Predictive Ability Test**
([Giacomini & White 2006](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2006.00718.x))

Standard DM tests **unconditional** superiority: "is A better than B on average across all time?"
Giacomini-White tests **conditional** superiority: "is A better than B *given current market state*?"

```
DM: E[d_t] = 0                   # unconditional — average over all t
GW: E[d_t | h_t] = 0             # conditional — given information h_t at time t

h_t could be: volatility regime, Hurst exponent, market cap, recent VR(q), etc.
```

*Why this is the most interesting test for YOUR thesis:* Your whole claim is that the
recommender's value is **conditional** — it knows WHEN to deploy the strategy, not just
whether the strategy is good on average. GW directly tests this claim.

Example result you could show: "On average (DM test), recommender is only marginally
better than unfiltered (p=0.12). But conditional on high-volatility regimes (GW test),
the recommender significantly outperforms (p=0.003)." This would be a strong thesis finding.

*Used in:* Phase 11 (RC3 — are models conditionally better in certain regimes?),
Phase 13 (RC4 — is the recommender conditionally better given market state?).
Condition on: volatility regime (GARCH σ), Hurst exponent, VR(q), permutation entropy.

---

#### C. "Am I fooling myself?" — Overfitting & Multiple Testing Controls

**C1. Deflated Sharpe Ratio (DSR)**
([Bailey & López de Prado 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551))

The standard Sharpe ratio is inflated by: (a) how many strategies you tried,
(b) non-normal returns (fat tails, skewness), (c) short sample periods.
DSR corrects for all three simultaneously.

```
DSR computes: P(SR* > 0 | SR_observed, N_trials, skew, kurtosis, T)

Inputs:
  SR_observed    = your best strategy's Sharpe
  N_trials       = total number of strategies/configs tried (be honest!)
  skew, kurtosis = of the return series (crypto has high kurtosis ≈ 5-15)
  T              = number of return observations

If DSR p-value > 0.05 → your best Sharpe is NOT significant after correction
```

*Why essential:* You test ~5 classifiers × ~6 regressors × ~3 horizons × ~20 assets =
potentially 1800 configurations. Without DSR, finding one with Sharpe > 2 by pure luck
is almost guaranteed. DSR tells you: "given that you tried 1800 things, is Sharpe 2.1
still impressive?" (Often: no.)

*Used in:* Phase 14. Applied to the best recommender configuration.

**C2. Probability of Backtest Overfitting (PBO)**
([Bailey & López de Prado 2015](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253))

Directly measures whether your model selection process is overfit.

```
Method:
1. Partition data into S equal-length subsets (e.g., S=16 months)
2. Form all C(S, S/2) combinations of "in-sample" (S/2 subsets) and "out-of-sample" (remaining S/2)
3. For each combination:
   a. Select the best model/config in-sample (by Sharpe, accuracy, etc.)
   b. Check its RANK out-of-sample
4. PBO = fraction of combinations where the IS-best model ranks below median OOS

PBO > 0.50 → model selection is no better than random → OVERFIT
PBO < 0.25 → model selection is robust → results likely genuine
PBO = 0.00 → IS-best always performs well OOS → very strong (rare in finance)
```

*Intuition:* If you pick the best strategy on training data and it consistently
performs below average on test data, you were just picking lucky noise patterns.

*Used in:* Phase 14. Applied to: classifier selection, regressor selection,
recommender hyperparameter tuning, bar type selection.

**C3. Combinatorial Purged Cross-Validation (CPCV)**
([López de Prado 2018, Ch. 12](https://en.wikipedia.org/wiki/Purged_cross-validation))

Standard k-fold CV is wrong for financial time series because:
1. Adjacent folds share temporal information (autocorrelation leakage)
2. Labels overlap with features across fold boundaries (label leakage)

CPCV fixes both with two mechanisms:

```
PURGING: If a test sample has label y_t = sign(r_{t→t+h}), then training samples
         in [t, t+h] are REMOVED because they overlap with the label window.

EMBARGO: After each test fold, an additional buffer of E bars is removed from
         training. This handles residual autocorrelation (e.g., volatility clustering).

COMBINATORIAL: Instead of sequential folds (1→2→3→4), CPCV uses all C(N, k)
               combinations, giving a full DISTRIBUTION of OOS performance —
               not a single number.
```

*Visual example with 6 blocks, k=2 (train on 4, test on 2):*
```
Combo 1: [TRAIN][TRAIN][purge|TEST|purge][TRAIN][purge|TEST|purge][TRAIN]
Combo 2: [TRAIN][purge|TEST|purge][TRAIN][TRAIN][purge|TEST|purge][TRAIN]
... etc, C(6,2) = 15 combinations total
```

*Used in:* Phases 9, 10, 12 — all model training and evaluation.

---

#### D. "Are my probabilistic forecasts honest?" — Calibration Tests

**D1. PIT (Probability Integral Transform) Test**

If a probabilistic forecast F(y) is perfectly calibrated, then u_t = F_t(y_t) (the CDF value
at the realized observation) should be Uniform(0,1). This is the PIT.

```
Compute: u_t = F_t(y_t) for each prediction-observation pair
Test: are {u_t} ~ Uniform(0,1)?
  - KS test or Anderson-Darling on {u_t}
  - Histogram of {u_t} should be flat

Common failures:
  U-shape histogram → overdispersed (intervals too wide, model underconfident)
  ∩-shape histogram → underdispersed (intervals too narrow, model overconfident)
  Right-skewed → model systematically overestimates returns
  Left-skewed  → model systematically underestimates returns
```

*Why better than reliability diagrams:* Reliability diagrams check specific quantiles
(e.g., "does my 90% interval cover 90% of the time?"). PIT checks the ENTIRE distribution
at once. A model can pass reliability checks at 50%, 90%, 95% but still fail PIT
(e.g., wrong shape between quantiles).

*Used in:* Phase 11 (RC3 — for all probabilistic regressors: quantile regression,
GRU-MDN, conformal intervals).

**D2. Kupiec Unconditional Coverage Test**
([Kupiec 1995](https://www.mathworks.com/help/risk/overview-of-var-backtesting.html))

Tests whether the observed violation rate of a prediction interval matches the target.

```
Target: α = 5% (for a 95% prediction interval)
Observed: x violations out of T predictions
H₀: p = α
Test: likelihood ratio LR = -2 ln[(α^x × (1-α)^(T-x)) / (p̂^x × (1-p̂)^(T-x))]
LR ~ χ²(1)
```

*Example:* Your 95% conformal interval has 127 violations out of 2000 predictions.
Expected: 100 violations (5%). Observed: 127 (6.35%). Is this significantly different?
Kupiec answers: p = 0.03 → yes, the interval undercovers.

*Used in:* Phase 11 (RC3 — test all conformal intervals).

**D3. Christoffersen Independence Test**
([Christoffersen 1998](https://www.sciencedirect.com/science/article/pii/S0304407698000165))

Kupiec tests the *rate* of violations. Christoffersen tests whether violations are
*independent* — i.e., does a violation today predict a violation tomorrow?

```
Build 2×2 transition matrix:
         | no violation_t+1 | violation_t+1
---------+------------------+--------------
no viol_t|      n_00        |    n_01
viol_t   |      n_10        |    n_11

H₀: P(violation_t+1 | violation_t) = P(violation_t+1 | no violation_t)
    i.e., violations are independent

If rejected → violations CLUSTER → model fails to adapt to regime changes
```

*Why essential for crypto:* Market crises cause clustered extreme moves. If your
conformal intervals fail 5 times in a row during a crash (even if overall rate is fine),
the model is useless exactly when you need it most. Christoffersen catches this.

*Combined test (Christoffersen 1998):* Joint test of unconditional coverage (Kupiec) +
independence → tests both rate AND clustering simultaneously. LR_cc = LR_uc + LR_ind.

*Used in:* Phase 11 (RC3 — especially important for GRU-MDN and conformal intervals).

---

#### E. "Is my Sharpe real?" — Sharpe Ratio Inference

**E1. Lo (2002) Autocorrelation-Corrected Sharpe Ratio**

Standard Sharpe SE assumes i.i.d. returns: SE = 1/√T. But crypto returns are
autocorrelated (especially at high frequency), which means the true SE is larger.

```
SE_corrected = SE_iid × √(1 + 2 × Σ_{k=1}^{q} ρ(k) × (1 - k/(q+1)))

where ρ(k) = autocorrelation at lag k, q = truncation lag

Typical impact on crypto (4h bars):
  ρ(1) ≈ 0.05, ρ(2) ≈ 0.03 → correction factor ≈ 1.15
  This means: SR = 1.96 (significant uncorrected) → SR = 1.70 (insignificant corrected)

For 1h bars with higher autocorrelation:
  correction factor ≈ 1.4-1.8 → many "significant" Sharpes become insignificant
```

*Used in:* Phase 7 (backtest metrics — always report corrected SE), Phase 14 (all Sharpe tests).

**E2. Minimum Backtest Length (MBL)**

Given observed Sharpe, non-normality, and desired significance level, how long a
backtest do you need?

```
MBL = [1 + (1 - γ₃×SR + (γ₄-1)/4 × SR²)] × (z_α / SR)²

where γ₃ = skewness, γ₄ = kurtosis (crypto: γ₄ ≈ 5-15)

Example: SR = 1.5, γ₃ = -0.3, γ₄ = 8, α = 0.05
  → MBL ≈ 6.2 years of daily data
  → If your test set is 6 months → INCONCLUSIVE regardless of observed Sharpe
```

*Used in:* Phase 14. Quick check — if MBL > available test period, the results are
honest but inconclusive. Document this.

---

#### F. "Do my features matter?" — Feature Justification Tests

**F1. Granger Causality Test**

Tests if past values of series X help predict series Y beyond Y's own past.

```
Restricted model: y_t = Σ α_i × y_{t-i} + ε_t
Full model:       y_t = Σ α_i × y_{t-i} + Σ β_j × x_{t-j} + ε_t
F-test: are all β_j = 0?

If rejected: X Granger-causes Y → including X as a feature for Y is justified
```

*Used in:* RC2 — test cross-asset relationships. "Does BTC Granger-cause ETH?"
"Does volume Granger-cause returns?" Provides formal justification for each cross-asset
feature in the recommender.

*Python:* `statsmodels.tsa.stattools.grangercausalitytests`

---

#### Summary: Where Each Test Is Used

| Test | Category | Used In | Answers |
|------|----------|---------|---------|
| Variance Ratio (Lo-MacKinlay) | A1 | RC1, Phase 4 (feature) | Is this asset predictable? |
| BDS Test | A2 | Phase 5, RC3 | Is there nonlinear structure in residuals? |
| Diebold-Mariano | B1 | RC3, RC4 | Is model A better than model B? |
| Model Confidence Set | B2 | RC3, RC4 | Which models are the "best" set? |
| Hansen SPA | B3 | RC4, Phase 14 | Does best model beat benchmark after correction? |
| Giacomini-White (conditional) | B4 | RC3, RC4 | Is model A better in specific regimes? |
| Deflated Sharpe Ratio | C1 | Phase 14 | Is best Sharpe significant after N trials? |
| PBO (Prob. of Backtest Overfit) | C2 | Phase 14 | Is model selection process overfit? |
| CPCV (purge + embargo) | C3 | Phase 9, 10, 12 | Is cross-validation honest for time series? |
| PIT Calibration | D1 | RC3 | Is the full predictive distribution calibrated? |
| Kupiec Coverage | D2 | RC3 | Do intervals fail at the right rate? |
| Christoffersen Independence | D3 | RC3 | Do interval failures cluster? |
| Lo (2002) Sharpe Correction | E1 | Phase 7, 14 | Is Sharpe significant given autocorrelation? |
| Minimum Backtest Length | E2 | Phase 14 | Do we have enough data for a conclusion? |
| Granger Causality | F1 | RC2 | Does feature X predict target Y? |

---

## Architecture Overview

```
src/app/
├── system/                    # [EXISTS] Cross-cutting infrastructure
│   ├── logging.py             # [EXISTS] Loguru logging
│   └── database/              # [EXISTS] DuckDB + Alembic
├── ohlcv/                     # [EXISTS] OHLCV domain module
│   ├── domain/                # [EXISTS] Entities, value objects, protocols
│   ├── application/           # [EXISTS] OHLCVService
│   └── infrastructure/        # [EXISTS] DuckDB repository
├── ingestion/                 # [EXISTS] Binance API data pipeline
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── bars/                      # [EXISTS] Lopez de Prado alternative bars
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── research/                  # [EXISTS] RC1 analysis services (coverage, returns, ACF, bar comparison)
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── features/                  # [NEW] Feature engineering + validation
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── profiling/                 # [NEW] Statistical profiling per asset
│   ├── domain/
│   └── application/
├── backtest/                  # [NEW] Trading simulation engine
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── strategy/                  # [NEW] Base trading strategies
│   ├── domain/
│   └── application/
├── forecasting/               # [NEW] Return regression models (NOT classification)
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── recommendation/            # [NEW] ML-trained recommendation system
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── evaluation/                # [NEW] Statistical proof, Monte Carlo, permutation tests
│   ├── domain/
│   └── application/
├── live/                      # [NEW] Live paper trading engine (Block III)
│   ├── domain/
│   ├── application/
│   └── infrastructure/
└── dashboard/                 # [NEW] Dashboard API + frontend (Block III)
    ├── infrastructure/
    └── frontend/

research/                      # [NEW] Jupyter notebooks for research checkpoints
├── RC1_data_and_bars.ipynb
├── RC2_features_and_profiling.ipynb
├── RC3_classification.ipynb        # Direction classifiers evaluation
├── RC3_regression.ipynb            # Return regressors evaluation (direction-conditional)
├── RC4_recommender_evaluation.ipynb
└── utils/                     # Shared plotting/analysis helpers
```

---

## Code Quality Standards

All code in the project adheres to strict quality gates enforced by pre-commit hooks
that run **automatically on every commit**. No code merges without passing all checks.

### Docstrings — Google Style (enforced by Ruff `D` + `DOC` rules)

Every public module, class, method, and function **must** have a Google-style docstring.

```python
def compute_log_returns(
    prices: pl.Series,
    *,
    horizon: int = 1,
) -> pl.Series:
    """Compute log returns over a given horizon.

    Calculates ln(P_t / P_{t-horizon}) for each price in the series.
    Returns are NOT shifted — caller is responsible for leakage prevention.

    Args:
        prices: Close price series, must be strictly positive.
        horizon: Number of periods for return calculation. Defaults to 1.

    Returns:
        Log return series of same length, with first `horizon` values as null.

    Raises:
        ValueError: If horizon < 1 or prices contain non-positive values.
    """
```

- Classes: docstring immediately after `class` line, describing purpose and invariants
- Modules: docstring at top of file, one-liner describing the module's role
- Protocols: docstring on the protocol AND on each abstract method
- Private methods (`_foo`): docstring optional, but add one if logic is non-trivial

### Type Hints — Python 3.14 (enforced by Pyright strict + Ruff `ANN`)

Use modern Python 3.14 type hint syntax throughout:

```python
# Built-in generics (PEP 585) — no typing.List, typing.Dict, etc.
def get_assets() -> list[Asset]: ...
def get_config() -> dict[str, Any]: ...

# Union with | (PEP 604) — no typing.Optional, typing.Union
def find_bar(bar_id: str) -> AggregatedBar | None: ...
def process(value: int | float) -> str: ...

# type keyword for aliases (PEP 695)
type OHLCVFrame = pl.DataFrame
type FeatureMatrix = pd.DataFrame

# Pydantic for all domain objects — no raw dataclasses
from pydantic import BaseModel

class FetchRequest(BaseModel, frozen=True):
    """Request to fetch OHLCV data from Binance."""

    asset: Asset
    timeframe: Timeframe
    date_range: DateRange
```

- **All** function signatures: parameters and return type annotated
- **All** class attributes: annotated (Pydantic models, protocols)
- **All** local variables: explicitly annotated — Pyright infers these automatically so
  pre-commit hooks will NOT catch missing local annotations, but this is a **manual project
  convention** enforced during code review. Every local variable gets a type hint:
  ```python
  # YES — explicit type on every local variable
  t0: float = time.perf_counter()
  sql: TextClause = text("SELECT ...")
  rows: Sequence[Row[Any]] = conn.execute(sql).fetchall()
  candles: list[OHLCVCandle] = [self._row_to_entity(r) for r in rows]

  # NO — implicit inference (not allowed in this project)
  t0 = time.perf_counter()
  sql = text("SELECT ...")
  rows = conn.execute(sql).fetchall()
  ```
- **Pydantic `BaseModel`** for all value objects, configs, DTOs — no `dataclasses`
- **No** `Any` unless interfacing with untyped third-party libs (and then comment why)
- Use `Self` (PEP 673) for methods returning the same class
- Use `Never` for functions that always raise
- Pyright runs in strict mode — zero type errors allowed

### Pre-Commit Pipeline (already configured)

The following hooks run on every `git commit`:

| Order | Hook | Tool | What It Does |
|-------|------|------|-------------|
| 1 | Formatter | `ruff --fix` | Auto-formats code (Black-compatible), 119 char line length, double quotes |
| 2 | Linter | `ruff` | Enforces ~20 rule categories: `F`, `ANN`, `D`, `DOC`, `S`, `B`, `UP`, `PERF`, `PLR`, `N`, `PT`, etc. |
| 3 | Type checker | `pyright --strict` | Full static type analysis, no `Unknown` types allowed |
| 4 | Import sorter | `isort` | Alphabetical, grouped by STDLIB → THIRDPARTY → FIRSTPARTY → LOCAL |

All hooks use `pyproject.toml` as the single source of configuration.
A commit that fails any hook is **rejected** — fix the issues, re-stage, commit again.

### Naming Conventions (enforced by Ruff `N` rules)

- **Modules:** `snake_case.py` (e.g., `binance_fetcher.py`, `duckdb_repository.py`)
- **Classes:** `PascalCase` (e.g., `OHLCVCandle`, `IngestionService`)
- **Functions/methods:** `snake_case` (e.g., `compute_log_returns`, `fetch_ohlcv`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BATCH_SIZE`, `MAX_RETRIES`)
- **Type aliases:** `PascalCase` via `type` keyword (e.g., `type FeatureMatrix = pl.DataFrame`)
- **Protocols:** `I`-prefix (e.g., `IOHLCVRepository`, `IBarAggregator`)

### DataFrame & Numerical Libraries — Split by Purpose

The project uses **different libraries for different concerns**, chosen for their strengths:

| Library | Where | Why |
|---------|-------|-----|
| **Polars** | ETL pipelines, data ingestion, bar construction, feature engineering, backtest engine, live trading | Zero-copy, lazy evaluation, no GIL, native Parquet. Best for high-throughput data pipelines where performance matters. |
| **Pandas** | Research notebooks (RC1–RC4), statistical profiling, model training/evaluation, experiment analysis | Richest ecosystem for `statsmodels`, `arch`, `scipy`, `sklearn`, `pytorch-forecasting`. Most ML/stats libraries expect Pandas. |
| **NumPy** | Vectorized math inside indicators, permutation tests, Monte Carlo simulation, matrix operations | Optimal for tight numerical loops: log returns, rolling windows, bootstrap sampling, variance ratio calculations. |

**Conversion boundaries** are explicit and minimal:

```python
# Polars → Pandas: at the boundary where stats/ML libraries need it
df_pandas: pd.DataFrame = df_polars.to_pandas()

# Pandas → Polars: when returning to the ETL pipeline
df_polars: pl.DataFrame = pl.from_pandas(df_pandas)

# NumPy ↔ both: zero-copy when possible
arr: np.ndarray = df_pandas["close"].to_numpy()
arr: np.ndarray = df_polars["close"].to_numpy()
```

**Rule of thumb:**
- If it's a **pipeline** (ingestion → bars → features → storage → backtest → live) → **Polars**
- If it's a **research notebook** or **model training** → **Pandas** (compatibility with ML ecosystem)
- If it's a **tight numerical computation** (indicators, bootstrap, Monte Carlo) → **NumPy**

### Data Models — Pydantic Everywhere

**All** value objects, configurations, DTOs, and domain entities use **Pydantic `BaseModel`**
instead of `dataclasses`. No raw `dataclass` usage in the project.

Why Pydantic over dataclasses:
- Built-in validation (type coercion, constraints, custom validators)
- Serialization to/from JSON, dict, YAML for free
- `frozen=True` for immutability (same as `dataclass(frozen=True)`)
- `.model_dump()`, `.model_validate()` — explicit and readable
- Settings management via `pydantic-settings` (already used for `BinanceSettings`)
- Pyright and Ruff understand Pydantic natively

```python
from pydantic import BaseModel, Field

class BarConfig(BaseModel, frozen=True):
    """Configuration for alternative bar construction."""

    bar_type: BarType
    threshold: float = Field(gt=0, description="Sampling threshold (e.g., dollar volume)")
    ewm_span: int = Field(default=100, ge=10, description="EWMA span for adaptive threshold")
```

### Database — Alembic for All Schema Changes

**Every** DuckDB schema change goes through Alembic migrations. No raw `CREATE TABLE`
or `ALTER TABLE` outside of migration files.

- New table → `alembic revision --autogenerate -m "add bars table"`
- Column change → new migration with `op.add_column()` / `op.alter_column()`
- Index creation → migration with `op.create_index()`
- All migrations are **idempotent** and **reversible** (both `upgrade()` and `downgrade()`)
- Migration naming: `{rev}_{description}.py` (e.g., `0002_add_aggregated_bars_table.py`)
- Run before app startup: `alembic upgrade head`

This applies to every phase that introduces new tables or modifies existing ones:
Phase 1 (OHLCV already migrated), Phase 2 (bars table), Phase 5 (features table),
Phase 7 (backtest results), Phase 9/10 (forecast outputs), Phase 12 (recommendations),
Phase 16 (live trading state).

---

## Phase 1: Data Ingestion Pipeline ✅ COMPLETED

**Goal:** Fetch historical OHLCV data from Binance and store it in DuckDB.

### 1A: Binance API Client

Port the working Binance client from the legacy project into the clean architecture.

- **`src/app/ingestion/domain/value_objects.py`** — `BinanceKlineInterval` enum (1m, 5m, 15m, 1h, 4h, 1d), `FetchRequest` (asset, timeframe, start_date, end_date)
- **`src/app/ingestion/domain/protocols.py`** — `IMarketDataFetcher` protocol with `fetch_ohlcv(request) -> list[OHLCVCandle]`
- **`src/app/ingestion/infrastructure/binance_fetcher.py`** — Concrete implementation using `python-binance`. Pagination (1000-bar batches), retry with exponential backoff, rate limit handling. Reads `BINANCE_API_KEY` / `BINANCE_API_SECRET` from env.
- **`src/app/ingestion/infrastructure/settings.py`** — `BinanceSettings` (Pydantic) for API keys, rate limits, retry config

### 1B: Ingestion Service

- **`src/app/ingestion/application/services.py`** — `IngestionService` orchestrating: fetch from Binance → validate → store in DuckDB via `IOHLCVRepository`. Supports: single asset, bulk (list of assets), incremental (only fetch missing data from last timestamp).
- **`src/app/ingestion/application/commands.py`** — `IngestAssetCommand`, `IngestUniverseCommand` (list of assets + timeframes + date range)

### 1C: CLI / Entry Point

- Update `main.py` or add a CLI command (via justfile) to run ingestion: `just ingest --assets BTCUSDT,ETHUSDT --timeframes 1h,4h --start 2020-01-01`
- Log progress: "Fetching BTCUSDT 1h: 45000/52000 bars..."
- Idempotent: re-running fills gaps without duplicating data (leverages INSERT OR IGNORE)

### 1D: Tests

- Unit test: mock Binance API responses, verify candle parsing
- Integration test: fetch small date range from real API, verify storage in in-memory DuckDB
- Edge cases: empty responses, API errors, timezone handling

**Dependencies:** `python-binance`, existing `ohlcv` module
**Estimated scope:** ~8 files, ~500 lines

---

## Phase 2: Alternative Bar Construction (Lopez de Prado) ✅ COMPLETED

**Goal:** Implement information-driven bar types that sample data based on market activity rather than fixed time intervals.

**Reference:** *Advances in Financial Machine Learning*, Ch. 2

### 2A: Bar Domain Model

- **`src/app/bars/domain/value_objects.py`** — `BarType` enum (TIME, TICK, VOLUME, DOLLAR, TICK_IMBALANCE, VOLUME_IMBALANCE, DOLLAR_IMBALANCE, TICK_RUN, VOLUME_RUN, DOLLAR_RUN), `BarConfig` (bar_type + parameters)
- **`src/app/bars/domain/entities.py`** — `AggregatedBar` entity (open, high, low, close, volume, bar_type, tick_count, buy_volume, sell_volume, vwap, start_ts, end_ts)
- **`src/app/bars/domain/protocols.py`** — `IBarAggregator` protocol with `aggregate(trades: DataFrame) -> list[AggregatedBar]`

### 2B: Standard Bars

- **`src/app/bars/application/tick_bars.py`** — Aggregate every N ticks into one bar. Parameters: `tick_count` (default 1000)
- **`src/app/bars/application/volume_bars.py`** — Aggregate until cumulative volume reaches threshold. Parameters: `volume_threshold`
- **`src/app/bars/application/dollar_bars.py`** — Aggregate until cumulative dollar volume (price * volume) reaches threshold. Parameters: `dollar_threshold`

### 2C: Information-Driven Bars

- **`src/app/bars/application/imbalance_bars.py`** — Tick/Volume/Dollar imbalance bars. Form a bar when the cumulative signed imbalance exceeds an adaptive EMA-based threshold. Parameters: `initial_threshold`, `ema_span`, `warmup_period`
- **`src/app/bars/application/run_bars.py`** — Tick/Volume/Dollar run bars. Form a bar when the longest run of same-direction ticks/volume exceeds expected length. Parameters: same as imbalance bars.

### 2D: Bar Storage

- Alembic migration: `aggregated_bars` table (asset, bar_type, bar_config_hash, start_ts, end_ts, open, high, low, close, volume, tick_count, buy_volume, sell_volume, vwap)
- **`src/app/bars/infrastructure/duckdb_repository.py`** — `DuckDBBarRepository` for storing and querying aggregated bars

### 2E: Tests

- Unit test: feed known tick sequences, verify bar boundaries match expected
- Property test: for any bar type, all input ticks must appear in exactly one bar
- Statistical test: verify dollar bars produce more uniform bar counts than time bars across volatile/calm periods

**Dependencies:** Raw trade data from Binance (1m OHLCV as proxy, or aggTrades endpoint)
**Estimated scope:** ~12 files, ~800 lines

---

## Phase 3: Research Checkpoint 1 — Data Quality & Bar Analysis ✅ COMPLETED

**Status:** COMPLETED (2026-03-12). Full results in `research/RC1_analysis.md`.

**Goal:** Stop and analyze what we have before building further. Generate charts, statistics,
assess data adequacy. This is a collaborative research session (notebook-driven).

### 3A: Notebook `research/RC1_data_and_bars.ipynb`

**Data coverage analysis:**
- Total bars per asset × timeframe → coverage heatmap
- Date range gaps detection → timeline visualization
- Volume profile across time → are there dead periods?
- Asset universe filtering: drop assets with <2 years of data or excessive gaps

**Bar type comparison (per asset):**
- Bar count distribution over time: time bars (fixed count/day) vs dollar bars (adaptive)
- Return distribution per bar type: QQ-plots, kurtosis, skewness comparison
- Normality tests (Jarque-Bera) per bar type → which produces closer-to-normal returns?
- Autocorrelation of returns per bar type → which reduces serial correlation?
- Bar duration distribution → how variable is bar timing?

**Charts produced:**
- Heatmap: asset × timeframe coverage (Bokeh, interactive)
- Gap timeline: per-asset gap bars on a timeline (Bokeh, interactive)
- Volume profile: hourly volume for BTCUSDT over time (Bokeh, interactive)
- Weekly bar count histogram: all bar types (Bokeh, interactive)
- Bar duration boxplot: per bar type (Matplotlib)
- Return statistics table: descriptive stats + JB test (Bokeh DataTable + pandas)
- QQ-plot grid: returns under each bar type vs normal (Matplotlib)
- Return distribution overlay: histograms + KDE per bar type (Matplotlib)
- ACF comparison grid: raw returns per bar type (Matplotlib)
- ACF comparison grid: squared returns per bar type (Matplotlib)
- Ljung-Box summary table: serial correlation + volatility clustering (pandas)
- RC1 summary comparison table: all metrics consolidated (pandas)

### 3B: RC1 Findings

#### Q1: Data Quality — GO ✅

All 4 assets pass quality filters:

| Asset   | Coverage (1h) | Data Span | Largest Gap |
|---------|--------------|-----------|-------------|
| BTCUSDT | 99.94%       | 2,263 days (~6.2 yr) | 6 hours |
| ETHUSDT | 99.94%       | 2,263 days (~6.2 yr) | 6 hours |
| LTCUSDT | 99.94%       | 2,263 days (~6.2 yr) | 6 hours |
| SOLUSDT | 99.96%       | 2,040 days (~5.6 yr) | 5 hours |

All exceed the 95% coverage threshold. 18 gaps total, all ≤ 6 hours (Binance maintenance
windows in 2020–2021). No gap exceeds the 48-hour danger threshold. 1h OHLCV is sufficient
as source data for bar construction — no need for 1m/aggTrades.

**Selected asset universe:** BTCUSDT, ETHUSDT, LTCUSDT, SOLUSDT (all 4 pass).

#### Q2: Alternative Bars Improve Distributions — YES ✅

Dollar bars reduce excess kurtosis from 53.3 (time_1h) → 6.7 (8× improvement).
Volume_imbalance achieves the lowest kurtosis (2.9) and near-zero skew (0.17).
Run bars make distributions worse (kurtosis 30–43).

| Bar Type          | N      | Skew   | Excess Kurt | JB Normal? |
|-------------------|--------|--------|-------------|------------|
| time_1h           | 54,277 | -0.94  | **53.3**    | No         |
| **dollar**        | 5,286  | -0.36  | **6.7**     | No         |
| **volume**        | 3,263  | -0.28  | **4.1**     | No         |
| **volume_imbalance** | 529 | 0.17   | **2.9**     | No         |
| **dollar_imbalance** | 568 | -0.73  | 8.3         | No         |
| dollar_run        | 434    | 2.66   | 30.5        | No         |
| volume_run        | 388    | 3.82   | 42.6        | No         |
| tick              | 55     | —      | —           | (too few)  |

#### Q3: Information-Driven Bars Reduce Serial Correlation — YES ✅

Time bars have strong serial correlation (Ljung-Box p=0.0000). Dollar and imbalance bars
eliminate it (p > 0.15) while preserving volatility clustering — the ideal combination
per López de Prado.

| Bar Type          | Serial Corr? | Vol Clustering? | Ideal? |
|-------------------|:---:|:---:|:---:|
| **dollar**        | No (p=0.20) | Yes (p=0.00) | **Yes** |
| **dollar_imbalance** | No (p=0.71) | Yes (p=0.00) | **Yes** |
| **volume_imbalance** | No (p=0.15) | Yes (p=0.00) | **Yes** |
| volume            | YES (p=0.049) | Yes (p=0.00) | Almost |
| dollar_run        | YES (p=0.01) | No (p=0.41) | No |
| volume_run        | No (p=0.94) | No (p=1.00) | No |
| time_1h           | YES (p=0.00) | Yes (p=0.00) | No |

#### Q4: Bar Types for Phase 4 — DECIDED ✅

**Proceed to Phase 4:**

| Bar Type              | N     | Role | Justification |
|-----------------------|-------|------|---------------|
| **dollar**            | 5,286 | Primary | Best overall — large sample, no serial corr, preserves vol clustering, kurtosis 6.7 |
| **volume**            | 3,264 | Secondary | Good sample, best kurtosis among high-N bars (4.1), borderline serial corr |
| **volume_imbalance**  | 530   | Information-driven | Best distributional properties (kurt 2.9, skew 0.17), clean serial structure |
| **dollar_imbalance**  | 568   | Information-driven | Clean serial properties, moderate sample |
| **time_1h**           | 54,277| Baseline | Required for comparison — serial corr present, extreme kurtosis |

**Disqualified:**
- **tick, tick_imbalance, tick_run** — threshold too high, ≤ 55 bars over 6 years. Would
  need recalibration (see Phase 2 action item below).
- **dollar_run, volume_run** — extreme kurtosis (30–43), run bars lose volatility clustering.

#### Phase 2 Action Item (from RC1)

> **Tick bar thresholds need recalibration.** Current thresholds produce ≤ 55 bars from
> ~54,000 hourly candles. If tick bars are desired for Phase 4, the tick_count threshold
> must be reduced by ~100–1000×. This is a Phase 2 configuration change, not a code change.
> **Decision: defer tick bar recalibration — dollar + volume families provide sufficient
> coverage for the thesis.**

**Estimated scope:** 1 notebook (~20 cells), 1 analysis report (`research/RC1_analysis.md`)

---

## Phase 4: Feature Engineering

**Goal:** Compute features from OHLCV / bar data. Target is now **future log return** (regression),
not direction (classification).

> **RC1 inputs (Phase 3):**
> - **Assets:** BTCUSDT, ETHUSDT, LTCUSDT, SOLUSDT (all 4 passed quality filters)
> - **Bar types:** dollar (primary, N=5,286), volume (N=3,264), volume_imbalance (N=530),
>   dollar_imbalance (N=568), time_1h (N=54,277, baseline only)
> - **Key consideration:** imbalance bars have ~500 samples — feature warmup windows must be
>   short enough to preserve sufficient training data. Use ≤ 50-bar lookbacks for imbalance
>   bar features (vs ≤ 200 for dollar/volume/time bars).
> - **Key consideration:** dollar bars produce ~2–3 bars/day on average — multi-horizon
>   targets should use bar-counts (1-bar, 4-bar, 24-bar) not wall-clock hours.

### 4A: Core Indicators (port + extend from legacy)

- **`src/app/features/application/indicators.py`** — Pure functions operating on Polars DataFrames (NumPy for vectorized math internally):
  - **Returns:** log returns (multi-horizon: 1, 4, 12, 24 bars)
  - **Volatility:** realized vol (multi-window), Garman-Klass vol, Parkinson vol, ATR (Wilder)
  - **Momentum:** EMA crossover (fast/slow), RSI (continuous), ROC (multi-window), rolling linear regression slope
  - **Volume:** volume z-score, OBV slope, Amihud illiquidity ratio
  - **Statistical:** rolling Hurst exponent, return z-score, Bollinger %B and width
  - **Cross-timeframe:** 1h/4h/1d feature stacking (for time_1h baseline only — not applicable to non-time bars)

### 4B: Regression Target Construction

- **`src/app/features/application/targets.py`** — Target variable builder:
  - `forward_log_return(horizon)` = ln(close_{t+h} / close_t) — the regression target
  - Multiple horizons: 1h, 4h, 24h (or 1-bar, 4-bar, 24-bar for non-time bars)
  - `forward_volatility(horizon)` = realized vol over next h bars (secondary target)
  - All targets clearly labeled as forward-looking, only used for training labels
  - `.shift(-horizon)` to align, then drop NaN rows at the end

### 4C: Feature Matrix Builder

- **`src/app/features/application/feature_matrix.py`** — `FeatureMatrixBuilder` that takes OHLCV DataFrame + config → outputs feature DataFrame with all computed columns. Config specifies which features and their parameters. All features computed from past data only (no future leakage — `.shift(1)` applied).
- **`src/app/features/domain/value_objects.py`** — `FeatureConfig` (list of feature specs with parameters), `FeatureSet` (computed features + metadata)

### 4D: Feature Validation (Permutation Testing)

- **`src/app/features/application/validation.py`** — Permutation test framework:
  1. **MI test:** mutual information of feature vs forward_log_return. Shuffle target 1000x → null distribution. BH-corrected p-values. Keep features with corrected p < 0.05.
  2. **Single-feature predictive power:** Train Ridge regression on single feature → measure **directional accuracy** (DA) and **direction-conditional MAE** (DC-MAE, i.e., MAE only on samples where predicted sign == actual sign). Shuffle target 500x → null. Features must beat shuffled baselines on DA.
  3. **Temporal stability:** repeat MI on yearly windows (2020, 2021, 2022, 2023). Flag features significant in <50% of windows.
  4. **Interaction test:** combined feature groups vs sum of individual contributions.
- **`src/app/features/domain/entities.py`** — `FeatureValidationResult` (feature_name, mi_pvalue, fdr_corrected_p, directional_accuracy, dc_mae, stability_score, keep: bool)

### 4E: Tests

- Unit test: each indicator against known reference values
- Property test: all features finite (no NaN/inf after warmup), correct shape
- Leakage test: verify no feature uses future data (correlation with shifted target should be ~0)
- Regression target test: forward return computed correctly on known price series

**Dependencies:** Phase 1 (OHLCV data), Phase 2 (bar data), Phase 3 (asset + bar type selection)
**Estimated scope:** ~12 files, ~1100 lines

---

## Phase 5: Statistical Profiling

**Goal:** Generate statistical profile per asset to understand data properties and justify
modeling choices. Every profiling result must connect to a downstream modeling decision
("Therefore..."). Phase 5 also hardens the Phase 4D validation pipeline and establishes
the project-level temporal partition that all subsequent phases must respect.

> **RC1 overlap:** Phase 3 already computed return distributions (JB test, kurtosis, skewness),
> ACF/PACF analysis, and Ljung-Box tests for BTCUSDT across all bar types. Phase 5 extends
> this to all 4 assets, adds GARCH modeling, alternative distribution fitting, and the
> Lo-MacKinlay variance ratio test. Phase 5 services should reuse the RC1 analysis classes
> from `src/app/research/application/` where applicable (e.g., `ReturnAnalyzer`,
> `AutocorrelationAnalyzer`) and extend them for the profiling-specific tests below.

> **Review-driven additions (2026-03-19):** Consolidated review by 5 quant-crypto-architect
> agents identified 16 critical/important issues across statistical methodology, thesis
> presentation, code gaps, crypto-specific concerns, and data-flow leakage. Phases 5pre
> through 5H below incorporate all fixes. See `4d_report.md` for the Phase 4D review and
> the consolidated review discussion for the full finding list.

### 5pre: Foundation & Validation Hardening

> Before any profiling begins, establish the temporal partition that prevents data leakage
> across phases, install missing dependencies, harden the Phase 4D validation pipeline,
> and add stationarity screening. Everything downstream depends on this.

#### 5pre.1 — Project-Level Temporal Partition

Define a `DataPartition` config (in `profiling/domain/value_objects.py` or a shared location)
that declares authoritative data boundaries respected by ALL subsequent phases:

| Partition | Period | Used By |
|-----------|--------|---------|
| **Feature selection** | 2020-01-01 → 2022-12-31 | Phase 4D validation (MI, Ridge DA, stability) |
| **Model development** | 2020-01-01 → 2023-12-31 | Phase 9-10 walk-forward (CPCV within this range) |
| **Final holdout** | 2024-01-01 → end of data | Phase 14 evaluation only — never touched before |

- Re-run Phase 4D validation restricted to the **feature selection partition** only.
  The current `kept_feature_names` was computed on full data (2020-2026), which contaminates
  the holdout. After re-running, the kept set may differ — document any changes in RC2.
- Update `ValidationConfig.temporal_windows` to cover only the feature selection period:
  `(2020, 2021), (2021, 2022), (2022, 2023)` (3 windows, not 4).
- Thread `DataPartition` through `ProfilingService` and downstream phases.

#### 5pre.2 — Install `arch` Package

Phase 5B (VR test) and 5C (GARCH) require the `arch` library. Run `just add arch` and
verify Python 3.14 compatibility (Cython extensions). If `arch` fails on 3.14, fall back
to `scipy.optimize`-based GARCH MLE (significantly more implementation work — test early).

#### 5pre.3 — Phase 4D Validation Hardening

**Fix C2 — Move `ridge_train_ratio` to config:**
- Add `ridge_train_ratio: float = 0.7` to `ValidationConfig`.
- Thread it through `_run_ridge_test` → `evaluate_single_feature_ridge` and
  `compute_ridge_null_distribution`. Remove the magic number from the function signature.

**Fix I4 — Remove DC-MAE null distribution waste:**
- Remove `null_dc_mae` from `compute_ridge_null_distribution` return.
- Simplify `_run_ridge_test` to only track DA nulls.
- Keep DC-MAE as a single observed diagnostic (no permutations).
- Drop `dc_mae_null_mean` from `FeatureValidationResult` or set to `float('nan')`.
- Net effect: Ridge permutation loop becomes ~2× faster.

**Fix: Block permutation for MI and Ridge null distributions:**
- Replace `rng.permutation(target)` with **block permutation** in both
  `compute_mi_null_distribution` and `compute_ridge_null_distribution`.
- Block size = max feature lookback (configurable via `ValidationConfig.permutation_block_size: int = 50`).
- Block permutation preserves local autocorrelation and volatility clustering while
  breaking the feature-target association. Without this, the null hypothesis is
  "no dependency vs i.i.d. noise" instead of "no dependency vs temporally structured noise,"
  making it too easy to beat on volatility-clustered crypto data.
- Alternative (simpler): phase-scramble surrogates (FFT → randomize phases → IFFT).
  Choose one approach and document the rationale.

**Tests:** Update existing Phase 4D tests to cover:
- Temporal split correctness (Ridge trains on first 70%, evaluates on last 30%)
- DC-MAE still computed as a diagnostic (no nulls)
- Block permutation preserves marginal distribution but breaks temporal alignment
- Re-run validation on feature selection partition produces valid results

#### 5pre.4 — Stationarity Screen

**`src/app/profiling/application/stationarity.py`** — Before MI/Ridge/profiling, verify
that all features and return series are stationary:

- Run ADF (null = unit root) and KPSS (null = stationarity) on each of the 23 features
  and on the return series, per (asset, bar_type).
- **Confirmation approach:** ADF rejects unit root AND KPSS fails to reject stationarity
  → stationary. Both reject → trend-stationary. Neither rejects → unit root.
- Features that fail stationarity:
  - `atr_14` → replace with `atr_14 / close` (percentage ATR) in `indicators.py`
  - `amihud_24` → apply rolling z-score (already done for volume; extend to Amihud)
  - `hurst_100`, `bbwidth_20_2.0` → apply fractional differentiation with minimum d
    that achieves stationarity (López de Prado Ch. 5), or first-difference
- Store results in `StationarityReport` Pydantic model: per-feature ADF statistic,
  ADF p-value, KPSS statistic, KPSS p-value, is_stationary flag, transformation applied.
- Output table goes directly into RC2.

#### 5pre.5 — Sample-Size Tier Classification

Define bar-type tiers based on usable sample size (after warmup/NaN removal) that gate
which tests are valid. Add to `ProfilingConfig`:

| Tier | N threshold | Test battery | Expected bar types |
|------|-------------|--------------|-------------------|
| **A** | N > 2000 | Full battery | dollar, volume, time_1h |
| **B** | 500 ≤ N ≤ 2000 | Reduced: Student-t only (no GH), VR capped at q=10, GARCH(1,1) only (no GJR), 2 stability half-windows instead of 3 yearly | — |
| **C** | N < 500 | Descriptive statistics only — no asymptotic inference | volume_imbalance, dollar_imbalance (after warmup) |

- `ProfilingService` classifies each (asset, bar_type) into a tier before dispatching tests.
- Tier is stored in `StatisticalReport` and displayed prominently in RC2.
- Imbalance bar results are explicitly labeled "exploratory" in all outputs.

### 5A: Return Distribution Analysis

- **`src/app/profiling/application/distribution.py`** — Per-asset, **per-bar-type**, **tier-gated**:
  - Log return computation (reuse `ReturnAnalyzer.compute_log_returns()`)
  - Normality test: **Jarque-Bera only** (Shapiro-Wilk removed — invalid for N > 5000,
    and at these sample sizes all normality tests trivially reject on financial data;
    Anderson-Darling adds no information beyond JB)
  - Report **effect sizes** alongside JB: excess kurtosis, skewness, QQ-deviation integral
  - **Alternative distribution fitting (Tier A/B only):**
    - Student-t via MLE → report fitted degrees of freedom ν (the primary tail-heaviness metric)
    - **No Generalized Hyperbolic** — 5-parameter GH is intractable on N < 1000, has poor
      Python support, and downstream models don't use the distributional parameters.
      If examiner asks: use NIG (Normal Inverse Gaussian, 4-param GH subfamily) as fallback.
    - Optional: 2-component Gaussian Mixture (captures bimodality from regime mixing)
  - **Model comparison:** AIC, BIC between Normal and Student-t.
    KS test statistic D_n reported as a **distance measure** (not for inference — KS p-values
    are anti-conservative for fitted distributions since critical values assume known parameters).
  - **QQ-plots against fitted Student-t** (not Normal — Normal QQ is already known to fail
    and provides no new information). Update `ReturnAnalyzer.compute_qq_data()` or add
    a new method that accepts the fitted distribution.
  - Output: `DistributionProfile` Pydantic model (frozen=True) with:
    `jb_stat`, `jb_pvalue`, `excess_kurtosis`, `skewness`, `student_t_nu`,
    `student_t_nu_ci_lower`, `student_t_nu_ci_upper`, `aic_normal`, `aic_student_t`,
    `bic_normal`, `bic_student_t`, `ks_d_statistic`, `tier`

### 5B: Autocorrelation & Serial Dependence

- **`src/app/profiling/application/autocorrelation.py`** — Per-asset, **per-bar-type**, **tier-gated**:
  - ACF/PACF of returns and squared returns (reuse `AutocorrelationAnalyzer`)
  - **Ljung-Box at multiple lags** `[5, 10, 20, 40]` (not just the max lag — single-max-lag
    Ljung-Box masks short-horizon serial dependence diluted by many zero-autocorrelation lags).
    Existing `AutocorrelationAnalyzer.compute_acf_analysis()` runs Ljung-Box at a single
    effective lag — extend or override.
  - **Lo-MacKinlay variance ratio test** — **heteroscedasticity-robust Z2 only** (Z1
    homoscedastic variant is inappropriate for crypto with GARCH effects):
    - **Horizons defined in calendar time**, converted to bar-count per bar type:
      | Calendar horizon | time_1h (24/day) | dollar (~2.3/day) | volume (~1.5/day) |
      |-----------------|-------------------|--------------------|--------------------|
      | 1 day | q=24 | q=2 | q=2 |
      | 3 days | q=72 | q=7 | q=5 |
      | 1 week | q=168 | q=16 | q=11 |
      | 2 weeks | q=336 | q=33 | q=22 |
    - For Tier B/C: cap at the "1 week" horizon (skip 2 weeks).
    - Consider **Chow-Denning (1993) multiple VR test** which tests all horizons jointly,
      avoiding the multiple-testing problem of 4 separate tests.
    - Use `arch.unitroot.VarianceRatio` with `robust=True`.
  - **Granger causality test** (Tier A only):
    - `statsmodels.tsa.stattools.grangercausalitytests`
    - Test: does BTC return Granger-cause ETH/LTC/SOL return? Does volume Granger-cause returns?
    - Lags = [1, 2, 4, 8]. Report F-statistic and p-value.
    - Referenced in statistical testing framework (F1) as "RC2" deliverable — build the
      service here so RC2 can consume it cleanly.
  - Output: `AutocorrelationProfile` with:
    `ljung_box_stats` (dict of lag → (Q, p)), `vr_stats` (dict of horizon → (VR, Z2, p)),
    `chow_denning_stat`, `chow_denning_p`, `granger_results` (list per pair),
    `acf_values`, `pacf_values`, `tier`

### 5C: Volatility Modeling

- **`src/app/profiling/application/volatility.py`** — **time_1h bars only** (GARCH assumes
  equally-spaced observations; dollar/volume/imbalance bars have non-uniform spacing with
  inter-bar duration CV = 0.93–2.0+, making GARCH methodologically unsound on them):
  - **Innovation distribution comparison:** Fit GARCH(1,1) with Normal, Student-t, and
    Skewed Student-t innovations (`arch` library: `dist='normal'`, `dist='t'`, `dist='skewt'`).
    Select best by AIC/BIC. **Always report the Student-t result** as the primary.
  - **Engle-Ng sign bias test** before GJR-GARCH: formally verify that leverage effects
    exist (negative shocks increase volatility more than positive). If sign bias is not
    significant, skip GJR and report GARCH(1,1) as sufficient.
  - **GJR-GARCH** (Tier A only, if sign bias significant):
    Fit with Student-t innovations. Report gamma (asymmetry), alpha, beta, omega.
  - **GARCH persistence check:** Report alpha + beta. Flag any (asset) where
    persistence ≥ 0.99 as potentially integrated (IGARCH). If alpha + beta ≥ 1,
    the unconditional variance does not exist → downstream VR test interpretation is affected.
  - **ARCH-LM test on GARCH residuals:** Verify that GARCH captured all volatility
    clustering (Ljung-Box on z_t² should not reject after GARCH).
  - **BDS test on standardized GARCH residuals** (embedding dimensions m = {2, 3, 4, 5}):
    If BDS rejects i.i.d. at multiple dimensions → nonlinear structure remains after
    GARCH → justifies using nonlinear ML models in Phase 9. If BDS does NOT reject →
    document honestly that linear volatility models may suffice.
  - **Rolling realized volatility** (all bar types, not just time_1h):
    - Reuse `rv_12`, `rv_24`, `rv_48` from Phase 4A feature set where applicable.
    - Window defined in **calendar time** (not bar count) for cross-bar-type comparability.
    - **Quantile-based regime labeling:** low-vol (below 25th percentile), normal (25th-75th),
      high-vol (above 75th). Store regime labels per bar for downstream use.
  - For **information-driven bars** (dollar, volume, imbalance): use existing volatility
    features (GK, Parkinson, ATR, RV) from Phase 4A as the volatility profile instead
    of GARCH. No GARCH fitting on non-uniform bars.
  - Output: `VolatilityProfile` (frozen=True) with:
    `garch_alpha`, `garch_beta`, `garch_omega`, `garch_persistence`, `garch_dist`,
    `garch_aic_normal`, `garch_aic_t`, `garch_aic_skewt`, `gjr_gamma` (if applicable),
    `sign_bias_p`, `arch_lm_p`, `bds_results` (dict of m → (stat, p)),
    `nonlinear_structure_detected`, `regime_labels`, `tier`

### 5D: Predictability Assessment

> New sub-phase. Directly confronts reference paper R5 (crypto ≈ Brownian noise) and
> provides the quantitative foundation for the RC2 "Is our data predictable?" section.
> Without this, a thesis examiner can challenge the entire modeling effort.

- **`src/app/profiling/application/predictability.py`** — Per-asset, **per-bar-type**:
  - **Permutation entropy** (Bandt & Pompe, 2002):
    - Embedding dimension d = {3, 4, 5, 6}, delay τ = 1.
    - Normalized permutation entropy H_norm ∈ [0, 1]. H_norm → 1 = Brownian noise.
    - Compare against R5's reported values for BTC/ETH. Position each (asset, bar_type)
      on the **complexity-entropy plane** (H vs. Jensen-Shannon complexity C).
    - If information-driven bars (dollar, volume) show lower entropy than time bars,
      that is a thesis-worthy finding: "information-driven sampling extracts structure
      that uniform sampling misses."
  - **Effective sample size** (Kish formula):
    `N_eff = N / (1 + 2 · Σ ρ_k)` for k = 1 to Bartlett bandwidth.
    Applied to return-level autocorrelation (for mean inference).
    Report `N_eff / N` ratio per (asset, bar_type) — this is the "autocorrelation tax."
  - **Minimum Detectable Effect (MDE) for directional accuracy:**
    Given N_eff, α = 0.05, power = 0.80, compute the minimum DA above 50% that is
    reliably detectable using a one-sided binomial test.
    - For dollar bars (N_eff ≈ 3000-5000): MDE ≈ 1.3-1.8% → DA ≥ 51.3%
    - For imbalance bars (N_eff ≈ 200-400): MDE ≈ 4-5% → DA ≥ 54% (too coarse to detect weak edges)
    Use `statsmodels.stats.power` or direct binomial calculation.
  - **Minimum viable DA from transaction costs:**
    Define break-even DA: `(2p - 1) · mean(|r_t|) > round_trip_cost` where
    round_trip_cost = 2 × 10 bps = 0.002 (Binance spot). Solve for p.
    This is the concrete, economically grounded threshold for the RC2 go/no-go.
  - **Signal-to-noise ratio:**
    Adjusted R² from multivariate Ridge regression of kept features on target,
    computed on a **temporal holdout** (last 30% of feature selection partition).
    Compare against R² from regressing on random Gaussian features (noise baseline).
  - Output: `PredictabilityProfile` (frozen=True) with:
    `permutation_entropies` (dict of d → H_norm), `js_complexity`, `n_eff`, `n_eff_ratio`,
    `mde_da`, `breakeven_da`, `snr_r2`, `snr_r2_noise_baseline`, `tier`

### 5E: Profiling Service & Config

- **`src/app/profiling/domain/value_objects.py`** — Pydantic config classes (no magic numbers):
  - `DistributionConfig` — JB alpha, list of alternative distributions, KS reporting
  - `AutocorrelationConfig` — Ljung-Box lags `[5, 10, 20, 40]`, VR calendar horizons,
    Granger max lags, significance alpha
  - `VolatilityConfig` — GARCH order (p,q), innovation distributions to compare,
    BDS embedding dimensions, RV window (calendar time), regime quantile thresholds
  - `PredictabilityConfig` — permutation entropy embedding dimensions, Kish bandwidth,
    power analysis parameters (alpha, power target), round-trip cost for break-even DA
  - `ProfilingConfig` — composite of the above + `DataPartition` reference + tier thresholds

- **`src/app/profiling/application/services.py`** — `ProfilingService`:
  - Takes `DataLoader` as dependency (same pattern as RC1 `CoverageAnalyzer`)
  - Iterates over (asset, bar_type) combinations via `DataLoader.get_available_bar_configs()`
  - Classifies each into Tier A/B/C based on usable N
  - Dispatches to distribution, autocorrelation, volatility, predictability analyzers
    with tier-appropriate test batteries
  - Aggregates into `StatisticalReport` (per-asset-bar-type profiles + cross-asset summary)
  - Conversion boundary: `pl.DataFrame → pd.DataFrame` at service entry point
    (same pattern as `validation.py`)

- **Multiple comparison correction across Phase 5:**
  - For **characterization tests** (normality, GARCH diagnostics): report effect sizes, not p-values.
    These tests are descriptive — binary reject/not-reject is uninformative at N > 1000.
  - For **inferential tests** (Ljung-Box, VR, Granger, BDS): apply **Benjamini-Hochberg FDR
    correction** across the full grid of (asset, bar_type, test, lag/horizon) combinations.
    Report both raw and corrected p-values. Store in `StatisticalReport.corrected_pvalues`.

### 5F: Tests

- **`src/tests/profiling/conftest.py`** — Synthetic data factories:
  - `make_normal_returns(n, seed)` — for normality test calibration
  - `make_student_t_returns(n, nu, seed)` — for distribution fitting validation
  - `make_ar1_returns(n, phi, seed)` — for autocorrelation test validation
  - `make_garch_returns(n, alpha, beta, seed)` — for GARCH fitting (known parameters)
  - `make_random_walk(n, seed)` — for VR test (should not reject VR=1)
  - `make_profiling_config()` — factory with fast defaults (reduced permutations, fewer lags)

- **Unit tests (per sub-module):**
  - `test_distribution.py`:
    - Normal data should not reject JB at α = 0.05
    - Student-t(5) data: fitted ν should be in [4, 7] confidence interval
    - AIC(Student-t) < AIC(Normal) on fat-tailed data
  - `test_autocorrelation.py`:
    - White noise: Ljung-Box should not reject at any lag
    - AR(1) with φ = 0.3: Ljung-Box should reject at lag 5
    - Random walk: VR(q) should be ≈ 1.0 for all q
    - Known Granger relationship: F-test should reject
  - `test_volatility.py`:
    - GARCH(1,1) on synthetic GARCH data: recovered α, β within 20% of true values
    - Constant series: α ≈ 0, β ≈ 0
    - BDS on i.i.d. normal: should NOT reject
    - BDS on GARCH residuals with remaining nonlinearity: should reject
  - `test_predictability.py`:
    - Random walk: permutation entropy H_norm → 1.0 (within 0.05)
    - Deterministic series (e.g., sin): H_norm → 0
    - ESS of i.i.d. data: N_eff ≈ N
    - ESS of AR(1) with φ = 0.5: N_eff ≈ N/3
  - `test_stationarity.py`:
    - Random walk (non-stationary): ADF should fail to reject
    - White noise (stationary): ADF should reject, KPSS should not reject
  - `test_services.py`:
    - `ProfilingService` tier classification: synthetic data with known N produces correct tier
    - Full pipeline on small synthetic dataset produces valid `StatisticalReport`

- **Sanity tests (integration-level):**
  - Real BTCUSDT dollar bar data: GARCH persistence < 1.0
  - Real data: permutation entropy within [0.85, 1.0] (consistent with R5)
  - Real data: stationarity report flags known non-stationary features

**Dependencies:** Phase 1 (data), Phase 4 (features), `scipy.stats`, `statsmodels`, `arch`
**Estimated scope:** ~15 files, ~1500 lines

---

## Phase 6: Research Checkpoint 2 — Features, Profiling & Data Adequacy

**Goal:** Deep analysis of features and statistical properties. Answer: "Is our data
sufficient and do our features carry signal for return regression?" Every section must
connect to a modeling decision ("Therefore..."). Negative results are documented honestly.

> **Thesis narrative structure:** RC2 is not a sequence of tables — it tells a story with
> three claims: (1) our features carry genuine information, (2) returns exhibit structure
> beyond random walks, (3) our sample is adequate for ML modeling. Each section builds
> evidence for or against these claims, culminating in a formal go/no-go decision.

> **Pre-registration (Nosek et al., 2018):** Before opening the RC2 notebook, write the
> decision rules in a markdown cell at the top. All feature selection, asset universe, and
> horizon choices must follow mechanical rules defined before seeing the data. Post-hoc
> decisions (human judgment after seeing results) are documented as "trials" for the Phase
> 14 Deflated Sharpe Ratio. This reduces researcher degrees of freedom.

### Theoretical Foundation

**VIF Analysis (Belsley, Kuh & Welsch, 1980):** Variance Inflation Factor measures how
much the variance of a regression coefficient is inflated due to collinearity.
`VIF_j = 1 / (1 - R²_j)` where `R²_j` is from regressing feature j on all other features.
VIF > 10 indicates severe collinearity. While Ridge regression (used in Phase 4D) handles
collinearity by shrinking coefficients, VIF quantifies the *degree* of multicollinearity
in the feature matrix, which affects: (a) instability of feature importance rankings,
(b) interpretability of MI scores (collinear features share MI), and (c) the effective
dimensionality of the feature space. For a thesis, VIF preempts the standard committee
question: "Are your features redundant?"

**Economic Significance vs Statistical Significance (Ziliak & McCloskey, 2008; Harvey
et al., 2016):** A p-value below 0.05 says "unlikely under the null" but says nothing
about whether the effect is large enough to matter economically. Harvey et al. propose
that financial studies should use `t > 3.0` (not 1.96) due to multiple testing across
the profession. For this thesis, the bridge is the break-even DA from transaction costs:
if the minimum profitable DA is 52.3% and the best feature achieves DA = 52.5%, the
statistical significance is irrelevant — the economic margin is 0.2 percentage points,
which disappears under any model uncertainty or regime shift. RC2 must frame every DA
result against this economic threshold.

**Notebook Scope (Tufte, 2001 — "data-ink ratio"):** Every chart and table should maximize
the ratio of information to ink. 500 cells means the examiner reads none carefully. 200-250
cells with interpretive "Therefore..." paragraphs means every result connects to a decision.
The pre-registration framework (Nosek et al., 2018) further requires that decision criteria
be stated before seeing results, converting exploratory analysis into confirmatory analysis
and reducing researcher degrees of freedom.

### 6A: Notebook `research/RC2_features_and_profiling.ipynb`

#### Section 1: Pre-Registration & Decision Rules

Define mechanical decision criteria before any analysis:
- "Keep all features that pass the three-gate validation on the feature selection partition"
- "Keep all assets with > 1000 usable bars on the primary bar type (dollar)"
- "Use forecast horizons where ACF of returns is significant at lag 1 after BH correction"
- "Minimum viable DA = break-even DA from Phase 5D (transaction-cost threshold)"
- "If no features pass validation → report honestly and discuss longer horizons or alternative targets"
- Document these rules in a dedicated cell. Any deviation is flagged as post-hoc.

#### Section 2: Stationarity Report

- Table: feature × (ADF p, KPSS p, is_stationary, transformation_applied) from Phase 5pre.4
- Highlight non-stationary features and their transformations
- **Therefore:** "All features entering validation and profiling are stationary, preventing
  spurious MI/Ridge results from shared trends."

#### Section 3: Feature Exploration (ALL features, not just kept)

> Show all 23 features side-by-side. The kept/dropped partition is a **color-coded overlay**,
> not a filter. This prevents confirmation bias from profiling only survivors.

- **Feature Rationale Table** — feature → economic intuition → literature reference → expected
  sign. Every feature must have a reason to exist before we look at its performance.
- Feature correlation matrix → heatmap (all features, kept features highlighted)
- **VIF analysis (Belsley et al., 1980):** Per-feature VIF, flag VIF > 10. Report alongside
  correlation matrix — VIF captures multivariate collinearity that pairwise correlations miss.
- Feature distributions → violin plots (all features, grouped by kept/dropped)
- Feature-target scatter plots → visual inspection (kept features only, to keep it readable)
- **MI results table** with columns: feature, MI (nats), raw p-value, BH-corrected p-value,
  mi_significant, **MI as % of target entropy** (effect size, not just significance)
- **Ridge DA table** with columns: feature, DA_observed, DA_null_mean, **DA excess (pp)**,
  DA p-value, da_beats_null. Highlight features where DA excess < 1 pp (statistically
  significant but practically negligible).
- **Stability heatmap:** feature × temporal window (from feature selection partition),
  colored by per-window MI significance. Annotate windows with regime labels (bull/bear/range).
- **Feature importance comparison across bar types:** x-axis = features, y-axis = MI score,
  grouped by bar type. If dollar bars consistently produce higher MI → thesis-worthy finding.
- **Validation confirmation on holdout preview:** Re-run MI and Ridge DA on 2023 data only
  (the first year of model development period, NOT the final holdout). Report how many
  features retain significance. If features lose significance → flag feature selection
  instability.
- **Multi-horizon comparison table:** MI and Ridge DA for `fwd_logret_1`, `fwd_logret_4`,
  `fwd_logret_24` side-by-side. Show where signal concentrates across horizons.
  Decision: which horizons proceed to modeling?

#### Section 4: Confronting R5 — Is Our Data Predictable?

> Dedicated section addressing the thesis's most dangerous question. Must be present and
> thorough — a thesis examiner will ask about this.

- **Permutation entropy table:** (asset, bar_type) → H_norm at d = {3, 4, 5, 6}
  Compare against R5's reported values for BTC/ETH.
- **Complexity-entropy plane:** Scatter plot positioning each (asset, bar_type) relative
  to the Brownian noise boundary. If information-driven bars show lower entropy than
  time bars → "information-driven sampling extracts structure that uniform sampling misses."
- **Variance ratio profile chart:** x-axis = calendar horizon, y-axis = VR(q), horizontal
  line at VR = 1 (random walk). Each asset gets a line. Interpret:
  VR < 1 at short horizons = mean reversion, VR > 1 at long horizons = momentum.
- **Interpretation paragraph:** "Our data shows [VR significantly different from 1 at
  horizons X, permutation entropy Y]. This [confirms/partially confirms/contradicts] R5.
  The recommender's value is knowing WHEN not to trade: in high-entropy regimes, the
  system should abstain."

#### Section 5: Statistical Profiling Results

- **Return distribution per asset per bar type:**
  - Histogram + fitted Student-t PDF overlay (not Normal — Normal overlay is uninformative)
  - QQ-plot against **fitted Student-t** quantiles
  - Table: JB statistic, excess kurtosis, skewness, fitted ν, AIC(Normal), AIC(Student-t),
    best-fit distribution per (asset, bar_type)
  - **Therefore:** "Returns follow Student-t with ν ≈ X, confirming fat tails. This motivates
    robust loss functions (Huber loss) in regression and Student-t error terms in GARCH."

- **Autocorrelation analysis:**
  - ACF/PACF plots for returns and squared returns (all 4 assets, primary bar type)
  - Ljung-Box results table at lags [5, 10, 20, 40] with BH-corrected p-values
  - **Therefore:** "Squared returns show significant autocorrelation → ARCH effects present
    → volatility features justified. Raw returns show [significant/no] autocorrelation
    at short horizons → [supports/does not support] short-horizon forecasting."

- **Variance ratio results:**
  - Table: (asset, bar_type, calendar_horizon) → VR, Z2, BH-corrected p-value
  - Present VR as a profile chart (not just a table) — immediate visual comparison
  - **Therefore:** "VR(1 day) ≠ 1 for [assets] → short-horizon predictability exists,
    supporting our choice of 1-4 bar forecast horizons."

- **Granger causality (BTC lead only):** BTC → {ETH, LTC, SOL} at lag 1.
  3 tests only (not 24). BTC is the dominant market driver; full cross-asset Granger
  is underpowered with N < 5000 and produces excessive multiple-testing correction.
  - **Therefore:** "BTC [does/does not] Granger-cause ETH → [supports/does not support]
    cross-asset features in the recommendation system."

- **Volatility dynamics (consolidated GARCH + BDS):**
  - Single subsection combining all volatility-related tests.
  - Parameter table: (asset) → α, β, ω, persistence, ν_innovation, best-fit dist (AIC)
  - Sign bias test results → GJR-GARCH justified for [assets] / not justified
  - ARCH-LM on residuals → GARCH captured all volatility clustering? [yes/no]
  - BDS test results: (asset, m) → BDS stat, p-value. Flag if nonlinear structure remains.
  - **One combined "Therefore:" paragraph:** "GARCH(1,1) with Student-t innovations
    captures volatility dynamics for [assets]. BDS [rejects/does not reject] i.i.d. on
    residuals → nonlinear ML models [are/are not] justified beyond GARCH."

- **Rolling volatility + regime labeling:**
  - Time series plot: rolling RV with regime bands (low/normal/high) overlaid
  - Per-regime summary statistics: mean return, volatility, sample count
  - **Therefore:** "We identify N regimes. Feature informativeness [varies/does not vary]
    across regimes → [supports/does not support] regime-conditional modeling."

#### Section 6: Data Adequacy Assessment

- **Sample size table:** (asset, bar_type) → N_raw, N_after_warmup, N_eff, N_eff/N ratio, tier
- **Minimum Detectable Effect table:** (asset, bar_type) → MDE (DA above 50%),
  break-even DA (from transaction costs). Flag bar types where MDE > break-even DA
  (cannot detect economically meaningful edges).
- **Signal-to-noise ratio:** (asset, bar_type) → adjusted R² (kept features vs target),
  R² noise baseline (random features). If R² ≈ R²_noise → features explain nothing.
- **Power analysis summary:** "With N_eff = X for dollar bars, we can detect DA ≥ Y%
  at 80% power. The break-even DA is Z%. Since Y [</>] Z, detection of economically
  meaningful edges [is/is not] feasible."
- **Cross-asset consistency:**
  - Rank correlation (Kendall's τ) of feature MI scores across assets. High τ = same
    features are informative everywhere → shared model justified. Low τ = asset-specific
    feature selection needed.
  - Heatmap: features × assets, colored by keep/drop → visual synthesis
  - Profiling metric correlation: are kurtosis, VR, GARCH parameters similar across assets?
  - **Synthesis paragraph:** "N features are universally informative, M features are
    asset-specific, K features should be dropped everywhere."

- **Imbalance bar viability verdict:**
  - Per-window N for imbalance bars. Flag windows with < 100 rows.
  - "Imbalance bars (Tier C, N ≈ 430 after warmup) cannot support reliable asymptotic
    inference. Phase 4D validation on imbalance bars has MDE ≈ X% — only very strong
    effects are detectable. Results are exploratory."
  - Decision: keep imbalance bars as exploratory or drop from modeling phases.

#### Section 7: Baselines & Economic Significance

> Establish the floor before claiming features add value.

- **Buy-and-hold return and Sharpe** per asset over the feature selection period
  (2020-2022). This is the bar to clear.
- **Random walk forecast baseline:** predict next-bar return = 0 → DA = 50.0%, DC-MAE = raw MAE.
  This is the null model for all subsequent DA comparisons.
- **Coin-flip baseline:** random ±1 direction predictions → expected DA = 50%, expected
  DC-MAE. This is the absolute floor for directional models.
- **Economic significance paragraph (Ziliak & McCloskey, 2008; Harvey et al., 2016):**
  Frame ALL feature DA results relative to the break-even DA from transaction costs.
  "Our best feature achieves DA = X% vs. random DA = 50.0%, a Y pp improvement (p = Z
  after BH correction). The break-even DA for dollar bars at 20 bps round-trip cost is
  W%. The economic margin is (X - W) pp, which [is/is not] robust to model uncertainty."

#### Section 8: Go/No-Go Decision

**Formal decision table** (mechanical, based on pre-registered rules):

| Criterion | Threshold | Result | Decision |
|-----------|-----------|--------|----------|
| Features passing validation | ≥ 5 | ? | go / no-go |
| DA excess over baseline | ≥ break-even DA for ≥ 1 bar type | ? | go / no-go |
| Permutation entropy | H_norm < 0.98 for ≥ 1 bar type | ? | go / no-go |
| N_eff | ≥ 1000 for primary bar type | ? | go / no-go |
| Cross-asset consistency (τ) | τ > 0 (significant) | ? | shared / asset-specific |
| BDS on GARCH residuals | rejects i.i.d. | ? | nonlinear / linear models |

**Decision output:**
- Final feature set per horizon (keep/drop with justification)
- Final asset universe (drop assets failing adequacy)
- Confirmed bar types (Tier A proceed to modeling, Tier C dropped or exploratory-only)
- Confirmed forecast horizons
- Model complexity recommendation: linear-only vs. nonlinear (based on BDS)
- Is regression feasible? Expected DA range. If DA excess < break-even for all
  combinations → "negative result — document honestly, discuss longer horizons,
  alternative targets, or the recommender's value as a NO-TRADE filter."

**Estimated scope:** 1 notebook, no artificial cell limits — be as thorough as needed (with mandatory "Therefore..." after each section)

---

## Phase 7: Backtest Engine

**Goal:** Simple, correct backtesting engine with realistic execution modeling.
Fill on next bar open, fixed costs, minimum trade threshold.

### Theoretical Foundation

**Look-Ahead Bias in Fill Price (Bailey et al., 2014; López de Prado, 2018 Ch. 11):**
Using the current bar's close as the fill price implicitly assumes you knew the bar's
close before it occurred. In crypto, the "close" of a dollar bar is the price at which
cumulative dollar volume crossed the threshold — a fact knowable only after the bar
completes. The correct specification: signal on bar[t], fill on bar[t+1] open. This is the
standard in academic backtesting (Harvey & Liu, 2015) and eliminates the most common
source of inflated backtests.

**Lo (2002) Autocorrelation-Corrected Sharpe Ratio:** The standard Sharpe ratio SE assumes
i.i.d. returns: `SE(SR) = sqrt((1 + SR²/2) / T)`. With autocorrelated returns (which
crypto exhibits, especially on information-driven bars with volatility clustering), this
SE is biased downward, inflating significance. Lo's correction:
`SR_corrected = SR × η(q)` where `η(q) = sqrt(q / (q + 2·Σ_{k=1}^{q} (q-k)·ρ_k))`
and `ρ_k` is the k-th autocorrelation. For crypto dollar bars with moderate return
autocorrelation, the correction factor is typically 1.15-1.8×, which can flip a
"significant" Sharpe to non-significant. This must be computed from day one so all
results are honest.

**Minimum Trade Count (Chordia et al., 2014):** Metrics computed on small samples are
unreliable. A Sharpe ratio from 8 trades has a standard error so large that even SR = 3.0
is not significant. The critical threshold depends on the desired precision, but 30 trades
is a standard minimum (CLT convergence). Below this, report "insufficient sample" rather
than misleading point estimates.

**Transaction Costs in Crypto (Makarov & Schoar, 2020):** Binance spot maker/taker fees
at standard tier are 0.1%/0.1% (10 bps each way = 20 bps round-trip). Effective spreads
vary by asset: BTC/ETH have tight spreads (1-3 bps), while LTC/SOL can have 5-15 bps
during low-liquidity periods. An asset-level cost multiplier (BTC/ETH 1×, LTC/SOL 1.5-2×)
captures this heterogeneity without over-engineering a full order book model.

**Random Strategy Baseline (White, 2000 — Reality Check):** A strategy that cannot
statistically beat random entry has no edge. The RandomStrategy generates signals with the
same frequency distribution as the real strategy but random timing. Comparing real vs.
random Sharpe via permutation test directly supports Phase 14's Monte Carlo validation and
is the first line of defense against overfitting.

### 7A: Core Domain Model

- **`src/app/backtest/domain/value_objects.py`** —
  - `Side` enum (LONG, SHORT)
  - `ExecutionConfig` — commission_bps (default 10), asset_cost_multiplier (dict), min_trade_count (default 30)
  - `TradeResult` — entry_price, exit_price, side, size, entry_time, exit_time, gross_pnl, net_pnl, commission_paid
  - `PortfolioSnapshot` — timestamp, equity, cash, positions, unrealized_pnl, drawdown

- **`src/app/backtest/domain/entities.py`** —
  - `Position` — asset, side, size, entry_price, entry_time, unrealized_pnl, stop_loss, take_profit
  - `Trade` — full lifecycle of a position from open to close
  - `EquityCurve` — time series of portfolio value

- **`src/app/backtest/domain/protocols.py`** —
  - `IStrategy` protocol — `on_bar(timestamp, features, portfolio) -> list[Signal]`
  - `IPositionSizer` protocol — `size(signal, portfolio, volatility) -> float`

### 7B: Execution Layer

- **`src/app/backtest/application/execution.py`** — `ExecutionEngine`:
  - Sequential bar loop: `for bar in bars: signal = strategy.on_bar(bar)` (~300 LOC total)
  - **Fill on next bar open** (López de Prado, 2018). Signal on bar[t], fill on bar[t+1].open.
    Non-negotiable. No exceptions.
  - Fixed commission (10 bps per side) + asset-level cost multiplier (Makarov & Schoar, 2020)
  - Bar staleness check — skip if gap > 2× median bar duration (crypto-specific: variable
    bar duration means stale bars carry no information)
  - Tracks equity curve, all trades, portfolio snapshots
  - No lookahead: strategy only sees data up to current bar

- **`src/app/backtest/application/position_sizer.py`** —
  - `FixedFractionalSizer` — `size = equity × fraction / price`. Signal → position size mapping.
  - Implements `IPositionSizer` protocol.

- **`src/app/backtest/application/cost_sweep.py`** —
  - `run_with_cost_sweep(strategy, data, fees=[5, 10, 15, 20])` → dict of results per fee level.
    ~50 lines. Shows at what cost level strategy alpha disappears.

### 7C: Metrics Layer (independently testable)

- **`src/app/backtest/application/metrics.py`** —
  - **Return metrics:** total return, annualized return, CAGR
  - **Risk metrics:** max drawdown, drawdown duration, annualized volatility, downside volatility
  - **Risk-adjusted:** Sharpe ratio (with Lo 2002 autocorrelation correction), Sortino ratio, Calmar ratio
  - **Trade metrics:** win rate, profit factor, avg win/loss ratio, max consecutive losses
  - **Min trade threshold** (default 30, Chordia et al., 2014): if N_trades < min_trades,
    report "insufficient sample — metrics unreliable" instead of point estimates
  - All metrics computed on the equity curve / trade list
  - **Buy-and-hold benchmark** auto-computed per run for comparison

### 7D: Baselines

- **`src/app/backtest/application/baselines.py`** —
  - `RandomStrategy` — generates signals with same frequency distribution as real strategy
    but random timing (White, 2000). ~30 lines.
  - `BuyAndHoldStrategy` — enter long at start, hold to end. The absolute floor.

### 7E: Walk-Forward Framework

- **`src/app/backtest/application/walk_forward.py`** — `WalkForwardRunner`:
  - Configurable window: expanding or rolling
  - For each window: fit strategy on train, generate signals on test, run through engine
  - Aggregate results across windows
  - Output: per-window metrics + aggregate metrics

### 7F: Tests

- Unit test: commission calculation on known inputs
- Unit test: position sizer produces correct sizes
- Integration test: run trivial strategy (always long) through engine, verify equity curve
  matches manual calculation with known commission
- Regression test: deterministic strategy on fixed data → known PnL
- Verify fill-on-next-open: signal at bar[t].close → fill at bar[t+1].open
- Edge cases: zero-volume bars, gaps, first/last bar handling, staleness skip
- Lo Sharpe correction: verify on synthetic autocorrelated returns

**Dependencies:** Phase 1 (data), Phase 4 (features for strategies)
**Estimated scope:** ~10 files, ~500 lines

---

## Phase 8: Base Trading Strategies

**Goal:** Implement three diverse strategies with orthogonal regime profiles for the
recommender to learn conditional deployment.

### Theoretical Foundation

**Strategy Diversity for Meta-Labeling (López de Prado, 2018 Ch. 3):** The recommender
(Phase 12) learns *when* to deploy each strategy. Its discriminative power depends entirely
on strategies having *different* return profiles across regimes. If two strategies win and
lose in the same conditions, the recommender has no signal to learn from. The current plan
had momentum crossover + DRTS — both EMA-based trend following. Functionally, they produce
near-identical label vectors (confirmed by examining legacy DRTS code: it uses
`ema_diff_norm + slope_norm + vol_ratio`, which is momentum crossover with a volatility
gate).

**Regime Theory in Crypto (Bouri et al., 2019; Wen et al., 2022):** Crypto markets exhibit
three dominant regimes: (1) trending (bull/bear), (2) range-bound/mean-reverting, (3) regime
transitions (breakouts from consolidation). A recommender needs at least one strategy per
regime type to learn conditional deployment:
- **Momentum crossover** → profits during established trends
- **Mean reversion (Bollinger)** → profits during range-bound periods
- **Donchian breakout** → profits at regime *transitions*

This gives three genuinely orthogonal regime profiles, maximizing the recommender's feature
space diversity.

**Hurst Exponent as Regime Filter (Mandelbrot, 1963; López de Prado, 2018 Ch. 5):** The
Hurst exponent H measures long-range dependence. H > 0.5 = trending, H < 0.5 = mean-
reverting, H ≈ 0.5 = random walk. Conditioning mean reversion on H < 0.5 restricts signals
to regimes where the underlying assumption (price returns to mean) is empirically supported.

**Fat Tails and Bollinger Width (Cont, 2001):** Standard Bollinger Bands assume returns are
approximately Gaussian — the 2σ bands correspond to 95.4% containment. With crypto kurtosis
~6.7 (Student-t ν ≈ 5-6), the probability mass beyond 2σ is ~2-3× higher than Gaussian,
meaning bands are violated much more frequently. Widening to 2.5σ or using Student-t
quantile-based bands (at the fitted ν from Phase 5A) restores the intended containment
probability.

### 8A: Strategy Interface

- **`src/app/strategy/domain/protocols.py`** — `IStrategy` protocol:
  - `generate_signals(feature_set: FeatureSet) -> pl.DataFrame` — strategies consume
    FeatureSet (from Phase 4), not raw OHLCV. Eliminates duplicate indicator computation.
  - `name() -> str`

### 8B: Strategies

- **`src/app/strategy/application/momentum_crossover.py`** — EMA crossover with ATR-based
  stops. Long when fast > slow, short when fast < slow. Parameters: fast_period, slow_period,
  atr_multiplier_sl, atr_multiplier_tp.

- **`src/app/strategy/application/donchian_breakout.py`** — Donchian channel breakout:
  enter long when close > highest high of N bars, exit at trailing ATR stop. ~80 lines.
  Breakout confirmation: close must be *above* the channel (not just touch). Crypto has
  false breakouts at round numbers ($50K, $100K).

- **`src/app/strategy/application/mean_reversion.py`** — Bollinger band bounce: enter when
  price crosses below lower band (long) or above upper band (short), exit at mean.
  **Hurst filter:** signal only when `hurst_100 < 0.5` (~5 lines).
  **Widen Bollinger to 2.5σ** (or Student-t quantile bands if available from Phase 5A).

### 8C: Tests

- Unit test: each strategy on synthetic trending/mean-reverting data
- Backtest: run each strategy through engine on historical data
- Verify strategies produce different signal patterns (pairwise Jaccard similarity < 0.5)

**Dependencies:** Phase 4 (features), Phase 7 (backtest engine)
**Estimated scope:** ~6 files, ~350 lines

---

## Phase 9: Direction Classification Models

**Goal:** Predict the **direction** of future price movement (up/down).
This is the first forecasting track — classification determines the **side** of the trade.

> **Two-track forecasting design (inspired by López de Prado's meta-labeling):**
> - **Track 1 (Phase 9):** Classification → predict direction (up/down) → determines SIDE
> - **Track 2 (Phase 10):** Regression → predict return magnitude → determines SIZE
> - **Combined:** Classification picks the side, regression estimates how much.
>   The recommendation system (Phase 12) consumes BOTH outputs.

### Theoretical Foundation

**CPCV with Purging and Embargo (López de Prado, 2018 Ch. 7, 12):** Standard k-fold CV on
financial time series produces wildly optimistic results because: (a) training and test folds
share temporal neighbors with autocorrelated features, (b) forward-looking labels (shift-based
targets) create overlap zones. CPCV fixes this with three mechanisms:
- **Purging:** Remove training samples whose label period overlaps with any test sample's
  feature period. For horizon h=1, purge at least 1 bar around each fold boundary.
- **Embargo:** After purging, remove an additional buffer to account for serial correlation
  in features. Embargo = autocorrelation decay length (from Phase 5 ACF analysis).
- **Combinatorial:** Test all C(N, k) train/test combinations for a full performance
  distribution, not just a point estimate.

**Cross-Asset Temporal Purging (Leakage Prevention):** When pooling 4 assets with BTC-ETH
correlation ~0.85, training on ETH at time `t` while testing on BTC at time `t` gives the
model a free preview of ~85% of the test signal. The correct implementation purges ALL assets
in the temporal window `[t - embargo, t + h + embargo]`, not just the test asset. This is the
single most impactful leakage vector identified by R5.

**Deep Learning on Small Tabular Data (Grinsztajn et al., 2022; Borisov et al., 2022):**
Systematic benchmarks show that gradient-boosted trees (XGBoost, LightGBM) outperform deep
learning on tabular data below ~10,000 samples. With ~5,000 dollar bars per asset (pooled to
~20,000), tree methods are expected to dominate. A Transformer with 50K+ parameters will
memorize noise. A small GRU (2-layer, 64 hidden) serves as a controlled negative-result
experiment: "Confirms R5: crypto noise does not reward model complexity." This is valid
thesis content — honest negative results demonstrate understanding.

**Confidence-Based Abstention (Chow, 1970; El-Yaniv & Wiener, 2010):** A classifier forced
to predict on every bar wastes capacity on the ~80% where direction is essentially random
(near-Brownian). Allowing abstention when `P(up) ∈ [0.4, 0.6]` concentrates evaluation on
high-conviction predictions. Reporting DA at multiple confidence thresholds
`{0.5, 0.55, 0.6, 0.65, 0.7}` alongside coverage shows the accuracy-coverage tradeoff —
a model with DA=56% on 30% of bars is vastly more economically useful than DA=51% on 100%.
This IS the two-track system's economic core: the classifier is a filter, not an oracle.

**Sign Target vs Triple Barrier (simplification rationale):** Triple barrier labeling
(López de Prado, 2018 Ch. 3) creates a 3-class problem (take-profit hit, stop-loss hit,
time expiry). The "time expiry" class typically captures 40-60% of samples, halving per-class
counts to ~1200. For a bachelor's thesis, binary `sign(fwd_logret_h)` is cleaner, easier to
defend, and sufficient. Triple barrier can be noted as future work.

### 9A: Classification Domain & CPCV Infrastructure

- **`src/app/forecasting/domain/value_objects.py`** —
  - `ForecastHorizon` enum (H1, H4, H24)
  - `DirectionForecast` — predicted_direction (+1/-1), confidence (probability), horizon
  - `ReturnForecast` — predicted_return (point estimate), prediction_std, quantiles, confidence_interval

- **`src/app/forecasting/domain/protocols.py`** —
  - `IDirectionClassifier` protocol: `fit(X, y_direction)`, `predict(X) -> list[DirectionForecast]`
  - `IReturnRegressor` protocol: `fit(X, y_return)`, `predict(X) -> list[ReturnForecast]`

- **`src/app/forecasting/infrastructure/cpcv.py`** — `CPCVSplitter`:
  - Shared infrastructure used in Phase 9, 10, 12
  - Parameters: n_blocks, purge_window, embargo_window
  - **Cross-asset temporal purging** — purge ALL assets in `[t - embargo, t + h + embargo]`
  - Returns train/test indices with proper leakage prevention

### 9B: Classification Target Construction

- **`src/app/features/application/targets.py`** — Extend with classification targets:
  - `forward_direction(horizon)` = `sign(fwd_logret_h)` → +1 or -1
  - Start with binary sign target. Triple barrier noted as future work.

### 9C: Classifiers (3 models)

- **`src/app/forecasting/application/logistic_baseline.py`** — Logistic regression.
  Interpretable baseline. Outputs calibrated probabilities.
- **`src/app/forecasting/application/random_forest_clf.py`** — Random Forest classifier.
  Non-linear, handles feature interactions. Feature importance for interpretability.
- **`src/app/forecasting/application/gradient_boosting_clf.py`** — LightGBM classifier.
  Strong tabular baseline. Outputs calibrated probabilities via Platt scaling.

### 9D: GRU Negative-Result Experiment

- **`src/app/forecasting/application/gru_classifier.py`** — GRU encoder (2 layers, 64
  hidden) → sigmoid head for direction probability. Loss: binary cross-entropy.
  **Purpose:** Controlled experiment to show tree dominance at this sample size
  (Grinsztajn et al., 2022). Expected to underperform LightGBM — this is valid thesis
  content documenting negative results.

### 9E: Classification Metrics

- **`src/app/forecasting/application/classification_metrics.py`** —
  - **Accuracy:** % correct direction predictions — must beat 50% (coin flip baseline)
  - **Precision / Recall / F1** per class (up/down): is the model biased toward one direction?
  - **Confidence-based abstention** — DA at thresholds `{0.5, 0.55, 0.6, 0.65, 0.7}` with
    coverage at each threshold. Accuracy-coverage tradeoff curves.
  - **Calibration:** predicted probability vs actual frequency (reliability diagram).
  - **AUC-ROC:** discrimination ability regardless of threshold
  - **Economic accuracy:** accuracy weighted by |actual return|
  - **Asymmetric class weighting** — penalize missed crashes more (crypto negative
    skewness -0.36). Report weighted F1 alongside standard.

### 9F: Sanity Checks

- **Shuffled-labels sanity (Ojala & Garriga, 2010):** Train on permuted targets, verify
  DA → 50%. If ANY model exceeds 50% on shuffled labels, the pipeline has a bug.
- **Naive benchmarks:** majority-class, persistence (predict yesterday's direction),
  momentum-sign (predict direction of trailing EMA).
- **Asset pooling comparison:** Pool 4 assets with `asset_id` categorical (~14000+ samples)
  vs per-asset training. Compare in RC3.

### 9G: Tests

- Unit test: logistic regression on linearly separable data → near-perfect accuracy
- Convergence test: GRU loss decreases over epochs
- Calibration test: predicted probabilities match observed frequencies
- Null test: model trained on shuffled labels → accuracy ≈ 50%
- CPCV test: verify no temporal leakage (train/test indices don't overlap after purging)
- Cross-asset purge test: verify ETH training data purged when BTC is in test window

**Dependencies:** Phase 4 (features + targets), `lightgbm`, `pytorch`
**Estimated scope:** ~12 files, ~900 lines (including shared CPCV infrastructure)

---

## Phase 10: Return Regression Models

**Goal:** Predict the **magnitude** of future price movement (how much will it move).
This is the second forecasting track — regression determines the SIZE of the position.

> **Key principle:** Regression metrics (MAE, RMSE, R²) are only meaningful WHEN the direction
> classifier is correct. A model predicting +2% when the actual move is -3% has low MAE but is
> useless. **Regression is evaluated conditionally on correct direction.**

### Theoretical Foundation

**Volatility-Normalized Targets (Bollerslev, 1986; Andersen & Bollerslev, 1998):** Raw
crypto log returns are extremely heteroscedastic — a +2% move during a 10% daily vol regime
is unremarkable; the same +2% during a 0.5% daily vol regime is massive. Normalizing:
`z_t = r_t / σ_t` (where σ_t is backward-looking realized vol) removes the time-varying
volatility component, leaving only the directional signal. This is standard in quantitative
finance ("risk-adjusted returns"). At inference, rescale:
`predicted_return = predicted_z × current_vol`. Use Garman-Klass or Parkinson vol (already
computed as features) — they are more efficient estimators with fewer observations than
simple return std.

**Huber Loss for Fat Tails (Huber, 1964):** MSE loss weights residuals quadratically,
meaning a single extreme observation (crypto crash) can dominate the entire gradient. With
kurtosis 5-15 on dollar bars, MSE-trained models systematically underweight typical
observations and overfit to tails. Huber loss transitions from quadratic (near zero) to
linear (beyond threshold δ) — it downweights tail events without ignoring them. This is
the standard robust regression approach for heavy-tailed financial data.

**MC Dropout vs Mixture Density Networks (Gal & Ghahramani, 2016; Bishop, 1994):** MDNs
model the conditional distribution as a mixture of Gaussians, learning means, variances,
and mixing coefficients. They are notoriously hard to train: mode collapse, numerical
instability in log-likelihood, and sensitivity to K. With N=5,000, an MDN will likely
degenerate to a single Gaussian. MC Dropout is simpler: run N stochastic forward passes
with dropout active at inference time. The variance across passes estimates epistemic
uncertainty. ~10 lines of code (set `model.train()` at inference), not a separate module.

**Quantile Crossing (Koenker, 2005; Chernozhukov et al., 2010):** Separate quantile
regressions at τ = {0.05, 0.25, 0.50, 0.75, 0.95} can produce crossing predictions
(Q90 < Q50 for some samples). This violates the monotonicity requirement for valid
predictive distributions. Fix: isotonic regression post-processing or LightGBM's built-in
quantile mode (which handles this more gracefully). The 5th/95th quantiles matter for crypto
because tail events (kurtosis ~6.7) are where risk management operates.

**Conformal Prediction Exchangeability (Vovk et al., 2005; Gibbs & Candes, 2021 — ACI):**
Standard split conformal prediction assumes exchangeable calibration residuals. Crypto
returns violate this — regime-dependent volatility, autocorrelated squared returns,
structural breaks. ACI (Adaptive Conformal Inference) partially addresses this by adapting
α_t online based on recent coverage. Report coverage per regime separately — overall 90%
can hide 60% during volatile regimes (when intervals matter most).

**Direction-Conditional Evaluation Bias:** DC-MAE/DC-RMSE are computed only where the
Phase 9 classifier is correct. This biased subsample is NOT random — it likely has lower
volatility and smaller absolute returns (the "easy" predictions). Report: (a) fraction
where classifier was correct, (b) mean |return| in correct vs wrong subsets, (c) KS test
between distributions. This prevents misleading DC-MAE values.

### 10A: Volatility-Normalized Target

- **`src/app/features/application/targets.py`** — Extend with:
  - `fwd_zret_h = fwd_logret_h / backward_rv_h` — volatility-normalized target
  - Runtime assertion: σ is backward-looking only (no future leakage)
  - **Winsorization at 1st/99th percentile** instead of hard [-0.15, 0.15] clip
    (which censors COVID crash, FTX, Luna events)

### 10B: Regressors (4 models)

- **`src/app/forecasting/application/ridge_baseline.py`** — Ridge regression. Simple, fast,
  interpretable. Provides point estimate + residual std for uncertainty. **Huber loss option.**
- **`src/app/forecasting/application/gradient_boosting_reg.py`** — LightGBM quantile
  regressor. 5 quantiles `{0.05, 0.25, 0.50, 0.75, 0.95}` with isotonic regression for
  monotonicity (Chernozhukov et al., 2010). Primary nonlinear model.
- **`src/app/forecasting/application/gru_regressor.py`** — GRU encoder (2 layers, 64 hidden)
  → linear head + **MC Dropout** (Gal & Ghahramani, 2016). N=50 forward passes at inference
  for epistemic uncertainty. Replaces GRU-MDN (saves weeks of training instability).
- **`src/app/forecasting/application/arima_garch.py`** — ARIMA for conditional mean + GARCH
  for conditional variance. **Time_1h bars only** — ARIMA assumes equal spacing.
  Univariate baseline for comparison.

### 10C: Calibration & Conformal Prediction

- **`src/app/forecasting/application/calibration.py`** —
  - Reliability diagrams: predicted quantile q → actual coverage ≈ q
  - ACI conformal prediction wrapper (Gibbs & Candes, 2021) — adapts α_t online
  - Residual diagnostics: homoscedasticity, normality
  - **Coverage per regime** — report separately for high-vol and low-vol periods

### 10D: Regression Metrics (Direction-Conditional)

> **These metrics are ALWAYS reported conditional on the direction classifier's prediction.**
> Two evaluation modes:
> - **Standalone regression:** evaluate on all samples, but report DA alongside MAE/RMSE
> - **Pipeline regression:** evaluate ONLY on samples where direction classifier was correct

- **`src/app/forecasting/application/regression_metrics.py`** —
  - **DC-MAE (Direction-Conditional MAE):** MAE only where sign(predicted) == sign(actual)
  - **DC-RMSE:** Same, penalizes large errors on correct-direction predictions
  - **WDL (Wrong-Direction Loss):** Average |predicted - actual| where direction is wrong
  - **PDR (Profitable Direction Ratio):** When predicted return > threshold AND direction
    correct, what is avg realized return?
  - **CRPS:** Full distributional metric — primary metric for probabilistic forecasters
  - **Economic Sharpe:** classifier picks side, regressor sizes position. Ultimate metric.
  - **Selection-bias characterization:** report correct-subset distribution vs full — fraction
    correct, mean |return| in correct vs wrong, KS test. Prevents misleading DC metrics.
  - **Prediction clipping** via Winsorization at inference

### 10E: Tests

- Unit test: Ridge on linear data → low DC-MAE
- Convergence test: GRU loss decreases
- Calibration test: conformal intervals achieve target coverage
- Null test: regressor on noise → DC-MAE ≈ unconditional MAE (no improvement)
- Volatility normalization test: verify backward-only σ computation
- Quantile monotonicity test: verify isotonic correction removes crossings

**Dependencies:** Phase 4 (features + targets), Phase 9 (direction classifier for conditional eval), `arch`, `statsmodels`, `lightgbm`, `pytorch`, `mapie`
**Estimated scope:** ~10 files, ~900 lines

---

## Phase 11: Research Checkpoint 3 — Classification & Regression Evaluation

**Goal:** Evaluate BOTH forecasting tracks. Compare classification vs regression approaches.
Can the classifier beat a coin flip? Does regression add value on top of correct direction?
Are the two complementary? Split into 3 focused notebooks.

### Theoretical Foundation

**Pre-Registration (Nosek et al., 2018; Chambers & Tzavella, 2022):** Pre-registration
separates confirmatory from exploratory analysis. Writing decision criteria before opening
notebooks converts researcher degrees of freedom into pre-committed mechanical rules. For
a bachelor's thesis, this is unusually rigorous and will impress committee members.

**Statistical Power (Cohen, 1988):** Power = P(reject H₀ | H₁ true). With N_test = 500
bars (one walk-forward fold for dollar bars) and α = 0.05, the minimum detectable accuracy
above 50% at 80% power is ~54.5%. If realistic DA is 52-53%, the test WILL fail to reject —
but this is a power problem, not a signal problem. Computing and reporting MDE per cell
prevents misinterpreting non-rejection as evidence of no effect (the classical Type II error).

**Regime-Stratified Evaluation vs Giacomini-White (2006):** The GW conditional predictive
ability test is elegant but requires choosing instrument variables, bandwidth, and has power
issues at small N. A simpler approach with identical insight: run standard Diebold-Mariano
tests separately on subsets (high-vol, low-vol, trending, ranging). This answers "is model A
conditionally better?" without econometric machinery that is hard to debug. The theoretical
cost: GW tests conditional superiority jointly; stratified DM tests it marginally per regime.
For a bachelor's thesis with N < 5000, the marginal approach has higher power per regime.

**Shuffled-Labels as Leakage Detector (Ojala & Garriga, 2010):** Training every model once
on permuted targets and showing accuracy → 50% is the single most convincing anti-leakage
proof. If ANY model exceeds 50% on shuffled labels, the pipeline has a bug. Cost: one extra
training loop per model. Value: strongest defense against "is this just overfitting?"

### 11A: Notebook `research/RC3_classification.ipynb`

**Pre-register go/no-go decision tree** — mechanical criteria before seeing data.

**Power analysis table** per (asset, bar_type) — MDE at 80% power. Know what effects are
detectable before interpreting results.

**Classification evaluation:**
- Per-model, per-asset, per-horizon: accuracy, precision, recall, F1, AUC-ROC
- Comparison table: all classifiers × all assets → which model wins?
- **Accuracy > 50% is the minimum bar** — binomial test: p-value for accuracy > 0.5
- Confidence-based abstention curves: DA at `{0.5, 0.55, 0.6, 0.65, 0.7}` with coverage
- Calibration plots: predicted probability vs actual frequency per model
- Pooled vs per-asset comparison (from Phase 9F)
- **Walk-forward equity curve visualization** — WHERE models fail, not just aggregate metrics

**Shuffled-labels sanity (Ojala & Garriga, 2010):** Every model trained on permuted targets.
Verify DA → 50%.

**Naive benchmark battery:** majority-class, persistence, momentum-sign.

**Classification decision output:**
- Best classifier(s) for direction prediction
- Best horizon for direction prediction
- Per-asset accuracy heatmap → which assets are predictable?
- Is classification viable? (any model significantly > 50%?)

### 11B: Notebook `research/RC3_regression.ipynb`

**Standalone regression evaluation:**
- Per-model: raw MAE, RMSE (for completeness only — NOT decision metrics)
- **DA (Directional Accuracy)** of regressors: do they implicitly get direction right?
- If regressor DA > classifier accuracy → regressor alone may be sufficient

**Pipeline evaluation (classifier + regressor combined):**
- Filter to samples where best classifier is correct → evaluate regressor DC-MAE, DC-RMSE
- Scatter plots: predicted vs actual return, **only for correct-direction samples**
- **Selection-bias characterization** — report correct-subset distribution vs full
- **PDR:** when classifier says "up" AND regressor says "> +1%", how often is realized
  return positive and > 0.5%?

**Uncertainty evaluation:**
- Calibration plots (reliability diagrams) per regressor
- CRPS per model
- Conformal interval coverage (overall + per regime)
- Interval sharpness: narrower is better at same coverage

**Shuffled-labels sanity** for regressors.

**Naive benchmark battery:** random walk, EWMA, historical mean.

### 11C: Notebook `research/RC3_combined_pipeline.ipynb`

**Economic evaluation (combined pipeline):**
- **Economic Sharpe:** classifier picks side, regressor sizes position → equity curve
- Compare: classifier-only (equal size), regressor-only (sign determines side), combined
- Profit factor: gross profit from correct trades / gross loss from incorrect
- **Walk-forward equity curve** showing WHERE combined outperforms/underperforms

**Statistical comparison:**
- **Regime-stratified Diebold-Mariano** — DM test separately on high-vol, low-vol, trending,
  ranging subsets. Same insight as Giacomini-White, more transparent at small N.
- **Combined vs separate test:** Does (classifier + regressor) significantly beat
  classifier-only? DM test on Economic Sharpe.

**Data adequacy:**
- Is any classifier's accuracy significantly > 50%? (binomial test)
- Is any regressor's DC-MAE significantly lower than unconditional MAE? (permutation test)
- Is Economic Sharpe of combined pipeline significantly > 0? (bootstrap CI)

**Decision output:**
- Select best classifier (or top-2) for the recommendation system
- Select best regressor (or top-2) for the recommendation system
- Confirm: does the combined pipeline outperform each alone?
- Determine best forecast horizon
- Assets where accuracy ≈ 50% → classification hopeless, flag for recommender

**Estimated scope:** 3 notebooks (no artificial cell limits — be as thorough as needed)

---

## Phase 12: ML Recommendation System

**Goal:** Train a machine learning model that learns to predict which assets the base strategy
will perform well on, given current market features and **both classification + regression
forecasts**. This is generalized meta-labeling — predicting expected strategy return
(continuous), enabling both deploy/skip decisions AND position sizing.

> **Key insight:** The recommender IS the ML model. It is not a formula or heuristic.
> It consumes BOTH forecasting tracks: (1) classifier's direction + confidence, (2) regressor's
> magnitude + uncertainty. It has training data, a loss function, train/val/test splits,
> and testable hypotheses.
>
> **This is generalized meta-labeling (López de Prado):** the primary models (classifier +
> regressor) generate signals, and the recommender is the secondary model that decides
> WHETHER to deploy the strategy and HOW to SIZE the position, based on predicted strategy
> performance.

### Theoretical Foundation

**Multi-Layer Walk-Forward and Stacking Leakage (van der Laan et al., 2007; Bojer &
Meldgaard, 2021):** The recommender's features include classifier/regressor predictions.
If these predictions were generated on data that overlaps with the recommender's training
labels, the features are artifactually good — the recommender learns a mapping that only
works when inputs are in-sample quality. Fix: strict temporal separation. L1 (classifier/
regressor) OOS predictions on `[t1+purge, t2]` become L2 (recommender) training features
for the same period, with labels from the same period. L3 (evaluation) on `[t2+purge, t3]`.
Key invariant: L1 predictions used as L2 features are ALWAYS genuinely out-of-sample for L1.

**Fixed-Horizon Labels (Khandani et al., 2010; Cont et al., 2005):** Position-level strategy
returns are path-dependent (stop-loss timing, slippage sequence) and wildly noisy. Fixed-
horizon returns at the decision point (net return over `[t, t+H]` including costs) are:
(a) comparable across time, (b) free from exit-logic noise, (c) aligned with rebalancing
frequency. Make `label_horizon` configurable.

**Generalized Meta-Labeling (López de Prado, 2018 Ch. 3; extended):** Original meta-labeling
is binary: the secondary model predicts whether to bet on the primary model's signal. This
thesis generalizes to regression: the recommender predicts *expected strategy return*
(continuous). This enables position SIZING proportional to conviction, not just binary
deploy/skip. Implementation: `size ∝ max(r_hat - threshold, 0) / sigma` (Kelly-adjacent).

**Conformal Prediction on Recommender Output (Vovk et al., 2005; Romano et al., 2019):**
Wrapping the recommender's point prediction in split conformal gives a principled deploy/skip
threshold: "deploy only if the lower bound of the 80% interval > 0." This is a novel
combination — conformal prediction applied to meta-label output.

**Ablation Studies (Meyes et al., 2019):** SHAP values show feature-prediction correlation;
ablation shows causation. Running the full pipeline minus one feature group (classifier
features, regressor features, regime features) with DM tests against the full model directly
answers: "Does combining tracks add value?" (H3).

### 12A: Training Data Construction

- **`src/app/recommendation/application/label_builder.py`** — For each (asset, time_window):
  1. **Fixed-horizon strategy return labels** at decision point (Khandani et al., 2010).
     Net return over `[t, t+H]` including transaction costs. `label_horizon` configurable.
  2. **Weekly windows** (not monthly) — 4 assets × 3 strategies × ~150 weeks ≈ 1800 labels.
  3. Walk-forward: only use past data for features, future window for labels.

- **`src/app/recommendation/application/feature_builder.py`** — Features for the recommender:
  - **Market state features:** all features from Phase 4 (volatility, momentum, Hurst, etc.)
  - **Classifier features (from Phase 9):** predicted direction, classifier confidence,
    classifier accuracy on recent N predictions
  - **Regressor features (from Phase 10):** predicted return magnitude, prediction uncertainty,
    quantile spread (Q95-Q05), conformal interval width
  - **Combined forecast features:** classifier agrees with regressor sign? |predicted return|
    × classifier confidence (conviction score)
  - **Regime features:** GARCH conditional volatility, volatility regime indicator, rolling
    permutation entropy
  - **Cross-asset features:** relative strength vs universe mean, beta to BTC,
    **rolling cross-asset correlation** (crypto correlations → 1.0 during crashes)
  - **Historical strategy features:** rolling strategy Sharpe on this asset, rolling win rate

### 12B: Recommender Domain

- **`src/app/recommendation/domain/value_objects.py`** —
  - `RecommendationInput` — asset, timestamp, feature_vector, direction_forecast, return_forecast
  - `Recommendation` — asset, predicted_strategy_return, confidence, deploy (bool),
    predicted_direction, predicted_magnitude, **position_size** (from generalized meta-label)
  - `RecommenderConfig` — model_type, train_window, retrain_frequency, min_threshold,
    label_horizon

- **`src/app/recommendation/domain/protocols.py`** — `IRecommender` protocol:
  - `fit(X_train: DataFrame, y_train: Series) -> None` (y = realized strategy return)
  - `predict(X: DataFrame) -> list[Recommendation]`

### 12C: Recommender Models

- **`src/app/recommendation/application/gradient_boosting_recommender.py`** —
  LightGBM regressor predicting strategy return per asset. Feature importance (SHAP) reveals
  what drives recommendations. Primary model — strong on tabular data at this sample size
  (Grinsztajn et al., 2022), interpretable, fast.

- **`src/app/recommendation/application/baseline_recommenders.py`** — Baselines:
  - `RandomRecommender` — randomly select assets (null hypothesis)
  - `AllAssetsRecommender` — deploy strategy on everything (unfiltered)
  - `ClassifierOnlyRecommender` — deploy based on classifier confidence alone
  - `RegressorOnlyRecommender` — deploy based on predicted return magnitude alone
  - `EqualWeightRecommender` — equal weight to all assets with positive forecast

### 12D: Walk-Forward Training Pipeline

- **`src/app/recommendation/application/pipeline.py`** — `RecommenderPipeline`:
  1. **Multi-layer walk-forward with explicit purging between layers.** Non-negotiable.
     L1 (classifier/regressor) OOS predictions become L2 (recommender) features.
  2. For each window:
     a. L1: Compute classifier + regressor OOS predictions on window data
     b. L2: Build recommender features from L1 predictions + market state
     c. L2: Train recommender on (features, fixed-horizon strategy returns)
     d. L3: On test window → recommender.predict() → deploy/skip + position size
     e. Run base strategy on recommended assets with sized positions
     f. Run all baselines for comparison
  3. Aggregate: per-window and overall metrics

### 12E: Recommendation Metrics & Ablation

- **`src/app/recommendation/application/metrics.py`** —
  - **Decision quality:** Precision of deploy=True decisions (of recommended, how many had
    positive strategy return?)
  - **Economic value:** Sharpe of recommended portfolio vs baselines — **the ultimate metric**
  - **Position sizing value:** Sharpe with sizing vs without (binary deploy/skip).
    Realizes the "generalized" meta-labeling claim.
  - **Conformal intervals on predictions** (~150 lines) — deploy only if lower bound > 0.

- **`src/app/recommendation/application/ablation.py`** —
  - **Structured ablation** — full vs remove-{classifier, regressor, regime} features.
    DM test per ablation against full model. Directly tests H3.

### 12F: Tests

- Unit test: label builder produces correct strategy returns
- Unit test: feature builder assembles features without leakage
- Integration test: full pipeline on small synthetic dataset
- Sanity test: recommender trained on noise → no better than random baseline
- Multi-layer leakage test: verify L1 predictions used as L2 features are genuinely OOS

**Dependencies:** Phase 4 (features), Phase 7 (backtest), Phase 8 (strategies), Phase 9 (classifier), Phase 10 (regressor)
**Estimated scope:** ~12 files, ~1000 lines

---

## Phase 13: Research Checkpoint 4 — Recommender Evaluation

**Goal:** Does the ML recommender actually add value? Does combining classification + regression
outperform each alone? Final charts, statistics, and honest evaluation.

### Theoretical Foundation

**Baseline Ladder (Harvey et al., 2016; López de Prado, 2019):** A single benchmark
(AllAssetsRecommender) is necessary but not sufficient. The real question is whether the ML
recommender beats the *best simple alternative*. Baseline ladder: Random → AllAssets →
ClassifierOnly → RegressorOnly → EqualWeight → ML Recommender. Each step up in complexity
must justify itself statistically (DM test). If the recommender only beats "deploy everything"
but loses to "just use the classifier," the meta-layer destroys value.

**Break-Even Analysis (Novy-Marx & Velikov, 2016):** The transaction cost level at which alpha
disappears is the single most practical number in the thesis. If break-even cost is 3 bps and
realistic costs are 10 bps, the recommender is dead regardless of statistical significance.
Compute per asset (BTC/ETH may survive, LTC/SOL may not).

**Pre-Registration of Honest Assessment (Chambers & Tzavella, 2022):** Pin thresholds before
results: "Delta-Sharpe > 0.15, hit rate improvement > 3pp, ≥ 3/4 assets positive." Any
deviation is documented as post-hoc.

### 13A: Notebook `research/RC4_recommender_evaluation.ipynb`

**Pre-register honest assessment criteria** in first notebook cell:
- "Delta-Sharpe > 0.15 between ML recommender and best baseline"
- "Hit rate improvement > 3 percentage points"
- "≥ 3 out of 4 assets show positive contribution"
- Any deviation from these criteria is documented as post-hoc.

**Baseline ladder** — 5 baselines in one comparison table:

| Baseline | What it tests |
|----------|---------------|
| `RandomRecommender` | Is the recommender better than random? (null hypothesis) |
| `AllAssetsRecommender` | Is filtering better than no filtering? |
| `ClassifierOnlyRecommender` | Does regression add value on top of direction? |
| `RegressorOnlyRecommender` | Does classification add value on top of magnitude? |
| `EqualWeightRecommender` | Is ML better than naive equal-weight positive forecasts? |

Per metric: Sharpe, total return, max drawdown, win rate. This table IS the thesis result.

**Value decomposition chart** — 5 cumulative PnL lines: buy-hold, classifier-only,
regressor-only, combined (no recommender), full ML recommender. Single most important
thesis figure.

**Hypothesis testing:**

*H₁: ML recommender selects assets with higher strategy returns than random selection*
- Permutation test: shuffle asset selections 10000×, compare real mean return vs null
- Report: p-value, effect size

*H₂: ML recommender produces higher portfolio Sharpe than unfiltered deployment*
- Block bootstrap: 95% CI for Sharpe difference (recommended - unfiltered)
- Report: p-value, CI

*H₃: Combined (classifier + regressor + ML recommender) outperforms classifier-only or
regressor-only*
- **Ablation test** — classifier-only vs regressor-only vs both inputs. DM test per ablation.
  This is the key thesis hypothesis.

*H₄: Recommendations are stable (not random noise)*
- Walk-forward consistency: Jaccard similarity of top-K sets across adjacent windows
- Rank correlation of asset scores across adjacent windows

**Break-even cost analysis** per asset (Novy-Marx & Velikov, 2016):
- Compute the fee level at which recommender alpha disappears, per asset
- If break-even < realistic costs → "recommender is not viable for this asset"

**Cross-asset decomposition** — per-asset marginal contribution to portfolio Sharpe.

**Conditional Sharpe ratio analysis** — rolling Sharpe by regime, not just aggregate.
Shows WHERE the recommender adds/destroys value.

**Honest assessment:**
- If recommender Sharpe CI includes 0 → "cannot claim the recommender adds economic value"
- If p-value > 0.05 → "cannot reject null hypothesis"
- If combined ≈ classifier-only → "regression adds no value beyond direction"
- If combined ≈ regressor-only → "classification adds no value beyond implicit direction"
- Document all negative results explicitly

**Estimated scope:** 1 notebook, no artificial cell limits — be as thorough as needed

---

## Phase 14: Statistical Proof & Final Report

**Goal:** Formal statistical tests aggregated into a reproducible report.
Prove the system works on real data AND prove it doesn't work on random data.
Honest power statements throughout.

### Theoretical Foundation

**Monte Carlo Null Hypothesis (Bailey & López de Prado, 2012):** If a strategy profits on
synthetic noise, it is overfit. GBM (constant volatility) is the standard first null, but it
is intentionally too simple for crypto — real crypto has GARCH persistence > 0.95 and jump
diffusion. A strategy exploiting volatility structure will correctly fail on GBM for the
"wrong reason." The proper null for a volatility-aware strategy is GARCH-bootstrapped paths:
resample real returns in blocks, preserving vol clustering but destroying predictive signal.
This tests: "does the strategy detect structure beyond what statistical memory explains?" A
third null — Politis-Romano stationary bootstrap — preserves ALL autocorrelation structure
and is the strongest test.

**Deflated Sharpe Ratio (Bailey & López de Prado, 2014):** DSR corrects observed Sharpe for:
(a) number of strategies tried (N_trials), (b) non-normality (skewness, kurtosis), (c) sample
length. Formula: `P(SR* > 0 | SR_observed, N_trials, skew, kurt, T)`. N_trials must be
honestly exhaustive — include ALL configurations explored across Phases 9-12, all
hyperparameter searches, all RC decision points. Undercounting N is the exact p-hacking DSR
exists to prevent. The trial count for this thesis will likely be 200-500+.

**Minimum Backtest Length (Bailey & López de Prado, 2012 — MBL):** Given observed Sharpe,
non-normality, and desired significance level, MBL computes the minimum test period needed for
a conclusive result. For crypto with kurtosis ~6.7 and a plausible Sharpe of 1.0, MBL may
exceed the available holdout period. If MBL > holdout → "results are indicative but
statistically inconclusive." This is honest science, not failure.

**Holdout Integrity (Cochrane, 2005; Leamer, 1983):** The holdout must be accessed exactly
once, with no iteration. Any model modification after seeing holdout results constitutes data
snooping. The holdout contamination audit is a programmatic check that no 2024+ data touched
any model decision — ~50 lines of code, highest ROI anti-leakage measure.

### 14A-pre: Holdout Contamination Audit

- **`src/app/evaluation/application/holdout_audit.py`** — Automated programmatic check (~50
  lines) that no holdout-period data (2024+) touched any model training or hyperparameter
  decision. Scans MLflow logs, training configs, and data access timestamps. Highest ROI
  anti-leakage measure. Run before any holdout evaluation.

### 14A: Monte Carlo Simulation on Synthetic Data

> **Key sanity check:** If our strategy/recommender "works" on pure random data,
> it's overfitting. A valid system should find NO signal in noise.

- **Pre-register primary hypothesis + correction method** — Monte Carlo on GBM is the
  primary test. Secondary tests (GARCH-bootstrap, Politis-Romano) get Benjamini-Hochberg
  correction.

- **`src/app/evaluation/application/monte_carlo.py`** —
  - **GBM price paths:** N=1000 synthetic paths with μ, σ calibrated to real crypto.
    No exploitable structure by construction.
  - **GARCH-bootstrapped paths:** Resample real returns in blocks, preserving volatility
    clustering but destroying predictive signal. Proper null for vol-aware strategies.
  - **Politis-Romano stationary bootstrap:** Preserves ALL autocorrelation structure.
    Strongest test — if strategy beats this, it uses structure beyond statistical memory.

- **`src/app/evaluation/application/monte_carlo_runner.py`** —
  1. Run the FULL pipeline on each synthetic path: features → classifier → regressor →
     recommender → backtest
  2. Record: Sharpe, total return, accuracy per synthetic path
  3. Build null distributions from the 1000 runs
  4. Compare real-data metrics against null distributions
  5. **Expected:** strategy Sharpe on GBM paths should be ~0
  6. **If strategy is profitable on random data → overfitting alarm**

### 14B: Permutation Tests on Real Data

- **`src/app/evaluation/application/permutation_tests.py`** —
  - **Test 1 — Shuffled returns:** Freeze recommendations, shuffle returns 10000× → null
  - **Test 2 — Shuffled selections:** Keep returns, random K selection 10000× → null
  - **Test 3 — Filtered vs unfiltered:** Permutation test on Sharpe difference
  - **Test 4 — Combined vs single-track:** Permutation on Sharpe(combined) - Sharpe(single)
  - **Test 5 — Strategy on random data (from 14A):** Compare real Sharpe against Monte Carlo null

### 14C: Bootstrap Confidence Intervals

- **`src/app/evaluation/application/bootstrap.py`** —
  - Block bootstrap (preserving autocorrelation via stationary bootstrap)
  - 95% CI for: Sharpe ratio, max drawdown, total return, Sharpe difference
  - If Sharpe CI includes 0 → cannot claim profitability
  - Bootstrap on each test period separately + pooled

### 14D: Walk-Forward Validation & Holdout

- **`src/app/evaluation/application/walk_forward_proof.py`** —
  - Entire pipeline on multiple non-overlapping test periods (2024 H1, 2024 H2, 2025 H1)
  - **Regime characterization of holdout** — 2024+ is post-halving bull. Decompose alpha vs beta.
  - **Evaluate ALL baselines on holdout** — strongest proof: ML beats simple alternatives on
    unseen data.
  - Report all periods honestly (including failures)
  - **Honest power statement:** 3 walk-forward periods = no formal statistical power.
    Frame descriptively, not as hypothesis tests.

### 14E: Model Comparison & DSR

- **`src/app/evaluation/application/model_comparison.py`** —
  - Diebold-Mariano test for each model pair
  - **DSR with honestly exhaustive N_trials** (Bailey & López de Prado, 2014) — count ALL
    configs including RC decision points, hyperparameter searches, model variants. Likely
    200-500+ trials.
  - **Minimum Backtest Length** — compute MBL given observed metrics. If MBL > holdout →
    "indicative but inconclusive."
  - Summary table: model × metric × significance

### 14F: Crypto Event Studies

- **`src/app/evaluation/application/event_studies.py`** —
  - **What did the system do during:** FTX collapse, Luna/UST crash, COVID March 2020,
    post-halving rally?
  - Per-event: positions held, recommendations made, PnL during event window.
  - Not statistical tests — narrative evidence that builds credibility.
  - "During the FTX collapse, the recommender [deployed/abstained]. The result was [X]."

### 14G: Tests

- Monte Carlo: strategy on 1000 GBM paths → Sharpe distribution centered at ~0
- Monte Carlo: strategy on paths with injected autocorrelation=0.10 → should detect signal
- Permutation test on synthetic "known profitable" recommender → should reject null
- Permutation test on random recommender → should NOT reject null
- Holdout audit: verify no 2024+ data in training configs

**Dependencies:** All previous phases
**Estimated scope:** ~14 files, ~1200 lines

---

---

# BLOCK III: Polishing & Production

> Everything below is post-thesis-validation. Only proceed after Phases 1–14 are complete
> and the research results are satisfactory. This block turns the research prototype into
> a production-grade paper trading system.

---

## Phase 15: Pipeline Runner & Integration

**Goal:** Single pipeline runner class that orchestrates the full pipeline from ingestion to
recommendation, with simple retry, resume, and logging. No production infrastructure
over-engineering.

### Theoretical Foundation

**YAGNI Principle + Thesis Scope (Beck, 2000):** Dead letter queues, circuit breakers,
rollback mechanisms, and separate checkpointing layers are production infrastructure patterns
for distributed systems. This thesis has one process, one database, four assets, and needs to
run for 72 hours. Every hour spent on production patterns is an hour NOT spent on statistical
analysis — which is what the committee evaluates.

**Idempotent Pipeline Steps (Kleppmann, 2017):** Each pipeline step writes to DuckDB. The
table IS the checkpoint. On restart, query the max processed timestamp and resume. This is
the append-only, idempotent pattern that eliminates the need for separate state management.

### 15A: Pipeline Runner

- **`src/app/system/pipeline/runner.py`** — `PipelineRunner` class (~200 LOC):
  - Sequential steps: ingest → bars → features → classify → regress → recommend → backtest
  - Retry with exponential backoff on transient failures
  - Log timing per step (Loguru structured logging)
  - Resume from high-water mark (query DuckDB for last processed timestamp)
  - SIGINT handler for graceful shutdown

- **`src/app/system/pipeline/protocols.py`** — `PipelineStep` Protocol:
  - `run() -> None`
  - `last_processed_timestamp() -> datetime | None`

- **`src/app/system/pipeline/config.py`** — `PaperTradingConfig` Pydantic settings:
  - Cycle interval, assets, bar types, model paths, commission settings

### 15B: Canary Run

- Run full pipeline once on historical data, verify output matches backtest results.
- This is the integration test: "does the pipeline produce the same numbers as the
  research notebooks?"

### 15C: Tests

- Integration test: pipeline runs end-to-end on small dataset
- Resume test: kill mid-run → restart → verify resumes from high-water mark
- Config validation test: invalid config raises immediately

**Dependencies:** Phases 1-14 (working pipeline)
**Estimated scope:** ~5 files, ~300 lines

---

## Phase 16: Live Paper Trading Engine

**Goal:** Real-time paper trading system that connects to Binance live data,
runs the full pipeline, and executes trades with simulated money on real market data.

### Theoretical Foundation

**Reconciliation Testing (Lopez, 2018; Tulchinsky et al., 2019):** The most insidious
production bug in quantitative systems is silent divergence between backtest and live feature
computation. A model trained on batch-computed features will produce garbage predictions on
slightly different live features. The reconciliation test — replay historical data through
the live pipeline, diff against batch output — is the standard proof of correctness.

**Flight Recorder Pattern (Leveson, 2011):** Logging every raw WebSocket message before
processing creates an immutable audit trail. Without it, a bug in live is unreproducible.
With it, any live session can be replayed deterministically for debugging.

### 16A: Real-Time Data Feed

- **`src/app/live/infrastructure/binance_websocket.py`** — WebSocket connection to Binance
  for real-time kline/candle data. Auto-reconnect on disconnect (handle Binance 24h forced
  disconnects). REST backfill on reconnect to fill gaps.
- **`src/app/live/infrastructure/flight_recorder.py`** — Log all raw WebSocket messages to
  DuckDB/parquet before processing. Immutable audit trail.

### 16B: Real-Time Bar Construction & Features

- **`src/app/live/application/bar_builder.py`** — `LiveBarBuilder`: receives raw candles from
  data feed, constructs bars in real-time. Emits completed bars.
- **`src/app/live/application/feature_engine.py`** — `LiveFeatureEngine`: maintains rolling
  windows, computes features incrementally on each new bar.

### 16C: Paper Trading Execution

- **`src/app/live/domain/value_objects.py`** —
  - `PaperAccount` — starting_balance, current_balance, positions, trade_history, equity_history
  - `PaperOrder` — asset, side, size, order_type, timestamp, fill_price (simulated)
  - `LiveTradeConfig` — initial_capital, max_position_pct, commission_bps

- **`src/app/live/application/paper_broker.py`** — `PaperBroker`: simulates order execution
  against live market prices. Applies commission model (same as backtest engine). Persists all
  trades and equity snapshots to DuckDB.

- **`src/app/live/application/trading_loop.py`** — `LiveTradingLoop`: main event loop.
  1. Receive new bar → compute features → classify → regress → recommend
  2. If deploy: generate order → execute via PaperBroker
  3. Log everything, persist state to DuckDB (crash-recoverable)
  4. **Heartbeat + drift monitor** — every 60s log status

### 16D: Model Loading & Validation

- **`src/app/live/application/model_loader.py`** — Load trained models from MLflow.
  Feature contract validation: verify live features match training schema (~20 lines).

### 16E: Session State

- **`paper_trading_sessions` DuckDB table** — session ID, start time, end time, equity curve,
  trades, model versions used. One table, no separate state management.

### 16F: Reconciliation Test (Shadow Mode)

- Replay 1 week of historical data through live pipeline, diff feature values and predictions
  against batch-computed results. Maximum acceptable divergence: 1e-6 relative error.

### 16G: Tests

- Integration test: feed historical data through LiveTradingLoop → verify trades match
  backtest results
- Reconnection test: simulate WebSocket disconnect → verify auto-reconnect + REST backfill
- Crash recovery test: kill loop → restart → verify resumes from DuckDB state
- Reconciliation test: live vs batch feature parity

**Dependencies:** Phase 15 (pipeline runner), all model phases
**Estimated scope:** ~12 files, ~1000 lines

---

## Phase 17: Dashboard Frontend

**Goal:** Read-only Streamlit dashboard for monitoring paper trading and thesis defense demo.
Five panels with clear information hierarchy.

### Theoretical Foundation

**Demo Risk Mitigation:** A thesis defense demo that crashes is worse than no demo. The
`--demo` flag loading a curated DuckDB snapshot eliminates the dependency on a running live
loop. Static fallback from the last paper trading session guarantees the demo works.

**Information Hierarchy (Tufte, 2001; Few, 2012):** A dashboard with 10 panels is a dashboard
with 0 panels. Five panels with clear hierarchy: (1) summary statistics card, (2) equity curve
with benchmarks and regime overlay, (3) recommendations with explanations, (4) trade log,
(5) model health.

### 17A: Streamlit Dashboard (read-only, 5 panels)

- **`src/app/dashboard/app.py`** — Single Streamlit application:

  **Panel 1: Summary Statistics Card** (first-glance verdict)
  - Key metrics: total return, Sharpe, max drawdown, win rate, # trades
  - Comparison vs buy-and-hold in same card

  **Panel 2: Equity Curve + Benchmarks + Regimes**
  - Equity curve with buy-and-hold overlay
  - **Regime overlay** from Phase 5C labels (color bands: low/normal/high vol)
  - **Comparison view** — ML recommender vs baselines cumulative PnL
  - This is the thesis argument visualized

  **Panel 3: Recommendations with Explanations**
  - Current asset recommendations with top-3 SHAP drivers per recommendation
  - Deploy/skip decision with confidence and predicted return
  - Shows interpretability of the recommender

  **Panel 4: Trade Log**
  - Scrollable table of recent trades (last 50)
  - Columns: timestamp, asset, side, entry/exit price, PnL, commission
  - Color-coded profit/loss

  **Panel 5: Model Health**
  - Active model versions with last retrain date
  - Data staleness indicator (time since last bar)
  - Feature drift monitor (simple distribution shift detection)

### 17B: Demo Mode & Export

- **`--demo` flag** — loads pre-baked DuckDB snapshot with curated paper trading results.
  Eliminates live dependency for thesis defense.
- **Exportable thesis figures** (PNG/SVG, publication-quality) — every chart has an export
  button for inclusion in the thesis document.

### 17C: Configuration

- Dashboard reads from the same DuckDB database as the trading engine
- No separate data store — single source of truth
- Configurable refresh interval (default: 30 seconds)

### 17D: Tests

- Dashboard renders without errors on empty data (no trades yet)
- Demo mode loads and displays correctly
- Export produces valid PNG/SVG files

**Dependencies:** Phase 16 (live trading engine), `streamlit`
**Estimated scope:** ~4 files, ~500 lines

---

## Implementation Order & Dependencies

```
═══════════════════════════════════════════════════════════════════════════
 BLOCK I: Data & Infrastructure
═══════════════════════════════════════════════════════════════════════════

Phase 1  (Ingestion) ✅ ───────────────────────────────────────────────────
    │
    ├── Phase 2  (Bars) ✅
    │       │
    │       └── Phase 3  [RC1: Data & Bar Analysis] ✅ COMPLETED
    │               Decisions: 4 assets, 5 bar types (dollar, volume,
    │               volume_imbalance, dollar_imbalance, time_1h baseline)
    │
    ├── Phase 4  (Features + Targets: classification & regression) ✅
    │       │
    │       ├── Phase 5  (Profiling) ◄── IN PROGRESS
    │       │       │
    │       │       └── Phase 6  [RC2: Features & Data Adequacy] ◄── RESEARCH STOP
    │       │
    │       └── Phase 7  (Backtest Engine)
    │               │
    │               └── Phase 8  (Base Strategies)

═══════════════════════════════════════════════════════════════════════════
 BLOCK II: Models, Recommendation & Proof
═══════════════════════════════════════════════════════════════════════════

    ├── Phase 9  (Direction Classification) ─────┐
    │                                             │
    ├── Phase 10 (Return Regression) ─────────────┤
    │                                             │
    │       └── Phase 11 [RC3: Classification     │
    │            & Regression Evaluation]          │
    │            ◄── RESEARCH STOP                │
    │                                             │
    ├── Phase 12 (ML Recommendation System) ◄─────┘
    │       │     (consumes BOTH tracks)
    │       │
    │       └── Phase 13 [RC4: Recommender Evaluation] ◄── RESEARCH STOP
    │
    └── Phase 14 (Statistical Proof + Monte Carlo)

═══════════════════════════════════════════════════════════════════════════
 BLOCK III: Polishing & Production (post-validation)
═══════════════════════════════════════════════════════════════════════════

    Phase 15 (Pipeline Runner) ──── sequential steps, retry, resume
    │                                from high-water mark, ~200 LOC
    │
    Phase 16 (Live Paper Trading) ── WebSocket feed, real-time bars,
    │                                paper broker, flight recorder
    │
    Phase 17 (Dashboard Frontend) ── read-only Streamlit, 5 panels,
                                     --demo mode, exportable figures

    RC = Research Checkpoint (notebook-driven, collaborative review)
```

**Current status (2026-03-19):** Phases 1–4 complete. Phase 5 in progress.

**Parallel work possible:**
- Phases 5 + 7 can proceed in parallel after Phase 4
- **Phases 9 + 10 can proceed in parallel** (classification and regression are independent tracks)
- Phase 11 (RC3) evaluates both tracks together, compares and combines
- Phase 12 needs Phases 7, 8, 9, 10 all done
- **Block III only starts after Phase 14** (all research validated first)

**Critical path:** Phase 5 → 6 → 9/10 → 11 → 12 → 13 → 14

**Estimated remaining:** ~5500 production LOC + ~5500 test LOC + ~950 notebook cells

---

## Per-Phase Deliverables

| Phase | Type | Deliverable | Status |
|-------|------|------------|--------|
| | | **BLOCK I: Data & Infrastructure** | |
| 1 | Build | DuckDB filled with multi-asset OHLCV data | ✅ Done |
| 2 | Build | Alternative bar types computed and stored | ✅ Done |
| **3** | **Research** | **RC1 notebook: data quality charts, bar comparison, coverage** | **✅ Done** |
| 4 | Build | Feature matrix + both targets (direction + return) | ✅ Done |
| 5 | Build | Statistical profile per asset | In progress |
| **6** | **Research** | **RC2 notebook: VIF, feature interactions, regime MI, economic significance, go/no-go** | **Yes: charts, go/no-go decision** |
| 7 | Build | Backtest engine (~500 LOC, fill-on-next-open, Lo Sharpe) | Yes: equity curves, PnL |
| 8 | Build | 3 diverse strategies (momentum, Donchian, mean reversion) | Yes: performance table |
| | | **BLOCK II: Models, Recommendation & Proof** | |
| 9 | Build | Direction classifiers (logistic, RF, LightGBM + GRU neg. result) | Yes: accuracy, abstention curves |
| 10 | Build | Return regressors (Ridge, LightGBM quantile, GRU+MCDropout, ARIMA) | Yes: DC-MAE, CRPS, uncertainty |
| **11** | **Research** | **RC3: 3 notebooks (clf, reg, combined), pre-registered, power analysis** | **Yes: dual-track comparison** |
| 12 | Build | ML recommendation system (generalized meta-labeling, position sizing) | Yes: SHAP, ablation, conformal |
| **13** | **Research** | **RC4 notebook: baseline ladder, break-even, value decomposition** | **Yes: final results, p-values** |
| 14 | Build | Statistical proof (GBM + GARCH null, DSR, holdout audit, event studies) | Yes: p-values, null distributions |
| | | **BLOCK III: Polishing & Production** | |
| 15 | Build | Pipeline runner (~300 LOC, retry, resume from high-water mark) | Yes: canary run |
| 16 | Build | Live paper trading (flight recorder, reconciliation test) | Yes: running system, live trades |
| 17 | Build | Read-only Streamlit dashboard (5 panels, --demo mode) | Yes: thesis defense demo |

---

## Key Design Decisions

1. **Two-track forecasting: classification + regression** — Classification predicts direction (SIDE), regression predicts magnitude (SIZE). Combined: richer signal than either alone. Inspired by López de Prado's meta-labeling, generalized from binary to continuous.
2. **Regression metrics are direction-conditional** — Raw MAE/RMSE/R² are misleading in finance. Regression is evaluated ONLY when direction is correct (DC-MAE, DC-RMSE). Economic Sharpe is the ultimate metric. A model with low MAE but DA ≤ 50% is useless.
3. **ML recommendation system** — The recommender is a trained LightGBM model consuming BOTH classifier + regressor outputs. Has its own multi-layer walk-forward pipeline with stacking leakage prevention. Testable hypotheses: H₁–H₄, including H₃ which tests whether combining both tracks beats each alone. Generalized meta-labeling: predicts expected strategy return (continuous), enabling position sizing.
4. **Research checkpoints** — 4 explicit stops (RC1–RC4) where we analyze data, produce charts, and make informed decisions before proceeding. Prevents building on bad foundations.
5. **Honest evaluation** — DA ≤ 50%, non-significant p-values, and Sharpe CIs including zero are valid and documented results.
6. **Polars for ETL, Pandas for experiments, NumPy for math** — Polars in pipelines (ingestion, bars, backtest, live) for performance. Pandas in research notebooks and model training for ML ecosystem compatibility. NumPy for vectorized numerical computations (indicators, bootstrap, Monte Carlo).
7. **Pydantic everywhere, no dataclasses** — All value objects, configs, DTOs use `BaseModel` with validation, serialization, and `frozen=True` immutability. No raw `dataclass` usage.
8. **DuckDB for storage** — Analytical queries, Parquet integration, no server needed.
9. **Protocol-based DI** — All modules depend on protocols, not concrete implementations.
10. **Config-driven** — Every parameter in Pydantic config classes, no magic numbers.
10. **MLflow for tracking** — Experiments, model registry, artifact storage.
11. **No future leakage** — Enforced by `TemporalSplit` value object + `.shift(1)` convention. Walk-forward throughout.
12. **Monte Carlo validation** — Strategy must NOT be profitable on synthetic GBM paths OR GARCH-bootstrapped paths. Deflated Sharpe Ratio with honestly exhaustive N_trials.
13. **Minimal production infrastructure** — Pipeline runner (~200 LOC), flight recorder, reconciliation testing. No circuit breakers, dead letter queues, or Prometheus — YAGNI for a thesis.
14. **Dashboard for observability** — Read-only Streamlit with 5 panels, --demo mode for thesis defense. Publication-quality exportable figures. Single source of truth (DuckDB).
15. **Google-style docstrings everywhere** — Every public module, class, method, function. Enforced by Ruff `D` + `DOC` rules. No code merges without docstrings.
16. **Python 3.14 type hints** — Modern syntax (`list[X]`, `X | None`, `type` aliases, `Self`, `Never`). Pyright strict mode with zero errors. Enforced on every commit.
17. **Pre-commit quality gates** — Ruff formatter → Ruff linter → Pyright strict → isort. All from `pyproject.toml`. Failed hook = rejected commit.
18. **Alembic for all schema changes** — Every DuckDB table/column/index change goes through a versioned, reversible Alembic migration. No raw DDL outside migrations.
