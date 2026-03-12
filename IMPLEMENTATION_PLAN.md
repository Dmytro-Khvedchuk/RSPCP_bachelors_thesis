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

**Goal:** Generate statistical profile per asset to understand data properties and justify modeling choices.

> **RC1 overlap:** Phase 3 already computed return distributions (JB test, kurtosis, skewness),
> ACF/PACF analysis, and Ljung-Box tests for BTCUSDT across all bar types. Phase 5 extends
> this to all 4 assets, adds GARCH modeling, alternative distribution fitting, and the
> Lo-MacKinlay variance ratio test. Phase 5 services should reuse the RC1 analysis classes
> from `src/app/research/application/` where applicable (e.g., `ReturnAnalyzer`,
> `AutocorrelationAnalyzer`) and extend them for the profiling-specific tests below.

### 5A: Return Distribution Analysis

- **`src/app/profiling/application/distribution.py`** — Per-asset, **per-bar-type** (dollar, volume, volume_imbalance, dollar_imbalance, time_1h):
  - Log return computation (already done in RC1 for BTCUSDT — extend to all 4 assets)
  - Normality tests: Jarque-Bera (already in RC1), Shapiro-Wilk, Anderson-Darling
  - Alternative distribution fitting: Student-t, Generalized Hyperbolic
  - Model comparison: AIC, BIC, KS goodness-of-fit
  - Output: `DistributionProfile` Pydantic model

### 5B: Autocorrelation & Serial Dependence

- **`src/app/profiling/application/autocorrelation.py`** — Per-asset, **per-bar-type**:
  - ACF/PACF of returns and squared returns (extend RC1's BTCUSDT results to all 4 assets)
  - Ljung-Box test for serial correlation (extend RC1 results)
  - **Lo-MacKinlay variance ratio test** at multiple horizons q = {2, 5, 10, 20} — not yet done in RC1
  - Output: `AutocorrelationProfile` with test statistics and p-values

### 5C: Volatility Modeling

- **`src/app/profiling/application/volatility.py`** — Per-asset:
  - GARCH(1,1) and GJR-GARCH fit
  - Residual diagnostics (i.i.d. tests)
  - Rolling realized volatility
  - Output: `VolatilityProfile`

### 5D: Profiling Service

- **`src/app/profiling/application/services.py`** — `ProfilingService` that runs all analyses for a list of assets, aggregates into a `StatisticalReport` DataFrame, logs/saves results.

### 5E: Tests

- Unit test: each test against known synthetic data (e.g., normal data should not reject Jarque-Bera)
- Sanity test: GARCH on constant series should have near-zero alpha/beta

**Dependencies:** Phase 1 (data), `scipy.stats`, `statsmodels`, `arch`
**Estimated scope:** ~8 files, ~600 lines

---

## Phase 6: Research Checkpoint 2 — Features, Profiling & Data Adequacy

**Goal:** Deep analysis of features and statistical properties. Answer: "Is our data
sufficient and do our features carry signal for return regression?"

### 6A: Notebook `research/RC2_features_and_profiling.ipynb`

**Feature exploration:**
- Feature correlation matrix → heatmap (identify redundant features)
- Feature distributions → violin plots, detect skew/outliers
- Feature-target scatter plots (feature vs forward_log_return) → visual inspection of relationships
- Mutual information results table with p-values → which features pass validation?
- Single-feature Ridge DA table → any feature alone predict direction better than 50%?
- Stability heatmap: feature × year → which features are stable across time?

**Statistical profiling results:**
- Return distribution per asset: histogram + fitted Student-t overlay
- Table: Jarque-Bera p-value, kurtosis, skewness, best-fit distribution per asset
- ACF/PACF plots for top-5 assets → exploitable autocorrelation?
- Ljung-Box results table → serial dependence significance
- GARCH parameter table (alpha, beta, omega) per asset
- Rolling volatility time series per asset → regime identification

**Data adequacy assessment:**
- Sample size per asset (after warmup/NaN removal)
- Effective sample size (accounting for autocorrelation)
- Signal-to-noise ratio estimate: how much variance do features explain?
- Power analysis: given sample size and effect size, can we detect real effects?
- Cross-asset consistency: do features behave similarly across assets?

**Decision output:**
- Final feature set (keep/drop decisions with justification)
- Final asset universe (drop assets with insufficient data or anomalous properties)
- Appropriate forecast horizons (based on autocorrelation analysis)
- Is regression feasible? Expected DA range? Is DA > 50% achievable?
- Bar type confirmation (RC1 selected: dollar, volume, volume_imbalance, dollar_imbalance + time_1h baseline — RC2 validates whether feature quality supports this selection or narrows it further)

**Estimated scope:** 1 notebook, ~400 cells

---

## Phase 7: Backtest Engine

**Goal:** Event-driven backtesting engine with realistic execution modeling
(slippage, commissions, position management).

### 7A: Core Domain Model

- **`src/app/backtest/domain/value_objects.py`** —
  - `Side` enum (LONG, SHORT)
  - `OrderType` enum (MARKET, LIMIT)
  - `ExecutionConfig` — slippage model choice + commission model choice + max position size + margin requirements. All adjustable via config.
  - `TradeResult` — entry_price, exit_price, side, size, entry_time, exit_time, gross_pnl, net_pnl, commission_paid, slippage_cost
  - `PortfolioSnapshot` — timestamp, equity, cash, positions, unrealized_pnl, drawdown

- **`src/app/backtest/domain/entities.py`** —
  - `Position` — asset, side, size, entry_price, entry_time, unrealized_pnl, stop_loss, take_profit
  - `Trade` — full lifecycle of a position from open to close
  - `EquityCurve` — time series of portfolio value

- **`src/app/backtest/domain/protocols.py`** —
  - `ISlippageModel` protocol — `estimate(price, size, volatility) -> slipped_price`
  - `ICommissionModel` protocol — `calculate(price, size, side) -> commission`
  - `IStrategy` protocol — `on_bar(timestamp, features, portfolio) -> list[Signal]`

### 7B: Execution Models

- **`src/app/backtest/application/slippage.py`** —
  - `FixedBpsSlippage` — constant basis points (default 5 bps)
  - `VolatilityScaledSlippage` — slippage proportional to recent ATR
  - `ZeroSlippage` — for comparison baseline

- **`src/app/backtest/application/commission.py`** —
  - `FixedRateCommission` — flat rate (default 0.1% maker/taker, Binance standard)
  - `TieredCommission` — volume-based tiers (Binance VIP levels)
  - `ZeroCommission` — for comparison baseline

### 7C: Backtest Engine

- **`src/app/backtest/application/engine.py`** — `BacktestEngine`:
  - Iterates through bars chronologically
  - On each bar: update positions (mark-to-market), check stop-loss/take-profit, call strategy.on_bar(), execute resulting signals with slippage + commission
  - Tracks equity curve, all trades, portfolio snapshots
  - Supports: long-only, long-short, single-asset, multi-asset
  - No lookahead: strategy only sees data up to current bar

### 7D: Performance Metrics

- **`src/app/backtest/application/metrics.py`** —
  - **Return metrics:** total return, annualized return, CAGR
  - **Risk metrics:** max drawdown, drawdown duration, annualized volatility, downside volatility
  - **Risk-adjusted:** Sharpe ratio, Sortino ratio, Calmar ratio
  - **Trade metrics:** win rate, profit factor, avg win/loss ratio, max consecutive losses
  - **Statistical:** Sharpe significance (Lo 2002 adjustment for autocorrelation)
  - All metrics computed on the equity curve / trade list

### 7E: Walk-Forward Framework

- **`src/app/backtest/application/walk_forward.py`** — `WalkForwardRunner`:
  - Configurable window: expanding or rolling
  - Monthly/quarterly rebalancing
  - For each window: fit strategy on train, generate signals on test, run through BacktestEngine
  - Aggregate results across windows
  - Output: per-window metrics + aggregate metrics

### 7F: Tests

- Unit test: slippage/commission models on known inputs
- Integration test: run trivial strategy (always long) through engine, verify equity curve matches manual calculation with known slippage + commission
- Regression test: deterministic strategy on fixed data → known PnL
- Edge cases: zero-volume bars, gaps, first/last bar handling

**Dependencies:** Phase 1 (data), Phase 4 (features for strategies)
**Estimated scope:** ~14 files, ~1200 lines

---

## Phase 8: Base Trading Strategies

**Goal:** Implement simple, interpretable strategies as baselines. The thesis focus is
the recommendation layer, not the strategy itself.

### 8A: Strategy Interface

- **`src/app/strategy/domain/protocols.py`** — `IStrategy` protocol:
  ```python
  def fit(train_features: pl.DataFrame) -> None
  def on_bar(timestamp, bar_features, portfolio) -> list[Signal]
  def name() -> str
  ```

### 8B: Strategies

- **`src/app/strategy/application/momentum_crossover.py`** — EMA crossover with ATR-based stops. Long when fast > slow, short when fast < slow. Parameters: fast_period, slow_period, atr_multiplier_sl, atr_multiplier_tp.

- **`src/app/strategy/application/dual_regime_trend.py`** — Port DRTS from legacy: volatility regime filter + dual EMA/slope confirmation + confidence scoring.

- **`src/app/strategy/application/mean_reversion.py`** — Bollinger band bounce: enter when price crosses below lower band (long) or above upper band (short), exit at mean. Hurst filter: only trade when H < 0.5.

### 8C: Tests

- Unit test: each strategy on synthetic trending/mean-reverting data
- Backtest: run each strategy through engine on historical data

**Dependencies:** Phase 4 (features), Phase 7 (backtest engine)
**Estimated scope:** ~6 files, ~400 lines

---

## Phase 9: Direction Classification Models

**Goal:** Predict the **direction** of future price movement (up/down).
This is the first forecasting track — classification determines the **side** of the trade.

> **Two-track forecasting design (inspired by López de Prado's meta-labeling):**
> - **Track 1 (Phase 9):** Classification → predict direction (up/down) → determines SIDE
> - **Track 2 (Phase 10):** Regression → predict return magnitude → determines SIZE
> - **Combined:** Classification picks the side, regression estimates how much.
>   The recommendation system (Phase 12) consumes BOTH outputs.

### 9A: Classification Domain

- **`src/app/forecasting/domain/value_objects.py`** —
  - `ForecastHorizon` enum (H1, H4, H24)
  - `DirectionForecast` — predicted_direction (+1/-1), confidence (probability), horizon
  - `ReturnForecast` — predicted_return (point estimate), prediction_std, quantiles, confidence_interval
- **`src/app/forecasting/domain/protocols.py`** —
  - `IDirectionClassifier` protocol: `fit(X, y_direction)`, `predict(X) -> list[DirectionForecast]`
  - `IReturnRegressor` protocol: `fit(X, y_return)`, `predict(X) -> list[ReturnForecast]`

### 9B: Classification Target Construction

- **`src/app/features/application/targets.py`** — Extend with classification targets:
  - `forward_direction(horizon)` = sign(close_{t+h} - close_t) → +1 or -1
  - Triple barrier labeling (López de Prado): upper profit barrier, lower stop-loss barrier, time barrier → +1/-1/0. Configurable barrier widths (ATR-based).
  - Both target types available for comparison: simple sign vs triple barrier

### 9C: Baseline Classifiers

- **`src/app/forecasting/application/logistic_baseline.py`** — Logistic regression. Interpretable baseline. Outputs calibrated probabilities.
- **`src/app/forecasting/application/random_forest_clf.py`** — Random Forest classifier. Non-linear, handles feature interactions. Feature importance for interpretability.
- **`src/app/forecasting/application/gradient_boosting_clf.py`** — LightGBM classifier. Strong tabular baseline. Outputs calibrated probabilities via Platt scaling.

### 9D: Deep Learning Classifiers

- **`src/app/forecasting/application/gru_classifier.py`** — GRU encoder (2 layers, 64-128 hidden) → sigmoid/softmax head for direction probability. Loss: binary cross-entropy. Sequence input captures temporal patterns.
- **`src/app/forecasting/application/transformer_classifier.py`** — Transformer encoder with classification head. Attention weights show which time steps matter most for direction prediction.

### 9E: Classification Metrics

- **`src/app/forecasting/application/classification_metrics.py`** —
  - **Accuracy:** % correct direction predictions — must beat 50% (coin flip baseline)
  - **Precision / Recall / F1** per class (up/down): is the model biased toward one direction?
  - **Precision@confidence:** among predictions with P > 0.6 (or 0.7, 0.8), what % are correct? Higher confidence should mean higher precision.
  - **Calibration:** predicted probability vs actual frequency (reliability diagram). P(up) = 0.7 should mean ~70% of those are actually up.
  - **AUC-ROC:** discrimination ability regardless of threshold
  - **Economic accuracy:** accuracy weighted by |actual return| — being right on big moves matters more than being right on tiny moves
  - **Confusion matrix with return overlay:** for each cell (TP/TN/FP/FN), show average |actual return| — how costly are the errors?

### 9F: Tests

- Unit test: logistic regression on linearly separable data → near-perfect accuracy
- Convergence test: GRU loss decreases over epochs
- Calibration test: predicted probabilities match observed frequencies
- Null test: model trained on shuffled labels → accuracy ≈ 50%
- Economic test: accuracy on days with |return| > 1% (big moves) vs small moves

**Dependencies:** Phase 4 (features + targets), `lightgbm`, `pytorch`
**Estimated scope:** ~10 files, ~800 lines

---

## Phase 10: Return Regression Models

**Goal:** Predict the **magnitude** of future price movement (how much will it move).
This is the second forecasting track — regression determines the SIZE of the position.

> **Key principle:** Regression metrics (MAE, RMSE, R²) are only meaningful WHEN the direction
> classifier is correct. A model predicting +2% when the actual move is -3% has low MAE but is
> useless. **Regression is evaluated conditionally on correct direction.**

### 10A: Baseline Regressors

- **`src/app/forecasting/application/ridge_baseline.py`** — Ridge regression. Simple, fast, interpretable. Provides point estimate + residual std for uncertainty.
- **`src/app/forecasting/application/arima_garch.py`** — ARIMA for conditional mean + GARCH for conditional variance → Gaussian predictive distribution. Auto order selection via AIC.
- **`src/app/forecasting/application/quantile_regression.py`** — Quantile regression at tau = {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}. Non-parametric uncertainty estimation.
- **`src/app/forecasting/application/gradient_boosting_reg.py`** — LightGBM/XGBoost regressor. Quantile regression mode for uncertainty.

### 10B: Deep Learning Regressors

- **`src/app/forecasting/application/gru_regressor.py`** — GRU encoder → linear head for predicted return. Optionally: MDN head (K=3-5 Gaussians) for multimodal return distributions. Loss: NLL.
- **`src/app/forecasting/application/transformer_regressor.py`** — TFT or simpler Transformer encoder. Multi-horizon output. Attention weights for interpretability.
- **`src/app/forecasting/application/mc_dropout.py`** — Wrapper around any DL model. Dropout at inference, 100 forward passes → epistemic vs aleatoric uncertainty decomposition.

### 10C: Calibration & Conformal Prediction

- **`src/app/forecasting/application/calibration.py`** —
  - Reliability diagrams: predicted quantile q → actual coverage ≈ q
  - Conformal prediction wrapper: guaranteed coverage ≥ (1-α)
  - Residual diagnostics: homoscedasticity, normality

### 10D: Regression Metrics (Direction-Conditional)

> **These metrics are ALWAYS reported conditional on the direction classifier's prediction.**
> Two evaluation modes:
> - **Standalone regression:** evaluate on all samples, but report DA alongside MAE/RMSE
> - **Pipeline regression:** evaluate ONLY on samples where direction classifier was correct

- **`src/app/forecasting/application/regression_metrics.py`** —
  - **DC-MAE (Direction-Conditional MAE):** MAE only where sign(predicted) == sign(actual). "When you get the direction right, how accurate is the magnitude?"
  - **DC-RMSE:** Same, penalizes large errors on correct-direction predictions
  - **WDL (Wrong-Direction Loss):** Average |predicted - actual| where direction is wrong. Quantifies cost of direction errors.
  - **PDR (Profitable Direction Ratio):** When predicted return > threshold AND direction correct, what is avg realized return?
  - **CRPS:** Full distributional metric — captures both direction and magnitude. Primary metric for probabilistic forecasters.
  - **Economic Sharpe:** Sharpe of long/short strategy: direction classifier picks side, regressor sizes position by |predicted return|. **The ultimate combined metric.**

### 10E: Tests

- Unit test: Ridge on linear data → low DC-MAE
- Convergence test: GRU loss decreases
- Calibration test: conformal intervals achieve target coverage
- Null test: regressor on noise → DC-MAE ≈ unconditional MAE (no improvement)

**Dependencies:** Phase 4 (features + targets), Phase 9 (direction classifier for conditional eval), `arch`, `statsmodels`, `lightgbm`, `pytorch`, `mapie`
**Estimated scope:** ~12 files, ~1200 lines

---

## Phase 11: Research Checkpoint 3 — Classification & Regression Evaluation

**Goal:** Evaluate BOTH forecasting tracks. Compare classification vs regression approaches.
Can the classifier beat a coin flip? Does regression add value on top of correct direction?
Are the two complementary?

### 11A: Notebook `research/RC3_classification.ipynb`

**Classification evaluation:**
- Per-model, per-asset, per-horizon: accuracy, precision, recall, F1, AUC-ROC
- Comparison table: all classifiers × all assets → which model wins?
- **Accuracy > 50% is the minimum bar** — binomial test: p-value for accuracy > 0.5
- Confusion matrix per model with **average |return|** overlay per cell (how costly are FP/FN?)
- Calibration plots: predicted probability vs actual frequency per model
- Precision@confidence curves: does higher confidence → higher precision?
- **Economic accuracy:** accuracy weighted by move magnitude
- Simple sign vs triple barrier labeling: which target produces better models?

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
- Scatter plots: predicted vs actual return, **only for correct-direction samples** (this is the signal)
- Scatter plots: predicted vs actual for wrong-direction samples (this is the noise we filter out)
- **PDR:** when classifier says "up" AND regressor says "> +1%", how often is realized return positive and > 0.5%?

**Economic evaluation (combined pipeline):**
- **Economic Sharpe:** classifier picks side, regressor sizes position → equity curve
- Compare: classifier-only (equal size), regressor-only (sign determines side), combined
- Profit factor: gross profit from correct trades / gross loss from incorrect

**Uncertainty evaluation:**
- Calibration plots (reliability diagrams) per regressor
- CRPS per model
- Conformal interval coverage (overall + direction-conditional)
- Interval sharpness: narrower is better at same coverage

**Statistical comparison:**
- Diebold-Mariano test: pairwise CRPS differences, H₀: equal predictive ability
- Model Confidence Set (Hansen 2011)
- **Combined vs separate test:** Does (classifier + regressor) significantly beat classifier-only? DM test on Economic Sharpe.

**Data adequacy:**
- Is any classifier's accuracy significantly > 50%? (binomial test)
- Is any regressor's DC-MAE significantly lower than unconditional MAE? (permutation test)
- Is Economic Sharpe of combined pipeline significantly > 0? (bootstrap CI)
- Cross-asset generalization: train on BTC, test on ETH

**Decision output:**
- Select best classifier (or top-2) for the recommendation system
- Select best regressor (or top-2) for the recommendation system
- Confirm: does the combined pipeline (classifier + regressor) outperform each alone?
- Determine best forecast horizon
- Assets where accuracy ≈ 50% → classification hopeless, flag for recommender

**Estimated scope:** 2 notebooks, ~500 cells total

---

## Phase 12: ML Recommendation System

**Goal:** Train a machine learning model that learns to predict which assets the base strategy
will perform well on, given current market features and **both classification + regression forecasts**.

> **Key insight:** The recommender IS the ML model. It is not a formula or heuristic.
> It consumes BOTH forecasting tracks: (1) classifier's direction + confidence, (2) regressor's
> magnitude + uncertainty. It has training data, a loss function, train/val/test splits,
> and testable hypotheses.
>
> **This is generalized meta-labeling (López de Prado):** the primary models (classifier + regressor)
> generate signals, and the recommender is the secondary model that decides WHETHER to deploy
> the strategy and HOW to size the position, based on predicted strategy performance.

### 12A: Training Data Construction

- **`src/app/recommendation/application/label_builder.py`** — For each (asset, time_window):
  1. Run the base strategy on that asset during that window (using backtest engine)
  2. Record the realized strategy return (net of slippage + commission)
  3. This realized return is the **regression target** for the recommender
  4. Alternatively: risk-adjusted return (Sharpe over the window) as target
  - Walk-forward: only use past data for features, future window for labels

- **`src/app/recommendation/application/feature_builder.py`** — Features for the recommender (per asset, per timestamp):
  - **Market state features:** all features from Phase 4 (volatility, momentum, Hurst, etc.)
  - **Classifier features (from Phase 9):** predicted direction, classifier confidence/probability, classifier accuracy on recent N predictions
  - **Regressor features (from Phase 10):** predicted return magnitude, prediction uncertainty (std), quantile spread (Q90-Q10), conformal interval width
  - **Combined forecast features:** classifier agrees with regressor sign? (agreement signal), |predicted return| × classifier confidence (conviction score)
  - **Regime features:** GARCH conditional volatility, volatility regime indicator, rolling permutation entropy
  - **Cross-asset features:** relative strength vs universe mean, correlation rank, beta to BTC
  - **Historical strategy features:** rolling strategy Sharpe on this asset (past N windows), rolling win rate, max drawdown

### 12B: Recommender Domain

- **`src/app/recommendation/domain/value_objects.py`** —
  - `RecommendationInput` — asset, timestamp, feature_vector, direction_forecast, return_forecast
  - `Recommendation` — asset, predicted_strategy_return, confidence, rank, deploy (bool), predicted_direction, predicted_magnitude
  - `RecommenderConfig` — model_type, train_window, retrain_frequency, top_k, min_threshold

- **`src/app/recommendation/domain/protocols.py`** — `IRecommender` protocol:
  - `fit(X_train: DataFrame, y_train: Series) -> None` (y = realized strategy return)
  - `predict(X: DataFrame) -> list[Recommendation]`
  - `rank(X: DataFrame) -> list[Recommendation]` (sorted by predicted return)

### 12C: Recommender Models

- **`src/app/recommendation/application/gradient_boosting_recommender.py`** —
  LightGBM regressor predicting strategy return per asset. Feature importance (SHAP) reveals what drives recommendations. Primary model — strong on tabular data, interpretable, fast.

- **`src/app/recommendation/application/neural_recommender.py`** —
  Small feedforward network (2-3 layers, 64-128 hidden). Takes asset features + classifier output + regressor output → predicted strategy return. Compared against LightGBM.

- **`src/app/recommendation/application/baseline_recommenders.py`** — Baselines for comparison:
  - `RandomRecommender` — randomly select K assets (null hypothesis)
  - `TopVolumeRecommender` — always pick highest-volume assets
  - `TopMomentumRecommender` — always pick highest-momentum assets
  - `AllAssetsRecommender` — deploy strategy on everything (unfiltered)
  - `ClassifierOnlyRecommender` — rank by classifier confidence (no regression, no ML recommender)
  - `RegressorOnlyRecommender` — rank by predicted return magnitude (no classifier, no ML recommender)

### 12D: Walk-Forward Training Pipeline

- **`src/app/recommendation/application/pipeline.py`** — `RecommenderPipeline`:
  1. Split timeline into expanding train / test windows (monthly or quarterly)
  2. For each window:
     a. Compute features on train data
     b. Run classifier + regressor on train data to generate forecast features
     c. Run base strategy on train data → get realized returns (labels)
     d. Train recommender on (market_features + forecast_features, realized_returns)
     e. On test window: run classifier + regressor → compute features → recommender.predict() → rank assets → select top-K
     f. Run base strategy ONLY on recommended assets → record performance
     g. Run base strategy on ALL assets → record unfiltered performance
     h. Run baseline recommenders → record their performance
  3. Aggregate: per-window and overall metrics

### 12E: Recommendation Metrics

- **`src/app/recommendation/application/metrics.py`** —
  - **Direction quality:** Did we correctly predict whether the strategy would be profitable on this asset? Precision of deploy=True decisions.
  - **Ranking quality:** NDCG@K, Spearman rank correlation between predicted and realized strategy returns
  - **Decision quality:** Precision@K (of recommended, how many had positive strategy return?), Recall@K (of profitable assets, how many were recommended?)
  - **Economic value:** Sharpe of recommended portfolio vs unfiltered vs random vs classifier-only vs regressor-only — **the ultimate metric**. Also: total return, max drawdown, profit factor.

### 12F: Tests

- Unit test: label builder produces correct strategy returns
- Unit test: feature builder assembles features without leakage (classifier/regressor outputs use only past data)
- Integration test: full pipeline on small synthetic dataset
- Sanity test: recommender trained on noise → no better than random baseline

**Dependencies:** Phase 4 (features), Phase 7 (backtest), Phase 8 (strategies), Phase 9 (classifier), Phase 10 (regressor)
**Estimated scope:** ~14 files, ~1200 lines

---

## Phase 13: Research Checkpoint 4 — Recommender Evaluation

**Goal:** Does the ML recommender actually add value? Does combining classification + regression
outperform each alone? Final charts, statistics, and honest evaluation.

### 13A: Notebook `research/RC4_recommender_evaluation.ipynb`

**Recommender model diagnostics:**
- Feature importance (LightGBM SHAP values) → what drives recommendations?
- Which features matter more: market state, classifier output, or regressor output?
- Predicted vs realized strategy return scatter plot
- Recommender accuracy per test window → is it stable?
- Learning curve: performance vs training data size

**Head-to-head comparison (the core thesis result):**
- Table: ML recommender vs every baseline:

| Baseline | What it tests |
|----------|---------------|
| `RandomRecommender` | Is the recommender better than random? (null hypothesis) |
| `AllAssetsRecommender` | Is filtering better than no filtering? |
| `ClassifierOnlyRecommender` | Does regression add value on top of direction? |
| `RegressorOnlyRecommender` | Does classification add value on top of magnitude? |
| `TopVolumeRecommender` | Is the ML better than a simple volume heuristic? |
| `TopMomentumRecommender` | Is the ML better than a simple momentum heuristic? |

- Per metric: Sharpe, total return, max drawdown, win rate, NDCG@K, Precision@K
- Equity curves: overlay all approaches

**Hypothesis testing:**

*H₁: ML recommender selects assets with higher strategy returns than random selection*
- Permutation test: shuffle asset selections 10000x, compare real mean return vs null
- Report: p-value, effect size

*H₂: ML recommender produces higher portfolio Sharpe than unfiltered deployment*
- Permutation test: shuffle returns 10000x, compare real Sharpe vs null
- Block bootstrap: 95% CI for Sharpe difference (recommended - unfiltered)
- Report: p-value, CI

*H₃: Combined (classifier + regressor + ML recommender) outperforms classifier-only or regressor-only*
- DM test on Sharpe: combined vs classifier-only, combined vs regressor-only
- This is the key thesis hypothesis — proves that the two-track approach adds value

*H₄: Recommendations are stable (not random noise)*
- Walk-forward consistency: Jaccard similarity of top-K sets across adjacent windows
- Rank correlation of asset scores across adjacent windows

**Robustness checks:**
- Sensitivity to slippage: rerun with 0, 5, 10, 20 bps → does conclusion change?
- Sensitivity to commission: rerun with 0%, 0.1%, 0.2% → does conclusion change?
- Sensitivity to top-K: results for K=3, 5, 10, 15 → is there an optimal K?
- Sensitivity to retrain frequency: monthly vs quarterly vs yearly

**Honest assessment:**
- If recommender Sharpe CI includes 0 → "cannot claim the recommender adds economic value"
- If p-value > 0.05 → "cannot reject null hypothesis"
- If combined ≈ classifier-only → "regression adds no value beyond direction"
- If combined ≈ regressor-only → "explicit classification adds no value beyond implicit direction in regression"
- Document all negative results explicitly

**Estimated scope:** 1 notebook, ~500 cells

---

## Phase 14: Statistical Proof & Final Report

**Goal:** Formal statistical tests aggregated into a reproducible report.
Prove the system works on real data AND prove it doesn't work on random data.

### 14A: Monte Carlo Simulation on Synthetic Data

> **Key sanity check:** If our strategy/recommender "works" on pure random data,
> it's overfitting. A valid system should find NO signal in noise.

- **`src/app/evaluation/application/monte_carlo.py`** —
  - **GBM price paths:** Generate N=1000 synthetic price series using Geometric Brownian Motion
    with parameters calibrated to match real crypto (μ, σ from historical BTC/ETH).
    These paths have NO exploitable structure by construction.
  - **Bootstrapped real paths:** Block-resample real returns to create synthetic paths
    that preserve volatility clustering but destroy any predictive signal.
  - **Variance ratio paths:** Generate paths with controlled autocorrelation
    (0, 0.01, 0.05, 0.10) to test at what signal strength the system detects it.

- **`src/app/evaluation/application/monte_carlo_runner.py`** —
  1. Run the FULL pipeline on each synthetic path: features → classifier → regressor → recommender → backtest
  2. Record: Sharpe, total return, accuracy, recommender Precision@K per synthetic path
  3. Build null distributions from the 1000 runs
  4. Compare real-data metrics against these null distributions
  5. **Expected:** strategy Sharpe on GBM paths should be ~0 (centered at 0, not significantly positive)
  6. **If strategy is profitable on random data → overfitting alarm → revisit model complexity**

### 14B: Permutation Tests on Real Data

- **`src/app/evaluation/application/permutation_tests.py`** —
  - **Test 1 — Shuffled returns:** Freeze recommendations, shuffle returns 10000x → null Sharpe distribution. H₀: recommender Sharpe no different from chance.
  - **Test 2 — Shuffled selections:** Keep returns, random K selection 10000x → null. H₀: recommender no better than random picking.
  - **Test 3 — Filtered vs unfiltered:** Permutation test on Sharpe difference (recommended vs all assets)
  - **Test 4 — Combined vs single-track:** Permutation test on Sharpe(combined) - Sharpe(classifier-only)
  - **Test 5 — Strategy on random data (from 14A):** Compare real Sharpe against Monte Carlo null. p-value = fraction of synthetic Sharpe ≥ real Sharpe.

### 14C: Bootstrap Confidence Intervals

- **`src/app/evaluation/application/bootstrap.py`** —
  - Block bootstrap (preserving autocorrelation via stationary bootstrap)
  - 95% CI for: Sharpe ratio, max drawdown, total return, Sharpe difference (recommended - unfiltered)
  - If Sharpe CI includes 0 → cannot claim profitability
  - Bootstrap on each test period separately + pooled

### 14D: Walk-Forward Validation

- **`src/app/evaluation/application/walk_forward_proof.py`** —
  - Entire pipeline on multiple non-overlapping test periods (2024 H1, 2024 H2, 2025 H1)
  - Report all periods honestly (including failures)
  - Consistency metric: is the system profitable in >50% of test periods?

### 14E: Model Comparison

- **`src/app/evaluation/application/model_comparison.py`** —
  - Diebold-Mariano test for each model pair (classifiers, regressors, recommenders)
  - Model Confidence Set (Hansen 2011)
  - Ablation results from MLflow
  - Summary table: model × metric × significance

### 14F: Tests

- Monte Carlo: strategy on 1000 GBM paths → Sharpe distribution centered at ~0
- Monte Carlo: strategy on paths with injected autocorrelation=0.10 → should detect signal
- Permutation test on synthetic "known profitable" recommender → should reject null
- Permutation test on random recommender → should NOT reject null

**Dependencies:** All previous phases
**Estimated scope:** ~12 files, ~1000 lines

---

---

# BLOCK III: Polishing & Production

> Everything below is post-thesis-validation. Only proceed after Phases 1–14 are complete
> and the research results are satisfactory. This block turns the research prototype into
> a production-grade paper trading system.

---

## Phase 15: Pipeline Hardening & Production Infrastructure

**Goal:** Harden the entire pipeline for continuous, unattended operation.
Make it crash-safe, observable, and recoverable.

### 15A: Error Handling & Recovery

- **`src/app/system/resilience/circuit_breaker.py`** — Circuit breaker pattern for external API calls (Binance). States: CLOSED → OPEN (after N failures) → HALF_OPEN (probe). Prevents hammering a dead API.
- **`src/app/system/resilience/retry.py`** — Unified retry decorator with exponential backoff, jitter, configurable max retries. Used across all I/O operations.
- **`src/app/system/resilience/dead_letter.py`** — Failed operations (e.g., missed candles, failed bar construction) are logged to a dead letter table in DuckDB for manual review and replay.

### 15B: Pipeline Orchestration

- **`src/app/system/pipeline/pipeline.py`** — `Pipeline` class: ordered sequence of steps, each step has `run()` and `rollback()`. If step N fails, steps 1..N-1 are rolled back (where applicable). Checkpointing: completed steps are recorded so re-runs skip finished work.
- **`src/app/system/pipeline/scheduler.py`** — Cron-like scheduler: trigger pipeline runs at configurable intervals (e.g., every 1h for 1h bars). Uses `APScheduler` or simple asyncio loop.
- **`src/app/system/pipeline/health.py`** — Health check endpoint: is the pipeline running? When was the last successful run? Any dead letters? Stale data detection.

### 15C: State Management

- **`src/app/system/state/checkpoint.py`** — Checkpoint manager: persist pipeline state to DuckDB. On restart, resume from last checkpoint. Handles: last ingested timestamp per asset, last bar constructed, last model prediction time.
- **`src/app/system/state/model_registry.py`** — Track which model version is active. On retrain: save new model artifact, validate on recent data, promote only if it passes validation gates (accuracy > threshold). Rollback to previous model if new one degrades.

### 15D: Observability

- Structured logging with Loguru: correlation IDs per pipeline run, log levels, JSON output for log aggregation
- Metrics: pipeline latency, API call count, model inference time, trade count — exposed via simple metrics file or Prometheus-compatible endpoint
- Alerting: log CRITICAL on pipeline failure, model degradation, API outage

### 15E: Tests

- Chaos test: kill pipeline mid-run → restart → verify it resumes from checkpoint
- Circuit breaker test: simulate 5 consecutive API failures → verify breaker opens → probe → closes
- Dead letter test: inject malformed candle data → verify it's captured, not silently dropped

**Dependencies:** Phases 1-14 (working pipeline)
**Estimated scope:** ~12 files, ~800 lines

---

## Phase 16: Live Paper Trading Engine

**Goal:** Real-time paper trading system that connects to Binance live data,
runs the full pipeline (bars → features → classifier → regressor → recommender → trades),
and executes trades with simulated (fake) money on real market data.

### 16A: Real-Time Data Feed

- **`src/app/live/infrastructure/binance_websocket.py`** — WebSocket connection to Binance for real-time kline/candle data. Supports multiple assets simultaneously. Auto-reconnect on disconnect. Heartbeat monitoring.
- **`src/app/live/infrastructure/binance_rest_poller.py`** — REST API fallback for when WebSocket is unavailable. Polls at configurable interval. Used for initial catchup and gap-filling.
- **`src/app/live/application/data_feed.py`** — `LiveDataFeed`: unified interface over WebSocket + REST. Maintains in-memory buffer of recent candles. Pushes new bars to subscribers (observer pattern).

### 16B: Real-Time Bar Construction

- **`src/app/live/application/bar_builder.py`** — `LiveBarBuilder`: receives raw candles/trades from data feed, constructs bars in real-time (time bars, dollar bars, etc.). Emits completed bars to subscribers. Handles partial bars (in-progress bar state).

### 16C: Real-Time Feature Computation

- **`src/app/live/application/feature_engine.py`** — `LiveFeatureEngine`: maintains rolling windows of data needed for feature computation. On each new bar: compute all features incrementally (not recompute from scratch). Emits feature vectors to the prediction pipeline.

### 16D: Paper Trading Execution

- **`src/app/live/domain/value_objects.py`** —
  - `PaperAccount` — starting_balance, current_balance, positions, trade_history, equity_history
  - `PaperOrder` — asset, side, size, order_type, timestamp, fill_price (simulated)
  - `LiveTradeConfig` — initial_capital, max_position_pct, slippage_model, commission_model

- **`src/app/live/application/paper_broker.py`** — `PaperBroker`: simulates order execution against live market prices. Applies slippage model + commission model (same as backtest engine). Manages PaperAccount state. Persists all trades and equity snapshots to DuckDB for dashboard.

- **`src/app/live/application/trading_loop.py`** — `LiveTradingLoop`: the main event loop:
  1. Receive new bar from LiveBarBuilder
  2. Compute features via LiveFeatureEngine
  3. Run classifier → get direction + confidence
  4. Run regressor → get magnitude + uncertainty
  5. Run recommender → get deploy/skip decision + rank
  6. If deploy: generate order via strategy → execute via PaperBroker
  7. Log everything: prediction, decision, trade, portfolio state
  8. Persist state to DuckDB (crash-recoverable)

### 16E: Model Management

- **`src/app/live/application/model_manager.py`** — `ModelManager`:
  - Load trained models (classifier, regressor, recommender) from MLflow registry
  - Periodic re-evaluation: every N hours, run latest model on recent data → compare with current model
  - Graceful model swap: new model takes over only after validation
  - Version tracking: which model version made which trade

### 16F: Tests

- Integration test: feed historical data through LiveTradingLoop as if it were real-time → verify trades match backtest results
- Reconnection test: simulate WebSocket disconnect → verify auto-reconnect + no data gaps
- Crash recovery test: kill LiveTradingLoop → restart → verify it resumes from last state
- Paper broker test: verify PnL accounting matches BacktestEngine on same data

**Dependencies:** Phase 15 (resilience), all model phases
**Estimated scope:** ~16 files, ~1500 lines

---

## Phase 17: Dashboard Frontend

**Goal:** Real-time web dashboard for monitoring the paper trading system.
Charts of equity, confidence, trade log, portfolio state, model diagnostics.

### 17A: Backend API

- **`src/app/dashboard/infrastructure/api.py`** — FastAPI backend serving dashboard data:
  - `GET /api/equity` — equity curve time series (paper account value over time)
  - `GET /api/trades` — recent trades with details (asset, side, size, price, PnL, slippage, commission)
  - `GET /api/positions` — current open positions
  - `GET /api/portfolio` — portfolio summary (total equity, cash, unrealized PnL, drawdown)
  - `GET /api/predictions` — latest classifier + regressor + recommender outputs per asset
  - `GET /api/recommendations` — current asset rankings with scores and deploy decisions
  - `GET /api/model-info` — active model versions, last retrain timestamp, validation metrics
  - `GET /api/health` — pipeline health, last data timestamp, any alerts
  - `WebSocket /ws/live` — real-time push of new bars, trades, equity updates

### 17B: Frontend Dashboard

- **`src/app/dashboard/frontend/`** — Streamlit or Plotly Dash application (single-page):

  **Equity & Performance panel:**
  - Equity curve chart (line chart, live-updating)
  - Drawdown chart below equity
  - Key metrics cards: total return, Sharpe, max drawdown, win rate, # trades today
  - Benchmark overlay: equity vs buy-and-hold BTC

  **Recommendations panel:**
  - Table: all assets ranked by recommender score
  - Columns: asset, direction (↑/↓), confidence, predicted return, recommendation (DEPLOY/SKIP), current price
  - Color-coded: green = deploy, red = skip, yellow = borderline
  - Refresh on each new bar

  **Trade Log panel:**
  - Scrollable table of recent trades (last 50)
  - Columns: timestamp, asset, side, entry price, exit price, size, PnL, commission, slippage
  - Color-coded: green = profit, red = loss
  - Expandable rows: show classifier/regressor/recommender predictions that led to the trade

  **Model Diagnostics panel:**
  - Active model versions (classifier, regressor, recommender) with last retrain date
  - Rolling accuracy / DA chart (last 100 predictions)
  - Confidence distribution histogram (are predictions concentrated or spread?)
  - Alert list: any model degradation warnings, pipeline errors, data gaps

  **Portfolio panel:**
  - Current positions table: asset, side, size, entry price, current price, unrealized PnL
  - Asset allocation pie chart
  - Exposure chart: long vs short exposure over time

### 17C: Configuration

- Dashboard reads from the same DuckDB database as the trading engine
- No separate data store — single source of truth
- Configurable refresh interval (default: 5 seconds for WebSocket, 30 seconds for REST polling)
- Dark theme (because finance)

### 17D: Tests

- API test: each endpoint returns correct schema
- Frontend test: dashboard renders without errors on empty data (no trades yet)
- Load test: dashboard handles 10000 trade records without lag
- WebSocket test: real-time updates arrive within 1 second of trade execution

**Dependencies:** Phase 16 (live trading engine), `fastapi`, `streamlit` or `plotly-dash`, `uvicorn`
**Estimated scope:** ~10 files, ~1200 lines

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
    ├── Phase 4  (Features + Targets: classification & regression) ◄── NEXT
    │       │
    │       ├── Phase 5  (Profiling)
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

    Phase 15 (Pipeline Hardening) ─── error handling, circuit breakers,
    │                                 checkpointing, observability
    │
    Phase 16 (Live Paper Trading) ─── WebSocket feed, real-time bars,
    │                                 paper broker, trading loop
    │
    Phase 17 (Dashboard Frontend) ─── FastAPI + Streamlit/Dash,
                                      equity charts, trade log, alerts

    RC = Research Checkpoint (notebook-driven, collaborative review)
```

**Current status (2026-03-12):** Phases 1–3 complete. Phase 4 is next.

**Parallel work possible:**
- Phases 5 + 7 can proceed in parallel after Phase 4
- **Phases 9 + 10 can proceed in parallel** (classification and regression are independent tracks)
- Phase 11 (RC3) evaluates both tracks together, compares and combines
- Phase 12 needs Phases 7, 8, 9, 10 all done
- **Block III only starts after Phase 14** (all research validated first)

---

## Per-Phase Deliverables

| Phase | Type | Deliverable | Status |
|-------|------|------------|--------|
| | | **BLOCK I: Data & Infrastructure** | |
| 1 | Build | DuckDB filled with multi-asset OHLCV data | ✅ Done |
| 2 | Build | Alternative bar types computed and stored | ✅ Done |
| **3** | **Research** | **RC1 notebook: data quality charts, bar comparison, coverage** | **✅ Done** |
| 4 | Build | Feature matrix + both targets (direction + return) | ◄ Next |
| 5 | Build | Statistical profile per asset | Yes: distribution plots, test tables |
| **6** | **Research** | **RC2 notebook: feature analysis, profiling, data adequacy verdict** | **Yes: charts, go/no-go decision** |
| 7 | Build | Backtest engine with slippage + commission | Yes: equity curves, PnL |
| 8 | Build | Base strategy backtests | Yes: performance table |
| | | **BLOCK II: Models, Recommendation & Proof** | |
| 9 | Build | Direction classifiers (logistic, RF, LightGBM, GRU, Transformer) | Yes: accuracy, precision, calibration |
| 10 | Build | Return regressors (Ridge, ARIMA-GARCH, LightGBM, GRU-MDN, TFT) | Yes: DC-MAE, CRPS, uncertainty |
| **11** | **Research** | **RC3: classification eval + regression eval + combined pipeline** | **Yes: dual-track comparison** |
| 12 | Build | ML recommendation system (consumes both tracks) | Yes: rankings, SHAP, feature importance |
| **13** | **Research** | **RC4 notebook: recommender vs baselines, H₁–H₄, equity curves** | **Yes: final results, p-values** |
| 14 | Build | Statistical proof + Monte Carlo on random data | Yes: p-values, null distributions |
| | | **BLOCK III: Polishing & Production** | |
| 15 | Build | Pipeline hardening (circuit breakers, checkpoints, observability) | Yes: health endpoint, recovery test |
| 16 | Build | Live paper trading engine (real-time Binance, paper broker) | Yes: running system, live trades |
| 17 | Build | Dashboard (equity charts, trade log, recommendations, alerts) | Yes: open in browser, live-updating |

---

## Key Design Decisions

1. **Two-track forecasting: classification + regression** — Classification predicts direction (SIDE), regression predicts magnitude (SIZE). Combined: richer signal than either alone. Inspired by López de Prado's meta-labeling, generalized from binary to continuous.
2. **Regression metrics are direction-conditional** — Raw MAE/RMSE/R² are misleading in finance. Regression is evaluated ONLY when direction is correct (DC-MAE, DC-RMSE). Economic Sharpe is the ultimate metric. A model with low MAE but DA ≤ 50% is useless.
3. **ML recommendation system** — The recommender is a trained model (LightGBM / neural net) consuming BOTH classifier + regressor outputs. Has its own train/test pipeline. Testable hypotheses: H₁–H₄, including H₃ which tests whether combining both tracks beats each alone.
4. **Research checkpoints** — 4 explicit stops (RC1–RC4) where we analyze data, produce charts, and make informed decisions before proceeding. Prevents building on bad foundations.
5. **Honest evaluation** — DA ≤ 50%, non-significant p-values, and Sharpe CIs including zero are valid and documented results.
6. **Polars for ETL, Pandas for experiments, NumPy for math** — Polars in pipelines (ingestion, bars, backtest, live) for performance. Pandas in research notebooks and model training for ML ecosystem compatibility. NumPy for vectorized numerical computations (indicators, bootstrap, Monte Carlo).
7. **Pydantic everywhere, no dataclasses** — All value objects, configs, DTOs use `BaseModel` with validation, serialization, and `frozen=True` immutability. No raw `dataclass` usage.
8. **DuckDB for storage** — Analytical queries, Parquet integration, no server needed.
9. **Protocol-based DI** — All modules depend on protocols, not concrete implementations.
10. **Config-driven** — Every parameter in Pydantic config classes, no magic numbers.
10. **MLflow for tracking** — Experiments, model registry, artifact storage.
11. **No future leakage** — Enforced by `TemporalSplit` value object + `.shift(1)` convention. Walk-forward throughout.
12. **Monte Carlo validation** — Strategy must NOT be profitable on synthetic GBM paths. Profitable on noise = overfitting.
13. **Production-grade live system** — Circuit breakers, checkpointing, dead letter queues, crash recovery. Paper trading with real Binance data before any real money.
14. **Dashboard for observability** — Real-time equity charts, trade log, model diagnostics, alerts. Single source of truth (DuckDB).
15. **Google-style docstrings everywhere** — Every public module, class, method, function. Enforced by Ruff `D` + `DOC` rules. No code merges without docstrings.
16. **Python 3.14 type hints** — Modern syntax (`list[X]`, `X | None`, `type` aliases, `Self`, `Never`). Pyright strict mode with zero errors. Enforced on every commit.
17. **Pre-commit quality gates** — Ruff formatter → Ruff linter → Pyright strict → isort. All from `pyproject.toml`. Failed hook = rejected commit.
18. **Alembic for all schema changes** — Every DuckDB table/column/index change goes through a versioned, reversible Alembic migration. No raw DDL outside migrations.
