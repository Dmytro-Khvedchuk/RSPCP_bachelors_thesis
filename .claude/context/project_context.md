Let me create a comprehensive context document that an AI agent can use to understand and help build this project.I'm having trouble with the file system right now. Let me provide the full context document directly — you can copy it into a file for your AI agent:

---

# Project Context: Probabilistic Crypto Strategy Recommender

## 1. Project Overview

### Title
**Recommendation System for Predicting Cryptocurrency Prices**

### One-Sentence Summary
A recommendation system that operates over a universe of crypto assets and, given a base trading strategy, uses statistically validated features and calibrated probabilistic forecasts to recommend on which assets and at what times the strategy should be deployed — with rigorous statistical proof of whether the filtering improves performance.

### Core Question
> "Given the current market state across N crypto assets and a base trading strategy, which assets should you deploy the strategy on right now, and with what confidence?"

### What This Project Is NOT
- NOT sentiment analysis or NLP — no text data is used
- NOT portfolio optimization — no weight allocation
- NOT arbitrage — no pairs trading
- NOT simple price prediction — probabilistic forecasting with calibrated uncertainty feeding into a recommendation layer

### What Makes This Strong
- Rigorous hypothesis testing at every stage
- Permutation tests on shuffled data to prove signal vs. noise
- Full probabilistic forecasts with calibrated uncertainty
- Conformal prediction guarantees
- Clean separation: forecasting engine → recommendation layer
- Honest reporting — negative results documented as valid science

---

## 2. Data Sources

### 2.1 OHLCV Data
- **Source:** Binance API
- **Assets:** 20-50 cryptos (BTC, ETH, SOL, BNB, AVAX, ADA, DOT, LINK, etc.)
- **Timeframes:** 1h, 4h, 1d
- **Fields:** Open, High, Low, Close, Volume
- **Depth:** 2020-present
- **Format:** Parquet

### 2.2 Macro News Calendar
- **Source:** Forex Factory scraper (already built)
- **Used as structured numeric features ONLY:**
  - Time until/since next/last event (hours)
  - Event type (one-hot)
  - Historical avg absolute return for event type
  - Historical avg volatility change for event type
  - Binary: did previous event of this type cause a >2σ move?

### 2.3 Data Pipeline
- Walk-forward train/val/test splits — NO future leakage
- Default: Train 2020-01 to 2023-06 / Val 2023-07 to 2023-12 / Test 2024-01 to 2024-06
- All configurable via YAML

---

## 3. Pipeline: Phase-by-Phase

### PHASE 1: Statistical Profiling

For each asset:

**Return distribution analysis:** Log returns → normality tests (Jarque-Bera, Shapiro-Wilk, Anderson-Darling). H₀: returns are normal → expect rejection. Fit alternatives (Student-t, Generalized Hyperbolic). Compare via AIC/BIC/KS test.

**Autocorrelation:** ACF/PACF of returns and squared returns. Ljung-Box tests. H₀: no serial correlation. Rejection = exploitable structure.

**Volatility:** GARCH(1,1) and GJR-GARCH fits. Test residuals for i.i.d. Rolling realized volatility.

**Macro impact:** For each event type, compute avg absolute return and volatility ratio around events. Mann-Whitney U test. H₀: event returns = normal returns. Rejection = macro features justified.

**Output:** Statistical profile per asset with formal test results.

**Libraries:** `scipy.stats`, `statsmodels`, `arch`

---

### PHASE 2: Feature Engineering & Validation

#### 2A: Features

**Price-based:** Log returns (multi-horizon), realized volatility (multi-window), Garman-Klass volatility (`0.5*ln(H/L)² - (2ln2-1)*ln(C/O)²`), Parkinson volatility (`ln(H/L)²/(4*ln2)`), return z-score.

**Momentum/mean-reversion:** RSI (continuous), ROC (multi-window), Bollinger width & %B, rolling Hurst exponent (H>0.5 trending, H≈0.5 random, H<0.5 mean-reverting).

**Volume:** Volume z-score, OBV slope, volume-price divergence, Amihud illiquidity (`|r|/V`).

**Multi-timeframe:** Key features at 1h, 4h, 1d stacked.

**Macro:** `hours_until_next_event`, `hours_since_last_event`, event type one-hot, `event_historical_impact`, `event_historical_vol_impact`, `prev_event_large_move` flag.

#### 2B: Validation — Permutation Testing Framework

**Step 1 — Mutual Information Test:** Compute MI(feature, target). Shuffle target 1000x → null distribution. P-value. Benjamini-Hochberg FDR correction. Keep only features with corrected p < 0.05.

**Step 2 — Single-Feature Predictive Power:** Train Ridge on single feature → OOS accuracy. Shuffle target 500x → null. Features must beat shuffled baselines.

**Step 3 — Stability:** Repeat MI test on 2020, 2021, 2022, 2023 separately. Flag features significant in <50% of windows.

**Step 4 — Interaction:** Compare combined feature groups vs. sum of individual contributions.

**Output:** Feature table with MI p-value, FDR-corrected p, predictive p, stability score, keep/drop.

**Libraries:** `sklearn.feature_selection`, `scipy.stats`, `sklearn.linear_model.Ridge`

---

### PHASE 3: Probabilistic Forecasting

**Target:** Future log return at horizon(s) (4h, 24h)

#### Baselines

**B1: ARIMA-GARCH** — conditional mean + variance → Gaussian distribution. Order by AIC. (`statsmodels`, `arch`)

**B2: Bayesian Linear Regression** — prior on coefficients, posterior via VI or MCMC → full posterior predictive. (`PyMC`/`NumPyro`)

**B3: Quantile Regression** — at τ={0.05,0.10,0.25,0.50,0.75,0.90,0.95} → non-parametric distribution. (`statsmodels`)

#### Deep Learning

**M1: GRU + Mixture Density Network (MDN)** — GRU encoder (2 layers, 64-128 hidden) → MDN head with K=3-5 Gaussians (mean, variance, weight per component). Loss: negative log-likelihood. Captures multimodal distributions. (`PyTorch`)

**M2: Temporal Fusion Transformer (TFT)** — static inputs (asset ID), known future (macro countdowns), observed (OHLCV features). Multi-horizon quantile output. Built-in variable selection & temporal attention for interpretability. (`pytorch-forecasting`)

**M3: Monte Carlo Dropout** — best architecture with dropout at inference, 100 forward passes. Epistemic uncertainty (model) vs. aleatoric (data). High epistemic → don't recommend.

#### Calibration

**Check:** For each predicted quantile q, verify actual coverage ≈ q. Reliability diagrams.

**Conformal Prediction:** Nonconformity scores on calibration set → guaranteed-coverage intervals. Coverage ≥ (1-α) regardless of model quality. (`mapie`)

---

### PHASE 4: Model Comparison

**Metrics:** CRPS (primary), log-likelihood, calibration error, interval sharpness, MAE, directional accuracy.

**Diebold-Mariano Test:** For each model pair, test if CRPS difference is significant. H₀: equal accuracy.

**Model Confidence Set (Hansen 2011):** Smallest set containing the best model at (1-α) confidence.

**Ablations (logged in W&B):** ±macro features, ±aggressive feature filtering, MDN vs. single Gaussian, TFT vs. GRU, single-asset vs. multi-asset.

---

### PHASE 5: Recommendation Layer — Strategy Gate

#### 5A: Base Strategy
Simple, well-defined strategy (momentum crossover, mean-reversion, breakout). The strategy is NOT the thesis focus — the recommendation system deciding where to deploy it IS.

#### 5B: Recommendation Score
Per asset per time step, combine:
- `expected_return / |CVaR|` (risk-adjusted return)
- `P(return > 0)` (profit probability)
- `1 / prediction_interval_width` (confidence)
- `regime_match` (Hurst vs. strategy type compatibility)
- `-epistemic_uncertainty` (penalize unknown patterns)

Weights tuned on validation set.

#### 5C: Output
Rank all assets → recommend top-K. Optional minimum threshold → "trade nothing" if nothing qualifies.

#### 5D: Metrics
- **NDCG@K:** ranking quality
- **Precision@K:** of recommended, how many profitable?
- **Recall@K:** of profitable, how many recommended?

---

### PHASE 6: Statistical Proof

**Test 1 — Shuffled Returns:** Freeze signals, shuffle returns 10000x, compare real Sharpe vs. null. H₀: no better than chance.

**Test 2 — Shuffled Signals:** Keep returns, random K selection 10000x. H₀: recommender no better than random.

**Test 3 — Filtered vs. Unfiltered:** Strategy on all assets vs. recommended only. Permutation test on difference.

**Test 4 — Bootstrap CIs:** Block bootstrap, 95% CI for Sharpe/drawdown/return. Sharpe CI includes 0 → can't claim profitability.

**Test 5 — Walk-Forward:** Entire pipeline on multiple non-overlapping test periods. Report all honestly.

**Test 6 — Baselines:** Compare against unfiltered, random, volatility filter, momentum filter, volume filter. Diebold-Mariano for each.

---

## 4. Tech Stack

- **Core:** Python 3.10+, pandas, numpy, Parquet, YAML configs
- **Stats:** scipy.stats, statsmodels, arch, sklearn
- **DL:** PyTorch, pytorch-forecasting, PyMC/NumPyro
- **Calibration:** mapie, custom MC Dropout
- **Tracking:** wandb or mlflow
- **Backtesting:** custom engine or vectorbt
- **Viz:** matplotlib, seaborn, plotly, optionally streamlit

## 5. Key Principles

1. **No future leakage.** All features from past only. `.shift(1)` before feeding to models.
2. **Statistical tests before claims.** No p-value = no claim.
3. **Shuffled data is gold standard.** Works on shuffled = not real.
4. **Calibration matters.** Conformal prediction guarantees honest uncertainty.
5. **Honest reporting.** Negative results are valid.
6. **Reproducibility.** Config + seed = same results.
7. **Clean code.** Type hints, docstrings, modular. This is a CV piece.

## 6. Deliverables

1. Clean GitHub repo with modular code, configs, docs
2. Feature validation report with all p-values
3. Model comparison table with DM test p-values
4. Calibration plots per model
5. Recommendation evaluation (NDCG@K, Precision@K, Recall@K)
6. Strategy proof (permutation tests, bootstrap CIs, walk-forward)
7. Results dashboard
