# RC1 — Data Quality & Bar Analysis: Results

**Research Checkpoint 1** for the RSPCP Bachelor's Thesis.
**Asset:** BTCUSDT (primary analysis asset) | **Date range:** 2020-01-01 to 2026-03-12

---

## Section 1: Data Coverage

**Purpose:** Validate that the ingested OHLCV data from Binance is complete enough for modelling.

**Assets:** 4 pairs — BTCUSDT, ETHUSDT, LTCUSDT, SOLUSDT

### Coverage Heatmap (Bokeh, interactive)

Shows a matrix of `(asset × timeframe)` with colour-coded coverage percentages. All cells are deep green:

- **1h:** 99.94% for BTC/ETH/LTC, 99.96% for SOL
- **4h:** 99.99% for BTC/ETH/LTC, 100% for SOL
- **1d:** 100% across the board

| Asset   | 1h       | 4h       | 1d      | Total bars (1h) | Date range             |
|---------|----------|----------|---------|-----------------|------------------------|
| BTCUSDT | 99.94%   | 99.99%   | 100%    | 54,278          | 2020-01-01 → 2026-03-12 |
| ETHUSDT | 99.94%   | 99.99%   | 100%    | 54,278          | 2020-01-01 → 2026-03-12 |
| LTCUSDT | 99.94%   | 99.99%   | 100%    | 54,278          | 2020-01-01 → 2026-03-12 |
| SOLUSDT | 99.96%   | 100%     | 100%    | 48,932          | 2020-08-11 → 2026-03-12 |

**Verdict:** Excellent coverage. All assets exceed the 95% threshold by a wide margin.

### Gap Timeline (Bokeh, interactive)

Shows 18 gaps as horizontal bars on a timeline per asset:

- **5 identical gap windows** hit BTC, ETH, and LTC (they share Binance downtime): Feb 2020, Jun 2020, Dec 2020, Apr 2021, Aug 2021
- SOL only has 3 gaps (it listed later, Aug 2020)
- **Largest gap: 6 hours** (Feb 19, 2020). All others are 4–5 hours
- No gap exceeds the 48-hour danger threshold

These are likely Binance maintenance windows — simultaneous across all pairs, short, and confined to 2020–2021.

### Volume Profile (Bokeh, interactive)

Shows hourly trading volume over time for BTCUSDT. Reveals market regime changes — volume spikes during bull runs (late 2020, early 2021) and quieter periods.

### Asset Filter Table

All 4 assets pass (criteria: >= 730 days, <= 5% gap ratio):

| Asset   | Included | Total Days | Coverage |
|---------|----------|-----------|----------|
| BTCUSDT | Yes      | 2,263     | 99.94%   |
| ETHUSDT | Yes      | 2,263     | 99.94%   |
| LTCUSDT | Yes      | 2,263     | 99.94%   |
| SOLUSDT | Yes      | 2,040     | 99.96%   |

---

## Section 2: Bar Construction

**Purpose:** Load Lopez de Prado alternative bars and compare their temporal properties.

### Bar Counts (9 bar types loaded)

| Bar Type          | Count  | Usable?                    |
|-------------------|--------|----------------------------|
| **dollar**        | 5,287  | Yes — strong sample        |
| **volume**        | 3,264  | Yes — solid                |
| dollar_imbalance  | 569    | Marginal                   |
| volume_imbalance  | 530    | Marginal                   |
| dollar_run        | 435    | Marginal                   |
| volume_run        | 389    | Marginal                   |
| tick              | 55     | Too few                    |
| tick_imbalance    | 2      | Unusable                   |
| tick_run          | 1      | Unusable                   |

**Key insight:** Tick-based bars have catastrophically low counts. The tick threshold is set way too high — it aggregates ~6 years of 1h candles into just 55 bars. Tick bars (and their imbalance/run variants) are **not viable** with current thresholds.

### Weekly Bar Count Histogram (Bokeh)

Shows how many bars each type produces per week:

| Bar Type          | Mean/Week | Std/Week | CV    |
|-------------------|-----------|----------|-------|
| dollar            | 16.3      | 11.8     | 0.72  |
| volume            | 10.1      | 11.0     | 1.09  |
| dollar_imbalance  | 1.8       | 3.0      | 1.68  |
| volume_imbalance  | 1.7       | 2.8      | 1.67  |
| volume_run        | 2.3       | 3.5      | 1.53  |
| dollar_run        | 1.4       | 3.5      | 2.58  |
| tick              | 0.17      | 0.38     | 2.21  |

Dollar bars produce ~16/week (~2–3/day) with moderate variability. Imbalance/run types produce irregular, bursty counts (CV > 1.5).

### Bar Duration Boxplot (Matplotlib)

Shows the spread of how long each bar takes to form:

| Bar Type          | Median (min) | Mean (min)  | CV    | Max (min)    |
|-------------------|-------------|-------------|-------|--------------|
| **dollar**        | 420         | 616         | 0.93  | 5,520        |
| **volume**        | 660         | 998         | 1.04  | 7,680        |
| dollar_imbalance  | 1,740       | 5,727       | 2.25  | 179,820      |
| volume_imbalance  | 1,500       | 6,148       | 3.03  | 205,440      |
| dollar_run        | 780         | 7,491       | 4.63  | 493,860      |
| volume_run        | 1,020       | 8,377       | 9.55  | 1,565,820    |
| tick              | 60,000      | 59,247      | 0.10  | 60,360       |

- **Dollar bars:** median ~7 hours, mean ~10 hours — the most regular
- **Volume bars:** median ~11 hours, mean ~17 hours — slightly wider spread
- **Imbalance/run bars:** enormous outliers — some bars span days or weeks (dollar_run max = 343 days!)
- **Tick bars:** nearly constant ~1,000 hours per bar — basically monthly bars

---

## Section 3: Return Distributions

**Purpose:** Compute log returns from each bar type's close prices and test whether they resemble a normal distribution (important for many statistical models).

### Return Statistics Table

| Bar Type          | N      | Std    | Skew   | Excess Kurt | JB p-value | Normal? |
|-------------------|--------|--------|--------|-------------|------------|---------|
| **time_1h**       | 54,277 | 0.0067 | -0.94  | **53.3**    | 0.0        | No      |
| **dollar**        | 5,286  | 0.0205 | -0.36  | **6.7**     | 0.0        | No      |
| **volume**        | 3,263  | 0.0266 | -0.28  | **4.1**     | 0.0        | No      |
| dollar_imbalance  | 568    | 0.0652 | -0.73  | 8.3         | 0.0        | No      |
| volume_imbalance  | 529    | 0.0653 | 0.17   | **2.9**     | 0.0        | No      |
| dollar_run        | 434    | 0.0790 | 2.66   | 30.5        | 0.0        | No      |
| volume_run        | 388    | 0.0802 | 3.82   | 42.6        | 0.0        | No      |
| tick*             | <=54   | —      | —      | —           | —          | (N/A)   |

**Key findings:**

1. **All bar types reject normality** (JB p-value = 0) except tick-based ones (too few samples to reject anything).
2. **Time bars have the worst kurtosis** (53.3) — extreme fat tails. This is expected: fixed-time sampling catches both quiet and volatile periods.
3. **Dollar bars reduce kurtosis by ~8x** (53.3 → 6.7) and skewness by ~2.6x (-0.94 → -0.36) — a massive improvement over time bars.
4. **Volume bars** are second-best (kurtosis 4.1, skew -0.28).
5. **Volume_imbalance** has the lowest excess kurtosis of all (2.9) and near-zero skew (0.17) — closest to Gaussian.
6. **Run bars** have terrible distributional properties (kurtosis 30–43, high positive skew) — they capture extreme moves and hold through them.

### QQ-Plot Grid (Matplotlib)

Each subplot plots observed quantiles (y) against theoretical normal quantiles (x). A perfect normal distribution sits on the 45-degree diagonal.

- **time_1h:** Extreme S-curve — massive departures in both tails. Far from normal.
- **dollar / volume:** Mild S-curve — tails are fatter than normal but much better than time bars.
- **volume_imbalance:** Closest to the diagonal — lightest tails.
- **run bars:** Sharp hooks at the tails — outlier-prone.
- **tick/tick_imbalance/tick_run:** "No data" or too few points for a meaningful plot.

### Return Distribution Overlay (Matplotlib)

Histograms + KDE curves overlaid for all bar types:

- Time_1h returns are extremely peaked and narrow (high kurtosis, small std)
- Dollar/volume bars have wider, flatter distributions (larger per-bar moves)
- The KDE curves reveal how much fatter the tails are vs. a Gaussian bell shape

---

## Section 4: Autocorrelation & Serial Dependence

**Purpose:** Test whether returns are predictable from their own past (serial correlation) and whether volatility clusters in time (ARCH effects). Per Lopez de Prado, good bars should have **no serial correlation in returns** but **preserve volatility clustering**.

### ACF Comparison Grid — Raw Returns (Matplotlib)

Stem plots of autocorrelation at each lag, with 95% confidence bands (dashed red lines). Spikes outside the bands = significant autocorrelation.

- **time_1h:** Multiple lags exceed the bands — significant serial correlation (exploitable or spurious patterns)
- **dollar:** All lags inside bands — no serial correlation (clean)
- **volume:** Borderline — just barely significant at lag=40
- **Imbalance bars:** Inside bands — clean

### ACF Comparison Grid — Squared Returns (Matplotlib)

Same layout but for r-squared. Significant ACF in squared returns = volatility clustering (ARCH effects).

- **time_1h:** Massive, slowly decaying ACF — strong volatility clustering (expected)
- **dollar:** Strong volatility clustering preserved (good!)
- **volume:** Strong volatility clustering preserved (good!)
- **volume_imbalance / dollar_imbalance:** Moderate clustering still present (good)
- **run bars:** No clustering detected — they may be averaging it away

### Ljung-Box Summary Table

| Bar Type          | LB stat (returns) | p-value (returns) | Serial Corr? | LB stat (r²) | p-value (r²) | Vol Clustering? | Ideal? |
|-------------------|-------------------:|------------------:|:---:|-------------:|--------------:|:---:|:---:|
| **dollar**        | 47.42              | 0.1957            | No  | 3,569.83     | 0.0000        | Yes | **Yes** |
| **dollar_imbalance** | 34.69           | 0.7078            | No  | 99.87        | 0.0000        | Yes | **Yes** |
| dollar_run        | 62.10              | 0.0141            | YES | 41.38        | 0.4101        | No  | No  |
| tick*             | 23.04              | 0.6309            | No  | 16.46        | 0.9246        | No  | (N/A) |
| **volume**        | 55.93              | 0.0485            | YES | 2,600.90     | 0.0000        | Yes | Almost |
| **volume_imbalance** | 49.15           | 0.1522            | No  | 572.98       | 0.0000        | Yes | **Yes** |
| volume_run        | 27.22              | 0.9383            | No  | 0.35         | 1.0000        | No  | No  |
| **time_1h**       | 218.77             | 0.0000            | YES | 18,395.43    | 0.0000        | Yes | No  |

**The ideal bar** has no serial correlation in returns (p > 0.05) but preserves volatility clustering (p < 0.05 on squared returns). Three bar types achieve this: **dollar, dollar_imbalance, and volume_imbalance**.

---

## Section 5: Conclusions & Go/No-Go Decisions

### Q1: Is the data quality sufficient for modelling?

**YES.** All 4 assets have >99.9% coverage, >5.5 years of data, and no gap exceeds 6 hours. All pass the filtering criteria (>= 730 days, <= 5% gap ratio).

### Q2: Do alternative bars improve distributional properties over time bars?

**YES, significantly.**

- Dollar bars reduce excess kurtosis from 53.3 → 6.7 (8x improvement)
- Volume_imbalance achieves kurtosis of 2.9 — closest to normal
- Skewness improves from -0.94 to near zero
- Run bars are the exception — they make distributions worse (kurtosis 30–43)

### Q3: Do information-driven bars reduce serial correlation in returns?

**YES.**

- Time bars have strong serial correlation (LB p=0.0000)
- Dollar and imbalance bars eliminate it (p > 0.15)
- Dollar and volume-based bars preserve volatility clustering — removing spurious time-based patterns while keeping genuine volatility structure

### Q4: Which bar types should proceed to Phase 4 (feature engineering)?

#### Recommended for Phase 4

| Bar Type              | N     | Serial Corr | Vol Clustering | Kurtosis | Verdict                                                        |
|-----------------------|-------|:-----------:|:--------------:|----------|----------------------------------------------------------------|
| **dollar**            | 5,286 | No          | Yes            | 6.7      | **Best overall** — large sample, clean returns, preserves vol  |
| **volume**            | 3,263 | Borderline  | Yes            | 4.1      | **Good** — needs monitoring for serial corr                    |
| **volume_imbalance**  | 529   | No          | Yes            | 2.9      | **Best distributional properties** — but small sample          |
| **dollar_imbalance**  | 568   | No          | Yes            | 8.3      | **Solid** — clean serial properties                            |
| **time_1h**           | 54,277| YES         | Yes            | 53.3     | **Keep as baseline** — needed for comparison                   |

#### Disqualified

- **tick, tick_imbalance, tick_run** — threshold too high, <55 bars. Need recalibration if tick bars are desired.
- **dollar_run, volume_run** — poor distributional properties (extreme kurtosis 30–43), and run bars lose volatility clustering.
