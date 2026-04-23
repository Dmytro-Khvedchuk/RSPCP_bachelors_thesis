# GML-RS — Recommendation System for Cryptocurrency Trading Strategy Deployment Based on Generalized Meta-Labeling

> Bachelor's thesis — a generalized meta-labeling recommendation system that decides which crypto trading signals to deploy.
> **Author:** Dmytro Khvedchuk

---

## Overview

GML-RS is a research-grade algorithmic trading stack built for a bachelor's thesis. The core contribution is a trained recommender that consumes two upstream forecasts — a direction classifier (SIDE) and a return regressor (SIZE) — and decides whether (and how aggressively) to deploy each candidate trading signal. The recommender is a generalization of López de Prado's binary meta-labeling to a continuous, sizing-aware decision. The evaluation framework is statistically disciplined: purged and embargoed walk-forward cross-validation, split conformal calibration, autocorrelation-corrected Sharpe (Lo, 2002), and shuffled-labels sanity checks throughout. Negative results are valid and documented.

---

## Repository layout

```
.
├── main.py                     Application entry point (logging + DuckDB smoke test)
├── justfile                    Task runner — every development command lives here
├── pyproject.toml              Dependencies, Python version, tool config
├── mkdocs.yml                  Documentation site (Material + mkdocstrings)
├── .example.env                Environment variable template
├── .pre-commit-config.yaml     Pre-commit hook pipeline
├── .github/workflows/ci.yml    CI: lint + type check + test on every PR
├── data/                       DuckDB store (gitignored; created on first run)
├── docs/                       MkDocs source tree
└── src/
    ├── app/
    │   ├── system/             Logging (Loguru) + DuckDB + Alembic
    │   ├── ohlcv/              OHLCV domain + DuckDB repository
    │   ├── ingestion/          Binance fetcher + service + CLI
    │   ├── bars/               Lopez de Prado alternative bars + CLI
    │   ├── features/           Indicators + targets + matrix builder + validation
    │   ├── profiling/          Per-asset statistical profiling
    │   ├── backtest/           Event-driven backtest engine
    │   ├── strategy/           Batch trading strategies
    │   ├── forecasting/        Direction (SIDE) + return (SIZE) forecasters
    │   └── recommendation/     Generalized meta-labeling recommender
    └── tests/                  Test suite mirroring src/app/
```

---

## Prerequisites

- Python 3.14+
- [`uv`](https://docs.astral.sh/uv/) package manager
- [`just`](https://github.com/casey/just) task runner
- Binance API credentials (read-only permissions are sufficient)

---

## Quick start

```bash
# 1. Install dependencies (Python 3.14 is pulled by uv if missing)
uv sync

# 2. Configure environment
cp .example.env .env
# Then edit .env: set BINANCE_API_KEY, BINANCE_SECRET_KEY, and optionally DUCKDB_PATH

# 3. Create the DuckDB data directory
mkdir -p data/

# 4. Apply database migrations
just migrate

# 5. Ingest OHLCV data
just ingest --assets BTCUSDT,ETHUSDT --timeframes 1h,4h --start 2020-01-01

# 6. Aggregate alternative bars
just bars --assets BTCUSDT --bar-types dollar,volume,dollar_imbalance

# 7. Run the test suite
just test

# 8. Start the documentation site locally
just serve
```

---

## Configuration

All runtime settings are loaded from environment variables (the `.env` file at project root).

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DUCKDB_PATH` | `data/market.duckdb` | Path to the DuckDB store |
| `DUCKDB_READ_ONLY` | `false` | Open the database in read-only mode |
| `DUCKDB_MEMORY_LIMIT` | `4GB` | Soft memory cap for DuckDB |
| `DUCKDB_THREADS` | `-1` | Thread count (`-1` = auto-detect) |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `DEBUG` | Loguru minimum level |
| `LOG_JSON` | `false` | Emit structured JSON logs when `true` |
| `LOG_FILE` | *(empty)* | Path to optional log file; stderr only when empty |

### Binance

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_API_KEY` | *(required)* | Binance REST API key |
| `BINANCE_SECRET_KEY` | *(required)* | Binance REST API secret |
| `BINANCE_BATCH_SIZE` | `1000` | Klines per API request (max 1000) |
| `BINANCE_MAX_RETRIES` | `5` | Retry attempts on transient failures |
| `BINANCE_RETRY_MIN_WAIT` | `1` | Minimum exponential backoff (seconds) |
| `BINANCE_RETRY_MAX_WAIT` | `10` | Maximum exponential backoff (seconds) |

---

## Commands reference

Every development action is a `just` recipe — run `just` with no arguments for the auto-generated list.

| Command | What it does |
|---------|--------------|
| `just run` | Execute `main.py` — logging bootstrap + DuckDB smoke test |
| `just ingest <args>` | Invoke the Binance ingestion CLI (`src/app/ingestion/cli.py`) |
| `just bars <args>` | Invoke the bar-aggregation CLI (`src/app/bars/cli.py`) |
| `just test [args]` | Run the full pytest suite against `src/tests/` |
| `just add <package>` | Add a new dependency via `uv add` |
| `just serve` | Start the MkDocs live-reload server at `http://127.0.0.1:8000` |
| `just install-hooks` | Install the pre-commit hook pipeline |
| `just uninstall-hooks` | Remove the pre-commit hook pipeline |
| `just lint` | Run the pre-commit pipeline against all files |
| `just migrate` | Apply every pending Alembic migration |
| `just migration "<msg>"` | Scaffold a new Alembic migration file |
| `just migrate-down` | Roll back one migration |

### CLI usage — ingestion

```bash
# Full historical ingest for multiple assets and timeframes
just ingest --assets BTCUSDT,ETHUSDT,SOLUSDT --timeframes 1h,4h,1d --start 2020-01-01

# Ingest up to a specific end date
just ingest --assets BTCUSDT --timeframes 1h --start 2020-01-01 --end 2024-01-01

# Incremental top-up — skips data already stored, fetches only what is missing
just ingest --assets BTCUSDT --timeframes 1h --start 2020-01-01 --incremental

# Adjust log verbosity
just ingest --assets BTCUSDT --timeframes 1d --start 2023-01-01 --log-level DEBUG
```

Supported timeframes: `1h`, `4h`, `1d`. Incremental mode queries the repository for the latest stored timestamp per `(asset, timeframe)` pair and advances the fetch cursor past that point, keeping the operation idempotent.

### CLI usage — bars

```bash
# Standard bars with default thresholds
just bars --assets BTCUSDT,ETHUSDT --bar-types tick,volume,dollar

# Information-driven bars with a custom threshold
just bars --assets BTCUSDT --bar-types tick_imbalance,tick_run --threshold 500

# All nine bar types
just bars --assets BTCUSDT --bar-types tick,volume,dollar,tick_imbalance,volume_imbalance,dollar_imbalance,tick_run,volume_run,dollar_run
```

---

## Architecture

The project follows Clean Architecture with Domain-Driven Design. Every `src/app/<module>/` layer folder respects strict inward dependency flow:

```
infrastructure  →  application  →  domain
(DuckDB, Binance)   (services)    (entities, protocols)
```

- The domain layer has **zero external dependencies**.
- Protocols (`typing.Protocol`, `I`-prefixed) invert dependencies between layers — there is no `abc.ABC`.
- All data classes — value objects, entities, configs, DTOs — are Pydantic `BaseModel`. No `@dataclass`.
- Pipeline code uses **Polars**; research/validation code that depends on statsmodels or sklearn uses **Pandas**; tight numerical kernels use **NumPy**.

---

## Modules

### `system/`

Cross-cutting infrastructure.

| Component | Path | Responsibility |
|-----------|------|----------------|
| Loguru setup | `system/logging.py` | Console / JSON / file sinks, level switch |
| `ConnectionManager` | `system/database/connection.py` | SQLAlchemy engine + session factory for DuckDB |
| `DatabaseSettings` | `system/database/settings.py` | `pydantic-settings`-driven DuckDB config |
| `BaseRepository` | `system/database/repositories.py` | Shared repository plumbing |
| Alembic | `system/database/alembic/` | All schema evolution; config at `alembic.cfg` |

Two migrations are applied in order by `just migrate`:

1. `001_create_ohlcv_table.py` — OHLCV table with composite primary key
2. `002_add_aggregated_bars_table.py` — Aggregated bars table for all nine bar types

---

### `ohlcv/`

Domain model and repository for raw candles.

```sql
CREATE TABLE ohlcv (
    asset       VARCHAR         NOT NULL,
    timeframe   VARCHAR         NOT NULL,
    timestamp   TIMESTAMPTZ     NOT NULL,
    open        DECIMAL(18, 8)  NOT NULL,
    high        DECIMAL(18, 8)  NOT NULL,
    low         DECIMAL(18, 8)  NOT NULL,
    close       DECIMAL(18, 8)  NOT NULL,
    volume      DOUBLE          NOT NULL,
    PRIMARY KEY (asset, timeframe, timestamp)
);
```

Prices are stored as `DECIMAL(18, 8)` to match Binance's 8-decimal precision. Volume is `DOUBLE`.

`DuckDBOHLCVRepository` satisfies `IOHLCVRepository` structurally:

| Method | Description |
|--------|-------------|
| `ingest(candles)` | Bulk `INSERT OR IGNORE`, returns rows written |
| `ingest_from_parquet(path, asset, timeframe)` | Bulk load via DuckDB `read_parquet()` |
| `query(asset, timeframe, date_range)` | Range query, ordered by `timestamp` |
| `query_split(asset, timeframe, split, partition)` | Query a single `TemporalSplit` partition |
| `query_cross_asset(assets, timeframe, date_range)` | Multi-asset query grouped by symbol |
| `get_available_assets()` | Distinct asset symbols in the store |
| `get_date_range(asset, timeframe)` | Min/max timestamp for an asset + timeframe |
| `count()` | Total row count |

---

### `ingestion/`

Binance REST ingestion pipeline.

```
cli.py
  └── IngestionService (application)
        ├── IMarketDataFetcher (domain protocol)
        │     └── BinanceFetcher (infrastructure) ← BinanceSettings
        └── IOHLCVRepository (domain protocol)
              └── DuckDBOHLCVRepository (infrastructure) ← ConnectionManager
```

| Component | Layer | Responsibility |
|-----------|-------|----------------|
| `BinanceKlineInterval` | domain | StrEnum mapping `Timeframe` → Binance interval strings |
| `FetchRequest` | domain | Frozen Pydantic value object: asset + timeframe + date range |
| `IMarketDataFetcher` | domain | `typing.Protocol` — structural interface for any market-data source |
| `IngestionError / FetchError / RateLimitError` | domain | Exception hierarchy (no external deps) |
| `IngestAssetCommand` | application | Frozen command for a single asset + timeframe |
| `IngestUniverseCommand` | application | Frozen command for an asset × timeframe Cartesian product |
| `IngestionService` | application | `ingest_asset`, `ingest_universe`, `ingest_incremental` |
| `BinanceFetcher` | infrastructure | Paginated kline fetching with `tenacity` exponential-backoff retries |
| `BinanceSettings` | infrastructure | `pydantic-settings.BaseSettings`, `BINANCE_` env prefix |

`BinanceFetcher.fetch_ohlcv()` paginates using the millisecond interval duration (`TIMEFRAME_INTERVAL_MS`) to advance the cursor after each batch. Rate-limit responses (HTTP 429) raise `RateLimitError` immediately; other transient errors are retried up to `max_retries` times with exponential backoff.

---

### `bars/`

All nine alternative bars from López de Prado, *Advances in Financial Machine Learning* (2018), §2.3.

| Type | Aggregator | Sampling trigger | Algorithm |
|------|-----------|------------------|-----------|
| `TICK` | `TickBarAggregator` | Every N input rows | Vectorised Polars cumsum |
| `VOLUME` | `VolumeBarAggregator` | Cumulative volume ≥ threshold | Vectorised Polars cumsum |
| `DOLLAR` | `DollarBarAggregator` | Cumulative `close × volume` ≥ threshold | Vectorised Polars cumsum |
| `TICK_IMBALANCE` | `ImbalanceBarAggregator` | `\|Σ direction\|` ≥ adaptive threshold | Sequential NumPy O(n) |
| `VOLUME_IMBALANCE` | `ImbalanceBarAggregator` | `\|Σ direction × volume\|` ≥ adaptive threshold | Sequential NumPy O(n) |
| `DOLLAR_IMBALANCE` | `ImbalanceBarAggregator` | `\|Σ direction × close × volume\|` ≥ adaptive threshold | Sequential NumPy O(n) |
| `TICK_RUN` | `RunBarAggregator` | Max consecutive run ≥ adaptive threshold | Sequential NumPy O(n) |
| `VOLUME_RUN` | `RunBarAggregator` | Max consecutive run volume ≥ adaptive threshold | Sequential NumPy O(n) |
| `DOLLAR_RUN` | `RunBarAggregator` | Max consecutive run dollar value ≥ adaptive threshold | Sequential NumPy O(n) |

Direction classification: a candle is buy (+1) if `close >= open`, sell (−1) otherwise. Adaptive thresholds update after warmup via EMA: `θ_t = α × |observed| + (1 − α) × θ_{t−1}`, with `α = 2 / (ewm_span + 1)`.

`BarConfig` is a frozen Pydantic model whose SHA-256 `config_hash` acts as a storage key for deduplication:

```python
from src.app.bars.domain.value_objects import BarConfig, BarType

config = BarConfig(
    bar_type=BarType.DOLLAR_IMBALANCE,
    threshold=500_000.0,
    ewm_span=100,       # EMA half-life (>= 10)
    warmup_period=50,    # fixed threshold before EMA kicks in (<= ewm_span)
)
print(config.config_hash)  # 16-char hex; used as storage key
```

```sql
CREATE TABLE aggregated_bars (
    asset           VARCHAR        NOT NULL,
    bar_type        VARCHAR        NOT NULL,
    bar_config_hash VARCHAR(16)    NOT NULL,
    start_ts        TIMESTAMPTZ    NOT NULL,
    end_ts          TIMESTAMPTZ    NOT NULL,
    open            DECIMAL(18, 8) NOT NULL,
    high            DECIMAL(18, 8) NOT NULL,
    low             DECIMAL(18, 8) NOT NULL,
    close           DECIMAL(18, 8) NOT NULL,
    volume          DOUBLE         NOT NULL,
    tick_count      INTEGER        NOT NULL,
    buy_volume      DOUBLE         NOT NULL,
    sell_volume     DOUBLE         NOT NULL,
    vwap            DECIMAL(18, 8) NOT NULL,
    PRIMARY KEY (asset, bar_type, bar_config_hash, start_ts)
);
```

`DuckDBBarRepository` offers `ingest`, `query`, `get_available_configs`, `get_date_range`, `get_latest_end_ts`, `count`, `count_by_config`, and `delete`.

---

### `features/`

Feature engineering: backward-looking technical indicators, forward-looking regression targets, a matrix builder that chains them into a clean `FeatureSet`, and a permutation-test validator that gates features before they reach any model.

**Indicator groups (21 features):**

| Group | Features | Column prefix |
|-------|----------|---------------|
| Returns (4) | Log returns at horizons 1, 4, 12, 24 bars | `logret_` |
| Volatility (5) | Realised vol (3 windows), Garman-Klass, Parkinson | `rv_`, `gk_vol_`, `park_vol_` |
| Momentum (4) | ATR-normalised EMA crossover, ROC at 3 periods | `ema_xover_`, `roc_` |
| Volume (3) | Volume z-score, OBV slope, Amihud illiquidity | `vol_zscore_`, `obv_slope_`, `amihud_` |
| Statistical (5) | Return z-score, Bollinger %B, Bollinger width, price slope, Hurst exponent | `ret_zscore_`, `bbpctb_`, `bbwidth_`, `slope_`, `hurst_` |

Rolling-map features (slope, OBV slope, Hurst) use NumPy callbacks via `rolling_map` in a separate pass. Every feature column is clipped to `[clip_lower, clip_upper]` (default `[-5, 5]`).

**Regression targets:**

| Target | Formula | Column |
|--------|---------|--------|
| Forward log return | `ln(C_{t+h} / C_t)` | `fwd_logret_{h}` |
| Forward realised volatility | `std(r_{t+1}, …, r_{t+h})` | `fwd_vol_{h}` |

Default horizons: returns at 1, 4, 24 bars; volatility at 4, 24. The `fwd_` prefix distinguishes targets from backward-looking indicators. Targets are never produced during live inference (`FeatureConfig.compute_targets = False`).

**Pipeline:**

```
raw OHLCV DataFrame
      │
      ▼
compute_all_indicators(df, IndicatorConfig)   ← Polars expressions, vectorised
      │  (two-pass: batch Expr then rolling_map; clip at end)
      ▼
compute_all_targets(df, TargetConfig)         ← forward-looking negative-shift Polars exprs
      │
      ▼
FeatureMatrixBuilder.build(df, FeatureConfig)
      │  (identify new columns, drop NaN rows, record row counts)
      ▼
FeatureSet(df, feature_columns, target_columns, n_rows_raw, n_rows_clean)
      │
      ▼
FeatureValidator.validate(feature_set, ValidationConfig)
      │  (four independent test batteries — see below)
      ▼
ValidationReport(feature_results, kept_feature_names, dropped_feature_names, …)
```

**Validation gates** — a feature must pass the first three to be kept:

| Battery | Method | Gate |
|---------|--------|------|
| MI permutation test | Mutual information vs. 1000-shuffle null, Phipson-Smyth empirical p-value | BH-corrected p < α |
| Ridge DA / DC-MAE | Single-feature Ridge on a 70/30 temporal split, DA vs. 500-shuffle null | DA empirical p < α |
| Temporal stability | Per-year-window MI significance across configurable boundaries | Significant in ≥ 50% of valid windows |
| Group interaction (informational) | Group vs. individual Ridge R² for synergy/redundancy | Does not affect `keep` |

Benjamini-Hochberg FDR correction controls the false discovery rate across all features simultaneously. A fallback keeps at least `min_features_kept` (default 5) features even when the statistical gates are stringent.

---

### `profiling/`

Per-asset, per-bar-type statistical characterisation of return dynamics — a dispatch that adapts its test battery to the sample size.

**Sample-size tiers:**

| Tier | Samples | Analyses available |
|------|---------|--------------------|
| A | > 2,000 | All: distribution, serial dependence, GARCH, GJR-GARCH, BDS, predictability, SNR |
| B | 500–2,000 | Distribution, serial dependence (VR ≤ 7-day horizon), GARCH, regime labelling, PE, MDE |
| C | < 500 | Descriptive stats, JB test, ACF/PACF, Ljung-Box, regime labelling |

**Analyzer batteries:**

| Analyzer | Key tests |
|----------|-----------|
| `StationarityScreener` | Joint ADF + KPSS → stationary / trend-stationary / unit-root / inconclusive |
| `DistributionAnalyzer` | Jarque-Bera, Student-t MLE, AIC/BIC comparison, KS distance |
| `SerialDependenceAnalyzer` | Multi-lag Ljung-Box, Lo-MacKinlay VR (robust Z₂), Chow-Denning, Granger causality |
| `VolatilityAnalyzer` | GARCH(1,1) with Normal/t/Skewed-t, Engle-Ng sign bias, GJR-GARCH, ARCH-LM, BDS |
| `PredictabilityAnalyzer` | Permutation entropy (H_norm), Jensen-Shannon complexity, Kish N_eff, MDE DA, SNR R² |
| `ProfilingService` | Orchestrator + Benjamini-Hochberg FDR correction across all inferential tests |

All analyzers are stateless. `ProfilingService` injects `DataLoader` and dispatches to every analyzer, collecting results into an immutable `StatisticalReport`. P-values from Ljung-Box (returns and squared), variance ratio, Granger causality, BDS, ARCH-LM, and the Engle-Ng joint F-test are pooled across all `(asset, bar_type)` combinations and corrected simultaneously at `fdr_alpha = 0.05`.

---

### `backtest/`

A self-contained event-driven backtest engine — no external simulation frameworks. BuyAndHold and Random baselines plus a walk-forward runner complete the evaluation loop.

**Domain model:**

| Component | Purpose |
|-----------|---------|
| `Side` | `LONG / SHORT / FLAT` enum |
| `ExecutionConfig` | Frozen config: commission rate, slippage, initial capital |
| `TradeResult` | Immutable closed-trade record: entry/exit price, side, PnL, return |
| `PortfolioSnapshot` | Per-bar equity, cash, position value, drawdown |
| `Signal` | Timestamped trading signal: asset, side, size hint |
| `Position` | Open position: entry price, side, quantity, unrealised PnL |
| `Trade` | In-flight trade; converts to `TradeResult` on close |
| `EquityCurve` | Time-indexed equity series with peak tracking |
| `IStrategy` | `Protocol` — `on_bar(bar, position) → Signal \| None` |
| `IPositionSizer` | `Protocol` — `size(signal, equity, bar) → float` |

**Application layer:**

| Component | File | Purpose |
|-----------|------|---------|
| `ExecutionEngine` | `execution.py` | Next-bar fill, commission + slippage, returns `BacktestResult` |
| `compute_metrics` | `metrics.py` | Lo 2002 AC-corrected Sharpe, max DD, Calmar, win rate, avg trade |
| `compute_buy_and_hold_metrics` | `metrics.py` | Absolute-floor baseline metrics on raw bar returns |
| `BuyAndHoldStrategy` | `baselines.py` | Enters long on first bar and holds — unconditional floor |
| `RandomStrategy` | `baselines.py` | Random long/flat signals — White (2000) null hypothesis |
| `FixedFractionalSizer` | `position_sizer.py` | Sizes by fixed fraction of equity |
| `RegimeConditionalSizer` | `position_sizer.py` | Scales fraction by volatility regime label |
| `cost_sweep` | `cost_sweep.py` | Grid-evaluates `BacktestMetrics` across a commission schedule |
| `WalkForwardRunner` | `walk_forward.py` | Expanding / rolling windows, chains equity across folds |

**Metrics** — `compute_metrics()` returns a `BacktestMetrics` value object containing:

- **Sharpe ratio** — annualised, autocorrelation-corrected per Lo (2002): `SR_AC = SR × √(1 + 2 Σ ρ_k)`
- **Max drawdown** — peak-to-trough percentage from the `EquityCurve`
- **Calmar ratio** — annualised return / max drawdown
- **Win rate** — fraction of `TradeResult` records with positive PnL
- **Average trade return** — mean `TradeResult.pnl_pct`
- **Total PnL** — sum of all closed-trade PnL in quote currency

**Walk-forward modes** — `WalkForwardRunner` supports two window modes via `WindowMode`:

| Mode | Train window grows? | Use case |
|------|---------------------|----------|
| `EXPANDING` | Yes — all history up to fold start | Sufficient data; no retraining cost |
| `ROLLING` | No — fixed lookback window | Regime-aware; prevents concept drift |

The runner receives an `IStrategyFactory` that constructs a fresh strategy per fold, letting training-dependent strategies (classifiers, regressors) be retrained on each fold's training slice before evaluation.

---

### `strategy/`

Five batch trading strategies. Each consumes a `FeatureSet` and emits a Polars DataFrame of `(timestamp, side, strength)` signals. The batch interface complements the backtest engine's per-bar `IStrategy` Protocol — these strategies process a full feature matrix at once rather than bar-by-bar.

| Strategy | Class | Signal logic | Regime profile |
|----------|-------|-------------|----------------|
| Momentum crossover | `MomentumCrossover` | Long when `ema_xover > threshold`, short when `< -threshold`, flat otherwise. Strength = clipped `\|ema_xover\|`. | Trending |
| Mean reversion | `MeanReversion` | Long when `close < lower_BB` AND `hurst < 0.5`; short when `close > upper_BB` AND `hurst < 0.5`. Strength = band-normalised distance. | Range-bound |
| Donchian breakout | `DonchianBreakout` | Long-only: long when `close > rolling_max(high.shift(1))`. Strength = ATR-normalised breakout distance. | Trending (breakouts) |
| Volatility targeting | `VolatilityTargeting` | Always long. Strength = `target_vol / realized_vol`, clipped to `[0, 1]`. | All regimes |
| No-trade | `NoTrade` | Always flat with avoidance confidence. PE gate (`pe_value > pe_threshold` → strength 1.0) or per-bar low-vol filter. | Transition / low-signal |

The `MeanReversion` Hurst filter (`hurst < 0.5`) suppresses signals during trending regimes where mean-reversion logic is unreliable. `DonchianBreakout` applies `.shift(1)` to the rolling high before comparison to eliminate look-ahead bias.

```python
class IStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def generate_signals(self, feature_set: FeatureSet) -> pl.DataFrame: ...
```

The returned DataFrame has three columns: `timestamp` (Datetime, aligned with the feature-set row index), `side` (`"long"` / `"short"` / `"flat"`), and `strength` (Float64 in `[0, 1]`). Signal diversity is validated with pairwise Jaccard similarity — each strategy pair must score below 0.5.

---

### `forecasting/`

Two complementary tracks, each fronted by a Protocol.

#### Direction classification (SIDE)

```python
class IDirectionClassifier(Protocol):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None: ...
    def predict(self, x_test: np.ndarray) -> list[DirectionForecast]: ...
```

`DirectionForecast` carries `predicted_direction` (+1 / −1), `confidence` ∈ [0, 1], and `horizon` (H1 / H4 / H24). Confidence is obtained from native calibration (Logistic), `CalibratedClassifierCV` with Platt / isotonic scaling (LightGBM), or MC Dropout at inference time (GRU).

| Classifier | Class | Key implementation detail |
|------------|-------|--------------------------|
| Logistic baseline | `LogisticBaseline` | sklearn `LogisticRegression`, L2 regularisation, natively calibrated |
| Random forest | `RandomForestClassifier` | sklearn RF with Gini feature importances |
| Gradient boosting | `GradientBoostingClassifier` | LightGBM + `CalibratedClassifierCV` (Platt / isotonic) |
| GRU | `GRUClassifier` | Multi-layer GRU, BCE loss, MC Dropout uncertainty, early stopping |

**Naive baselines:** `MajorityClassifier` (most frequent training label), `PersistenceClassifier` (last observed direction), `MomentumSignClassifier` (sign of the momentum feature).

**Classification metrics** (`classification_metrics.py`): accuracy / precision / recall / F1 (macro and weighted), trapezoidal AUC-ROC with no sklearn dependency, abstention curves at five confidence thresholds, reliability diagrams + ECE, economic accuracy (return-weighted), and asymmetric class weighting (crash penalty 1.5× reflecting crypto negative skewness).

**Label overlap** (`label_overlap.py`) implements López de Prado Ch. 4: sequential bootstrap via conditional-uniqueness sampling, non-overlapping subsampling, Kish effective sample size, and the indicator matrix.

**CPCV splitter** (`infrastructure/cpcv.py`): Combinatorial Purged Cross-Validation (López de Prado, *AFML* Ch. 7 & 12) with purging, embargo, and cross-asset temporal purging for correlated pairs.

**Sanity checks** (`sanity_checks.py`) run the Ojala & Garriga (2010) shuffled-labels permutation test: after fitting on permuted labels, directional accuracy must collapse to `[0.48, 0.52]`.

#### Return regression (SIZE)

| Regressor | Purpose |
|-----------|---------|
| `RidgeRegressor` | Linear baseline with L2 regularisation |
| `LightGBMQuantileRegressor` | Quantile regression with isotonic-regression calibration |
| `GRURegressor` | Multi-layer GRU with MC Dropout for predictive uncertainty |
| HAR-RV | Heterogeneous autoregressive model for realised volatility |
| ARIMA-GARCH(1,1) | Volatility forecasting via ARMA + GARCH(1,1) |
| ACI (`calibration.py`) | Adaptive Conformal Inference — online coverage calibration |

**Standalone regression metrics** (no direction dependency): MAE, RMSE, R², CRPS, QLIKE, Mincer-Zarnowitz R². Direction-conditional metrics (DC-MAE, DC-RMSE) apply only where the direction prediction is correct.

---

### `recommendation/`

Generalised meta-labeling — the thesis centerpiece. A LightGBM recommender takes upstream classifier + regressor outputs and decides whether (and how aggressively) to deploy each candidate signal.

**Domain:**

| Component | Purpose |
|-----------|---------|
| `RecommendationInput` | Frozen input bundle: feature vector + direction forecast + return forecast per `(asset, timestamp)` |
| `Recommendation` | Frozen output: deploy flag + position size + calibrated confidence |
| `IRecommender` | `Protocol` with `fit(...)` and `recommend(...)` |
| `RecommenderConfig` | Pydantic config covering model hyperparameters and deployment thresholds |

**Application:**

| Component | Purpose |
|-----------|---------|
| `GradientBoostingRecommender` | LightGBM recommender with Kelly-adjacent position sizing |
| `RecommenderFeatureBuilder` (`feature_builder.py`) | Assembles L2 features from classifier / regressor / regime groups |
| `label_builder.py` | Constructs the generalised meta-label from realised PnL |
| `pipeline.py` | Expanding walk-forward pipeline with multi-layer temporal purging; L1 OOS predictions are fed as L2 features |
| `metrics.py` | Lo 2002 AC-corrected Sharpe, sizing-value quantification, deployment precision/recall |
| `ablation.py` | Structured ablation with Diebold-Mariano tests across classifier / regressor / regime feature groups |
| `baseline_recommenders.py` | Random, AllAssets, ClassifierOnly, RegressorOnly, EqualWeight baselines |

Split conformal calibration produces a deployment decision with a controlled false-positive rate. The expanding walk-forward pipeline enforces temporal discipline end-to-end — L1 models are trained on each fold's train slice, their out-of-sample predictions become L2 features for the recommender, and purging + embargo eliminate label-overlap leakage.

---

## Data flow

```
Binance REST API
      │  (paginated klines, batch_size=1000)
      ▼
BinanceFetcher.fetch_ohlcv()
      │  (tenacity retries, rate-limit detection)
      ▼
list[OHLCVCandle]   ←── domain entities (Pydantic, Decimal prices)
      │
      ▼
IngestionService.ingest_asset() / ingest_universe() / ingest_incremental()
      │
      ▼
DuckDBOHLCVRepository.ingest()
      │  (INSERT OR IGNORE, composite PK deduplication)
      ▼
data/market.duckdb   (ohlcv table)
      │
      ▼
Bar Aggregators (Tick / Volume / Dollar / Imbalance / Run)
      │  (Polars vectorised or NumPy sequential with adaptive EMA)
      ▼
list[AggregatedBar]   ←── domain entities (Decimal prices, VWAP, buy/sell volume)
      │
      ▼
DuckDBBarRepository.ingest()
      │  (INSERT OR IGNORE, config_hash deduplication)
      ▼
data/market.duckdb   (aggregated_bars table)
      │
      ▼
FeatureMatrixBuilder.build(df, FeatureConfig)
      │  (indicators → targets → NaN drop → FeatureSet)
      ▼
FeatureValidator.validate(feature_set, ValidationConfig)
      │  (MI permutation, Ridge DA / DC-MAE, temporal stability, BH correction)
      ▼
ValidationReport   ←── kept_feature_names, per-feature keep/drop decisions
      │
      ▼
ProfilingService.profile_all(assets, config, partition)
      │  (tier classification, five analyzers per asset-bar pair, BH FDR correction)
      ▼
StatisticalReport   ←── AssetBarProfile per (asset, bar_type), corrected p-values
      │
      ▼
IStrategy.generate_signals(feature_set)
      │  (batch signal generation: momentum, mean-reversion, breakout, vol-target, no-trade)
      ▼
Signal DataFrame   ←── timestamp, side ("long" / "short" / "flat"), strength [0, 1]
      │
      ▼
ExecutionEngine.run(signals, bars, config, sizer)
      │  (next-bar fill, commission + slippage, FixedFractional / RegimeConditional sizing)
      ▼
BacktestResult   ←── EquityCurve, list[Trade], list[PortfolioSnapshot]
      │
      ▼
compute_metrics(result)                           compute_buy_and_hold_metrics(bars)
      │  (Lo 2002 AC-corrected Sharpe, max DD,          │  (absolute-floor baseline)
      │   Calmar, win rate, avg trade, PnL)             │
      ▼                                                 ▼
BacktestMetrics   ←── paired with BuyAndHoldStrategy / RandomStrategy baselines
      │
      ▼
WalkForwardRunner.run(factory, bars, config)
      │  (expanding / rolling windows, equity chaining, IStrategyFactory per fold)
      ▼
WalkForwardResult   ←── per-fold WindowResult, combined EquityCurve, aggregate metrics
      │
      ├──── Return regression (SIZE) ────────────────────────────────────────────────────┐
      │                                                                                   │
      ▼                                                                                   │
IRegressor.fit / IVolatilityForecaster.fit                                              │
      │  (Ridge, LightGBM quantile, GRU+MC Dropout, HAR-RV, ARIMA-GARCH(1,1))           │
      ▼                                                                                   │
RegressionMetrics   ←── MAE, RMSE, R², CRPS, QLIKE, DC-MAE, Mincer-Zarnowitz R²         │
      │                                                                                   │
      └───────────────────────────────────────────────────────────────────────────────────┘
      │
      ├──── Direction classification (SIDE) ─────────────────────────────────────────────┐
      │                                                                                   │
      ▼                                                                                   │
LabelOverlap (sequential bootstrap / Kish N_eff)                                        │
      │  (López de Prado Ch. 4 — deduplication of overlapping forward-return labels)     │
      ▼                                                                                   │
CPCV splitter (infrastructure)                                                          │
      │  (purging + embargo + cross-asset temporal purging)                              │
      ▼                                                                                   │
IDirectionClassifier.fit / predict                                                      │
      │  (Logistic, RandomForest, LightGBM+calibration, GRU+MC Dropout,                 │
      │   Majority, Persistence, MomentumSign baselines)                                 │
      ▼                                                                                   │
ClassificationMetrics   ←── accuracy, AUC-ROC, ECE, abstention curve, economic DA      │
      │                                                                                   │
      ▼                                                                                   │
SanityCheckReport   ←── shuffled-labels permutation test (DA must collapse to ~0.50)   │
      │                                                                                   │
      └───────────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
RecommenderFeatureBuilder → GradientBoostingRecommender.fit / recommend
      │  (L2 features from SIDE+SIZE+regime; split-conformal deployment decision)
      ▼
Recommendation   ←── deploy flag, position size, calibrated confidence
```

---

## Testing

The test suite mirrors the application tree under `src/tests/`:

```
src/tests/
├── conftest.py           # Shared factories: make_asset, make_date_range, make_candle
├── backtest/             # Engine, execution, metrics, baselines, position sizer, walk-forward
├── bars/                 # domain / application / infrastructure splits per bar type
├── features/             # Indicators, targets, matrix, validation, leakage
├── forecasting/          # Regressors, classifiers, naive baselines, CPCV, sanity checks
├── ingestion/
│   ├── conftest.py       # Fakes: FakeMarketDataFetcher, FakeOHLCVRepository; kline builders
│   ├── unit/             # Unit tests for every ingestion component
│   └── e2e/              # End-to-end CLI tests against an in-memory DuckDB
├── profiling/            # Distribution, serial dependence, volatility, predictability, stationarity
├── recommendation/       # Domain, feature builder, label builder, models, pipeline, metrics, ablation
├── research/             # Analysis utilities
└── strategy/             # All five strategies + signal-diversity checks
```

**Pytest markers** defined in `pyproject.toml`:

| Marker | Meaning |
|--------|---------|
| `integration` | Requires a real DuckDB instance (in-memory by default) |
| `e2e` | End-to-end CLI / pipeline tests |

**Scoping runs:**

```bash
just test                           # Full suite
just test -v                        # Verbose
just test src/tests/backtest/       # One module
just test -k "test_execution"       # By name pattern
just test -m integration            # Only integration-marked tests
just test -m "not e2e"              # Exclude e2e tests
```

Unit tests use in-memory fakes for every Protocol — no network, no disk. Integration and e2e tests spin up an in-memory DuckDB instance.

---

## Development standards

### Pre-commit pipeline

Runs on every `git commit` (install with `just install-hooks`):

| Order | Hook | Tool |
|-------|------|------|
| 1 | Formatter | `ruff format` — 119-char lines, double quotes |
| 2 | Linter | `ruff` — ~25 rule categories, including `D` (Google docstrings), `DOC`, `ANN`, `S`, `N`, `PERF`; imports sorted via the `I` group |
| 3 | Type checker | `ty` (Astral, strict mode; excludes `src/tests/`) |

All project configuration lives in `pyproject.toml`. `just lint` runs the same pipeline against every file at once.

### Type hints (Python 3.14)

- `list[X]`, `dict[K, V]`, `X | None`, `X | Y` — not `typing.List`, `typing.Optional`, `typing.Union`.
- PEP 695 type aliases: `type OHLCVFrame = pl.DataFrame`.
- Every local variable carries an explicit type annotation (project convention).
- No `Any` unless interfacing with an untyped third-party library — comment why.
- `from __future__ import annotations` at the top of every module.

### Docstrings

Google-style docstrings are required on every public module, class, method, and function. A one-line module-level docstring tops every `.py` file.

---

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request to `main` or `dev`:

- **Lint job:** `ruff format --check`, `ruff check`, and `ty` strict.
- **Test job:** full `uv run pytest src/tests/ -v`.

Both jobs are the same commands as `just lint` and `just test` — local success is a faithful preview of CI.

A second workflow, `.github/release.yml`, auto-generates release notes grouped by PR labels when a GitHub release is cut.

---

## Documentation site

The MkDocs Material site lives under `docs/` with configuration in `mkdocs.yml`. `mkdocstrings` auto-generates API reference pages from source docstrings; MathJax renders the formulae referenced throughout the forecasting and metrics modules.

```bash
just serve
# → Live-reload server at http://127.0.0.1:8000
```

---

## Author

**Dmytro Khvedchuk** — <dmytro.khvedchuk.dev@gmail.com>
