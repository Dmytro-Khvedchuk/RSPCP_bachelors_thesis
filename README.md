# RSPCP — Recommendation System for Predicting Cryptocurrency Prices

> Bachelor's thesis — a probabilistic ML recommendation system for cryptocurrency strategy deployment.
> **Author:** Dmytro Khvedchuk

---

## Overview

RSPCP is a research-grade algorithmic trading system built for a bachelor's thesis. It combines
two complementary forecasting tracks — directional classification (SIDE) and return regression (SIZE) —
into a trained recommendation system that selects which trading signals to act on. The evaluation
framework is statistically rigorous: walk-forward cross-validation, Monte Carlo permutation tests,
and deflated Sharpe ratios guard against overfitting.

**Current state:** Phases 1–8 complete — OHLCV ingestion, López de Prado alternative bars,
RC1 research checkpoint, full feature engineering pipeline (21 indicators after Phase 7 audit,
regression targets, feature matrix builder, and permutation-test validation), statistical
profiling (distribution, serial dependence, volatility modeling, predictability assessment),
RC2 profiling closure (6 audit gaps resolved), and a complete event-driven backtest engine
(domain model, execution layer with next-bar fill semantics, Lo 2002 corrected metrics,
BuyAndHold + Random baselines, and walk-forward runner with expanding/rolling windows).
1,429 tests passing. Phase 9 (Base Trading Strategies) is next.

---

## Architecture

The project follows Clean Architecture with Domain-Driven Design. Each module is split into three
layers with strict inward dependency flow:

```
infrastructure  →  application  →  domain
(DuckDB, Binance)  (services)    (entities, protocols)
```

Domain layer has zero external dependencies. Protocols (`typing.Protocol`, `I`-prefixed) invert
dependencies between layers. All data classes use Pydantic `BaseModel` — no raw dataclasses.

### Planned module roadmap

| Phase | Module | Purpose | Status |
|-------|--------|---------|--------|
| — | `system/` | Logging, DB connection, Alembic | Done |
| 1 | `ohlcv/` | Candle entities + DuckDB repository | Done |
| 1 | `ingestion/` | Binance fetcher + ingestion service + CLI | Done |
| 2 | `bars/` | Lopez de Prado alternative bars | Done |
| 3 | `research/` | RC1 analysis (coverage, returns, ACF, bar comparison, charts) | Done |
| 4 | `features/` | Technical indicators, regression targets, matrix builder, validation | Done |
| 5 | `profiling/` | Statistical profiling per asset | Done |
| 6–7 | (research) | RC2 profiling closure — 6 audit gaps, stationarity policy, Tier B protocol | Done |
| 8 | `backtest/` | Event-driven backtest engine | Done |
| 9 | `strategy/` | Momentum, DRTS, mean-reversion strategies | Next |
| 10–11 | `forecasting/` | Classification (SIDE) + regression (SIZE) | Planned |
| 12 | `recommendation/` | ML recommendation system (meta-labeling) | Planned |
| 14 | `evaluation/` | Monte Carlo, PBO, DSR, MCS | Planned |
| 16 | `live/` | Live paper trading engine | Planned |
| 17 | `dashboard/` | FastAPI + Streamlit/Dash | Planned |

---

## Project Structure

```
RSPCP_bachelors_thesis/
├── src/
│   ├── app/
│   │   ├── bars/                # Phase 2 — López de Prado alternative bars
│   │   │   ├── domain/          # BarType, BarConfig, AggregatedBar, IBarAggregator, IBarRepository
│   │   │   ├── application/     # Tick/Volume/Dollar/Imbalance/Run bar aggregators
│   │   │   └── infrastructure/  # DuckDBBarRepository
│   │   ├── features/            # Phase 4 — feature engineering + validation
│   │   │   ├── domain/          # IndicatorConfig, TargetConfig, FeatureConfig, FeatureSet, ValidationConfig
│   │   │   │                    # FeatureValidationResult, InteractionTestResult, ValidationReport
│   │   │   └── application/     # indicators.py, targets.py, feature_matrix.py, validation.py
│   │   ├── profiling/           # Phase 5 — statistical profiling per asset
│   │   │   ├── domain/          # DataPartition, SampleTier, TierConfig, all profile value objects
│   │   │   │                    # DistributionProfile, AutocorrelationProfile, VolatilityProfile
│   │   │   │                    # PredictabilityProfile, StationarityReport, StatisticalReport
│   │   │   └── application/     # distribution.py, serial_dependence.py, volatility.py,
│   │   │                        # predictability.py, stationarity.py, services.py
│   │   ├── backtest/            # Phase 8 — event-driven backtest engine
│   │   │   ├── domain/          # Side, ExecutionConfig, TradeResult, PortfolioSnapshot,
│   │   │   │                    # Signal, Position, Trade, EquityCurve, IStrategy, IPositionSizer
│   │   │   └── application/     # ExecutionEngine, metrics.py, baselines.py, position_sizer.py,
│   │   │                        # cost_sweep.py, walk_forward.py
│   │   ├── ingestion/           # Phase 1 — Binance OHLCV ingestion
│   │   │   ├── domain/          # BinanceKlineInterval, FetchRequest, exceptions, IMarketDataFetcher
│   │   │   ├── application/     # IngestionService, IngestAssetCommand, IngestUniverseCommand
│   │   │   ├── infrastructure/  # BinanceFetcher (tenacity retries), BinanceSettings
│   │   │   └── cli.py           # Typer CLI entry point (just ingest)
│   │   ├── ohlcv/               # OHLCV domain model + DuckDB repository
│   │   │   ├── domain/          # OHLCVCandle, Asset, Timeframe, DateRange, TemporalSplit
│   │   │   └── infrastructure/  # DuckDBOHLCVRepository
│   │   ├── research/            # Phase 3 — RC1 analysis services
│   │   └── system/              # Cross-cutting concerns
│   │       ├── logging.py       # Loguru setup
│   │       └── database/        # ConnectionManager, DatabaseSettings, BaseRepository, Alembic
│   └── tests/
│       ├── conftest.py          # Shared factories: make_asset, make_date_range, make_candle
│       ├── backtest/            # 186 tests — domain, execution, metrics, baselines,
│       │                        # position sizer, cost sweep, walk-forward
│       ├── bars/                # 260 tests — domain, application, infrastructure, statistical
│       ├── features/            # 195 tests — indicators, targets, matrix, validation, leakage
│       ├── profiling/           # 188 tests — distribution, serial dependence, volatility,
│       │                        # predictability, stationarity, service orchestration
│       └── ingestion/
│           ├── conftest.py      # Fakes: FakeMarketDataFetcher, FakeOHLCVRepository; kline builders
│           ├── unit/            # Unit tests for all ingestion components
│           └── e2e/             # End-to-end CLI tests
├── data/
│   └── market.duckdb            # Persistent DuckDB store (gitignored)
├── docs/                        # MkDocs source (mkdocs-material + mkdocstrings)
├── legacy_project/              # Reference code only — do not modify
├── main.py                      # Application entry point
├── pyproject.toml               # All tool config: ruff, pyright, isort, pytest, deps
├── justfile                     # Task runner commands
├── mkdocs.yml                   # Documentation site config
├── .pre-commit-config.yaml      # Pre-commit hooks
└── IMPLEMENTATION_PLAN.md       # Full 17-phase plan with statistical test framework
```

---

## Prerequisites

- Python 3.14+
- [`uv`](https://docs.astral.sh/uv/) package manager
- [`just`](https://github.com/casey/just) task runner
- Binance API key and secret (read-only permissions sufficient)

---

## Quick Start

```bash
# 1. Clone and install dependencies
uv sync

# 2. Configure environment
cp .example.env .env
# Edit .env: set DUCKDB_PATH, BINANCE_API_KEY, BINANCE_SECRET_KEY

# 3. Create the data directory
mkdir -p data/

# 4. Run database migrations
just migrate

# 5. Ingest historical OHLCV data
just ingest --assets BTCUSDT,ETHUSDT --timeframes 1h,4h --start 2020-01-01
```

---

## Configuration

All settings are loaded from environment variables (`.env` file at project root).

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DUCKDB_PATH` | `:memory:` | Path to the DuckDB file (e.g. `data/market.duckdb`) |
| `DUCKDB_MEMORY_LIMIT` | `4GB` | DuckDB memory cap |
| `DUCKDB_THREADS` | auto | Thread count (`-1` = auto-detect) |
| `DUCKDB_READ_ONLY` | `false` | Open in read-only mode |

### Binance

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_API_KEY` | required | Binance REST API key |
| `BINANCE_SECRET_KEY` | required | Binance REST API secret |
| `BINANCE_BATCH_SIZE` | `1000` | Klines per API request (max 1000) |
| `BINANCE_MAX_RETRIES` | `5` | Retry attempts on transient failures |
| `BINANCE_RETRY_MIN_WAIT` | `1` | Minimum backoff seconds |
| `BINANCE_RETRY_MAX_WAIT` | `10` | Maximum backoff seconds |

---

## Usage

### Data Ingestion

The `just ingest` command fetches the Cartesian product of `--assets x --timeframes`
from the Binance REST API and writes candles to DuckDB.

```bash
# Full historical ingest for multiple assets and timeframes
just ingest --assets BTCUSDT,ETHUSDT,SOLUSDT --timeframes 1h,4h,1d --start 2020-01-01

# Ingest up to a specific end date
just ingest --assets BTCUSDT --timeframes 1h --start 2020-01-01 --end 2024-01-01

# Incremental top-up — skips already-stored data, fetches only what is missing
just ingest --assets BTCUSDT --timeframes 1h --start 2020-01-01 --incremental

# Adjust log verbosity
just ingest --assets BTCUSDT --timeframes 1d --start 2023-01-01 --log-level DEBUG
```

**Supported timeframes:** `1h`, `4h`, `1d`

**Incremental mode** queries the repository for the latest stored timestamp per
`asset + timeframe` pair and advances the fetch start past that point, avoiding
redundant API calls while remaining idempotent.

### Database Migrations

```bash
just migrate                        # Apply all pending migrations
just migration "add bars table"     # Create a new migration
just migrate-down                   # Rollback one step
```

Migrations live in `src/app/system/database/alembic/versions/`. The OHLCV table
schema (migration `001`) defines a composite primary key on `(asset, timeframe, timestamp)`
and a matching index for fast range queries.

---

## Development

### Install dev dependencies and hooks

```bash
uv sync --dev
just install-hooks
```

### Run all tests

```bash
just test
just test -v                        # verbose
just test src/tests/ingestion/      # scope to one module
```

Tests are structured into `unit/` and `e2e/` subdirectories. Unit tests use
in-memory fakes (`FakeMarketDataFetcher`, `FakeOHLCVRepository`) — no network or
database required. E2E tests wire the full CLI against a real in-memory DuckDB.

### Lint and type-check

```bash
just lint           # ruff format, ruff lint (incl. import sorting), pyright — same as CI
```

Pre-commit hooks run the same checks automatically on every `git commit`.

### Code quality standards

| Tool | Rule |
|------|------|
| `ruff format` | 119-char lines, double quotes |
| `ruff lint` | ~20 rule categories including ANN, D (Google), N, UP, S, B, I (imports) |
| `pyright --strict` | Full static type checking (Python 3.14) |

All public modules, classes, methods, and functions require Google-style docstrings.
Every local variable must carry an explicit type annotation.

### Documentation site

```bash
just serve          # Live-reloading MkDocs server at http://127.0.0.1:8000
```

---

## Ingestion Module — Technical Detail

### Dependency graph

```
cli.py
  └── IngestionService (application)
        ├── IMarketDataFetcher (domain protocol)
        │     └── BinanceFetcher (infrastructure) ← BinanceSettings
        └── IOHLCVRepository (domain protocol)
              └── DuckDBOHLCVRepository (infrastructure) ← ConnectionManager
```

### Key components

| Component | Layer | Responsibility |
|-----------|-------|----------------|
| `BinanceKlineInterval` | domain | StrEnum mapping `Timeframe` → Binance interval strings |
| `FetchRequest` | domain | Frozen Pydantic value object: asset + timeframe + date range |
| `IMarketDataFetcher` | domain | `typing.Protocol` — structural interface for any market data source |
| `IngestionError / FetchError / RateLimitError` | domain | Exception hierarchy (no external deps) |
| `IngestAssetCommand` | application | Frozen command for a single asset + timeframe |
| `IngestUniverseCommand` | application | Frozen command for an asset × timeframe Cartesian product |
| `IngestionService` | application | `ingest_asset`, `ingest_universe`, `ingest_incremental` |
| `BinanceFetcher` | infrastructure | Paginated kline fetching with tenacity exponential-backoff retries |
| `BinanceSettings` | infrastructure | `pydantic-settings.BaseSettings`, `BINANCE_` env prefix |

`BinanceFetcher.fetch_ohlcv()` paginates using the millisecond interval duration
(`TIMEFRAME_INTERVAL_MS`) to advance the cursor after each batch. Rate-limit responses
(HTTP 429) are raised immediately as `RateLimitError` without retry; other transient
errors are retried up to `max_retries` times with exponential backoff via `tenacity`.

---

## OHLCV Module — Technical Detail

### Schema

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

Prices are stored as `DECIMAL(18,8)` matching Binance's 8-decimal precision. Volume
is `DOUBLE` because sub-satoshi precision is unnecessary there.

### Repository API

`DuckDBOHLCVRepository` satisfies `IOHLCVRepository` via structural subtyping:

| Method | Description |
|--------|-------------|
| `ingest(candles)` | Bulk `INSERT OR IGNORE`, returns rows written |
| `ingest_from_parquet(path, asset, timeframe)` | Bulk-load via DuckDB `read_parquet()` |
| `query(asset, timeframe, date_range)` | Range query, ordered by timestamp |
| `query_split(asset, timeframe, split, partition)` | Query a single `TemporalSplit` partition |
| `query_cross_asset(assets, timeframe, date_range)` | Multi-asset query grouped by symbol |
| `get_available_assets()` | Distinct asset symbols in the store |
| `get_date_range(asset, timeframe)` | Min/max timestamp for an asset + timeframe |
| `count()` | Total row count |

---

## Features Module — Technical Detail

Implements the full Phase 4 feature engineering pipeline: backward-looking technical
indicators, forward-looking regression targets, a matrix builder that chains them into a
clean `FeatureSet`, and a permutation-test validator that gates features before they reach
any model.

### Feature groups (21 features — post Phase 7 audit)

`atr_14` and `rsi_14` were dropped in Phase 7.2 due to universal degeneracy across all
assets and bar types. 7 features require a stationarity transformation before modeling
(`amihud_24`, `bbwidth_20_2.0`, `gk_vol_24`, `park_vol_24`, `rv_12`, `rv_24`, `rv_48`).

| Group | Features | Column prefix |
|-------|----------|---------------|
| Returns (4) | Log returns at horizons 1, 4, 12, 24 bars | `logret_` |
| Volatility (5) | Realized vol (3 windows), Garman-Klass, Parkinson | `rv_`, `gk_vol_`, `park_vol_` |
| Momentum (4) | ATR-normalised EMA crossover, ROC at 3 periods | `ema_xover_`, `roc_` |
| Volume (3) | Volume z-score, OBV slope, Amihud illiquidity ratio | `vol_zscore_`, `obv_slope_`, `amihud_` |
| Statistical (5) | Return z-score, Bollinger %B, Bollinger width, price slope, Hurst exponent | `ret_zscore_`, `bbpctb_`, `bbwidth_`, `slope_`, `hurst_` |

Rolling-map features (slope, OBV slope, Hurst) use NumPy callbacks via `rolling_map` and
are applied in a separate pass. All feature columns are clipped to `[clip_lower, clip_upper]`
(default `[-5, 5]`) to prevent outliers from dominating downstream models.

### Regression targets

| Target | Formula | Column |
|--------|---------|--------|
| Forward log return | `ln(C_{t+h} / C_t)` | `fwd_logret_{h}` |
| Forward realized volatility | `std(r_{t+1}, …, r_{t+h})` | `fwd_vol_{h}` |

Default horizons: returns at 1, 4, 24 bars; volatility at 4, 24 bars. The `fwd_` prefix
distinguishes targets from backward-looking indicators. Targets are never present during
live inference (`FeatureConfig.compute_targets=False`).

### Pipeline

```
raw OHLCV DataFrame
      │
      ▼
compute_all_indicators(df, IndicatorConfig)   ← Polars expressions, vectorised
      │  (two-pass: batch Expr then rolling_map; clip at end)
      ▼
compute_all_targets(df, TargetConfig)         ← forward-looking, negative-shift Polars exprs
      │
      ▼
FeatureMatrixBuilder.build(df, FeatureConfig)
      │  (identify new columns, drop NaN rows, record row counts)
      ▼
FeatureSet(df, feature_columns, target_columns, n_rows_raw, n_rows_clean)
      │
      ▼
FeatureValidator.validate(feature_set, ValidationConfig)
      │  (four independent test batteries, see below)
      ▼
ValidationReport(feature_results, kept_feature_names, dropped_feature_names, …)
```

### Validation pipeline (Phase 4D)

A feature passes all three gates to be kept (`FeatureValidationResult.keep = True`):

| Battery | Method | Gate |
|---------|--------|------|
| MI permutation test | Mutual information vs. 1000-shuffle null, Phipson-Smyth empirical p-value | BH-corrected p < α |
| Ridge DA / DC-MAE | Single-feature Ridge on temporal 70/30 split, DA vs. 500-shuffle null | DA empirical p < α |
| Temporal stability | Per-year-window MI significance across configurable year boundaries | Significant in ≥ 50% of valid windows |
| Group interaction (informational) | Group vs. individual Ridge R-squared for synergy/redundancy | Does not affect `keep` |

Benjamini-Hochberg FDR correction is applied to the MI p-values to control false discovery
rate across all features simultaneously. A fallback ensures at least `min_features_kept`
(default 5) features are kept even if the statistical gates are too strict for the dataset.

---

## Profiling Module — Technical Detail

Implements Phase 5 statistical profiling: per-asset, per-bar-type characterization of return
dynamics to validate bar-type suitability before model training (RC2 checkpoint).

### Tier classification

Every `(asset, bar_type)` combination is classified into a sample-size tier before any test
is run. The tier gates which analyses are available:

| Tier | Samples | Analyses available |
|------|---------|--------------------|
| A | > 2,000 | All: distribution, serial dependence, GARCH, GJR-GARCH, BDS, predictability, SNR |
| B | 500–2,000 | Distribution, serial dependence (VR ≤ 7-day horizon), GARCH, regime labeling, PE, MDE |
| C | < 500 | Descriptive stats, JB test, ACF/PACF, Ljung-Box, regime labeling |

### Analysis batteries

| Phase | Analyzer | Key tests |
|-------|----------|-----------|
| 5pre | `StationarityScreener` | Joint ADF + KPSS → stationary / trend_stationary / unit_root / inconclusive |
| 5A | `DistributionAnalyzer` | Jarque-Bera, Student-t MLE, AIC/BIC comparison, KS distance |
| 5B | `SerialDependenceAnalyzer` | Multi-lag Ljung-Box, Lo-MacKinlay VR (robust Z2), Chow-Denning, Granger causality |
| 5C | `VolatilityAnalyzer` | GARCH(1,1) with Normal/t/Skewed-t, Engle-Ng sign bias, GJR-GARCH, ARCH-LM, BDS |
| 5D | `PredictabilityAnalyzer` | Permutation entropy (H_norm), Jensen-Shannon complexity, Kish N_eff, MDE DA, SNR R² |
| 5E | `ProfilingService` | Orchestration + Benjamini-Hochberg FDR correction across all inferential tests |

All analyzers are stateless (no constructor dependencies). The `ProfilingService` orchestrator
injects `DataLoader` and dispatches to each analyzer, collecting results into an immutable
`StatisticalReport`.

### FDR correction

P-values from Ljung-Box (returns and squared), variance ratio, Granger causality, BDS,
ARCH-LM, and sign bias joint F-test are pooled across all `(asset, bar_type)` combinations
and corrected simultaneously via Benjamini-Hochberg at `fdr_alpha=0.05`. Each `CorrectedPValue`
object stores both the raw and corrected p-value with pre/post-correction significance flags.

---

## Backtest Module — Technical Detail

Implements Phase 8: a self-contained event-driven backtest engine with no external
simulation frameworks. Two complementary baseline strategies (BuyAndHold, Random) and a
walk-forward runner cover the full evaluation path from domain model through to fold-level
metrics.

### Domain model

| Component | Layer | Purpose |
|-----------|-------|---------|
| `Side` | domain | `LONG / SHORT / FLAT` enum |
| `ExecutionConfig` | domain | Frozen config: commission rate, slippage, initial capital |
| `TradeResult` | domain | Immutable closed-trade record: entry/exit price, side, PnL, return |
| `PortfolioSnapshot` | domain | Per-bar equity, cash, position value, drawdown |
| `Signal` | domain | Timestamped trading signal: asset, side, size hint |
| `Position` | domain | Open position: entry price, side, quantity, unrealised PnL |
| `Trade` | domain | In-flight trade; converts to `TradeResult` on close |
| `EquityCurve` | domain | Time-indexed equity series with peak-tracking for drawdown |
| `IStrategy` | domain | `typing.Protocol` — `on_bar(bar, position) → Signal \| None` |
| `IPositionSizer` | domain | `typing.Protocol` — `size(signal, equity, bar) → float` |

### Application layer

| Component | File | Purpose |
|-----------|------|---------|
| `ExecutionEngine` | `execution.py` | Next-bar fill; applies commission + slippage; returns `BacktestResult` |
| `compute_metrics` | `metrics.py` | Lo (2002) AC-corrected Sharpe, max drawdown, Calmar, win rate, avg trade |
| `compute_buy_and_hold_metrics` | `metrics.py` | Absolute-floor baseline metrics on raw bar returns |
| `BuyAndHoldStrategy` | `baselines.py` | Enters long on first bar and holds — unconditional floor |
| `RandomStrategy` | `baselines.py` | Random long/flat signals — White (2000) null hypothesis |
| `FixedFractionalSizer` | `position_sizer.py` | Sizes by fixed fraction of equity |
| `RegimeConditionalSizer` | `position_sizer.py` | Scales fraction by volatility regime label |
| `cost_sweep` | `cost_sweep.py` | Grid-evaluates `BacktestMetrics` across a commission schedule |
| `WalkForwardRunner` | `walk_forward.py` | Expanding/rolling windows; chains equity across folds |

### Metrics

`compute_metrics()` returns a `BacktestMetrics` value object with:

- **Sharpe ratio** — annualised, autocorrelation-corrected per Lo (2002): `SR_AC = SR × √(1 + 2 Σ ρ_k)`
- **Max drawdown** — peak-to-trough percentage from the `EquityCurve`
- **Calmar ratio** — annualised return / max drawdown
- **Win rate** — fraction of `TradeResult` records with positive PnL
- **Average trade return** — mean `TradeResult.pnl_pct`
- **Total PnL** — sum of all closed-trade PnL in quote currency

### Walk-forward modes

`WalkForwardRunner` supports two window modes via `WindowMode`:

| Mode | Train window grows? | Use case |
|------|---------------------|----------|
| `EXPANDING` | Yes — all history up to fold start | Sufficient data; no retraining cost |
| `ROLLING` | No — fixed lookback window | Regime-aware; prevents concept drift |

The runner receives an `IStrategyFactory` protocol instance that constructs a fresh strategy
per fold, enabling training-dependent strategies (classifiers, regressors) to be retrained
on each fold's training slice before evaluation on the test slice.

---

## CI/CD

Two GitHub Actions workflows run on pull requests to `main`:

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yml` | PR to `main` | Lint & type check (ruff format, ruff lint, pyright), then full test suite |
| `release.yml` | GitHub release | Auto-generates release notes grouped by PR labels |

The CI pipeline mirrors the local `just lint` + `just test` commands exactly.

---

## Bars Module — Technical Detail

Implements all nine alternative bar types from López de Prado, *Advances in Financial
Machine Learning* (2018), §2.3, plus a reserved time bar type.

### Bar types

| Type | Aggregator | Sampling trigger | Algorithm |
|------|-----------|------------------|-----------|
| `TICK` | `TickBarAggregator` | Every N input rows | Vectorized Polars cumsum |
| `VOLUME` | `VolumeBarAggregator` | Cumulative volume ≥ threshold | Vectorized Polars cumsum |
| `DOLLAR` | `DollarBarAggregator` | Cumulative `close × volume` ≥ threshold | Vectorized Polars cumsum |
| `TICK_IMBALANCE` | `ImbalanceBarAggregator` | `\|Σ direction\|` ≥ adaptive threshold | Sequential NumPy O(n) |
| `VOLUME_IMBALANCE` | `ImbalanceBarAggregator` | `\|Σ direction × volume\|` ≥ adaptive threshold | Sequential NumPy O(n) |
| `DOLLAR_IMBALANCE` | `ImbalanceBarAggregator` | `\|Σ direction × close × volume\|` ≥ adaptive threshold | Sequential NumPy O(n) |
| `TICK_RUN` | `RunBarAggregator` | Max consecutive run ≥ adaptive threshold | Sequential NumPy O(n) |
| `VOLUME_RUN` | `RunBarAggregator` | Max consecutive run volume ≥ adaptive threshold | Sequential NumPy O(n) |
| `DOLLAR_RUN` | `RunBarAggregator` | Max consecutive run dollar value ≥ adaptive threshold | Sequential NumPy O(n) |

**Direction classification:** a candle is buy (+1) if `close >= open`, sell (−1) otherwise.

**Adaptive threshold:** after a configurable warmup period, the threshold is updated via
EMA: `θ_t = α × |observed| + (1 − α) × θ_{t−1}`, where `α = 2 / (ewm_span + 1)`.

### Configuration

`BarConfig` is a frozen Pydantic model with SHA-256 config hash for storage deduplication:

```python
from src.app.bars.domain.value_objects import BarConfig, BarType

config = BarConfig(
    bar_type=BarType.DOLLAR_IMBALANCE,
    threshold=500_000.0,
    ewm_span=100,       # EMA half-life (>= 10)
    warmup_period=50,    # Fixed threshold before EMA kicks in (<= ewm_span)
)
print(config.config_hash)  # 16-char hex, used as storage key
```

### Schema

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

### Repository API

`DuckDBBarRepository` satisfies `IBarRepository` via structural subtyping:

| Method | Description |
|--------|-------------|
| `ingest(bars, config_hash)` | Bulk `INSERT OR IGNORE`, returns rows written |
| `query(asset, bar_type, config_hash, date_range)` | Range query ordered by `start_ts` |
| `get_available_configs(asset)` | Distinct `(bar_type, config_hash)` pairs |
| `get_date_range(asset, bar_type, config_hash)` | Min/max `start_ts` range |
| `get_latest_end_ts(asset, bar_type, config_hash)` | Latest `end_ts` for incremental ingestion |
| `count()` / `count_by_config(...)` | Row counts (total or filtered) |
| `delete(asset, bar_type, config_hash)` | Remove bars for re-computation |

---

## Data Flow

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
Bar Aggregators (Tick/Volume/Dollar/Imbalance/Run)
      │  (Polars vectorized or NumPy sequential with adaptive EMA)
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
      │  (MI permutation, Ridge DA/DC-MAE, temporal stability, BH correction)
      ▼
ValidationReport   ←── kept_feature_names, per-feature keep/drop decisions
      │
      ▼
ProfilingService.profile_all(assets, config, partition)
      │  (tier classification, 5 analyzers per asset-bar pair, BH FDR correction)
      ▼
StatisticalReport   ←── AssetBarProfile per (asset, bar_type), corrected p-values
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
```

---

## Implementation Plan

The full plan is in [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md). Summary:

**Block I — Data & Infrastructure (Phases 1–8)**

Ingestion → alternative bars → RC1 → features → profiling → RC2 closure ✓ → backtest engine ✓ → strategies

**Block II — Models & Recommendation (Phases 9–14)**

Direction classification → return regression → RC3 → ML recommendation system → RC4 → statistical proof

**Block III — Polishing & Production (Phases 15–17)**

Pipeline hardening → live paper trading → FastAPI + Streamlit dashboard

Research checkpoints (RC1–RC4) are explicit go/no-go decision points interleaved between
building phases. Negative results are valid and will be documented.
