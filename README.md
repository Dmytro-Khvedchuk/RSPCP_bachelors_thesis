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

**Current state:** Phase 1 complete — full Binance OHLCV ingestion pipeline with DuckDB storage,
Clean Architecture throughout, 152 tests passing, and CI running on every PR.

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
| 2 | `bars/` | Lopez de Prado alternative bars | Planned |
| 4 | `features/` | Feature engineering + targets | Planned |
| 5 | `profiling/` | Statistical profiling per asset | Planned |
| 7 | `backtest/` | Event-driven backtest engine | Planned |
| 8 | `strategy/` | Momentum, DRTS, mean-reversion strategies | Planned |
| 9–10 | `forecasting/` | Classification (SIDE) + regression (SIZE) | Planned |
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
│   │   ├── ingestion/           # Phase 1 — Binance OHLCV ingestion
│   │   │   ├── domain/          # BinanceKlineInterval, FetchRequest, exceptions, IMarketDataFetcher
│   │   │   ├── application/     # IngestionService, IngestAssetCommand, IngestUniverseCommand
│   │   │   ├── infrastructure/  # BinanceFetcher (tenacity retries), BinanceSettings
│   │   │   └── cli.py           # Typer CLI entry point (just ingest)
│   │   ├── ohlcv/               # OHLCV domain model + DuckDB repository
│   │   │   ├── domain/          # OHLCVCandle, Asset, Timeframe, DateRange, TemporalSplit
│   │   │   └── infrastructure/  # DuckDBOHLCVRepository
│   │   └── system/              # Cross-cutting concerns
│   │       ├── logging.py       # Loguru setup
│   │       └── database/        # ConnectionManager, DatabaseSettings, BaseRepository, Alembic
│   └── tests/
│       ├── conftest.py          # Shared factories: make_asset, make_date_range, make_candle
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
just lint           # ruff format, ruff lint, pyright, isort — same as CI
```

Pre-commit hooks run the same checks automatically on every `git commit`.

### Code quality standards

| Tool | Rule |
|------|------|
| `ruff format` | 119-char lines, double quotes |
| `ruff lint` | ~20 rule categories including ANN, D (Google), N, UP, S, B |
| `pyright --strict` | Full static type checking (Python 3.14) |
| `isort` | STDLIB → THIRDPARTY → FIRSTPARTY → LOCAL |

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

## CI/CD

Two GitHub Actions workflows run on pull requests to `main`:

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yml` | PR to `main` | Lint & type check (ruff, pyright, isort), then full test suite |
| `release.yml` | GitHub release | Auto-generates release notes grouped by PR labels |

The CI pipeline mirrors the local `just lint` + `just test` commands exactly.

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
```

---

## Implementation Plan

The full plan is in [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md). Summary:

**Block I — Data & Infrastructure (Phases 1–8)**

Ingestion → alternative bars → RC1 → features → profiling → RC2 → backtest engine → strategies

**Block II — Models & Recommendation (Phases 9–14)**

Direction classification → return regression → RC3 → ML recommendation system → RC4 → statistical proof

**Block III — Polishing & Production (Phases 15–17)**

Pipeline hardening → live paper trading → FastAPI + Streamlit dashboard

Research checkpoints (RC1–RC4) are explicit go/no-go decision points interleaved between
building phases. Negative results are valid and will be documented.
