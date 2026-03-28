# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> The full implementation plan is in `IMPLEMENTATION_PLAN.md` (17 phases, 3 blocks).

## Project Identity

- **Title:** Recommendation System for Predicting Cryptocurrency Prices (RSPCP)
- **Type:** Bachelor's thesis — probabilistic ML recommendation system for crypto strategy deployment
- **Python:** 3.14, managed by `uv` (not pip, not poetry)

## Commands

```bash
# Development
just run                    # Run main.py
just lint                   # Run all pre-commit hooks (ruff format + ruff lint + ty)
just test                   # Run full test suite (uv run pytest src/tests/)
just test src/tests/research/  # Run specific test module
just test -k "test_name"    # Run single test by name
just test -m integration    # Run only integration tests
just test -m "not e2e"      # Exclude e2e tests
just serve                  # MkDocs live-reload server

# Data pipeline
just ingest --assets BTCUSDT,ETHUSDT --timeframes 1h,4h --start 2020-01-01
just bars --assets BTCUSDT --bar-types tick,volume,dollar

# Database
just migrate                # Run all pending Alembic migrations
just migration "msg"        # Create new migration
just migrate-down           # Rollback one migration

# Dependencies
just add <package>          # Add dependency via uv
```

## Architecture

### Clean Architecture + DDD

Every module under `src/app/<module>/` follows the same layered structure:

```
domain/          # Entities, value objects, protocols — NO external deps
application/     # Services, use cases — depends on domain only
infrastructure/  # Concrete implementations — depends on domain + external libs
```

- Dependencies flow inward: infrastructure → application → domain
- `typing.Protocol` for dependency inversion (not ABC). `I`-prefix: `IOHLCVRepository`, `IBarAggregator`
- `pydantic.BaseModel` for ALL value objects, configs, DTOs, entities. NO `dataclasses`.
- `frozen=True` for immutable value objects, `pydantic-settings.BaseSettings` for env config

### Module Status

| Module | Status | Purpose |
|--------|--------|---------|
| `system/` | ✅ | Logging (Loguru), database (DuckDB + SQLAlchemy + Alembic) |
| `ohlcv/` | ✅ | OHLCV domain entities, repository, service |
| `ingestion/` | ✅ | Binance API client, ingestion service, CLI (`src/app/ingestion/cli.py`) |
| `bars/` | ✅ | Lopez de Prado alternative bars (tick, volume, dollar, imbalance, run) + CLI (`src/app/bars/cli.py`) |
| `research/` | ✅ | RC1–RC2 analysis services (coverage, returns, ACF, bar comparison, charts) |
| `features/` | ✅ | Feature engineering (indicators, targets, matrix builder) + validation (MI, BH, DA) |
| `profiling/` | ✅ | Statistical profiling per asset (distributions, serial dependence, volatility, stationarity) |
| `backtest/` | Phase 8 🔧 | Event-driven backtest engine (domain model done, execution + metrics next) |
| `strategy/` | Phase 9 | Base trading strategies |
| `forecasting/` | Phase 10–11 | Classification + regression models |
| `recommendation/` | Phase 12 | ML recommendation system |
| `evaluation/` | Phase 14 | Monte Carlo, permutation tests, statistical proof |

### DataFrame Libraries — Split by Concern

| Library | Where | Why |
|---------|-------|-----|
| **Polars** | Pipeline code (ingestion → bars → features → backtest → live) | Performance, lazy eval, no GIL |
| **Pandas** | Research notebooks (RC1–RC4), `src/app/research/`, validation (`features/application/validation.py`) | ML ecosystem compat (statsmodels, sklearn) |
| **NumPy** | Vectorized math (indicators, bootstrap, Monte Carlo) | Tight numerical loops |

Conversion boundaries are explicit: `df_polars.to_pandas()`, `pl.from_pandas(df_pandas)`, `.to_numpy()`.

### Database

- **DuckDB** for ALL analytical storage — no Postgres, no SQLite
- `src/app/system/database/connection.py` — `ConnectionManager` (SQLAlchemy + DuckDB)
- `.env` contains `DUCKDB_PATH`, `BINANCE_API_KEY`, `BINANCE_API_SECRET` (see `.example.env` for template)
- **Every** schema change goes through Alembic migrations (no raw DDL outside migrations)
- Migrations in `src/app/system/database/alembic/versions/`, config at `src/app/system/database/alembic.cfg`

### Test Architecture

Tests mirror the source structure under `src/tests/<module>/`. Pytest markers: `integration`, `e2e`.

**Test patterns used throughout:**
- Factory functions in `src/tests/conftest.py`: `make_asset()`, `make_date_range()`, `make_candle()`
- Fake implementations for Protocols in per-module `conftest.py` files (e.g., `FakeMarketDataFetcher`, `FakeOHLCVRepository`)
- Shared fixtures: `btc_asset`, `eth_asset`, `date_range`, `h1_timeframe`
- In-memory DuckDB for integration/e2e tests

## Code Standards

### Pre-Commit Pipeline (runs on every `git commit`)

| Order | Hook | Tool |
|-------|------|------|
| 1 | Formatter | `ruff format` (119 char lines, double quotes) |
| 2 | Linter | `ruff` (~25 rule categories incl. `D`, `DOC`, `ANN`, `S`, `N`, `PERF`) |
| 3 | Type checker | `ty` (Astral, excludes `src/tests/` and `src/app/research/`) |

Import sorting is handled by ruff's `I` rules (no separate isort hook). All config in `pyproject.toml`.

### Type Hints — Python 3.14 syntax

- `list[X]`, `dict[K, V]`, `X | None`, `X | Y` — NOT `typing.List`, `typing.Optional`, `typing.Union`
- PEP 695 type aliases: `type OHLCVFrame = pl.DataFrame`
- **All local variables** must have explicit type annotations (project convention, not enforced by linter)
- No `Any` unless interfacing with untyped third-party libs (comment why)
- `from __future__ import annotations` in every file
- Type-ignore comments: use `# type: ignore[rule]` (for pyright compat) AND `# ty: ignore[rule]` side-by-side when suppressing third-party typing issues

### Docstrings — Google style (enforced by Ruff `D` + `DOC`)

Every public module, class, method, and function. One-liner at top of every `.py` file.

### Per-File Lint Relaxations

See `pyproject.toml` `[tool.ruff.lint.per-file-ignores]` for the full list. Key relaxations:
- `src/tests/*` — no `S101`, `ANN`, `D`, `DOC`, `PLR2004` (and several others)
- `src/app/research/*` — no `ANN`, `PLR6301`
- `research/*.ipynb` — no `T201`, `E402`, `ANN`, `D`, `DOC`, `F401`, `PLR2004`

## Key Design Decisions

1. **Two-track forecasting:** Classification → direction (SIDE), Regression → magnitude (SIZE). Combined by ML recommendation system.
2. **Regression metrics are direction-conditional:** DC-MAE, DC-RMSE only where direction is correct. Economic Sharpe is the ultimate metric.
3. **ML recommendation system (not a formula):** Generalized meta-labeling — predicts expected strategy return (continuous), not binary bet/no-bet.
4. **Research checkpoints (RC1–RC4):** Explicit analysis stops with charts, statistics, go/no-go decisions.
5. **No future leakage:** `TemporalSplit` value object + `.shift(1)` convention. Walk-forward throughout. CPCV with purging and embargo.
6. **Monte Carlo validation:** Strategy MUST NOT be profitable on synthetic GBM paths.
7. **Honest evaluation:** Negative results are valid and documented.
8. **Config-driven:** Every parameter in Pydantic config classes. No magic numbers.

## Current Progress (Phases 1–7 complete, Phase 8 in progress)

**Completed phases:**
- **Phase 1:** Data ingestion pipeline (Binance API → DuckDB)
- **Phase 2:** Alternative bar construction (López de Prado tick/volume/dollar/imbalance/run bars)
- **Phase 3 (RC1):** Data quality & bar analysis — 4 assets selected (BTCUSDT, ETHUSDT, LTCUSDT, SOLUSDT)
- **Phase 4:** Feature engineering + validation pipeline (MI, BH correction, DA, DC-MAE, temporal stability)
- **Phase 5:** Statistical profiling (return distributions, serial dependence, volatility modeling, stationarity)
- **Phase 6 (RC2):** Features & data adequacy audit — 6 audit gaps identified
- **Phase 7:** All 6 RC2 audit gaps resolved (cost sensitivity, constant-feature fix, stationarity policy, MI normalization, LTCUSDT volume-bar profiling, SOLUSDT tier-B protocol). See `research/RC7_analysis.md`.

**Phase 8 (in progress):** Backtest engine
- **8A done:** Domain model (`backtest/domain/`) — `Side`, `ExecutionConfig`, `TradeResult`, `PortfolioSnapshot`, `Signal`, `Position`, `Trade`, `EquityCurve`, `IStrategy`, `IPositionSizer`
- **8B next:** Execution layer (fill on next bar open, cost sweep)
- **8C next:** Metrics layer (Sharpe with Lo 2002 correction, drawdown, trade stats)

## What NOT to Do

- Do NOT use `dataclasses` — use Pydantic `BaseModel`
- Do NOT use `typing.Optional`, `typing.Union`, `typing.List`, `typing.Dict`
- Do NOT write raw DDL outside Alembic migrations
- Do NOT use Pandas in pipeline code — use Polars
- Do NOT use Polars in research/notebooks where stats libs need Pandas
- Do NOT leave local variables untyped
- Do NOT use ABC/abstractmethod — use `typing.Protocol`
- Do NOT modify files in `legacy_project/` — reference only
