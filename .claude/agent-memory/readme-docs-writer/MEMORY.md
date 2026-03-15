# README Docs Writer Memory

## Project: RSPCP Bachelor's Thesis

### Key Paths Confirmed to Exist
- `src/app/ingestion/` — domain, application, infrastructure, cli.py
- `src/app/ohlcv/` — domain (entities, value_objects, repositories), infrastructure
- `src/app/system/` — logging.py, database/ (connection, settings, exceptions, repository, alembic/)
- `src/tests/conftest.py` — shared factories (make_asset, make_date_range, make_candle)
- `src/tests/ingestion/conftest.py` — fakes (FakeMarketDataFetcher, FakeOHLCVRepository), kline builders
- `src/tests/ingestion/unit/` — 7 test files (binance_fetcher, cli, commands, exceptions, services, settings, value_objects)
- `src/tests/ingestion/e2e/` — test_cli_e2e.py
- `data/market.duckdb` — persistent DuckDB store
- `docs/` — MkDocs source (empty dir, present)
- `legacy_project/` — reference only
- `main.py`, `pyproject.toml`, `justfile`, `mkdocs.yml`, `IMPLEMENTATION_PLAN.md`
- `.github/workflows/ci.yml`, `.github/release.yml`

### Does NOT exist yet
- `research/` — planned for RC1–RC4 notebooks, not created yet. Do not reference in README.

### Justfile Commands (verified)
- `just run` — python main.py
- `just ingest *args` — python -m src.app.ingestion.cli
- `just test *args` — uv run pytest src/tests/
- `just add <package>` — uv add
- `just serve` — mkdocs serve
- `just install-hooks` / `just uninstall-hooks` — pre-commit install/uninstall
- `just lint` — pre-commit run --all-files
- `just migrate` — alembic upgrade head
- `just migration <message>` — alembic revision
- `just migrate-down` — alembic downgrade -1

### Alembic Config Path
`src/app/system/database/alembic.cfg` (referenced in justfile via `_alembic_cfg`)

### OHLCV Table Schema
Composite PK on (asset, timeframe, timestamp). Prices as DECIMAL(18,8), volume as DOUBLE.
Index: idx_ohlcv_asset_tf_ts on (asset, timeframe, timestamp).

### BinanceFetcher Retry Logic
Tenacity retries on: BinanceAPIException, BinanceRequestException, Timeout, ConnectionError, HTTPError.
HTTP 429 → RateLimitError (NOT retried, raised immediately).
After max_retries exhausted → FetchError.

### TIMEFRAME_INTERVAL_MS Keys
M1=60000, H1=3600000, H4=14400000, D1=86400000.
Note: M1 is only in BinanceKlineInterval (for bar construction), NOT in Timeframe enum.
Timeframe enum only has: H1, H4, D1.

### Env Prefix Conventions
- DUCKDB_ prefix for DatabaseSettings
- BINANCE_ prefix for BinanceSettings (secret is BINANCE_SECRET_KEY, not BINANCE_API_SECRET)

### bars Module (Phase 2) — Confirmed Structure
- `src/app/bars/` — domain, application (5 aggregators + shared _aggregation.py), infrastructure
- `src/tests/bars/` — conftest.py, domain/, application/, infrastructure/ (260 tests total)
- Alembic migration: `src/app/system/database/alembic/versions/002_add_aggregated_bars_table.py`
- aggregated_bars table: composite PK (asset, bar_type, bar_config_hash, start_ts); DECIMAL(18,8) prices
- bar_config_hash is SHA-256[:16] of BarConfig JSON — 16-char hex, stored as VARCHAR(16)
- Covering index: idx_bars_asset_type_hash_ts on (asset, bar_type, bar_config_hash, start_ts)
- Standard bars use Polars cumsum pipeline; information-driven bars use sequential NumPy O(n) loop
- Both ImbalanceBarAggregator and RunBarAggregator live in their own files but share build_bar_from_arrays
- EMA adaptive threshold formula: θ_t = α × |Θ_t| + (1 − α) × θ_{t-1}, where α = 2/(ewm_span+1)
- READMEs written: src/app/bars/README.md, src/tests/bars/README.md

### README Style Decisions
- No badges (project has no live CI badge URLs)
- Data flow section as ASCII diagram using boxes and arrows
- Module roadmap table with Phase, Module, Purpose, Status columns
- Two detailed technical sections for ingestion and OHLCV modules
- Implementation plan summary kept brief, points to IMPLEMENTATION_PLAN.md
- For module READMEs: include algorithm pseudocode for complex logic (e.g., imbalance/run sequential loop)
- Bar type tables use 4-column format: BarType | Class | Sampling trigger | Algorithm
- Test READMEs include fixture table with 3 columns: Fixture/Helper | Returns | Purpose
