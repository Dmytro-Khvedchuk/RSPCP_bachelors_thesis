# RSPCP Project — Agent Memory

## Key Architectural Patterns

### Module structure (src/app/<module>/)
- `domain/` — entities, value_objects, protocols (zero external deps)
- `application/` — services, commands (Pydantic BaseModel, frozen=True)
- `infrastructure/` — concrete implementations

### Dependency Injection Pattern
- Protocols in domain, implementations in infrastructure
- Services accept protocols, not concretions
- CLI layer wires everything via `_build_service(cm: ConnectionManager)` helper

### Typer CLI Pattern (confirmed working)
- `app: typer.Typer = typer.Typer(name=..., help=..., no_args_is_help=True)`
- Use `Annotated[str, typer.Option(...)]` syntax — works with `from __future__ import annotations`
- Use `""` (empty string) as default for optional str options, NOT `str | None` — avoids annotation evaluation issues
- Check empty string in command body: `if end == "" else _parse_date(end)`
- Guard: `if __name__ == "__main__": app()`
- Run via: `python -m src.app.ingestion.cli ingest {{args}}`

### Pydantic Validation in CLI
- Wrap `Asset(symbol=...)` in `try/except ValidationError` → `raise typer.BadParameter(...)`
- Validate Timeframe via `{tf.value for tf in Timeframe}` set membership check

### Datetime / Timezone Rules
- Always use `datetime.now(UTC)` not `datetime.utcnow()` (DTZ rule)
- `datetime.fromisoformat(raw)` then `.replace(tzinfo=UTC)` if `tzinfo is None`
- `DateRange` validator enforces UTC and start < end
- **DuckDB TIMESTAMPTZ gotcha:** DuckDB returns TIMESTAMPTZ values converted to system local TZ (e.g. CET), NOT UTC. Must use `dt.astimezone(UTC)` not just `dt.replace(tzinfo=UTC)`. See `_to_utc()` helper in `src/app/bars/infrastructure/duckdb_repository.py`.
- **DuckDB DELETE rowcount:** SQLAlchemy driver returns -1 for DELETE rowcount. Workaround: query COUNT before DELETE.

## Key Files
- `src/app/ingestion/cli.py` — Typer CLI entry point (Phase 1, Step 1C)
- `src/app/ingestion/application/services.py` — `IngestionService` (ingest_asset, ingest_universe, ingest_incremental)
- `src/app/ingestion/application/commands.py` — `IngestAssetCommand`, `IngestUniverseCommand`
- `src/app/ingestion/infrastructure/binance_fetcher.py` — `BinanceFetcher`
- `src/app/ingestion/infrastructure/settings.py` — `BinanceSettings` (pydantic-settings, BINANCE_ prefix)
- `src/app/ohlcv/domain/value_objects.py` — `Asset`, `Timeframe` (StrEnum: 1h/4h/1d), `DateRange`, `TemporalSplit`
- `src/app/ohlcv/infrastructure/duckdb_repository.py` — `DuckDBOHLCVRepository`
- `src/app/system/database/connection.py` — `ConnectionManager` (context manager)
- `src/app/system/database/settings.py` — `DatabaseSettings` (DUCKDB_ prefix)
- `src/app/bars/domain/protocols.py` — `IBarAggregator`, `IBarRepository` protocols
- `src/app/bars/domain/entities.py` — `AggregatedBar` (frozen BaseModel)
- `src/app/bars/domain/value_objects.py` — `BarType` (StrEnum), `BarConfig` (frozen BaseModel, `config_hash` property)
- `src/app/bars/infrastructure/duckdb_repository.py` — `DuckDBBarRepository` (Step 2D)
- `src/app/features/domain/value_objects.py` — `IndicatorConfig` (frozen BaseModel, all indicator params)
- `src/app/features/application/indicators.py` — 23 indicator functions + `compute_all_indicators()` orchestrator
- `src/app/system/database/alembic/versions/002_add_aggregated_bars_table.py` — migration
- `src/app/system/logging.py` — `setup_logging(*, level, json_output, log_file)`
- `justfile` — task runner (just run/lint/migrate/ingest/...)

## Linting Rules (all enforced via pre-commit)
- ruff format (119 char lines, double quotes)
- ruff lint (~20 rule categories incl. ANN, D, DTZ, N, S, UP, etc.)
- pyright strict
- isort (FUTURE → STDLIB → THIRDPARTY → FIRSTPARTY → LOCALFOLDER)
- Every local variable MUST have explicit type annotation: `x: int = 5`

## Domain Values
- `Timeframe` is a `StrEnum` with values `"1h"`, `"4h"`, `"1d"`
- `Asset.symbol` pattern: `^[A-Z0-9]{2,20}$`
- `DateRange` requires both bounds to be UTC-aware, start < end
