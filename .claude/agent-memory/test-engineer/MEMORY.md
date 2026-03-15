# Test Engineer Memory — RSPCP Bachelors Thesis

## Test Location & Structure
- Tests live in `src/tests/` (NOT `tests/` at project root)
- Mirror the src structure: `src/tests/ingestion/`, etc.
- `src/tests/__init__.py` and `src/tests/ingestion/__init__.py` required
- pytest configured via `pyproject.toml` — no `pytest.ini` needed
- ruff ignores `S101` (assert) in `src/tests/*` — already in pyproject.toml

## Test Dependencies
- `pytest` is a dev dependency managed via `uv add --dev pytest`
- No extra test libs needed — use `unittest.mock.MagicMock` and `unittest.mock.patch`
- `monkeypatch` fixture is built into pytest (no import needed in test files)

## Key Patterns Confirmed Working

### Protocol-based Fakes for Service Tests
- Create fake classes that implement ALL protocol methods (even unused ones)
- `FakeMarketDataFetcher` tracks `.calls: list[FetchRequest]` for assertion
- `FakeOHLCVRepository` tracks `.ingested`, `.ingest_call_count` for assertion
- See `src/tests/ingestion/test_services.py` for full working examples

### Mocking `binance.client.Client`
- Always use `patch("src.app.ingestion.infrastructure.binance_fetcher.Client", autospec=True)`
- Must create the fetcher INSIDE the `with patch(...)` block so `__init__` uses the mock
- After the `with` block exits, `fetcher._client` still holds the mock instance — calls still work
- `BinanceFetcher._retryer` can be replaced after construction for fast retry tests

### BinanceSettings Testing Pattern
- `BinanceSettings.model_construct(...)` bypasses env loading — use for unit tests needing a settings object
- For testing required-field validation, create a local `_BinanceSettingsNoEnvFile` subclass with `env_file=None`
- `monkeypatch.delenv()` alone is insufficient because `env_file=".env"` silently provides the value
- See `src/tests/ingestion/test_settings.py` for the `_BinanceSettingsNoEnvFile` pattern

## Tenacity + BinanceFetcher Behavior (Important!)
- `BinanceFetcher._fetch_klines_batch` uses `reraise=True` in its `Retrying` instance
- With `reraise=True`, tenacity re-raises the ORIGINAL exception (not `RetryError`) after all attempts fail
- The `except RetryError` branch in `_fetch_klines_batch` is DEAD CODE with the current config
- Tests should expect the underlying exception (e.g. `ConnectionError`) to propagate, not `FetchError`
- Confirmed via: `test_retryable_error_exhausts_retries_propagates_original_exception`

## Pagination Loop Logic (`fetch_ohlcv`)
- Loop condition: `while current_start < end_ms`
- After last batch: `next_start = last_open_time + interval_ms`
- Loop exits when `next_start >= end_ms` (NOT when API returns empty)
- To force a 3rd API call: set `end_ms = batch2_start + batch_size * interval_ms + 1`

## `ingest_incremental` Semantics
- `incremental_start = existing_end - timedelta(seconds=1)`
- "Already up-to-date" check: `incremental_start >= date_range.end`
- For up-to-date test: `existing_end` must be `>= date_range.end + 1 second`
- "Advance start" check: `incremental_start > date_range.start`
- Both conditions are strictly checked — off-by-one matters

## Ruff/Pyright Notes for Tests
- `from __future__ import annotations` — MANDATORY in every test file
- All local variables need explicit type annotations (pyright strict enforces this)
- `# type: ignore[misc]` needed when assigning to frozen Pydantic model fields in tests
- `# type: ignore[call-arg]` needed when calling `BinanceSettings()` without required args
- `# type: ignore[arg-type]` needed when passing wrong types to test error paths

## DateRange Validation
- Both bounds must be UTC-aware: `datetime(2024, 1, 1, tzinfo=UTC)` not naive
- `DateRange` uses `UTC` (from `datetime` module), NOT `timezone.utc`
- `start < end` is strictly enforced — use clearly separated datetimes

## Files Created (Phase 1 Tests)
- `src/tests/__init__.py`
- `src/tests/ingestion/__init__.py`
- `src/tests/ingestion/test_value_objects.py` — 152 total tests across all files
- `src/tests/ingestion/test_exceptions.py`
- `src/tests/ingestion/test_commands.py`
- `src/tests/ingestion/test_services.py`
- `src/tests/ingestion/test_binance_fetcher.py`
- `src/tests/ingestion/test_settings.py`
- `src/tests/ingestion/test_cli.py`

## Files Created (Phase 2 Tests — bars module, 260 tests)
- `src/tests/bars/__init__.py`, `domain/__init__.py`, `application/__init__.py`, `infrastructure/__init__.py`
- `src/tests/bars/conftest.py` — DataFrame builders, AggregatedBar factory, in-memory DuckDB fixture
- `src/tests/bars/domain/test_value_objects.py` — BarType, BarConfig validation, is_information_driven, config_hash
- `src/tests/bars/domain/test_entities.py` — AggregatedBar invariants, happy-path construction
- `src/tests/bars/application/test_standard_bars.py` — tick, volume, dollar bars (boundaries, OHLCV, property tests)
- `src/tests/bars/application/test_information_driven_bars.py` — imbalance and run bars
- `src/tests/bars/application/test_statistical.py` — statistical properties of dollar bars vs tick bars
- `src/tests/bars/infrastructure/test_duckdb_repository.py` — ingest, query, delete, count, get_latest_end_ts, get_available_configs

## Bars Module Critical Gotchas
- `BarConfig` with `ewm_span=N` requires `warmup_period <= N` — always set compatible warmup when changing ewm_span
- Dollar bar bar_id formula: `floor((cumsum_before_row) / threshold)` — Polars float division at exact boundary (e.g. 420000/420000) returns 0.999... not 1.0, so threshold is effectively EXCLUSIVE; need one extra row to cross to next bar_id
- `pytest.approx()` does NOT work with `Decimal` values — use `float(val)` before comparing with approx
- Statistical test correctness: tick bars have CV=0 (insensitive to volatility) while dollar bars have high CV (sensitive to volume). The López de Prado uniformity claim applies to PRICE LEVEL differences, not volatility/volume differences
- Infrastructure tests use `bar_connection_manager` + `bar_repository` fixtures from `src/tests/bars/conftest.py`; the table is created inline (not via Alembic) for isolation
- pytest markers `integration` and `e2e` must be registered in `pyproject.toml` under `[tool.pytest.ini_options]` to avoid PytestUnknownMarkWarning
- Ruff `PLR0904` (too many public methods) and `PLR0913` (too many arguments) must be added to `per-file-ignores` for `src/tests/*` — test classes routinely exceed these limits
