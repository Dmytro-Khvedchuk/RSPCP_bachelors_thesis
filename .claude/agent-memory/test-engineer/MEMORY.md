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

## Files Created (Phase 4E Tests — features module, 195 tests)
- `src/tests/features/__init__.py`
- `src/tests/features/conftest.py` — OHLCV DataFrame factories (random walk, trend, mean-reverting, OU), small-window config factories
- `src/tests/features/test_indicators.py` — unit tests for each indicator against known values
- `src/tests/features/test_indicators_property.py` — property tests: finite after warmup, shape, clipping, determinism
- `src/tests/features/test_targets.py` — forward return and volatility tests on known prices
- `src/tests/features/test_leakage.py` — future leakage detection (correlation analysis, shuffle test, time-reversal)
- `src/tests/features/test_feature_matrix.py` — FeatureMatrixBuilder integration tests
- `src/tests/features/test_validation_helpers.py` — MI score, empirical p-value, BH correction, DA, DC-MAE, Ridge, group classification
- `src/tests/features/test_validation_integration.py` — FeatureValidator full pipeline with @pytest.mark.integration
- `src/tests/features/test_value_objects.py` — IndicatorConfig, TargetConfig, ValidationConfig, FeatureConfig, FeatureSet
- `src/tests/features/test_entities.py` — FeatureValidationResult, InteractionTestResult, ValidationReport

## Features Module Critical Gotchas
- Frozen Pydantic models raise `ValidationError` (not `TypeError` or `PydanticFrozenInstanceError`) on mutation in pydantic v2.12+
- `ValidationConfig` minimum constraints: `n_permutations_mi>=100`, `n_permutations_ridge>=50`, `n_permutations_stability>=50` — "fast" test configs must respect these
- Bollinger %B for CONSTANT prices: std=0 so upper==lower==middle==close; formula gives `(close-lower)/(0+EPS) = 0/EPS = 0`, NOT 0.5 — test for %B=0.5 needs a non-constant series where close==rolling_mean
- `PLR0914` (too many local variables) fires on helper functions with 16+ vars — use `# noqa: PLR0914`
- `B905` zip-without-strict fires in tests too — always use `zip(..., strict=True)`
- `E741` fires on variable name `l` (ambiguous) — use `lo` or `low_val` instead
- Pre-commit pyright hook (v1.1.408) has a pre-existing failure on `src/app/system/database/repository.py` (PEP 695 syntax) unrelated to tests — confirmed present before Phase 4E changes
- `compute_all_indicators` with `hurst_window=40`, `slope_window=5`, `obv_slope_window=5` works well for 150-row test DataFrames
- Leakage tests: use `np.corrcoef` with `strict=False` NaN handling (some features have zero variance after warmup, causing NaN correlations that should be skipped)

## Files Added (Phase 5 Tests — profiling module, 861 total tests)
- Added to `src/tests/profiling/conftest.py`: `make_random_walk()` (price-level cumsum, distinct from `make_random_walk_returns`) and `make_profiling_config()` (fast defaults factory)
- Added to `src/tests/profiling/test_volatility.py`: `TestGARCHParameterRecovery` — `test_garch_recovers_alpha_beta_within_20_percent` + `test_constant_series_garch_fits_none_or_near_zero`
- Added to `src/tests/profiling/test_stationarity.py`: `TestADFAndKPSSPValues` — 4 tests checking individual ADF and KPSS p-value fields on random walk vs white noise
- Added to `src/tests/profiling/test_predictability.py`: `TestPermutationEntropyNearZero` (sine wave H_norm at d=5,6) + `TestEffectiveSampleSizeAR1` (phi=0.5 N_eff in [0.1, 0.7])

## Profiling Module Critical Gotchas
- `make_random_walk` (conftest) returns PRICE LEVELS (cumsum); `make_random_walk_returns` returns i.i.d. white noise (first differences) — do not confuse them
- `make_profiling_config()` reduces Ljung-Box lags, GARCH distributions, PE dimensions, and min_samples_garch for fast test execution
- Permutation entropy of a sine wave at d=3 is ~0.45 (NOT near 0) — only at d=5,6 does it drop below 0.20; use pe_dimensions=(5,6) for near-zero tests
- `N806` (uppercase variable names) fires in test methods even for named constants — use lowercase names (e.g. `tolerance`, `h_norm_max`, not `TOLERANCE`)
- GARCH on constant (all-zero) series: the optimizer either fails (garch_fits is None) OR converges to degenerate values — test must handle both cases
- KPSS p-value is bounded/clamped to [0, 1] in the screener; random walk reliably gives kpss_pvalue < 0.05 (n=1000, seed=42 confirmed)

## Files Created (Phase 8F Tests — backtest module, 147 tests)
- `src/tests/backtest/__init__.py`
- `src/tests/backtest/conftest.py` — `make_bars()`, `make_snapshot()`, `AlwaysLongStrategy`, `NeverTradeStrategy`, `SingleSignalStrategy`, `FixedNotionalSizer`
- `src/tests/backtest/test_domain.py` — Side, ExecutionConfig, TradeResult, PortfolioSnapshot, Signal, Position, Trade, EquityCurve
- `src/tests/backtest/test_position_sizer.py` — FixedFractionalSizer, RegimeConditionalSizer (parametrized known values)
- `src/tests/backtest/test_execution.py` — commission formula, fill-on-next-open, SL/TP logic, staleness, edge cases, equity curve manual verification
- `src/tests/backtest/test_metrics.py` — total return, max drawdown, Sharpe, Lo correction, trade stats, buy-and-hold
- `src/tests/backtest/test_baselines.py` — BuyAndHoldStrategy (first-bar only), RandomStrategy (seed, frequency, both sides)
- `src/tests/backtest/test_walk_forward.py` — expanding/rolling modes, equity chaining, no-lookahead, insufficient bars error
- `src/tests/backtest/test_cost_sweep.py` — fee-level sweep, economics, base_config inheritance

## Backtest Module Critical Gotchas
- Trade validator: `exit_time > entry_time` STRICTLY — if position opens and is liquidated at the same timestamp the Trade construction raises ValidationError. Ensure at least N+2 bars when signal emits at bar[0], fills at bar[1], liquidates at bar[N+1].
- `AlwaysLongStrategy` MUST check `portfolio.positions` before emitting — if it always emits, the engine overwrites the existing same-side position and opens a new one at the last bar, causing entry_time==exit_time on liquidation.
- `_annualized_return` raises OverflowError when duration is sub-hour (exponent = 365.25/days is enormous). Use DAILY intervals in equity curves for metrics tests, not hourly.
- Cost sweep tests with shared `SingleSignalStrategy` instance: the strategy's `_call_count` persists across fee-level runs so only the first run sees any signal. Use stateless `AlwaysLongStrategy` or instantiate a fresh strategy per fee level.
- `N814` ruff error: importing CamelCase class with underscore alias (e.g. `ExecutionEngine as _EE`) — keep the import at top-level without renaming.
- `compute_metrics` equity curve with <2 points returns all-None metrics — tests using single-bar curves should not check computed metrics.
- Lo correction factor < 1.0 requires STRONG positive autocorrelation (phi >= 0.7 in AR(1)) with >=300 returns. With IID returns, factor ≈ 1.0 ± 0.2.
- `_check_sl_tp` is a module-private function in execution.py — importable for direct unit testing via `from src.app.backtest.application.execution import _check_sl_tp`.

## Files Created (Phase 9C Tests — strategy module, 101 tests)
- `src/tests/strategy/__init__.py` — empty
- `src/tests/strategy/conftest.py` — `make_feature_set()`, `_make_ohlcv_base()`, scenario fixtures (trending_up, trending_down, mean_reverting, flat, high_vol, low_vol, mixed_regime)
- `src/tests/strategy/test_momentum_crossover.py` — name, schema, long/short/flat logic, strength clipping, custom config
- `src/tests/strategy/test_mean_reversion.py` — name, schema, Hurst gate, BB band logic, synthetic data
- `src/tests/strategy/test_donchian_breakout.py` — name, schema, long-only constraint, channel logic, shift prevents lookahead, strength
- `src/tests/strategy/test_volatility_targeting.py` — name, schema, always-long, strength=target/rv clipped
- `src/tests/strategy/test_no_trade.py` — name, schema, always-flat, PE gate, per-bar low-vol filter
- `src/tests/strategy/test_signal_diversity.py` — pairwise Jaccard, guaranteed-distinct pairs (VT vs NT), cross-scenario

## Strategy Module Critical Gotchas
- DonchianBreakout triggers ONLY when `close > rolling_max(shift(1)(high))`. With `high = close + offset` and `close` growing by `step`, breakouts require `step > offset`. The shared `trending_up_feature_set` has `high = close + 200` with `step=100` — close NEVER exceeds the channel. Always craft explicit breakout scenarios for Donchian channel tests.
- Jaccard similarity for signal diversity: MeanReversion (high Hurst → flat), DonchianBreakout (no breakouts on flat data → flat), and NoTrade (always flat) can all agree 99%+ on many feature sets. Test diversity by comparing only guaranteed-distinct pairs: VolatilityTargeting (all long) vs NoTrade (all flat) → Jaccard=0.0.
- NoTrade `pe_value > pe_threshold` uses strictly `>` — `pe_value == pe_threshold` does NOT trigger the gate.
- FeatureSet validators: `n_rows_clean == len(df)` AND `n_rows_clean <= n_rows_raw` AND all feature/target cols must exist in df. Build FeatureSet AFTER adding all required columns.
- `make_feature_set()` in strategy conftest accepts `xover_values`, `hurst_values`, `atr_values`, `rv_values` as overrides with defaults (0.0, 0.5, 200.0, 0.1).
