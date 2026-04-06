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

### features Module (Phase 4) — Confirmed Structure
- `src/app/features/` — domain (value_objects.py, entities.py), application (indicators.py, targets.py, feature_matrix.py, validation.py)
- `src/tests/features/` — 195 tests, flat structure (no unit/e2e subdirs): test_indicators.py, test_indicators_property.py, test_targets.py, test_feature_matrix.py, test_validation_helpers.py, test_validation_integration.py, test_value_objects.py, test_entities.py, test_leakage.py
- 23 features with default IndicatorConfig: Returns(4) + Volatility(6) + Momentum(5) + Volume(3) + Statistical(5)
- Column naming: `logret_N`, `rv_N`, `gk_vol_N`, `park_vol_N`, `atr_N`, `ema_xover_F_S`, `rsi_N`, `roc_N`, `vol_zscore_N`, `obv_slope_N`, `amihud_N`, `ret_zscore_N`, `bbpctb_W_STD`, `bbwidth_W_STD`, `slope_N`, `hurst_N`
- Target columns prefixed `fwd_`: `fwd_logret_{h}`, `fwd_vol_{h}`. TARGET_PREFIX = "fwd_" constant.
- FeatureMatrixBuilder is stateless — no constructor deps; pure computation via .build(df, FeatureConfig) → FeatureSet
- Validation gates (all three must pass for keep=True): MI BH-corrected p < α, Ridge DA empirical p < α, temporal stability ≥ 50% windows
- Group interaction test (4th battery) is informational only — does NOT affect keep flag
- Fallback: if fewer than min_features_kept pass all gates, top N by composite score are force-kept
- FeatureValidator uses Pandas + NumPy (not Polars) — research/ML path per CLAUDE.md convention
- No infrastructure layer in features/ — no I/O, no DB. Domain + application only.

### Does NOT exist yet
- `research/` directory at repo root — planned for RC1–RC4 notebooks, not created yet. Do not reference in README.
  Note: `src/app/research/` exists (RC1 analysis services), but the top-level `research/` dir does not.

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

### profiling Module (Phase 5) — Confirmed Structure
- `src/app/profiling/` — domain (value_objects.py only), application (5 analyzers + services.py)
- No infrastructure layer — pure computation; data loading delegated to DataLoader injection
- `src/tests/profiling/` — 188 tests, flat structure: 8 test files + conftest.py
  - test_data_partition.py, test_tier_classification.py, test_distribution.py
  - test_serial_dependence.py, test_volatility.py, test_predictability.py
  - test_stationarity.py, test_services.py
- Tier thresholds: A > 2000 samples, B: 500–2000, C < 500
- GARCH only on time bars (bar_type.startswith("time_")), not on alternative bars
- GJR-GARCH only fitted when sign_bias.has_leverage_effect=True (Tier A + time bars)
- BDS test only on Tier A time bars; Granger causality not called by profile_all() — callers supply returns_dict
- Variance ratio: arch.unitroot.VarianceRatio expects price levels (cumulative sum of returns), not returns directly
- FDR correction pools: Ljung-Box (returns), Ljung-Box (squared), variance ratio, Granger, BDS, ARCH-LM, sign bias joint F-test
- ProfilingService loads feature_selection period from DataPartition for data loading
- README written: src/app/profiling/README.md

### backtest Module (Phase 8) — Confirmed Structure
- `src/app/backtest/` — domain (value_objects.py, entities.py, protocols.py), application (6 files)
- No infrastructure layer — no DB, pure computation and simulation
- `src/tests/backtest/` — 186 tests, flat structure: conftest.py + 7 test files
  - test_domain.py, test_execution.py, test_metrics.py, test_baselines.py
  - test_position_sizer.py, test_cost_sweep.py, test_walk_forward.py
- Domain value objects: Side (LONG/SHORT/FLAT), ExecutionConfig, TradeResult, PortfolioSnapshot
- Domain entities: Signal, Position, Trade, EquityCurve
- Domain protocols: IStrategy (on_bar), IPositionSizer (size)
- Application: ExecutionEngine (next-bar fill), compute_metrics / compute_buy_and_hold_metrics
  BacktestMetrics, BuyAndHoldStrategy, RandomStrategy, FixedFractionalSizer, RegimeConditionalSizer
  cost_sweep (grid over commission schedule), WalkForwardRunner (EXPANDING/ROLLING via WindowMode)
- IStrategyFactory protocol enables per-fold retraining in WalkForwardRunner
- Lo (2002) formula: SR_AC = SR × √(1 + 2 Σ ρ_k)

### strategy Module (Phase 9) — Confirmed Structure
- `src/app/strategy/` — domain (protocols.py only), application (5 strategy files)
- No infrastructure layer — no DB, pure computation
- `src/tests/strategy/` — 101 tests, flat structure: conftest.py + 6 test files
  - test_momentum_crossover.py, test_mean_reversion.py, test_donchian_breakout.py
  - test_volatility_targeting.py, test_no_trade.py, test_signal_diversity.py
- IStrategy protocol: name property + generate_signals(FeatureSet) → pl.DataFrame
- Signal output columns: timestamp (Datetime), side (Utf8: "long"/"short"/"flat"), strength (Float64 [0,1])
- MeanReversion uses Hurst filter (hurst < 0.5) to suppress signals in trending regimes
- DonchianBreakout applies shift(1) to rolling high before comparison — prevents look-ahead bias
- VolatilityTargeting is always-long; NoTrade is always-flat (recommendation system baseline)
- Signal diversity test: pairwise Jaccard similarity < 0.5 across all 5 strategy pairs
- Note: strategy/ IStrategy protocol is DIFFERENT from backtest/ IStrategy — batch vs per-bar interface

### README Style Decisions
- No badges (project has no live CI badge URLs)
- Data flow section as ASCII diagram using boxes and arrows — extend it as new pipeline stages are added
- Module roadmap table with Phase, Module, Purpose, Status columns
- Detailed technical section per module (Ingestion, OHLCV, Bars, Features) in the root README
- Implementation plan summary kept brief, points to IMPLEMENTATION_PLAN.md
- Feature group tables use 3-column format: Group (count) | Features | Column prefix
- Validation pipeline tables use 3-column format: Battery | Method | Gate
- For module READMEs: include algorithm pseudocode for complex logic (e.g., imbalance/run sequential loop)
- Bar type tables use 4-column format: BarType | Class | Sampling trigger | Algorithm
- Test READMEs include fixture table with 3 columns: Fixture/Helper | Returns | Purpose
