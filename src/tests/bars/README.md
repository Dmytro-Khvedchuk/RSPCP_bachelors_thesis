# tests/bars
> Test suite for the `bars` module — 260 tests across domain, application, and infrastructure layers

## Overview

Full test coverage for Phase 2 (López de Prado alternative bars). Covers domain invariant enforcement, all nine aggregator implementations, statistical properties from *Advances in Financial Machine Learning* §2.3, and DuckDB repository integration.

## Structure

```
tests/bars/
├── conftest.py                          # Shared fixtures and DataFrame factories
├── domain/
│   ├── test_value_objects.py            # 56 tests — BarType, BarConfig validation
│   └── test_entities.py                 # 22 tests — AggregatedBar invariants
├── application/
│   ├── test_standard_bars.py            # 72 tests — tick, volume, dollar aggregators
│   ├── test_information_driven_bars.py  # 80 tests — imbalance and run aggregators
│   └── test_statistical.py              #  8 tests — statistical bar properties
└── infrastructure/
    └── test_duckdb_repository.py        # 32 tests — DuckDB integration (in-memory)
```

**Total: 260 tests**

## Test Layers

### Domain (78 tests)

`test_value_objects.py` — validates `BarConfig` construction rules:
- All 10 `BarType` values are valid
- `threshold` must be positive
- `ewm_span` must be >= 10
- `warmup_period` must be >= 1 and <= `ewm_span`
- `config_hash` is deterministic and 16 hex characters
- `is_information_driven` returns correct value per bar type

`test_entities.py` — validates `AggregatedBar` model invariants:
- `high >= low` enforced
- `tick_count >= 1` enforced
- `buy_volume + sell_volume <= volume` enforced (1e-9 tolerance)
- `start_ts < end_ts` enforced
- Valid bars construct without error
- `Decimal` precision preserved for prices

### Application (152 tests)

`test_standard_bars.py` — tick, volume, and dollar aggregator correctness:
- Empty input returns `[]`
- Missing column raises `ValueError`
- Bar boundaries: row N divides exactly into `floor(N / threshold)` bars plus one partial
- OHLCV correctness: first open, max high, min low, last close, sum volume
- `tick_count` sums to total input row count
- `end_ts` is `last_candle_start + candle_period` (not the same as `start_ts` of next bar)
- Buy/sell estimation: buy fraction → 1.0 when `close == high`, 0.0 when `close == low`
- VWAP equals close when all rows have identical price

`test_information_driven_bars.py` — imbalance and run aggregator correctness:
- Purely bullish input triggers imbalance bars at predictable intervals
- Purely bearish input triggers at symmetric intervals
- Alternating input suppresses imbalance, produces fewer bars
- Wrong `bar_type` in config raises `ValueError`
- Adaptive threshold reduces bar size in sustained directional markets
- Run bars: dominant run correctly identified as `max(buy_run, sell_run)`
- All three imbalance variants (tick/volume/dollar) tested independently
- All three run variants tested independently
- `tick_count` invariant: sum across all bars equals input row count

`test_statistical.py` — properties from López de Prado §2.3:
- Tick bar count per period is deterministic: `floor(N / threshold) ± 1` regardless of price volatility or volume
- Dollar bar count increases with dollar flow (10× volume → more bars)
- Dollar bars normalise across price-level differences: constant dollar flow at price P₁ vs 2P₁ (with halved volume) yields similar bar counts (< 50% relative difference)
- Dollar bars have higher coefficient of variation (CV) than tick bars across alternating calm/active regimes
- `tick_count` sum equals `n_rows` for parameterized inputs of size 100 and 500

### Infrastructure (32 tests)

`test_duckdb_repository.py` — `DuckDBBarRepository` against in-memory DuckDB:
- `ingest` returns count of rows written
- `INSERT OR IGNORE` skips duplicates on repeated ingest
- `query` filters correctly by `(asset, bar_type, config_hash, date_range)`
- `get_available_configs` returns distinct `(bar_type, config_hash)` pairs
- `get_date_range` returns `None` when no data exists, correct range otherwise
- `get_latest_end_ts` returns `None` when empty, max `end_ts` otherwise
- `count` and `count_by_config` return accurate row totals
- `delete` removes the correct rows and returns the deleted count
- Timezone handling: naive datetimes from DuckDB are normalized to UTC

## Fixtures (`conftest.py`)

| Fixture / Helper | Returns | Purpose |
|---|---|---|
| `make_trades_df(n, ...)` | `pl.DataFrame` | Uniform OHLCV rows, optional price step |
| `make_bullish_trades_df(n, ...)` | `pl.DataFrame` | All candles: `close = open + 10` |
| `make_bearish_trades_df(n, ...)` | `pl.DataFrame` | All candles: `close = open - 10` |
| `make_alternating_trades_df(n, ...)` | `pl.DataFrame` | Alternating buy/sell direction |
| `make_varying_volume_df(volumes, ...)` | `pl.DataFrame` | Explicit per-row volumes |
| `make_aggregated_bar(...)` | `AggregatedBar` | Minimal entity with sensible defaults |
| `bar_connection_manager` | `ConnectionManager` | In-memory DuckDB (`:memory:`) |
| `bar_repository` | `DuckDBBarRepository` | Repository with `aggregated_bars` table pre-created |
| `btc_asset` / `eth_asset` | `Asset` | BTCUSDT / ETHUSDT fixtures |
| `tick_bar_config` / `volume_bar_config` / `dollar_bar_config` | `BarConfig` | Pre-built standard bar configs |
| `imbalance_bar_config` / `run_bar_config` | `BarConfig` | Pre-built information-driven configs |

## Running the Tests

```bash
# All bars tests
just test src/tests/bars/

# Domain only
just test src/tests/bars/domain/

# Application only
just test src/tests/bars/application/

# Statistical tests only
just test src/tests/bars/application/test_statistical.py

# Infrastructure integration tests only
just test src/tests/bars/infrastructure/
```
