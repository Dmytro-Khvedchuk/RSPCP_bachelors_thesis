# bars
> López de Prado alternative bar aggregation (Phase 2)

## Overview

Implements all nine alternative bar types from *Advances in Financial Machine Learning* (López de Prado, 2018), §2.3, plus time bars. Standard bars (tick, volume, dollar) use a fully vectorized Polars cumsum pipeline. Information-driven bars (imbalance, run) use a sequential NumPy O(n) loop with an adaptive EMA threshold that adjusts after a configurable warmup period.

Bars are stored in DuckDB under composite key `(asset, bar_type, bar_config_hash, start_ts)`, keyed by a SHA-256 config hash so multiple parameter sets coexist without collision.

## Architecture

```
bars/
├── domain/
│   ├── value_objects.py    # BarType (StrEnum, 10 variants), BarConfig (frozen Pydantic)
│   ├── entities.py         # AggregatedBar (frozen Pydantic, Decimal prices)
│   └── protocols.py        # IBarAggregator, IBarRepository (typing.Protocol)
├── application/
│   ├── _aggregation.py     # Shared: validate_input, aggregate_by_metric, build_bar_from_arrays
│   ├── tick_bars.py        # TickBarAggregator
│   ├── volume_bars.py      # VolumeBarAggregator
│   ├── dollar_bars.py      # DollarBarAggregator
│   ├── imbalance_bars.py   # ImbalanceBarAggregator (tick/volume/dollar imbalance)
│   └── run_bars.py         # RunBarAggregator (tick/volume/dollar run)
└── infrastructure/
    └── duckdb_repository.py  # DuckDBBarRepository
```

Dependencies flow inward: `infrastructure` → `application` → `domain`. The domain layer has zero external dependencies.

## Bar Types

| BarType | Class | Sampling trigger | Algorithm |
|---|---|---|---|
| `TICK` | `TickBarAggregator` | Every N input rows | Vectorized cumsum |
| `VOLUME` | `VolumeBarAggregator` | Cumulative base-asset volume ≥ threshold | Vectorized cumsum |
| `DOLLAR` | `DollarBarAggregator` | Cumulative `close × volume` ≥ threshold | Vectorized cumsum |
| `TICK_IMBALANCE` | `ImbalanceBarAggregator` | `|Σ direction|` ≥ adaptive threshold | Sequential NumPy |
| `VOLUME_IMBALANCE` | `ImbalanceBarAggregator` | `|Σ direction × volume|` ≥ adaptive threshold | Sequential NumPy |
| `DOLLAR_IMBALANCE` | `ImbalanceBarAggregator` | `|Σ direction × close × volume|` ≥ adaptive threshold | Sequential NumPy |
| `TICK_RUN` | `RunBarAggregator` | Max consecutive run count ≥ adaptive threshold | Sequential NumPy |
| `VOLUME_RUN` | `RunBarAggregator` | Max consecutive run volume ≥ adaptive threshold | Sequential NumPy |
| `DOLLAR_RUN` | `RunBarAggregator` | Max consecutive run dollar volume ≥ adaptive threshold | Sequential NumPy |
| `TIME` | — | Fixed calendar interval | (reserved for future use) |

**Direction classification** (shared by imbalance and run): a candle is classified as a buy (+1) if `close >= open`, sell (−1) otherwise.

## Key Components

### `BarConfig`

Frozen Pydantic model controlling bar construction parameters:

```python
from src.app.bars.domain.value_objects import BarConfig, BarType

# Standard bar — only threshold is required
tick_cfg = BarConfig(bar_type=BarType.TICK, threshold=1000.0)

# Information-driven bar — EWM and warmup matter
imb_cfg = BarConfig(
    bar_type=BarType.TICK_IMBALANCE,
    threshold=500.0,
    ewm_span=100,       # EMA half-life for adaptive threshold; must be >= 10
    warmup_period=50,   # Fixed-threshold warmup bars before EMA kicks in; must be <= ewm_span
)

# Config hash for deduplication storage key (16-char hex)
print(tick_cfg.config_hash)          # e.g. "3a7f2b91c4e05d88"
print(tick_cfg.is_information_driven)  # False
```

| Field | Type | Constraint | Purpose |
|---|---|---|---|
| `bar_type` | `BarType` | — | Aggregation variant |
| `threshold` | `float` | `> 0` | Sampling threshold |
| `ewm_span` | `int` | `>= 10` | EMA span for adaptive threshold |
| `warmup_period` | `int` | `>= 1`, `<= ewm_span` | Fixed-threshold warmup count |

### `AggregatedBar`

Frozen Pydantic entity produced by every aggregator:

| Field | Type | Notes |
|---|---|---|
| `asset` | `Asset` | Trading-pair symbol |
| `bar_type` | `BarType` | Aggregation variant |
| `start_ts` | `datetime` | Inclusive, UTC |
| `end_ts` | `datetime` | Exclusive (`last_candle_start + candle_period`) |
| `open/high/low/close` | `Decimal` | Exchange-precision DECIMAL(18,8) |
| `volume` | `float` | Total base-asset volume |
| `tick_count` | `int` | Number of input candles in the bar |
| `buy_volume` | `float` | Estimated via close position in H-L range |
| `sell_volume` | `float` | `volume × (1 − buy_fraction)` |
| `vwap` | `Decimal` | `Σ(typical_price × volume) / Σ(volume)` |

Model-level invariants are enforced at construction: `high >= low`, `tick_count >= 1`, `buy_volume + sell_volume <= volume`, `start_ts < end_ts`.

## Usage

### Standard bars

```python
import polars as pl
from src.app.bars.application.dollar_bars import DollarBarAggregator
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.app.ohlcv.domain.value_objects import Asset

asset = Asset(symbol="BTCUSDT")
config = BarConfig(bar_type=BarType.DOLLAR, threshold=1_000_000.0)
agg = DollarBarAggregator()

# trades: pl.DataFrame with columns timestamp, open, high, low, close, volume
bars = agg.aggregate(trades, asset=asset, config=config)
```

### Information-driven bars

```python
from src.app.bars.application.imbalance_bars import ImbalanceBarAggregator
from src.app.bars.domain.value_objects import BarConfig, BarType

config = BarConfig(
    bar_type=BarType.DOLLAR_IMBALANCE,
    threshold=500_000.0,
    ewm_span=100,
    warmup_period=50,
)
agg = ImbalanceBarAggregator()
bars = agg.aggregate(trades, asset=asset, config=config)
```

### Persisting and querying

```python
from src.app.bars.infrastructure.duckdb_repository import DuckDBBarRepository
from src.app.ohlcv.domain.value_objects import DateRange
from datetime import datetime, UTC

repo = DuckDBBarRepository(connection_manager=connection_manager)

# Persist (INSERT OR IGNORE — safe to call repeatedly)
written = repo.ingest(bars, config_hash=config.config_hash)

# Query a date range
date_range = DateRange(
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 2, 1, tzinfo=UTC),
)
result = repo.query(asset, BarType.DOLLAR_IMBALANCE, config.config_hash, date_range)

# Incremental ingestion — find where to resume
latest = repo.get_latest_end_ts(asset, BarType.DOLLAR_IMBALANCE, config.config_hash)
```

## Algorithms

### Standard bars — vectorized Polars pipeline (`_aggregation.py`)

1. Compute per-row metric: `pl.lit(1)` (tick), `pl.col("volume")` (volume), `close × volume` (dollar).
2. Cumulative sum → `_cumsum`.
3. Bar ID: `floor((_cumsum − metric) / threshold)` — the row that crosses the threshold is the *last* row of its bar.
4. Group by `_bar_id`: first open, max high, min low, last close, sum volume, len tick_count.
5. Buy/sell estimation: `buy_frac = (close − low) / (high − low)`, fallback 0.5 when `high == low`.
6. VWAP: `Σ(typical_price × volume) / Σ(volume)` where `typical = (high + low + close) / 3`.
7. `end_ts = last_candle_start + candle_period` (period inferred from first two timestamps).

### Information-driven bars — sequential NumPy loop

Both `ImbalanceBarAggregator` and `RunBarAggregator` share the same structure:

```
threshold ← config.threshold
alpha ← 2 / (ewm_span + 1)
cumulative ← 0

for each row i:
    accumulate signed metric (imbalance) or track run length (run)
    if trigger_condition:
        emit bar [bar_start … i]
        bars_formed += 1
        if bars_formed >= warmup_period:
            threshold ← alpha × observed_value + (1 − alpha) × threshold   # EMA update
        reset state
remaining rows → one final (partial) bar
```

**Imbalance trigger:** `|cumulative| >= threshold`
**Run trigger:** `max(max_buy_run, max_sell_run) >= threshold`

The EMA update formula is `θ_t = α × |Θ_t| + (1 − α) × θ_{t−1}`, where `Θ_t` is the observed imbalance (or dominant run) at bar closure and `α = 2 / (span + 1)`.

## Database Schema

Migration: `src/app/system/database/alembic/versions/002_add_aggregated_bars_table.py`

```sql
CREATE TABLE aggregated_bars (
    asset           VARCHAR        NOT NULL,
    bar_type        VARCHAR        NOT NULL,
    bar_config_hash VARCHAR(16)    NOT NULL,   -- SHA-256[:16] of BarConfig JSON
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
-- Covering index for the primary query pattern
CREATE INDEX idx_bars_asset_type_hash_ts
    ON aggregated_bars (asset, bar_type, bar_config_hash, start_ts);
```

Apply with:

```bash
just migrate
```

## Dependencies

| Dependency | Layer | Purpose |
|---|---|---|
| `polars` | application | Vectorized pipeline for standard bars |
| `numpy` | application | Sequential loop for information-driven bars |
| `pydantic` | domain | Value objects and entity validation |
| `src.app.ohlcv.domain` | domain | `Asset`, `DateRange` value objects |
| `src.app.system.database` | infrastructure | `ConnectionManager`, `BaseRepository` |
| `loguru` | infrastructure | Ingestion timing logs |
| `sqlalchemy` | infrastructure | Parameterized DuckDB queries |
