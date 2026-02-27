# Recommendation System for Predicting Cryptocurrency Prices
Bachelor's Thesis
---
Author: Dmytro Khvedchuk

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- [just](https://github.com/casey/just) command runner

## Installation

```bash
uv sync
```

## Environment Setup

Copy the example environment file and adjust values as needed:

```bash
cp .example.env .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `DUCKDB_PATH` | `:memory:` | Path to persistent DuckDB file |
| `DUCKDB_MEMORY_LIMIT` | `4GB` | DuckDB memory limit |
| `LOG_LEVEL` | `DEBUG` | Minimum log level |

## Data Directory

Create the data directory for persistent storage:

```bash
mkdir -p data/
```

## Database Migrations

Run all pending migrations:

```bash
just migrate
```

Create a new migration:

```bash
just migration "describe the change"
```

Rollback one step:

```bash
just migrate-down
```

## Usage

```bash
just run
```

## Linting

```bash
just lint
```
