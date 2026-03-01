# Set PowerShell as the default shell on Windows for better compatibility.
set windows-powershell := true
set dotenv-load

# List all available commands. This is the default recipe.
default:
    @just --list

# ==============================================================================
# 🚀 Application
# ==============================================================================
# Commands for running the main application.
run:
    python main.py

# ==============================================================================
# 📥 Data Ingestion
# ==============================================================================
# Commands for fetching OHLCV data from Binance into DuckDB.

# Ingest OHLCV data from Binance.
# Usage: just ingest --assets BTCUSDT,ETHUSDT --timeframes 1h,4h --start 2020-01-01
ingest *args:
    python -m src.app.ingestion.cli {{args}}

# ==============================================================================
# 📦 Dependency Management
# ==============================================================================
# Commands for managing project dependencies with uv.

# Add a new package to the project's virtual environment.
# Usage: just add <package-name>
add package:
    uv add {{package}}

# ==============================================================================
# 📖 Documentation
# ==============================================================================
# Commands for building and serving the project documentation.

# Start a live-reloading local server for documentation.
serve:
    mkdocs serve

# ==============================================================================
# ✅ Git Hooks Management
# ==============================================================================
# Commands for setting up and tearing down Git pre-commit hooks.

# Install pre-commit hooks into the .git/ directory.
install-hooks:
    pre-commit install

# Uninstall pre-commit hooks from the .git/ directory.
uninstall-hooks:
    pre-commit uninstall

# Run all pre-commit hooks against all files.
lint:
    pre-commit run --all-files

# ==============================================================================
# 🗄️  Database Migrations
# ==============================================================================
# Commands for managing DuckDB schema migrations via Alembic.

_alembic_cfg := "src/app/system/database/alembic.cfg"

# Run all pending Alembic migrations to bring the database up to date.
migrate:
    alembic -c {{_alembic_cfg}} upgrade head

# Create a new Alembic migration file.
# Usage: just migration "short description of the change"
migration message:
    alembic -c {{_alembic_cfg}} revision -m "{{message}}"

# Downgrade the database by one migration step.
migrate-down:
    alembic -c {{_alembic_cfg}} downgrade -1
