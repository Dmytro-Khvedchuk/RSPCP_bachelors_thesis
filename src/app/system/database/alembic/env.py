"""Alembic environment configuration for DuckDB migrations."""

from __future__ import annotations

from alembic import context
from alembic.ddl.impl import DefaultImpl
from sqlalchemy import create_engine, pool

from src.app.system.database.settings import DatabaseSettings


class DuckDBImpl(DefaultImpl):
    """Register DuckDB as a known dialect for Alembic."""

    __dialect__ = "duckdb"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout."""
    settings = DatabaseSettings()
    context.configure(
        url=settings.sqlalchemy_url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DuckDB connection."""
    settings = DatabaseSettings()
    connectable = create_engine(
        settings.sqlalchemy_url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
