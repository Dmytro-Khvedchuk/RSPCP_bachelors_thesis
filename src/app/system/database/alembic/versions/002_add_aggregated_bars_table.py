"""Add aggregated_bars table.

Revision ID: 002
Revises: 001
Create Date: 2026-03-12
"""

from __future__ import annotations

from alembic import op


revision: str = "002"
down_revision: str = "001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the aggregated_bars table and supporting indices."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS aggregated_bars (
            asset           VARCHAR        NOT NULL,
            bar_type        VARCHAR        NOT NULL,
            bar_config_hash VARCHAR(16)    NOT NULL,
            start_ts        TIMESTAMPTZ    NOT NULL,
            end_ts          TIMESTAMPTZ    NOT NULL,
            open            DECIMAL(18, 8) NOT NULL,
            high            DECIMAL(18, 8) NOT NULL,
            low             DECIMAL(18, 8) NOT NULL,
            close           DECIMAL(18, 8) NOT NULL,
            volume          DOUBLE         NOT NULL,
            tick_count      INTEGER        NOT NULL,
            buy_volume      DOUBLE         NOT NULL,
            sell_volume      DOUBLE         NOT NULL,
            vwap            DECIMAL(18, 8) NOT NULL,
            PRIMARY KEY (asset, bar_type, bar_config_hash, start_ts)
        );
        """
    )
    # Index for the most common query pattern: filter by asset + bar_type + config_hash,
    # then range-scan on start_ts.  The PK already provides this ordering, but an
    # explicit index makes the intent clear and enables covering-index scans if
    # DuckDB decides to use it.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_bars_asset_type_hash_ts
        ON aggregated_bars (asset, bar_type, bar_config_hash, start_ts);
        """
    )


def downgrade() -> None:
    """Drop the aggregated_bars table and its indices."""
    op.execute("DROP INDEX IF EXISTS idx_bars_asset_type_hash_ts;")
    op.execute("DROP TABLE IF EXISTS aggregated_bars;")
