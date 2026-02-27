"""Create ohlcv table.

Revision ID: 001
Revises: None
Create Date: 2026-02-19
"""

from __future__ import annotations

from alembic import op


revision: str = "001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade migration."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            asset       VARCHAR   NOT NULL,
            timeframe   VARCHAR   NOT NULL,
            timestamp   TIMESTAMPTZ NOT NULL,
            open        DECIMAL(18, 8) NOT NULL,
            high        DECIMAL(18, 8) NOT NULL,
            low         DECIMAL(18, 8) NOT NULL,
            close       DECIMAL(18, 8) NOT NULL,
            volume      DOUBLE    NOT NULL,
            PRIMARY KEY (asset, timeframe, timestamp)
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ohlcv_asset_tf_ts
        ON ohlcv (asset, timeframe, timestamp);
        """
    )


def downgrade() -> None:
    """Downgrade migration."""
    op.execute("DROP INDEX IF EXISTS idx_ohlcv_asset_tf_ts;")
    op.execute("DROP TABLE IF EXISTS ohlcv;")
