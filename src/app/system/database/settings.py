"""DuckDB database settings loaded from environment / ``.env``."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Configuration for the DuckDB analytical database.

    Every field is read from an environment variable with the ``DUCKDB_``
    prefix (e.g. ``DUCKDB_PATH``, ``DUCKDB_READ_ONLY``).
    """

    model_config = SettingsConfigDict(
        env_prefix="DUCKDB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    path: str = ":memory:"
    read_only: bool = False
    memory_limit: str = "4GB"
    threads: int = -1  # -1 → DuckDB auto-detects

    @property
    def sqlalchemy_url(self) -> str:
        """Return a SQLAlchemy-compatible connection string."""
        if self.path == ":memory:":
            return "duckdb:///:memory:"
        return f"duckdb:///{self.path}"
