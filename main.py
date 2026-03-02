"""Application entry-point — logging bootstrap + DuckDB smoke test."""

from __future__ import annotations

from loguru import logger

from src.app.system.database import ConnectionManager, DatabaseSettings
from src.app.system.logging import setup_logging


def main() -> None:
    """Start of the program."""
    setup_logging(level="DEBUG")

    logger.info("Starting RSPCP bachelors thesis application")

    settings = DatabaseSettings()
    with ConnectionManager(settings):
        logger.success("DuckDB smoke test passed (path={})", settings.path)


if __name__ == "__main__":
    main()
