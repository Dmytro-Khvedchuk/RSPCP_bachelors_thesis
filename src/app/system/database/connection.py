"""DuckDB connection manager backed by SQLAlchemy Core."""

from __future__ import annotations

from types import TracebackType
from typing import Self

from loguru import logger
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import Engine

from src.app.system.database.exceptions import DatabaseConnectionError
from src.app.system.database.settings import DatabaseSettings


class ConnectionManager:
    """Manages a single SQLAlchemy :class:`Engine` pointing at a DuckDB file.

    Intended to be used as a context manager::

        with ConnectionManager(settings) as cm:
            conn = cm.connect()
            ...
    """

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        """Initialise the manager with optional database settings.

        Args:
            settings: DuckDB configuration. Falls back to env-based defaults
                when *None*.
        """
        self._settings = settings or DatabaseSettings()
        self._engine: Engine | None = None

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> Self:
        """Initialise the engine on context entry.

        Returns:
            The manager instance.
        """
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Dispose the engine on context exit."""
        self.dispose()

    # -- public API ----------------------------------------------------------

    def initialize(self) -> None:
        """Create the SQLAlchemy engine and verify connectivity.

        Raises:
            DatabaseConnectionError: If the connection cannot be established.
        """
        cfg = self._settings
        connect_args: dict[str, object] = {}
        if cfg.read_only:
            connect_args["read_only"] = True

        config: dict[str, str] = {}
        if cfg.memory_limit:
            config["memory_limit"] = cfg.memory_limit
        if cfg.threads > 0:
            config["threads"] = str(cfg.threads)
        if config:
            connect_args["config"] = config

        logger.info("Creating DuckDB engine (path={})", cfg.path)
        try:
            self._engine = create_engine(
                cfg.sqlalchemy_url,
                connect_args=connect_args,
            )
            # Smoke test.
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("DuckDB connection verified successfully")
        except Exception as exc:
            logger.error("Failed to connect to DuckDB: {}", exc)
            raise DatabaseConnectionError(str(exc)) from exc

    def dispose(self) -> None:
        """Dispose of the engine and release resources."""
        if self._engine is not None:
            self._engine.dispose()
            logger.info("DuckDB engine disposed")
            self._engine = None

    def connect(self) -> Connection:
        """Return a new :class:`~sqlalchemy.Connection`.

        Returns:
            An open SQLAlchemy connection to the DuckDB database.

        Raises:
            DatabaseConnectionError: If the manager has not been initialised.
        """
        if self._engine is None:
            raise DatabaseConnectionError("ConnectionManager has not been initialised — call initialize() first")
        return self._engine.connect()

    @property
    def engine(self) -> Engine:
        """Return the underlying SQLAlchemy engine.

        Returns:
            The active SQLAlchemy :class:`Engine`.

        Raises:
            DatabaseConnectionError: If the manager has not been initialised.
        """
        if self._engine is None:
            raise DatabaseConnectionError("ConnectionManager has not been initialised — call initialize() first")
        return self._engine
