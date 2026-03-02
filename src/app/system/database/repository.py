"""Abstract base repository for DuckDB-backed data access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import Connection, CursorResult, text

from src.app.system.database.connection import ConnectionManager


class BaseRepository[T](ABC):
    """Abstract repository that owns a reference to a :class:`ConnectionManager`.

    Subclasses **must** define :pyattr:`TABLE_NAME` and implement at least
    :meth:`count`.
    """

    TABLE_NAME: str

    def __init__(self, connection_manager: ConnectionManager) -> None:
        """Initialise the repository with a connection manager.

        Args:
            connection_manager: Shared connection manager for database access.
        """
        self._cm: ConnectionManager = connection_manager

    def _get_connection(self) -> Connection:
        """Open and return a new database connection.

        Returns:
            A SQLAlchemy connection from the underlying manager.
        """
        return self._cm.connect()

    def _ensure_table_exists(self) -> bool:
        """Return *True* if :pyattr:`TABLE_NAME` exists in the database.

        Returns:
            Whether the table exists in the information schema.
        """
        with self._get_connection() as conn:
            result: CursorResult[Any] = conn.execute(
                text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = :name"),
                {"name": self.TABLE_NAME},
            )
            return bool(result.scalar())

    @abstractmethod
    def count(self) -> int:
        """Return the total number of rows in the table.

        Returns:
            Row count.
        """
