"""Database-layer exception hierarchy."""

from __future__ import annotations


class DatabaseError(Exception):
    """Base exception for all database-related errors."""


class DatabaseConnectionError(DatabaseError):
    """Raised when a database connection cannot be established."""


class DataSourceNotFoundError(DatabaseError):
    """Raised when the requested data source (table / file) does not exist."""


class DataIntegrityError(DatabaseError):
    """Raised on constraint violations or data-integrity issues."""


class QueryError(DatabaseError):
    """Raised when a SQL query fails to execute."""


class MigrationError(DatabaseError):
    """Raised when an Alembic migration fails."""


class TemporalLeakageError(DatabaseError):
    """Raised when a temporal split would cause data leakage between sets."""
