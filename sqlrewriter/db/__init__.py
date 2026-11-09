"""Database backends."""

from __future__ import annotations

from ..config import DatabaseSettings
from .base import DatabaseBackend, PlanInfo, QueryError, QueryTimeout, ResultSet


def create_backend(settings: DatabaseSettings) -> DatabaseBackend:
    backend = settings.backend.lower()
    if backend == "sqlite":
        from .sqlite_backend import SQLiteBackend

        return SQLiteBackend(settings.sqlite)
    if backend == "oracle":
        from .oracle_backend import OracleBackend

        return OracleBackend(settings.oracle)
    raise ValueError(f"Unknown database backend {settings.backend!r} (expected 'sqlite' or 'oracle')")


__all__ = ["DatabaseBackend", "PlanInfo", "QueryError", "QueryTimeout", "ResultSet", "create_backend"]
