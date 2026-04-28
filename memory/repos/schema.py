"""Schema / connection management — scaffolded re-export.

The implementation still lives in :mod:`memory.__init__`. This module
is the future home for ``init_db``, ``get_db``, and migrations.
"""

from __future__ import annotations

from memory import (  # noqa: F401
    get_db,
    get_schema_version,
    init_db,
)

__all__ = ["get_db", "get_schema_version", "init_db"]
