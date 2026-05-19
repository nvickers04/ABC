"""
Execution layer — IBKR broker connector and order types.

Stable infrastructure: changes here affect the whole system.
Market data lives in :mod:`data` (not this package).
"""

from execution.ibkr_core import IBKRConnector, get_ibkr_connector
from execution.ibkr_gateway import IBKRGateway
from execution.order_types import IBKROrderType, is_stop_order

__all__ = [
    "IBKRConnector",
    "get_ibkr_connector",
    "IBKRGateway",
    "IBKROrderType",
    "is_stop_order",
]
