# src/execution/__init__.py
"""
Layer 1: Execution (Stable Infrastructure)

⚠️  STABILITY WARNING: This package contains stable infrastructure code.
    Only modify if absolutely necessary. Changes here affect the entire system.

This package contains:
- IBKR broker connector (ibkr_core, ibkr_orders, ibkr_queries, ibkr_options, ibkr_utils)
- Order types enum (order_types)

Note: marketdata_client moved to layer2_data (2026-01-28) - it's data access, not execution.

These modules rarely change and provide the foundation for all trading operations.
"""

from execution.ibkr_core import IBKRConnector, get_ibkr_connector
from execution.ibkr_gateway import IBKRGateway
from execution.order_types import IBKROrderType, is_stop_order

__all__ = [
    'IBKRConnector',
    'get_ibkr_connector',
    'IBKRGateway',
    'IBKROrderType',
    'is_stop_order',
]
