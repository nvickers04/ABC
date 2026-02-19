"""
IBKR Order Types - Centralized enum for all Interactive Brokers order types.

This module provides type-safe access to IBKR order type strings, preventing
typos and inconsistencies across the codebase.

Usage:
    from execution.order_types import IBKROrderType, is_stop_order

    # Use enum value for IBKR API calls
    order.orderType = IBKROrderType.LIMIT.value  # "LMT"

    # Check if order is a stop variant
    if is_stop_order(order_type):
        ...
"""

from enum import Enum
from typing import Union


class IBKROrderType(str, Enum):
    """
    IBKR order types - use .value for IBKR API calls.

    The enum names are descriptive, but the values are the exact strings
    expected by the IBKR API.
    """

    # Basic order types
    LIMIT = "LMT"                    # Standard limit order
    MARKET = "MKT"                   # Market order, fills at best available
    STOP = "STP"                     # Stop order, triggers market order at aux_price
    STOP_LIMIT = "STP LMT"          # Stop-limit, triggers limit order at aux_price

    # Trailing stop orders
    TRAIL = "TRAIL"                  # Trailing stop, tracks price by amount/percent
    TRAIL_LIMIT = "TRAIL LIMIT"     # Trailing stop with limit offset protection

    # Time-based orders (auction orders)
    MARKET_ON_CLOSE = "MOC"          # Executes at closing auction price
    LIMIT_ON_CLOSE = "LOC"           # Closing auction with limit price protection
    MARKET_ON_OPEN = "MOO"           # IBKR: MKT + tif='OPG' (opening auction)
    LIMIT_ON_OPEN = "LOO"            # IBKR: LMT + tif='OPG' (opening auction with limit)

    # Smart/Pegged order types
    MIDPRICE = "MIDPRICE"            # Pegs to bid/ask midpoint, auto-adjusts
    RELATIVE = "REL"                 # Pegs to bid (buy) or ask (sell) with offset
    SNAP_MID = "SNAP MID"           # Sets limit to midpoint at submission time

    # Fill constraint order types
    FILL_OR_KILL = "FOK"             # Must fill entire qty immediately or cancel
    IMMEDIATE_OR_CANCEL = "IOC"      # Fills what it can immediately, cancels rest

    # Algorithmic order types (use algoStrategy param, not orderType)
    VWAP = "VWAP"                    # IBKR: MKT + algoStrategy='Vwap', targets volume-weighted avg
    TWAP = "TWAP"                    # IBKR: MKT + algoStrategy='Twap', spreads evenly over time
    ICEBERG = "ICEBERG"              # IBKR: LMT + displaySize param, hides total quantity


# Set of order types that are stop-based (trigger on price movement)
STOP_ORDER_TYPES = frozenset({
    IBKROrderType.STOP,
    IBKROrderType.STOP_LIMIT,
    IBKROrderType.TRAIL,
    IBKROrderType.TRAIL_LIMIT,
})

# String values for quick lookup
_STOP_ORDER_VALUES = frozenset(t.value for t in STOP_ORDER_TYPES)


def is_stop_order(order_type: Union[str, IBKROrderType]) -> bool:
    """
    Check if an order type is a stop variant (STP, STP LMT, TRAIL, TRAIL LIMIT).

    Args:
        order_type: Either an IBKROrderType enum or a string value

    Returns:
        True if the order type is a stop-based order

    Example:
        >>> is_stop_order(IBKROrderType.STOP)
        True
        >>> is_stop_order("TRAIL")
        True
        >>> is_stop_order("LMT")
        False
    """
    if isinstance(order_type, IBKROrderType):
        return order_type in STOP_ORDER_TYPES
    return order_type in _STOP_ORDER_VALUES


def is_limit_order(order_type: Union[str, IBKROrderType]) -> bool:
    """
    Check if an order type is limit-based (has price protection).

    Args:
        order_type: Either an IBKROrderType enum or a string value

    Returns:
        True if the order type has limit price protection
    """
    limit_types = {
        IBKROrderType.LIMIT.value,
        IBKROrderType.STOP_LIMIT.value,
        IBKROrderType.TRAIL_LIMIT.value,
        IBKROrderType.LIMIT_ON_CLOSE.value,
        IBKROrderType.LIMIT_ON_OPEN.value,
    }
    if isinstance(order_type, IBKROrderType):
        return order_type.value in limit_types
    return order_type in limit_types


def is_market_order(order_type: Union[str, IBKROrderType]) -> bool:
    """
    Check if an order type executes at market price (no limit protection).

    Args:
        order_type: Either an IBKROrderType enum or a string value

    Returns:
        True if the order type executes at market price
    """
    market_types = {
        IBKROrderType.MARKET.value,
        IBKROrderType.STOP.value,  # Stop becomes market when triggered
        IBKROrderType.TRAIL.value,  # Trail stop becomes market when triggered
        IBKROrderType.MARKET_ON_CLOSE.value,
        IBKROrderType.MARKET_ON_OPEN.value,
    }
    if isinstance(order_type, IBKROrderType):
        return order_type.value in market_types
    return order_type in market_types


def is_algo_order(order_type: Union[str, IBKROrderType]) -> bool:
    """
    Check if an order type is an algorithmic execution order.

    Args:
        order_type: Either an IBKROrderType enum or a string value

    Returns:
        True if the order type is an IBKR algo order
    """
    algo_types = {
        IBKROrderType.VWAP.value,
        IBKROrderType.TWAP.value,
        IBKROrderType.ICEBERG.value,
    }
    if isinstance(order_type, IBKROrderType):
        return order_type.value in algo_types
    return order_type in algo_types
