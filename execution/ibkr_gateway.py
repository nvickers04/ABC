"""
IBKR Gateway — Adapter implementing the Broker Gateway for Interactive Brokers.

Wraps the existing IBKRConnector (which uses ib_insync) and adds:
  - Capability checking before every broker method call
  - LiveState event wiring
  - State refresh coordination

This is the ONLY file that Layer 2 indirectly uses to reach IBKR.
Layer 2 code never imports this directly — it goes through create_gateway().
"""

import logging
from typing import Any, Dict

from data.broker_gateway import (
    BrokerCapabilities,
    check_capability,
)
from execution.ibkr_core import get_ibkr_connector

logger = logging.getLogger(__name__)


class IBKRGateway:
    """
    Broker gateway adapter for Interactive Brokers via ib_insync.

    Uses __getattr__ to delegate all broker method calls to the underlying
    IBKRConnector while checking capabilities first.  Explicitly defined
    properties and methods take priority over __getattr__.
    """

    def __init__(self, capabilities: BrokerCapabilities, config: dict):
        self.capabilities = capabilities
        self._config = config
        self._connector = None  # Set on connect()

    # =====================================================================
    # CONNECTION
    # =====================================================================

    async def connect(self) -> bool:
        """Create/get the IBKRConnector singleton and connect to TWS."""
        self._connector = get_ibkr_connector()
        return await self._connector.connect()

    async def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._connector:
            await self._connector.disconnect()

    # =====================================================================
    # PROPERTIES — explicitly defined so they don't go through __getattr__
    # =====================================================================

    @property
    def is_connected(self) -> bool:
        if self._connector is None:
            return False
        return self._connector.is_connected()

    @property
    def account_id(self) -> str:
        return self._connector.account_id if self._connector else ''

    @property
    def day_trades_remaining(self) -> int:
        return self._connector.day_trades_remaining if self._connector else 0

    @property
    def net_liquidation(self) -> float:
        return self._connector.net_liquidation if self._connector else 0.0

    @property
    def available_funds(self) -> float:
        return getattr(self._connector, 'available_funds', 0.0) if self._connector else 0.0

    @available_funds.setter
    def available_funds(self, value: float):
        """Allow ToolExecutor._check_cash to write back fresh values."""
        if self._connector:
            self._connector.available_funds = value

    @property
    def cash_value(self) -> float:
        return self._connector.cash_value if self._connector else 0.0

    # =====================================================================
    # SYNC DELEGATES — called directly without capability check
    # =====================================================================

    def get_cached_account_values(self) -> list:
        return self._connector.get_cached_account_values() if self._connector else []

    def get_cached_portfolio(self) -> list:
        return self._connector.get_cached_portfolio() if self._connector else []

    def get_cached_trades(self) -> list:
        return self._connector.get_cached_trades() if self._connector else []

    # =====================================================================
    # LIVE STATE WIRING
    # =====================================================================

    async def wire_live_state(self) -> None:
        """
        Wire LiveState to broker events.

        Delegates to the existing wire_to_broker() in live_state.py,
        passing the raw connector.  LiveState event handlers still
        receive ib_insync objects for now — a future pass can abstract
        those into generic event types.
        """
        from data.live_state import wire_to_broker
        await wire_to_broker(self._connector)

    # =====================================================================
    # STATE REFRESH
    # =====================================================================

    async def refresh_state(self) -> None:
        """
        Force-refresh LiveState from broker after an order placement.

        Delegates to LiveState.refresh_after_order() which polls the
        broker for fresh account/portfolio/order data.
        """
        from data.live_state import get_live_state
        await get_live_state().refresh_after_order()

    # =====================================================================
    # TRUTHINESS — handler guards: `if not executor.gateway: ...`
    # =====================================================================

    def __bool__(self) -> bool:
        return self._connector is not None and self._connector.is_connected()

    # =====================================================================
    # DYNAMIC DELEGATION — all broker methods go through here
    # =====================================================================

    def __getattr__(self, name: str):
        """
        Delegate attribute access to IBKRConnector with capability checking.

        This is called only for attributes NOT found on the gateway itself
        (i.e., not __init__ args, explicit properties, or defined methods).

        Flow:
          1. Skip private/dunder names (raise AttributeError).
          2. Check if the method is capability-gated.
          3. Delegate to the connector.
        """
        if name.startswith('_'):
            raise AttributeError(name)

        # Capability check — raises CapabilityUnavailableError if blocked
        check_capability(self.capabilities, name)

        # Delegate to the underlying connector
        if self._connector is None:
            raise RuntimeError("IBKRGateway: not connected (no connector)")
        return getattr(self._connector, name)
