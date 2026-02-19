"""
Broker Gateway — Capability-configurable broker abstraction.

Separates Layer 2 (tools/state) from Layer 1 (broker implementation).
Capabilities use sensible defaults; settings come from .env.

Safety invariant: if you can open a position, you can close it.
Validation runs ONCE at startup, not every loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# EXCEPTIONS
# =========================================================================

class BrokerConfigError(Exception):
    """Raised when broker capabilities configuration is invalid."""
    pass


class CapabilityUnavailableError(Exception):
    """Raised when a requested broker capability is not available."""
    def __init__(self, capability: str, group: str):
        self.capability = capability
        self.group = group
        super().__init__(
            f"Capability '{capability}' is not available in '{group}' "
            f"for this broker configuration."
        )


# =========================================================================
# CAPABILITIES
# =========================================================================

@dataclass
class BrokerCapabilities:
    """
    Broker capability flags (sensible defaults).

    Organized into groups:
      - opening_equity:  Order types for entering equity positions
      - closing_equity:  Order types for exiting equity positions
      - protective:      Stop/trailing/OCA orders for risk management
      - opening_options: Option strategies for opening positions
      - closing_options: Closing option positions
      - management_options: Rolling, adjusting options
      - query:           Account/position/order/data lookups
    """

    opening_equity: Dict[str, bool] = field(default_factory=lambda: {
        'market_order': True, 'limit_order': True, 'adaptive_algo': True,
        'midprice_peg': True, 'relative_peg': True, 'vwap_algo': True,
        'twap_algo': True, 'iceberg': True, 'snap_mid': True,
        'moo': True, 'loo': True, 'fok': True, 'ioc': True,
        'bracket_order': True,
    })

    closing_equity: Dict[str, bool] = field(default_factory=lambda: {
        'market_order': True, 'limit_order': True, 'moc': True,
        'loc': True, 'adaptive_algo': True, 'flatten_all': True,
        'flatten_limits': True,
    })

    protective: Dict[str, bool] = field(default_factory=lambda: {
        'stop_order': True, 'stop_limit': True, 'trailing_stop': True,
        'trailing_stop_limit': True, 'oca_pair': True, 'gtd_order': True,
    })

    opening_options: Dict[str, bool] = field(default_factory=lambda: {
        'single_leg': True, 'vertical_spread': True, 'iron_condor': True,
        'iron_butterfly': True, 'straddle': True, 'strangle': True,
        'butterfly': True, 'calendar_spread': True, 'diagonal_spread': True,
        'covered_call': True, 'cash_secured_put': True, 'protective_put': True,
        'collar': True, 'ratio_spread': True, 'jade_lizard': True,
    })

    closing_options: Dict[str, bool] = field(default_factory=lambda: {
        'close_option': True,
    })

    management_options: Dict[str, bool] = field(default_factory=lambda: {
        'roll_option': True,
    })

    query: Dict[str, bool] = field(default_factory=lambda: {
        'positions': True, 'account_summary': True, 'open_orders': True,
        'executions': True, 'realtime_data': True, 'option_chain': True,
        'option_greeks': True, 'position_greeks': True,
    })

    def is_capable(self, group: str, capability: str) -> bool:
        """Check if a specific capability is enabled."""
        group_dict = getattr(self, group, None)
        if group_dict is None:
            return False
        return group_dict.get(capability, False)

    def validate(self) -> None:
        """
        Validate capability safety invariants at startup.

        Runs ONCE when the gateway is created.  Raises BrokerConfigError
        listing every violated invariant so the operator can fix them all
        in one pass.
        """
        violations: List[str] = []

        # 1. If ANY opening equity is true → must be able to close with a market order
        if any(self.opening_equity.values()):
            if not self.closing_equity.get('market_order', False):
                violations.append(
                    "opening_equity has enabled capabilities but closing_equity.market_order "
                    "is false. You must be able to close equity positions you open."
                )

        # 2. If ANY opening options is true → must be able to close options
        if any(self.opening_options.values()):
            if not self.closing_options.get('close_option', False):
                violations.append(
                    "opening_options has enabled capabilities but closing_options.close_option "
                    "is false. You must be able to close option positions you open."
                )
            if not self.query.get('option_chain', False):
                violations.append(
                    "opening_options has enabled capabilities but query.option_chain "
                    "is false. Cannot trade options without chain lookup."
                )

        # 3. Bracket orders require stop orders
        if self.opening_equity.get('bracket_order', False):
            if not self.protective.get('stop_order', False):
                violations.append(
                    "opening_equity.bracket_order is true but protective.stop_order "
                    "is false. Bracket orders require stop order capability."
                )

        # 4. Protective orders need open_orders query to manage them
        if any(self.protective.values()):
            if not self.query.get('open_orders', False):
                violations.append(
                    "protective has enabled capabilities but query.open_orders "
                    "is false. Need to see open orders to manage protective orders."
                )

        # 5. Rolling options needs both close + open
        if self.management_options.get('roll_option', False):
            if not self.closing_options.get('close_option', False):
                violations.append(
                    "management_options.roll_option is true but "
                    "closing_options.close_option is false."
                )
            if not any(self.opening_options.values()):
                violations.append(
                    "management_options.roll_option is true but no "
                    "opening_options capabilities are enabled."
                )

        if violations:
            msg = "Broker capability validation failed:\n" + "\n".join(
                f"  - {v}" for v in violations
            )
            raise BrokerConfigError(msg)

    @classmethod
    def from_config(cls, config: dict) -> 'BrokerCapabilities':
        """Load capabilities from the full parsed YAML config dict."""
        caps = config.get('broker', {}).get('capabilities', {})
        if not caps:
            return cls()  # all defaults (everything enabled)

        instance = cls()
        for group_name in (
            'opening_equity', 'closing_equity', 'protective',
            'opening_options', 'closing_options', 'management_options', 'query',
        ):
            if group_name in caps:
                group_dict = getattr(instance, group_name)
                for key, value in caps[group_name].items():
                    if key in group_dict:
                        group_dict[key] = bool(value)
        return instance


# =========================================================================
# CAPABILITY MAP — broker method name → allowed capability group(s)
# =========================================================================
# If a method appears here, at least ONE of its listed capabilities must be
# enabled.  Methods NOT in this map are always allowed (cancel, management, etc).

_TOOL_CAPABILITY_MAP: Dict[str, List[tuple]] = {
    # --- Equity orders (can be used for opening OR closing) ---
    'place_market_order':       [('opening_equity', 'market_order'), ('closing_equity', 'market_order')],
    'place_limit_order':        [('opening_equity', 'limit_order'), ('closing_equity', 'limit_order')],
    'place_adaptive':           [('opening_equity', 'adaptive_algo'), ('closing_equity', 'adaptive_algo')],
    'place_midprice':           [('opening_equity', 'midprice_peg')],
    'place_relative':           [('opening_equity', 'relative_peg')],
    'place_vwap':               [('opening_equity', 'vwap_algo')],
    'place_twap':               [('opening_equity', 'twap_algo')],
    'place_iceberg_order':      [('opening_equity', 'iceberg')],
    'place_snap_to_midpoint':   [('opening_equity', 'snap_mid')],
    'place_market_on_open':     [('opening_equity', 'moo')],
    'place_limit_on_open':      [('opening_equity', 'loo')],
    'place_fill_or_kill':       [('opening_equity', 'fok')],
    'place_immediate_or_cancel':[('opening_equity', 'ioc')],
    'place_bracket_order':      [('opening_equity', 'bracket_order')],

    # --- Closing-only equity ---
    'place_market_on_close':    [('closing_equity', 'moc')],
    'place_limit_on_close':     [('closing_equity', 'loc')],
    'flatten_all':              [('closing_equity', 'flatten_all')],
    'flatten_limits':           [('closing_equity', 'flatten_limits')],

    # --- Protective ---
    'place_stop_order':         [('protective', 'stop_order')],
    'place_stop_limit':         [('protective', 'stop_limit')],
    'place_trailing_stop':      [('protective', 'trailing_stop')],
    'place_trailing_stop_limit':[('protective', 'trailing_stop_limit')],
    'place_oca':                [('protective', 'oca_pair')],
    'place_limit_order_gtd':    [('protective', 'gtd_order')],
    'modify_stop_price':        [('protective', 'stop_order')],

    # --- Options opening ---
    'buy_option':               [('opening_options', 'single_leg')],
    'place_vertical_spread':    [('opening_options', 'vertical_spread')],
    'place_iron_condor':        [('opening_options', 'iron_condor')],
    'place_iron_butterfly':     [('opening_options', 'iron_butterfly')],
    'place_straddle':           [('opening_options', 'straddle')],
    'place_strangle':           [('opening_options', 'strangle')],
    'place_butterfly':          [('opening_options', 'butterfly')],
    'place_calendar_spread':    [('opening_options', 'calendar_spread')],
    'place_diagonal_spread':    [('opening_options', 'diagonal_spread')],
    'place_covered_call':       [('opening_options', 'covered_call')],
    'sell_cash_secured_put':    [('opening_options', 'cash_secured_put')],
    'place_protective_put':     [('opening_options', 'protective_put')],
    'place_collar':             [('opening_options', 'collar')],
    'place_ratio_spread':       [('opening_options', 'ratio_spread')],
    'place_jade_lizard':        [('opening_options', 'jade_lizard')],

    # --- Options closing ---
    'close_option_position':    [('closing_options', 'close_option')],

    # --- Options management ---
    'roll_option_position':     [('management_options', 'roll_option')],

    # --- Queries ---
    'get_positions':            [('query', 'positions')],
    'get_position':             [('query', 'positions')],
    'get_stock_positions':      [('query', 'positions')],
    'get_options_positions':    [('query', 'positions')],
    'get_account_summary':      [('query', 'account_summary')],
    'get_available_funds':      [('query', 'account_summary')],
    'get_open_orders':          [('query', 'open_orders')],
    'get_recent_executions':    [('query', 'executions')],
    'get_completed_trades':     [('query', 'executions')],
    'get_option_chain':         [('query', 'option_chain')],
    'qualify_option_contract':  [('query', 'option_chain')],
    'get_option_greeks':        [('query', 'option_greeks')],
    'get_position_greeks':      [('query', 'position_greeks')],
}


def check_capability(capabilities: BrokerCapabilities, method_name: str) -> None:
    """
    Check if a broker method is allowed by current capabilities.

    Methods NOT in _TOOL_CAPABILITY_MAP are always allowed
    (e.g., cancel_order, cancel_stops — management operations).

    Raises CapabilityUnavailableError if the method is mapped and
    NONE of its required capabilities are enabled.
    """
    mappings = _TOOL_CAPABILITY_MAP.get(method_name)
    if not mappings:
        return  # Not in map → unrestricted

    for group, capability in mappings:
        if capabilities.is_capable(group, capability):
            return  # At least one capability is enabled

    # All mapped capabilities are disabled
    group, capability = mappings[0]
    raise CapabilityUnavailableError(capability, group)


# =========================================================================
# GATEWAY FACTORY
# =========================================================================

async def create_gateway(config: dict):
    """
    Create, validate, connect, and return a broker gateway.

    Reads broker.adapter from config to choose the adapter (default: 'ibkr').
    Validates capabilities at startup — raises BrokerConfigError if
    safety invariants are violated.
    """
    adapter_name = config.get('broker', {}).get('adapter', 'ibkr')
    capabilities = BrokerCapabilities.from_config(config)

    # Safety validation — runs ONCE, not every loop
    capabilities.validate()
    logger.info("Broker capabilities validated OK")

    if adapter_name == 'ibkr':
        from execution.ibkr_gateway import IBKRGateway
        gateway = IBKRGateway(capabilities, config)
        connected = await gateway.connect()
        if not connected:
            raise ConnectionError("Failed to connect to IBKR broker")
        logger.info(f"Connected to broker via {adapter_name} gateway: {gateway.account_id}")
        return gateway
    else:
        raise BrokerConfigError(f"Unknown broker adapter: '{adapter_name}'")
