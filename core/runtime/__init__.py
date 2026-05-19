"""Runtime modules — orchestration, state assembly, safety, scheduling.

This package isolates concerns previously mixed inside ``core.agent``:

* ``interfaces``       — Protocol-style adapter contracts (broker / market hours / cost).
* ``state_context``    — :class:`StateContextBuilder` (extracted from agent).
* ``safety``           — :class:`SafetyController` (loss / drawdown / cost gates).
* ``operating_context`` — :class:`OperatingContext` (researcher mode, context quality).
* ``working_memory_access`` — routes WM to Postgres or local JSON fallback.
* ``local_memory_fallback`` — JSON store used when Postgres is down.

Future additions (per stabilization plan): ``coordinator``, ``scheduler``.

NOTE: Behavior parity with the original ``core.agent`` implementation is enforced
by ``tests/test_runtime_characterization.py``. Do not change semantics in this
package without updating that test suite.
"""

from core.runtime.interfaces import (
    AccountSummary,
    BrokerGatewayProtocol,
    CostTrackerProtocol,
    MarketHoursProtocol,
    WakeBusProtocol,
)
from core.runtime.operating_context import (
    ContextQuality,
    MemorySource,
    OperatingContext,
    QualityMatrix,
    get_operating_context,
    reset_operating_context_for_tests,
)
from core.runtime.safety import SafetyController, SafetyVerdict
from core.runtime.state_context import StateContextBuilder

__all__ = [
    "AccountSummary",
    "BrokerGatewayProtocol",
    "ContextQuality",
    "CostTrackerProtocol",
    "MarketHoursProtocol",
    "MemorySource",
    "OperatingContext",
    "QualityMatrix",
    "SafetyController",
    "SafetyVerdict",
    "StateContextBuilder",
    "WakeBusProtocol",
    "get_operating_context",
    "reset_operating_context_for_tests",
]
