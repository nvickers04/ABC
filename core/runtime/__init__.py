"""Runtime modules — orchestration, state assembly, safety, scheduling.

This package isolates concerns previously mixed inside ``core.agent``:

* ``interfaces``       — Protocol-style adapter contracts (broker / market hours / cost).
* ``state_context``    — :class:`StateContextBuilder` (extracted from agent).
* ``safety``           — :class:`SafetyController` (loss / drawdown / cost gates).
* ``operating_context`` — :class:`OperatingContext` + :class:`QualityMatrix` (PR2 orchestration driver: risk, model config, tool policy, provenance).

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
from core.runtime.safety import SafetyController, SafetyVerdict
from core.runtime.state_context import StateContextBuilder
from core.runtime.operating_context import (
    ContextQuality,
    OperatingContext,
    QualityMatrix,
    get_operating_context,
)

# Note: ToolUsageSnapshot removed (PR1 unification: superseded by ToolUsageRecord +
# DecisionProvenanceSnapshot in core.quality.quality_matrix). Old references should
# migrate to the canonical provenance types.

__all__ = [
    "AccountSummary",
    "BrokerGatewayProtocol",
    "ContextQuality",
    "CostTrackerProtocol",
    "MarketHoursProtocol",
    "OperatingContext",
    "QualityMatrix",
    "SafetyController",
    "SafetyVerdict",
    "StateContextBuilder",
    "WakeBusProtocol",
    "get_operating_context",
]
