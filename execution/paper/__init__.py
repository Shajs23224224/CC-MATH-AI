"""Public API for F42 paper trading; no live execution adapters are exposed."""

from execution.paper.contracts import (
    AuditEvent,
    ExecutionQuote,
    PaperAccountSnapshot,
    PaperFill,
    PaperOrder,
    PaperOrderRequest,
    PaperPosition,
)
from execution.paper.engine import PaperTradingEngine
from execution.policies.paper import PaperExecutionPolicy

__all__ = [
    "AuditEvent",
    "ExecutionQuote",
    "PaperAccountSnapshot",
    "PaperExecutionPolicy",
    "PaperFill",
    "PaperOrder",
    "PaperOrderRequest",
    "PaperPosition",
    "PaperTradingEngine",
]
