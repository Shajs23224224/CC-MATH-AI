"""Paper trading and future broker/exchange adapters.

This package exposes only the paper simulator. No live broker adapter is implemented.
"""

from execution.paper import (
    ExecutionQuote,
    PaperExecutionPolicy,
    PaperTradingEngine,
    PaperOrderRequest,
)

__all__ = [
    "ExecutionQuote",
    "PaperExecutionPolicy",
    "PaperOrderRequest",
    "PaperTradingEngine",
]
