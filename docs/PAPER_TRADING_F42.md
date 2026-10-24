# F42 — Paper Trading + Execution

## Scope and safety boundary

F42 adds a deterministic **in-memory paper simulator**. It does not send live orders, connect to brokers/exchanges, load credentials, or provide a route from an LLM to an execution venue. The public API only exposes the paper engine.

The order path requires a typed F38 `RiskGovernorResult`. A `BLOCK` decision, missing symbol, blocked/zero approved position, stale decision, unsupported order type, quantity above the approved target, insufficient holdings, or insufficient cash prevents a fill. There is no configuration switch that converts a risk veto into an approval.

## Contracts

- `PaperOrderRequest`: normalized symbol, BUY/SELL side, positive quantity, MARKET/LIMIT type, strategy/signal IDs, idempotency key and timezone-aware submission timestamp.
- `PaperOrder`: explicit ACCEPTED/FILLED/CANCELLED/REJECTED state, source F38 action/fingerprint, approved quantity and rejection reason.
- `ExecutionQuote`: timezone-aware bid/ask quote with non-crossed market validation.
- `PaperFill`: reference and fill prices, quantity, notional, commission, slippage and timestamp.
- `PaperAccountSnapshot`: cash, marked positions, market value, equity, realized/unrealized PnL and cumulative commissions.
- `AuditEvent`: append-only event sequence and SHA-256 hash chain.

## Lifecycle and matching

1. Submit an idempotent order with the actual F38 decision. A repeated key plus identical payload returns the original order; reusing the key with a different payload raises an error.
2. Accepted orders wait for quotes for the same normalized symbol.
3. MARKET orders fill at ask for buys / bid for sells plus configurable adverse slippage. LIMIT buys require ask <= limit; LIMIT sells require bid >= limit. Limit fills respect the limit and do not add slippage beyond it.
4. Commission is charged on fill notional. The simulator uses full fills only: it does not model partial fills, queue position or market impact.
5. BUY orders require sufficient cash and cannot exceed the F38 approved target after held and pending buy quantity. SELL orders cannot exceed unreserved paper holdings.
6. A cancellation is allowed only before terminal state. Quote and account event times must be timezone-aware, fresh and monotonic.

## Accounting

The simulator is long-only and single-currency. Buy cash flows include notional plus commission; sell proceeds subtract commission. Buy-side commission is included in the weighted average cost basis; sell-side commission is deducted from realized PnL. Equity is cash plus the latest marked position values. Quotes mark positions at the bid/ask midpoint; that midpoint is a bookkeeping mark, not a guaranteed liquidation price.

## Reproducibility and audit

Order IDs derive from account ID and idempotency key. Fill IDs derive from the order ID. Audit events serialize canonical JSON and hash the sequence, type, timestamp, entity, payload and previous hash. `verify_audit_chain()` detects changed event content, missing events, sequence changes and broken links. The ledger is in memory; persistence, crash recovery, event-store replication and cryptographic signing are out of scope.

## Validation

The dedicated F42 GitHub Actions workflow runs behavior tests, Ruff lint, Ruff format and mypy on Python 3.11 and 3.12. Passing synthetic tests verifies implementation behavior, not investment performance or real-world execution quality.

## Explicit limitations

- No live broker/exchange adapters or credential handling.
- No derivatives, shorts, FX conversion, taxes, financing, corporate actions or borrow fees.
- No stop orders, partial fills, market impact, queue priority or venue-specific rules.
- No durable storage or restart recovery; restart loses the in-memory state.
- F38 approved target quantity is enforced as a hard ceiling; this initial simulator does not implement risk-veto liquidation workflows.
- Quote age and F38 decision age limits are configurable policy assumptions, not guarantees of market-data quality.
