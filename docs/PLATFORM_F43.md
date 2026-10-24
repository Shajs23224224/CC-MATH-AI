# F43 — C-MATH-AI Platform 1.0

## Objective

Provide a deterministic application-level readiness contract that consolidates observed health evidence from the existing research, portfolio, risk, backtest, paper-execution and audit layers. F43 is an integration boundary, not a claim that every upstream subsystem has been automatically probed or that the platform has investment alpha.

## Deliverables

- Immutable, typed ComponentCheck records with evidence identifiers and criticality.
- build_readiness(): stable aggregation with required-component coverage, duplicate rejection, fail-closed critical gates and a SHA-256 fingerprint of canonical JSON.
- Immutable PlatformReadiness snapshot exposing status, missing components, failed critical components, fingerprint and execution capability flags.
- Unit tests for healthy, missing, failed-critical, optional-degraded, duplicate, determinism and invalid-identifier cases.

## Required observed checks

Data quality; quant research; model evaluation; decision engine; portfolio; Risk Governor; backtest; paper execution; audit integrity.

A required check is considered present only when the caller submits a ComponentCheck; configuration alone is not evidence of success. A present critical check marked passed=False blocks readiness. Missing required checks also block readiness. A failed optional check yields DEGRADED, which is not paper-trading-ready.

## Status and safety semantics

- READY: all required checks pass and no optional check failed.
- DEGRADED: required checks pass but at least one optional check fails.
- BLOCKED: a required component is missing or a critical check fails.

Only READY reports can_paper_trade=True. F43 always reports can_live_trade=False and execution_mode=PAPER_ONLY. The readiness object does not place orders, instantiate broker adapters, or bypass the F38 Risk Governor / F42 order contract.

## Fingerprint and limitations

The fingerprint is SHA-256 over canonical JSON containing sorted check records, status, missing components, critical failures and execution mode. It supports reproducible comparison, not authenticity or non-repudiation; there is no signature or durable audit store in this phase. Callers are responsible for truthful evidence IDs and actual health probes. This module does not automatically call all subsystems or provide a dashboard/API.

## Validation

Run pytest tests/unit/test_platform_readiness.py -q, Ruff lint/format, and mypy on apps/platform/readiness.py. The dedicated workflow runs tests and static checks on Python 3.11 and 3.12. Synthetic tests verify code behavior, not future returns or trading performance.
