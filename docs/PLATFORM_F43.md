# F43 — C-MATH-AI Platform 1.0 (Release Candidate)

## Purpose and current boundary

F43 composes implemented modules into a repeatable, observable **research-to-paper vertical slice**. The first release candidate is intentionally conservative. It validates market data, computes point-in-time features, reports the actual algorithm catalog by lifecycle, runs the F39 backtest, optionally runs a causal F40 target-persistence baseline, runs F41 robustness checks, and can evaluate F38 plus submit orders to the F42 paper simulator.

The strategy's timestamped target positions are explicit caller inputs. F43 does **not** invent trading signals or silently train/use every algorithm in the registry. F35/F36/F37 outputs and a covariance/forecasting universe must be provided by the upstream strategy workflow; when a serialized F38 request is supplied, F43 invokes the real F38 governor. Catalog metadata in `experimental` lifecycle is not an implemented algorithm and is not evidence of investment performance.

No real order, broker/exchange adapter, credential retrieval, or live execution path exists. There is no profitability claim.

## Included components

- `apps/platform/contracts.py`: bounded, versioned API payload contracts; strict timestamp, OHLCV, target-time and cross-stage input checks.
- `apps/platform/service.py`: orchestration, data-quality gate, F32 feature transforms, F33 registry inventory, F39 backtest, optional F40 OOS baseline, F41 bootstrap/adversarial analysis, F38 risk evaluation and F42 paper execution.
- `apps/platform/store.py`: SQLite job store, idempotent job IDs, explicit state transitions and per-job SHA-256 hash-chained audit events.
- `apps/platform/api.py`: dependency-light WSGI routes for health/readiness, algorithm discovery, synchronous jobs, job retrieval, audit events and Prometheus-style job counters.
- `apps/platform/cli.py`: operator CLI.
- `.github/workflows/f43-validation.yml`: Python 3.11/3.12 tests, Ruff lint/format and mypy.
- Integration and unit tests that check reproducibility, job idempotency, audit-chain integrity, routes, the data-quality gate and the integrated pipeline.

## Install and operate

Install development dependencies and the local package:

```bash
python -m pip install -e ".[dev]"
c-math-ai health
c-math-ai algorithms --lifecycle implemented --limit 50
c-math-ai algorithms --lifecycle experimental --limit 20
c-math-ai run --request examples/platform-run.json
c-math-ai job <job-id>
c-math-ai audit <job-id> --show
c-math-ai serve --host 127.0.0.1 --port 8080
```

Set `CMATH_PLATFORM_DB` or pass `--db` to select a persistent SQLite file. The default is `~/.local/state/c-math-ai/platform.sqlite3`. Use a writable directory with restricted permissions and include the database in the operator's backup plan.

### Minimal request shape

```json
{
  "job_id": "research-smoke-0001",
  "dataset_id": "sample-bars-v1",
  "symbol": "XYZ",
  "as_of": "2026-01-12T14:30:00Z",
  "bars": [
    {"timestamp": "2026-01-09T14:30:00Z", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 100000.0},
    {"timestamp": "2026-01-12T14:30:00Z", "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 105000.0}
  ],
  "targets": [
    {"timestamp": "2026-01-09T14:30:00Z", "target_units": 1}
  ],
  "backtest": {"initial_capital": 100000.0, "commission_bps": 1.0, "slippage_bps": 2.0, "periods_per_year": 252},
  "monte_carlo": {"simulations": 100, "seed": 42, "confidence_level": 0.95, "method": "moving_block", "block_size": 5},
  "walk_forward": {"enabled": true}
}
```

The F39 backtest is intentionally a single OHLC stream. For paper order execution, the request must additionally include an actual `risk_request` compliant with F38, `paper_orders`, and timestamped `paper_quotes`. F43 invokes `evaluate_risk_governor` itself. A BLOCK, stale evidence, missing approved position or order size above the approved quantity cannot be overridden by the orchestration layer. The paper engine starts a new in-memory account for each job; the SQLite job store persists job/audit metadata, not an ongoing brokerage account or position ledger.

## HTTP endpoints

The WSGI app supports:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | release, execution mode, registry lifecycle counts and job counts |
| GET | `/ready` | readiness view |
| GET | `/algorithms?lifecycle=implemented&limit=50` | algorithm metadata discovery |
| POST | `/jobs` | validate and execute a synchronous platform request |
| GET | `/jobs/{job_id}` | persisted job state/result |
| GET | `/jobs/{job_id}/audit` | per-job event chain |
| GET | `/jobs/{job_id}/audit/verify` | verify event sequence and SHA-256 links |
| GET | `/metrics` | job-state counters in Prometheus text format |

Request bodies are limited to 8 MiB. Job IDs are idempotency keys: submitting the same ID and same request fingerprint returns the stored record; reusing an ID for different input returns a conflict. Current job execution is synchronous in the WSGI request process; SQLite does not make a background worker/queue appear automatically.

## Data and reproducibility controls

1. Input bars and target timestamps must be timezone-aware; bars are strictly ordered, OHLCV validated and capped at 50,000 rows.
2. The data-quality engine checks required fields, duplicate symbol/timestamp keys, numeric quality, chronology, and the supplied `as_of`. An INVALID report or look-ahead flag fails the job before features/backtesting.
3. Features are deterministic trailing close returns and five-bar return volatility. The report records input/output fingerprints, feature definitions, warm-up missing rows and point-in-time metadata. Warm-up NaN values are not silently backfilled.
4. Registry output records the actual catalog fingerprint and lifecycle counts; catalog metadata does not imply code execution.
5. F39/F41 outputs and the feature lineage are returned with request fingerprint and a deterministic Monte Carlo seed. Wall-clock audit timestamps are operational metadata and are not expected to be identical across independent runs.
6. The F40 output in this release is specifically a last-known-target persistence baseline, labeled as such. It is not reported as the caller's strategy OOS performance.
7. A per-job SQLite hash chain records job creation, stage summaries, transitions and final result fingerprints. It detects local changes to recorded rows but is not a digital signature or protection against an administrator rewriting the database and recomputing every hash.

## Operational and safety limits

- The built-in WSGI server has no authentication or user/tenant authorization. Keep it on loopback or behind an authenticated, access-controlled reverse proxy; it is a development/operator API, not an internet-facing production deployment.
- Jobs execute synchronously. Large computations can occupy a request worker; a distributed queue, worker supervision, concurrency limits, cancellation, retries, and back-pressure are not included.
- The job store retains request JSON and summaries. Configure filesystem permissions, encryption/backup policy and retention for any non-public market/portfolio data.
- SQLite supports local durability but has no multi-region replication or centralized secret management. No credentials are accepted by these API contracts.
- The F42 paper ledger is new and in memory per job, not a durable continuous account. Only full fills and long-only single-currency exposure are simulated.
- No live broker adapter, production deployment manifests, dashboard, or automated rollback controller is introduced in this release candidate. Rollback means redeploying the previous versioned commit; validate schema compatibility and preserve/back up the SQLite store before doing so.
- Historical and synthetic tests validate software behavior, not strategy validity, causal efficacy or future profit.

## Release closure criteria

F43 is not considered complete until the F43 workflow, general CI, Security, relevant F38/F39/F40/F41/F42 regressions, integration smoke tests and audit-chain checks pass on the final PR head. This document describes scope, not an assertion that unobserved checks passed.
