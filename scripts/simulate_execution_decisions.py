"""Command-line entrypoint for D07.2 execution decision simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from execution.research.d07_simulation import simulate_execution_decisions


def _parse_cost_scenarios(value: str) -> tuple[float, ...]:
    try:
        scenarios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cost scenarios must be comma-separated numbers"
        ) from exc
    if not scenarios:
        raise argparse.ArgumentTypeError("at least one cost scenario is required")
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate D07.1 audited decisions across transaction-cost scenarios."
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/execution_simulations"),
    )
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--notional-per-trade", type=float, default=10000.0)
    parser.add_argument(
        "--cost-scenarios-bps-per-side",
        type=_parse_cost_scenarios,
        default=(0.0, 5.0, 10.0, 20.0),
        help="comma-separated all-in bps per side; default: 0,5,10,20",
    )
    parser.add_argument("--trading-days-per-year", type=int, default=252)
    args = parser.parse_args()
    try:
        summary = simulate_execution_decisions(
            args.audit_dir,
            output_root=args.output_root,
            initial_capital=args.initial_capital,
            notional_per_trade=args.notional_per_trade,
            cost_scenarios_bps_per_side=args.cost_scenarios_bps_per_side,
            trading_days_per_year=args.trading_days_per_year,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(
            f"D07.2 SIMULATION FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
