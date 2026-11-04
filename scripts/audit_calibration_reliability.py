"""Command-line interface for the independent D07.3 calibration audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from execution.research.d07_calibration import audit_calibration_reliability


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently audit D06.3 temperature calibration and reliability."
    )
    parser.add_argument("--policy-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/execution_policy_calibration_audits"),
    )
    args = parser.parse_args()
    try:
        summary = audit_calibration_reliability(
            args.policy_run,
            args.dataset,
            output_root=args.output_root,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(
            f"D07.3 CALIBRATION AUDIT FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
