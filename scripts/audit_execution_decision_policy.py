"""Command-line entrypoint for the D07.1 D06/D04 integrity audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from execution.research.d07_audit import audit_decision_policy_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit D06 execution-policy predictions against the verified D04 dataset."
    )
    parser.add_argument("--policy-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/execution_policy_audits"),
    )
    args = parser.parse_args()
    try:
        summary = audit_decision_policy_run(
            args.policy_run,
            args.dataset,
            output_root=args.output_root,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(
            f"D07.1 AUDIT FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
