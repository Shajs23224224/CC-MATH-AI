"""Bootstrap architecture tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_required_architecture_files_exist() -> None:
    required = [
        "SYSTEM_SPEC.md",
        "README.md",
        "docs/ARCHITECTURE.md",
        "core",
        "data",
        "quant",
        "ml",
        "signals",
        "portfolio",
        "risk",
        "backtest",
        "execution",
    ]
    for relative_path in required:
        assert (ROOT / relative_path).exists(), relative_path
