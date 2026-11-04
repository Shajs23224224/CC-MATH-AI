"""D07.2 end-to-end cost, accounting and failure-path tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from execution.research.d07_audit import audit_decision_policy_run
from execution.research.d07_simulation import (
    _simulate_account,
    simulate_execution_decisions,
)
from tests.unit.d07_fixture_factory import (
    _manifest_entry,
    _write_json,
    build_d04_d06_artifacts,
)


def _build_audit(tmp_path):
    artifacts = build_d04_d06_artifacts(tmp_path)
    audit = audit_decision_policy_run(
        artifacts["policy_dir"],
        artifacts["dataset_path"],
        output_root=tmp_path / "audits",
    )
    return artifacts, audit, tmp_path / "audits" / audit["output_directory"].split("/")[-1]


def test_simulation_compares_policies_across_cost_scenarios(tmp_path):
    _, _, audit_dir = _build_audit(tmp_path)
    result = simulate_execution_decisions(
        audit_dir,
        output_root=tmp_path / "simulations",
        initial_capital=100000.0,
        notional_per_trade=10000.0,
        cost_scenarios_bps_per_side=(0.0, 5.0, 10.0, 20.0),
    )

    assert result["status"] == "COMPLETED"
    assert result["simulation_type"] == "D07.2_cost_sensitivity_one_session"
    assert result["accounts_simulated"] == 8
    assert result["sessions_per_algorithm"] == 6
    assert result["cost_scenarios_bps_per_side"] == [0.0, 5.0, 10.0, 20.0]

    output_dir = tmp_path / "simulations" / result["simulation_id"]
    metrics = pd.read_csv(output_dir / "strategy_metrics.csv")
    sensitivity = pd.read_csv(output_dir / "cost_sensitivity.csv")
    ledger = pd.read_csv(output_dir / "trade_ledger.csv")
    assert len(metrics) == 8
    assert len(sensitivity) == 8
    assert len(ledger) == 48

    argmax = metrics.loc[
        (metrics["strategy"] == "argmax")
        & (metrics["cost_bps_per_side"] == 5.0)
    ].iloc[0]
    policy = metrics.loc[
        (metrics["strategy"] == "neutral_policy")
        & (metrics["cost_bps_per_side"] == 5.0)
    ].iloc[0]
    assert argmax["trades"] == 4
    assert argmax["total_gross_pnl"] == pytest.approx(650.0)
    assert argmax["total_transaction_costs"] == pytest.approx(40.0)
    assert argmax["total_net_pnl"] == pytest.approx(610.0)
    assert argmax["final_equity"] == pytest.approx(100610.0)
    assert argmax["break_even_cost_bps_per_side"] == pytest.approx(81.25)

    assert policy["trades"] == 3
    assert policy["neutral_signals"] == 3
    assert policy["total_gross_pnl"] == pytest.approx(450.0)
    assert policy["total_transaction_costs"] == pytest.approx(30.0)
    assert policy["total_net_pnl"] == pytest.approx(420.0)

    neutral = ledger.loc[
        (ledger["strategy"] == "neutral_policy")
        & (ledger["cost_bps_per_side"] == 5.0)
        & (ledger["signal"] == "NEUTRAL")
    ]
    assert len(neutral) == 3
    assert (neutral["transaction_costs"] == 0.0).all()
    assert (neutral["executed_direction"] == 0).all()

    manifest = json.loads(
        (output_dir / "simulation_manifest.json").read_text(encoding="utf-8")
    )
    for filename, entry in manifest["outputs"].items():
        path = output_dir / filename
        assert path.is_file()
        assert entry["bytes"] == path.stat().st_size
        assert len(entry["sha256"]) == 64


def test_simulation_cost_increase_cannot_improve_net_pnl(tmp_path):
    _, _, audit_dir = _build_audit(tmp_path)
    result = simulate_execution_decisions(
        audit_dir,
        output_root=tmp_path / "simulations",
        cost_scenarios_bps_per_side=(0.0, 5.0, 10.0, 20.0),
    )
    output_dir = tmp_path / "simulations" / result["simulation_id"]
    metrics = pd.read_csv(output_dir / "strategy_metrics.csv")
    for _, group in metrics.groupby(["algorithm", "strategy"]):
        ordered = group.sort_values("cost_bps_per_side")
        net = ordered["total_net_pnl"].to_numpy(dtype=float)
        assert (net[1:] <= net[:-1] + 1e-10).all()


def test_simulation_rejects_tampered_d07_1_artifact(tmp_path):
    _, _, audit_dir = _build_audit(tmp_path)
    with (audit_dir / "audited_predictions.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(
        ValueError,
        match="byte-count verification failed|SHA-256 verification failed",
    ):
        simulate_execution_decisions(
            audit_dir,
            output_root=tmp_path / "simulations",
        )


def test_simulation_rejects_failed_audit_checks(tmp_path):
    _, _, audit_dir = _build_audit(tmp_path)
    summary_path = audit_dir / "audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checks"][0]["passed"] = False
    _write_json(summary_path, summary)
    manifest_path = audit_dir / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["audit_summary.json"] = _manifest_entry(summary_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="failed or incomplete integrity checks"):
        simulate_execution_decisions(
            audit_dir,
            output_root=tmp_path / "simulations",
        )


def test_simulation_stops_opening_positions_after_bankruptcy():
    rows = pd.DataFrame(
        {
            "fold": [1, 1],
            "row_index": [0, 1],
            "feature_date": ["2024-01-01", "2024-01-02"],
            "entry_date": ["2024-01-02", "2024-01-03"],
            "actual_execution_return": [1.5, -0.02],
            "entry_open": [100.0, 100.0],
            "exit_close": [250.0, 98.0],
            "argmax_prediction": ["DOWN", "UP"],
            "policy_prediction": ["DOWN", "UP"],
            "selected_neutral_threshold": [0.4, 0.4],
        }
    )
    ledger = _simulate_account(
        rows,
        algorithm="test_classifier",
        strategy="argmax",
        prediction_column="argmax_prediction",
        initial_capital=1000.0,
        notional_per_trade=1000.0,
        cost_bps_per_side=5.0,
    )

    assert ledger.loc[0, "execution_status"] == "PORTFOLIO_BROKE"
    assert ledger.loc[0, "portfolio_equity_after"] < 0.0
    assert ledger.loc[1, "execution_status"] == "SKIPPED_BANKRUPTCY"
    assert ledger.loc[1, "active_notional"] == 0.0
    assert ledger.loc[1, "transaction_costs"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_capital": 0.0}, "initial_capital must be positive"),
        ({"notional_per_trade": 200000.0}, "notional_per_trade"),
        ({"cost_scenarios_bps_per_side": ()}, "at least one transaction-cost scenario"),
        ({"cost_scenarios_bps_per_side": (5.0, 5.0)}, "must not contain duplicates"),
        ({"cost_scenarios_bps_per_side": (-1.0,)}, "between 0 and 1000"),
        ({"trading_days_per_year": 0}, "trading_days_per_year"),
    ],
)
def test_simulation_rejects_invalid_configuration(tmp_path, kwargs, message):
    _, _, audit_dir = _build_audit(tmp_path)
    with pytest.raises(ValueError, match=message):
        simulate_execution_decisions(
            audit_dir,
            output_root=tmp_path / "simulations",
            **kwargs,
        )
