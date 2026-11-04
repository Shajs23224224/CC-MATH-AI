"""Tests for the D07.2 cost-aware execution decision simulator."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.simulate_execution_decisions import simulate_execution_decisions


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _refresh_predictions_hash(audit_dir):
    manifest_path = audit_dir / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions_path = audit_dir / "audited_predictions.csv"
    manifest["outputs"]["audited_predictions.csv"] = {
        "sha256": _sha256(predictions_path),
        "bytes": predictions_path.stat().st_size,
    }
    _write_json(manifest_path, manifest)


def create_audit_fixture(tmp_path):
    feature_dates = pd.bdate_range("2024-02-01", periods=4)
    entry_dates = feature_dates + pd.offsets.BDay(1)
    returns = np.asarray([0.02, -0.01, 0.005, -0.02], dtype=np.float64)
    entry_open = np.full(4, 100.0, dtype=np.float64)
    exit_close = entry_open * (1.0 + returns)

    predictions = pd.DataFrame(
        {
            "algorithm": ["test_classifier"] * 4,
            "fold": [1, 1, 2, 2],
            "row_index": [10, 11, 12, 13],
            "feature_date": feature_dates.strftime("%Y-%m-%d"),
            "entry_date": entry_dates.strftime("%Y-%m-%d"),
            "actual": ["UP", "DOWN", "UP", "DOWN"],
            "argmax_prediction": ["UP", "DOWN", "NEUTRAL", "UP"],
            "policy_prediction": ["UP", "NEUTRAL", "DOWN", "UP"],
            "selected_neutral_threshold": [0.4, 0.4, 0.45, 0.45],
            "actual_execution_return": returns,
            "entry_open": entry_open,
            "exit_close": exit_close,
        }
    )

    audit_dir = tmp_path / "test_d07_1_audit"
    audit_dir.mkdir()
    predictions_path = audit_dir / "audited_predictions.csv"
    predictions.to_csv(
        predictions_path,
        index=False,
        float_format="%.17g",
    )

    fold_audit = pd.DataFrame(
        {
            "algorithm": ["test_classifier"] * 2,
            "fold": [1, 2],
            "test_samples": [2, 2],
        }
    )
    fold_audit.to_csv(audit_dir / "fold_audit.csv", index=False)

    audit_summary = {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "audit_type": "D07.1_D06_prediction_integrity",
        "ticker": "TEST",
        "decision_policy_run_id": "test_policy_run",
        "execution_dataset_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "rows_audited": 4,
        "algorithm_count": 1,
        "algorithms": ["test_classifier"],
        "fold_count": 2,
        "checks_passed": 12,
        "checks_total": 12,
    }
    summary_path = audit_dir / "audit_summary.json"
    _write_json(summary_path, audit_summary)

    audit_manifest = {
        "schema_version": "1.0",
        "input": {
            "decision_policy_run_id": "test_policy_run",
            "execution_dataset_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
        },
        "outputs": {
            path.name: {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                predictions_path,
                audit_dir / "fold_audit.csv",
                summary_path,
            )
        },
    }
    _write_json(audit_dir / "audit_manifest.json", audit_manifest)
    return audit_dir


def test_simulator_compares_argmax_and_neutral_policy_with_round_trip_costs(
    tmp_path,
):
    audit_dir = create_audit_fixture(tmp_path)

    summary = simulate_execution_decisions(
        audit_dir,
        output_root=tmp_path / "simulations",
        initial_capital=100000.0,
        notional_per_trade=10000.0,
        cost_bps_per_side=5.0,
    )

    assert summary["status"] == "COMPLETED"
    assert summary["simulation_type"] == "D07.2_one_session_decision_simulation"
    assert summary["algorithm_count"] == 1
    assert summary["sessions_per_algorithm"] == 4
    assert summary["simulation_rows"] == 8
    assert summary["cost_model"]["round_trip_cost_bps"] == pytest.approx(10.0)

    output_dir = tmp_path / "simulations" / summary["simulation_id"]
    metrics = pd.read_csv(output_dir / "simulation_metrics.csv")
    decisions = pd.read_csv(output_dir / "simulated_decisions.csv")

    argmax = metrics.loc[metrics["strategy"] == "argmax"].iloc[0]
    policy = metrics.loc[metrics["strategy"] == "neutral_policy"].iloc[0]

    assert argmax["trades"] == 3
    assert argmax["long_trades"] == 2
    assert argmax["short_trades"] == 1
    assert argmax["no_trade_sessions"] == 1
    assert argmax["total_gross_pnl"] == pytest.approx(100.0)
    assert argmax["total_transaction_costs"] == pytest.approx(30.0)
    assert argmax["total_net_pnl"] == pytest.approx(70.0)
    assert argmax["final_equity"] == pytest.approx(100070.0)

    assert policy["trades"] == 3
    assert policy["long_trades"] == 2
    assert policy["short_trades"] == 1
    assert policy["no_trade_sessions"] == 1
    assert policy["total_gross_pnl"] == pytest.approx(-50.0)
    assert policy["total_transaction_costs"] == pytest.approx(30.0)
    assert policy["total_net_pnl"] == pytest.approx(-80.0)
    assert policy["final_equity"] == pytest.approx(99920.0)

    flat_rows = decisions.loc[decisions["execution_status"] == "NO_TRADE"]
    assert len(flat_rows) == 2
    assert (flat_rows["transaction_cost_pnl"] == 0.0).all()

    manifest = json.loads((output_dir / "simulation_manifest.json").read_text(encoding="utf-8"))
    for filename, entry in manifest["outputs"].items():
        path = output_dir / filename
        assert path.is_file()
        assert _sha256(path) == entry["sha256"]
        assert path.stat().st_size == entry["bytes"]


def test_simulator_rejects_tampered_d07_1_prediction_file(tmp_path):
    audit_dir = create_audit_fixture(tmp_path)
    predictions_path = audit_dir / "audited_predictions.csv"

    with predictions_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        simulate_execution_decisions(
            audit_dir,
            output_root=tmp_path / "simulations",
        )


def test_simulator_rejects_algorithms_with_different_fold_assignments(tmp_path):
    audit_dir = create_audit_fixture(tmp_path)
    predictions_path = audit_dir / "audited_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    second_algorithm = predictions.copy()
    second_algorithm["algorithm"] = "second_classifier"
    second_algorithm["fold"] = [1, 2, 2, 2]
    predictions = pd.concat([predictions, second_algorithm], ignore_index=True)
    predictions.to_csv(
        predictions_path,
        index=False,
        float_format="%.17g",
    )

    summary_path = audit_dir / "audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["algorithms"] = ["test_classifier", "second_classifier"]
    summary["algorithm_count"] = 2
    summary["rows_audited"] = len(predictions)
    _write_json(summary_path, summary)

    manifest_path = audit_dir / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in ("audited_predictions.csv", "audit_summary.json"):
        path = audit_dir / filename
        manifest["outputs"][filename] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="identical fold assignments"):
        simulate_execution_decisions(
            audit_dir,
            output_root=tmp_path / "simulations",
        )


def test_simulator_rejects_failed_d07_1_integrity_check(tmp_path):
    audit_dir = create_audit_fixture(tmp_path)
    summary_path = audit_dir / "audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checks"][0] = {
        "check": "tampered_check",
        "passed": False,
        "detail": "synthetic failure",
    }
    _write_json(summary_path, summary)

    manifest_path = audit_dir / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["audit_summary.json"] = {
        "sha256": _sha256(summary_path),
        "bytes": summary_path.stat().st_size,
    }
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="failed or incomplete checks"):
        simulate_execution_decisions(
            audit_dir,
            output_root=tmp_path / "simulations",
        )


def test_simulator_rejects_execution_return_price_mismatch(tmp_path):
    audit_dir = create_audit_fixture(tmp_path)
    predictions_path = audit_dir / "audited_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions.loc[0, "actual_execution_return"] = 0.5
    predictions.to_csv(
        predictions_path,
        index=False,
        float_format="%.17g",
    )
    _refresh_predictions_hash(audit_dir)

    with pytest.raises(ValueError, match="do not match exit_close"):
        simulate_execution_decisions(
            audit_dir,
            output_root=tmp_path / "simulations",
        )


def test_simulator_stops_opening_positions_after_bankruptcy(tmp_path):
    audit_dir = create_audit_fixture(tmp_path)
    predictions_path = audit_dir / "audited_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions.loc[0, "argmax_prediction"] = "DOWN"
    predictions.loc[0, "actual_execution_return"] = 1.5
    predictions.loc[0, "exit_close"] = 250.0
    predictions.to_csv(
        predictions_path,
        index=False,
        float_format="%.17g",
    )
    _refresh_predictions_hash(audit_dir)

    summary = simulate_execution_decisions(
        audit_dir,
        output_root=tmp_path / "simulations",
        initial_capital=1000.0,
        notional_per_trade=1000.0,
        cost_bps_per_side=5.0,
    )
    output_dir = tmp_path / "simulations" / summary["simulation_id"]
    decisions = pd.read_csv(output_dir / "simulated_decisions.csv")
    argmax = decisions.loc[decisions["strategy"] == "argmax"].sort_values("row_index")

    assert argmax.iloc[0]["execution_status"] == "PORTFOLIO_BROKE"
    assert argmax.iloc[0]["portfolio_equity_after"] < 0.0
    assert argmax.iloc[1]["execution_status"] == "SKIPPED_BANKRUPTCY"
    assert argmax.iloc[1]["executed_direction"] == 0
    assert argmax.iloc[1]["active_notional"] == 0.0
    assert argmax.iloc[1]["transaction_cost_pnl"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_capital": 0.0}, "initial_capital must be positive"),
        ({"notional_per_trade": 200000.0}, "notional_per_trade"),
        ({"cost_bps_per_side": -1.0}, "cost_bps_per_side"),
        ({"cost_bps_per_side": 1001.0}, "cost_bps_per_side"),
        ({"trading_days_per_year": 0}, "trading_days_per_year"),
    ],
)
def test_simulator_rejects_invalid_configuration(tmp_path, kwargs, message):
    audit_dir = create_audit_fixture(tmp_path)
    with pytest.raises(ValueError, match=message):
        simulate_execution_decisions(
            audit_dir,
            output_root=tmp_path / "simulations",
            **kwargs,
        )
