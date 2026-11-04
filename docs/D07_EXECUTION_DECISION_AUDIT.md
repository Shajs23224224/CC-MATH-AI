# D07.1 — D06 execution decision-policy audit

## Purpose

D07.1 audits the integrity and traceability of the nested chronological predictions produced by D06. It verifies the persisted D06 artifacts against their SHA-256 manifest, reconciles every out-of-sample prediction with the original D04 execution dataset, and recomputes classification metrics from row-level predictions.

The audit enriches each OOS prediction with its realized next-session execution return, entry date, assumed entry open, and exit close. These values are retained as auditable outcome evidence for D07.2. D07.1 does not convert them into trading PnL.

## Inputs

- A completed D06 run directory containing `decision_policy_manifest.json`, `decision_policy_summary.json`, `fold_metrics.csv`, `aggregate_metrics.csv`, and `oos_predictions.csv`.
- The exact D04 `execution_dataset.csv` used by that D06 run.
- The sibling D04 `execution_dataset_manifest.json`.

The D06 manifest must identify the same D04 dataset and source-manifest hashes. D04's loader independently validates the execution-dataset SHA-256 and the feature/target schema.

Generated CSVs and run artifacts remain experiment data and should not be committed to Git unless there is a specific, reviewed reason to publish a small sanitized sample.

## Checks

The audit fails closed when any of the following conditions is violated:

1. Every required D06 output matches the SHA-256 digest and byte count recorded in `decision_policy_manifest.json`.
2. The D06 run summary and manifest refer to the supplied D04 dataset and source manifest.
3. D04's target return equals `exit_close / entry_open - 1`, and its three-class target label follows the configured threshold.
4. D06 completed every declared algorithm/fold evaluation and emitted the declared number of OOS prediction rows.
5. There are no duplicate `(algorithm, row_index)` observations.
6. Each OOS row index and feature date map to the same D04 row, and the stored actual label matches `target_execution_direction`.
7. Probability values are finite, lie in `[0, 1]`, and sum to one per row.
8. Stored `argmax_prediction` and `policy_prediction` values can be reproduced from the probability columns and selected neutral threshold.
9. Per-fold metrics are reproduced from row-level predictions; selected aggregate means match the fold metrics.
10. Calibration suffixes remain inside the training prefix, outer test windows are chronological and non-overlapping, and outer test data is not marked as selection data.

## Usage

Run from the repository root after D04 and D06 artifacts are available locally:

```bash
python -m scripts.audit_execution_decision_policy \
  --policy-run artifacts/execution_policy_runs/spy_decision_policy_20261104T031601400863Z \
  --dataset artifacts/execution_datasets/spy_execution_20261103T042714129985Z/execution_dataset.csv \
  --output-root artifacts/execution_policy_audits
```

The paths above identify the SPY experiment already generated in the development environment. Use the actual D04 and D06 artifact paths when auditing a different run.

## Outputs

D07.1 creates a new timestamped directory below `artifacts/execution_policy_audits/`:

- `audited_predictions.csv`: D06 OOS rows joined with D04 realized execution returns and price assumptions. It also contains correctness flags, selected-class probabilities, maximum probability, and probability margin.
- `fold_audit.csv`: independently recomputed `argmax` and D06 policy metrics for each algorithm and outer fold.
- `audit_summary.json`: input hashes, integrity checks, per-algorithm pooled diagnostic metrics, realized-return descriptive statistics, and limitations.
- `audit_manifest.json`: SHA-256 hashes and byte counts for the D07.1 output files.

Writes use temporary files followed by atomic replacement. Existing run directories are never overwritten.

## Interpretation boundaries

D07.1 is an integrity audit, not a backtest and not a trading strategy evaluation. It does not apply positions, leverage, sizing, capital constraints, commissions, spread, slippage, market impact, or portfolio accounting. The realized execution return is joined only to prove row alignment and preserve the target outcome for the next phase.

The historical walk-forward windows were already inspected in D03–D06; they are not an untouched prospective test. Integrity checks and metric reproduction do not establish market alpha, profitability, or future predictive performance.

After D07.1 completes successfully, its output directory is the only accepted input to [D07.2 — Cost-aware execution decision simulation](D07_2_EXECUTION_DECISION_SIMULATION.md). D07.2 applies a transparent cost and notional model to compare the original argmax predictions with the D06 neutral-threshold policy.

## Tests

```bash
python -m pytest tests/unit/test_audit_execution_decision_policy.py -q
ruff check scripts/audit_execution_decision_policy.py tests/unit/test_audit_execution_decision_policy.py
ruff format --check scripts/audit_execution_decision_policy.py tests/unit/test_audit_execution_decision_policy.py
```
