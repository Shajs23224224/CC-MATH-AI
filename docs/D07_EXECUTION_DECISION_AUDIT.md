# D07.1 — Execution decision integrity audit

## Objective

D07.1 independently validates a completed D06 decision-policy run against the exact
D04 execution dataset used to train and evaluate it. The audit fails closed: no audit
package is emitted when an input hash, row mapping, class decision, temporal boundary
or persisted metric does not reconcile.

The implementation lives in execution/research/d07_audit.py and is available through
scripts/audit_execution_decision_policy.py.

## Inputs

D07.1 requires the exact files from one D06 run:

- decision_policy_manifest.json
- decision_policy_summary.json
- fold_metrics.csv
- aggregate_metrics.csv
- oos_predictions.csv

It also requires the matching D04 folder containing:

- execution_dataset.csv
- execution_dataset_manifest.json

The D06 input manifest must identify the same D04 dataset filename, dataset SHA-256
and D04 source-manifest SHA-256. D04's own loader validates the dataset artifact,
feature contract, dates and three-class target schema.

## Audit sequence

1. Verify the D06 manifest, required output files, byte counts and SHA-256 digests.
2. Verify the D06 run is complete: every declared algorithm/fold must have a completed
   fold record and the expected number of OOS predictions.
3. Verify the D04 artifact hash, byte count, chronological feature index, unique dates,
   positive prices and finite feature/target values.
4. Recompute next-session open-to-close return as exit_close / entry_open - 1
   and reconstruct UP/DOWN/NEUTRAL labels using D04's exact threshold band.
5. Check expanding-window boundaries and the nested calibration suffix. Training,
   calibration and outer-test indices must remain within their declared chronological
   ranges. Training, calibration and test samples must each contain the expected
   classes; outer-test data must not be marked as threshold-selection data.
6. Require complete, unique OOS row-index coverage for every algorithm/fold. Each
   feature date and actual label must map to the same D04 row.
7. Verify probability vectors are finite, bounded in [0, 1] and sum to one.
8. Reproduce argmax and the D06 adaptive-neutral policy from the persisted class
   probabilities and selected threshold.
9. Recompute all persisted per-fold classification diagnostics, including class-wise
   precision/recall/F1, support, prediction counts/shares and confusion counts.
10. Recompute the D06 aggregate means, standard deviations, minima, maxima and
    successful-fold counts from the fold records.

## Outputs

A successful audit creates a new timestamped directory under
artifacts/execution_policy_audits/:

- audited_predictions.csv: exact OOS decisions joined to the D04 execution date,
  entry open, exit close and realized gross execution return. Correctness flags,
  maximum probability and probability margin are included.
- fold_audit.csv: independently recomputed per-algorithm/per-fold metrics and
  verified temporal boundaries.
- audit_summary.json: source identities, passed checks, row/fold counts, descriptive
  target-return statistics and interpretation limits.
- audit_manifest.json: SHA-256 hashes and byte counts for each output.

Writes are atomic. Run directories are unique and never overwritten. Generated
experiment data remains outside Git.

## Usage

    python -m scripts.audit_execution_decision_policy \
      --policy-run artifacts/execution_policy_runs/<d06_run> \
      --dataset artifacts/execution_datasets/<d04_run>/execution_dataset.csv \
      --output-root artifacts/execution_policy_audits

D07.1 is an integrity audit only. It does not create positions or calculate PnL.

## Integrity boundary

SHA-256 and byte-count checks detect changes relative to the supplied manifests.
The manifests are unsigned: they do not authenticate who created them or prevent an
actor from replacing both an artifact and its manifest. Preserve the original
D04/D06 inputs and run the audit in a trusted environment when provenance matters.

## Tests

The D07 rebuild workflow tests production-shaped D04/D06 manifests and artifacts,
including tampered hashes, misaligned OOS indexes, unreproducible policy decisions,
outer-test leakage, aggregate-metric changes and modified D04 data.

See D07_2_EXECUTION_DECISION_SIMULATION.md for the next stage, which accepts only
a completed D07.1 output package.
