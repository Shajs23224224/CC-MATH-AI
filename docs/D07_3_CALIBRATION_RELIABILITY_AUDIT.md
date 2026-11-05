# D07.3 — Independent calibration and reliability audit

## Objective

D07.3 independently verifies the probability-calibration layer emitted by D06.3. It does not trust the temperature, selected neutral threshold, calibrated probability columns, fold calibration metrics, aggregate statistics, or reliability bins simply because they appear in D06 outputs.

The auditor requires a completed D06.3 run and the exact D04 execution dataset used by that run. It verifies the D06 manifest hashes and byte counts, then reuses D07.1's strict D04 contract verification: execution-return arithmetic, threshold/label reconciliation, chronological dates, and next-observed-session entry mapping. Only after those checks does it map calibration and OOS probability rows to D04 indices, dates, and actual labels.

## Independent recomputation

For each algorithm and expanding fold, D07.3:

1. Rebuilds the calibration suffix boundaries from the fold metadata and declared calibration fraction.
2. Validates calibration raw probabilities as finite three-class simplex rows and matches row indexes, dates, and labels to D04.
3. Fits a scalar temperature again using bounded optimization over the exact manifest bounds [0.05, 20]. The implementation independently reproduces D06's objective, optimizer tolerance, and deterministic endpoint/identity candidate tie-breaks.
4. Recomputes raw and temperature-scaled calibration log-loss and multiclass Brier scores.
5. Independently repeats macro-F1 neutral-threshold selection, including balanced-accuracy, neutral-precision, and higher-threshold tie breaks.
6. Applies the independently fitted temperature to every saved raw OOS probability vector and compares it with the calibrated OOS probabilities published by D06.
7. Recomputes OOS log-loss, Brier score, and all one-vs-rest reliability bins. It checks the exact algorithm/fold/class/bin key grid, bin boundaries, sample counts, means, observed frequencies, calibration gaps, ECE, and MCE.
8. Recomputes the calibration and probability metric aggregates in `aggregate_metrics.csv`.

Equal-width reliability bins use `floor(probability × bin_count)`, with probability 1.0 assigned to the final bin. Empty bins must have zero observations and null means/frequency/gap. For each class, ECE is the bin-gap sum weighted by the fraction of all OOS rows in each bin; MCE is the largest absolute gap among non-empty bins. Multiclass ECE is the mean of the three classwise ECE values, and multiclass MCE is the maximum of the three classwise MCE values.

## Required D06.3 evidence

The four primary D06 artifacts retain their existing names and purpose. D06.3 also persists:

- `calibration_predictions.csv`: the raw DOWN/NEUTRAL/UP probability vector, actual label, row index and feature date for every calibration-suffix row.
- `raw_probability_down`, `raw_probability_neutral`, and `raw_probability_up` in `oos_predictions.csv`: the pre-temperature probability vector for each OOS row.

Both additions are covered by the D06 output manifest. Without these raw observations, an independent temperature fit and independent application to the OOS segment cannot be reproduced from aggregate metrics alone.

## Usage

```bash
python -m scripts.audit_calibration_reliability \
  --policy-run artifacts/execution_policy_runs/<d06_run> \
  --dataset artifacts/execution_datasets/<d04_run>/execution_dataset.csv \
  --output-root artifacts/execution_policy_calibration_audits
```

## Outputs

Each run creates a unique directory containing:

- `calibration_fold_audit.csv`: published versus independently recomputed temperature, threshold, calibration scores and OOS metrics for each algorithm/fold.
- `reliability_bins_audit.csv`: published versus recomputed bin count, confidence, observed frequency and absolute gap for every class/bin.
- `calibration_audit_summary.json`: completion state, checks, coverage counts, source identity and interpretation limits.
- `calibration_audit_manifest.json`: SHA-256 and byte counts for the audit outputs, plus verified D06 source hashes.

Writes are atomic. The D06 source artifacts are read-only; this phase creates a separate audit package and never rewrites D04 or D06 outputs.

## Interpretation limits

The audit establishes internal consistency and reproducibility of saved historical calculations. It does not establish forecast quality, a calibration guarantee, an economic edge, or trading profitability. D06 uses the same historical calibration suffix to fit temperature and select its neutral threshold, so those suffix scores are selection diagnostics rather than unbiased validation. Equal-width bins can be noisy when counts are small; a global scalar temperature cannot guarantee classwise calibration or immunity to distribution shift. Existing historical OOS folds may have been inspected during development and are not a prospective untouched test. SHA-256 manifests detect file changes but are unsigned and do not authenticate artifact origin.
