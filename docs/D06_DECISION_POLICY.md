# D06.2 — Nested chronological decision-policy evaluation

## Purpose

D06 compares two deterministic decisions from the same classifier probabilities:

1. **Argmax:** choose the class with the largest probability in canonical
   \`DOWN, NEUTRAL, UP\` order.
2. **Adaptive neutral policy:** choose \`NEUTRAL\` when its probability reaches
   a selected threshold; otherwise choose the more likely of \`DOWN\` and \`UP\`.

D06 does not change the training labels or classifier probabilities when it
changes the decision policy.

## Nested temporal protocol

Each outer evaluation fold is an expanding chronological split. Its training
prefix ends exactly where its non-overlapping outer test window begins.

Inside the outer training prefix, D06 withholds a chronological suffix for
threshold selection:

- The estimator used for threshold selection is fitted only on the prefix before
  that suffix.
- Probabilities on the suffix are aligned to the canonical three-class order.
- A declared threshold grid is evaluated using macro F1. Ties are resolved by
  balanced accuracy, neutral precision, then the higher threshold.
- A fresh estimator is fitted on the complete outer training prefix.
- The outer test is evaluated once and cannot select the threshold, estimator,
  or hyperparameters.

The implementation checks class coverage in the pre-calibration training prefix,
calibration suffix and outer test. It validates exact split boundaries, row/date
alignment, probability dimensions, finite values, bounds and row sums.

**Important terminology:** the inner suffix selects a decision threshold; it does
not perform post-hoc probability calibration. Log loss and Brier score are
descriptive scores of the model's emitted probabilities. Calibration-suffix
selection metrics are not unbiased OOS estimates because the same suffix selects
the threshold.

## Outputs and traceability

A new directory is created under \`artifacts/execution_policy_runs/\`:

- \`fold_metrics.csv\`: temporal boundaries, selected threshold, calibration
  diagnostics, per-class metrics, confusion counts, and probability scores.
- \`aggregate_metrics.csv\`: mean, standard deviation, minimum and maximum for
  successful fold metrics.
- \`oos_predictions.csv\`: exact row index/date, actual label, argmax and policy
  decisions, selected threshold, and all three canonical probabilities.
- \`decision_policy_summary.json\`: run configuration, fold coverage, source
  identity and interpretation boundaries.
- \`decision_policy_manifest.json\`: D04 dataset and manifest identity, full
  effective configuration, runtime versions, SHA-256 and byte count for every
  output.

Writes are atomic; run directories are unique. The manifest preserves the
\`input\`, \`configuration\` and \`outputs\` fields required by D07. In particular,
\`input.source_manifest_sha256\` is the digest of the exact D04
\`execution_dataset_manifest.json\`, not the upstream training manifest.

## Usage

\`\`\`bash
python -m scripts.run_execution_decision_policy \\
  --dataset artifacts/execution_datasets/<d04_run>/execution_dataset.csv \\
  --output-root artifacts/execution_policy_runs \\
  --seed 28 \\
  --folds 4 \\
  --test-fraction 0.10 \\
  --calibration-fraction 0.20
\`\`\`

## Interpretation limits

The outer historical folds may have been inspected during development and are not
an untouched prospective test. A successful run proves that the declared
software protocol completed; it does not establish an edge or trading
profitability. D06 does not include transaction costs, portfolio sizing, risk
approval or broker execution. Those belong to later, separately validated stages.
