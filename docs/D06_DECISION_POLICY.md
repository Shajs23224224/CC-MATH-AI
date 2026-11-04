# D06.3 — Calibrated chronological decision-policy evaluation

## Purpose

D06 compares two deterministic decisions from the same classifier probabilities:

1. **Argmax:** choose the class with the largest probability in canonical
   \`DOWN, NEUTRAL, UP\` order.
2. **Adaptive neutral policy:** choose \`NEUTRAL\` when its probability reaches
   a selected threshold; otherwise choose the more likely of \`DOWN\` and \`UP\`.

D06 does not change the training labels. D06.3 post-hoc calibrates classifier
probabilities with one scalar temperature before comparing the two decision rules.

## Nested temporal protocol

Each outer evaluation fold is an expanding chronological split. Its training
prefix ends exactly where its non-overlapping outer test window begins.

Inside the outer training prefix, D06 withholds a chronological suffix:

- The estimator is fitted only on the prefix before that suffix.
- Its suffix probabilities are aligned to the canonical DOWN, NEUTRAL, UP order.
- A single multiclass temperature is fitted by minimizing suffix log loss over
  the bounded interval [0.05, 20].
- The calibrated suffix probabilities are then used to select the neutral
  threshold by macro F1. Ties are resolved by balanced accuracy, neutral
  precision, then the higher threshold.
- The exact same estimator and fitted temperature produce the outer OOS
  probabilities. The estimator is not refitted on its calibration suffix.
- The outer test is evaluated once and cannot fit the temperature, select the
  threshold, estimator, or hyperparameters.

The implementation checks class coverage in the pre-calibration training prefix,
calibration suffix and outer test. It validates exact split boundaries, row/date
alignment, probability dimensions, finite values, bounds and row sums.

**Selection caveat:** the same inner suffix fits the temperature and selects
the neutral threshold. Its calibration scores are descriptive and optimistic,
not an unbiased performance estimate. The untouched outer OOS probabilities
receive multiclass log loss, Brier score, equal-width one-vs-rest reliability
bins, expected calibration error (ECE), and maximum calibration error (MCE).

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

The same past-only suffix is used to fit temperature and select the neutral
threshold, so suffix metrics are not independent validation. Temperature scaling
is a one-parameter adjustment; it may not correct class-specific distortions or
distribution shift. The outer historical folds may have been inspected during
development and are not an untouched prospective test. A successful run proves
that the declared software protocol completed; it does not establish an edge or
trading profitability. D06 does not include transaction costs, portfolio sizing,
risk approval or broker execution. Those belong to later, separately validated
stages.


## D07 calibration-audit evidence

D06.3 also persists calibration_predictions.csv with the original three-class probabilities for every calibration-suffix observation. The OOS prediction file includes both the pre-temperature probabilities and the calibrated probabilities. These auxiliary raw values are retained in the output manifest so the downstream D07.3 auditor can independently refit the temperature, reproduce calibration scores, verify its application to OOS probabilities, and rebuild reliability bins from the actual rows. The primary four D06 artifacts and their existing schemas remain available to downstream consumers.
