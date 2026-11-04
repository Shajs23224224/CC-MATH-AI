# D04.2 — Auditable execution dataset

## Purpose

D04 converts a verified historical feature snapshot into an execution-aware,
three-class classification dataset. A feature row is available after the close of
its feature session. The label is derived from the next **observed** OHLCV session:
the theoretical entry is that session's open and the exit is that same session's
close.

\[
r_{t+1}^{exec} = \frac{Close_{t+1}}{Open_{t+1}} - 1
\]

With threshold \(b\) basis points, the label is:

- \`UP\` when the return is strictly greater than \(b/10{,}000\).
- \`DOWN\` when the return is strictly less than \(-b/10{,}000\).
- \`NEUTRAL\` otherwise, including equality at either boundary.

The threshold is a **target-label band**, not a commission, spread, slippage, or
transaction-cost estimate.

## Inputs and provenance

The builder accepts the raw \`raw_ohlcv.csv\` and \`dataset_snapshot.csv\` from
one training run. It requires the adjacent \`dataset_manifest.json\` and verifies
both input artifact names and SHA-256 hashes. When byte counts are declared, they
must match as well.

The builder also verifies:

- Strictly increasing, unique date indexes and an exact mapping of feature dates
  to the raw OHLCV sessions.
- The source snapshot's declared feature names and usable row count.
- Finite numeric feature values, source regression targets, and positive finite
  open/close prices.
- The source regression target against the next-session close-to-previous-close
  chronology before deriving the separate open-to-close execution target.
- A following observed OHLCV session for every feature row.
- Reserved and future execution columns are never admitted as model features.

## Outputs

Each run creates a new directory under \`artifacts/execution_datasets/\`:

- \`execution_dataset.csv\`: causal model features, feature date, next-session
  entry date/open, same-session exit close, gross execution return and class.
- \`execution_dataset_manifest.json\`: versioned contract, input identities,
  threshold units, execution assumptions, row/class statistics, environment,
  output SHA-256 and byte count, and explicit interpretation limits.

CSV writes use round-trip floating-point precision and atomic replacement. Existing
run directories are never overwritten. The output manifest preserves the D04
fields consumed by D05, D06 and D07.

## Usage

\`\`\`bash
python -m scripts.build_execution_dataset \\
  --raw artifacts/training_runs/<run>/raw_ohlcv.csv \\
  --snapshot artifacts/training_runs/<run>/dataset_snapshot.csv \\
  --output-root artifacts/execution_datasets \\
  --threshold-bps 10
\`\`\`

## Interpretation limits

The entry and exit prices are theoretical historical prices, not observed broker
fills. D04 does not model commissions, spread, slippage, market impact, financing,
borrow, margin or taxes. A SHA-256 manifest detects content changes relative to
the supplied manifest; it is not a signature and does not authenticate the
origin of the artifacts. Historical labels and later model metrics do not
establish future profitability.
