# F32 — Feature Engineering

## Scope and API

F32 provides a deterministic, allow-listed feature pipeline for tabular time series and
multi-asset panels. It uses the existing `FeatureDefinition` contract and returns both an
augmented pandas DataFrame and a Pydantic lineage report. No new runtime dependency is added.

```python
import pandas as pd

from core.contracts.feature_engineering import FeatureTransformSpec
from core.contracts.quality import FeatureDefinition
from ml.features import FeatureEngineer

spec = FeatureTransformSpec(
    definition=FeatureDefinition(
        feature_id="return_5d",
        name="Five-period return",
        version="1.0.0",
        description="Trailing price return",
        dtype="float64",
        source_columns=("close",),
        frequency="1D",
        lookback=5,
        point_in_time=True,
        methodology="close[t] / close[t-5] - 1",
    ),
    operation="pct_change",
    category="price_return",
    periods=5,
)
frame = pd.DataFrame({"timestamp": ..., "close": ...})
features, report = FeatureEngineer().transform(
    frame, (spec,), dataset_id="market-snapshot-v1"
)
```

## Allow-listed operations

- `identity`, `lag`, `pct_change`, `log_return`.
- Trailing `rolling_mean/std/min/max/sum/zscore` and `ewm_mean`.
- Controlled two-column `interaction` and `ratio`.
- Per-timestamp `cross_sectional_rank` and `cross_sectional_zscore` for multi-asset data.

All window features use trailing observations and preserve row count. Rolling statistics require a
full initial window; warm-up values remain NaN. Zero-denominator ratios and zero-variance
z-scores produce NaN rather than infinity. Invalid inputs and infinite outputs fail explicitly.

## Point-in-time and leakage controls

- Rows must arrive in non-decreasing timestamp order; the pipeline never silently sorts or drops rows.
- With `asset_column` configured, timestamp/asset pairs must be unique and rolling/lag operations
  are isolated per asset. Cross-sectional transforms only compare assets at the same timestamp.
- Fundamental, macro, regime and risk categories require an availability timestamp column or an
  explicit row availability lag. If an availability column is supplied, values not yet released
  are masked and only previously available values are carried forward.
- The source must already reflect point-in-time universe membership, corporate-action adjustments,
  revised macro/fundamental releases and feature-specific frequency alignment. The pipeline cannot
  infer whether upstream data vendors supplied a survivorship-free historical universe.
- No learned preprocessing is fitted here. Any scaler, imputer, selector or dimensionality reducer
  downstream must be fitted on the training prefix only.
- Same-timestamp market features are assumed available at that timestamp; callers predicting the
  next timestamp must align labels/horizons accordingly.

## Lineage and boundaries

Every feature uses the existing versioned `FeatureDefinition` fields: source columns, frequency,
lookback, point-in-time declaration and methodology. The report provides stable pipeline ID,
input/output SHA-256 fingerprints, exact feature identifiers, row counts and warm-up diagnostics.
It preserves input columns and row/index order and refuses to overwrite existing columns.

The pipeline does not fetch data, infer causal relationships, select investment strategies, train an
ML model, emit orders or override F21 risk controls. Cross-sectional values can be biased by a
survivorship-affected universe. Synthetic tests demonstrate implementation behavior, not predictive
value or investment profitability.

## Acceptance criteria

F32 is ready for integration when its dedicated workflow, general Python 3.11/3.12 CI, Security,
regression tests and leakage-boundary tests pass on the final PR head.
