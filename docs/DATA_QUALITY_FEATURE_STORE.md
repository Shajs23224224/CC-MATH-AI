# F12 — Data Quality + Feature Store

## Objective

Prevent low-quality or temporally invalid data from silently entering quantitative models, and persist reproducible feature definitions and values.

## Data-quality checks

F12 provides deterministic checks for:

- missing required columns;
- duplicate keys;
- invalid or missing numeric values;
- invalid timestamps;
- future observations relative to an analysis cutoff;
- large absolute jumps;
- robust statistical outliers;
- stale repeated values;
- missing expected universe members;
- explicit survivorship-bias risk when historical universe metadata is unavailable.

Severity levels are INFO, WARNING and ERROR.

ERRORs or detected temporal leakage make a report INVALID. Warnings do not automatically reject a dataset.

## Leakage

A dataset has detected leakage when timestamped records occur after the requested analysis cutoff.

The report records leakage_detected, issue codes, counts and overall status.

Downstream code can require report.is_usable before feature or model computation.

## Survivorship bias

The engine does not infer survivorship bias from a price table alone.

When an expected historical universe is supplied, missing members are reported. If historical universe metadata is unavailable, the report sets the survivorship-bias risk flag. This is a risk indicator, not proof that bias exists.

## Outliers and large jumps

Outliers use a robust median/MAD score. Large percentage changes use a configurable threshold.

The quality engine reports anomalies; it does not silently delete them. Genuine splits or extraordinary market events can be real observations and require explicit treatment.

## Stale data

Repeated identical numeric values over a configurable run length are flagged. This is particularly relevant to sparse, illiquid or slowly published data.

## Feature Store

F12 introduces the chain:

FeatureDefinition -> FeatureRecord -> FeatureSet

A feature definition records:

- feature ID;
- semantic name;
- version;
- description;
- source columns;
- frequency;
- lookback;
- point-in-time property;
- methodology.

A feature record records:

- feature ID/version;
- asset symbol;
- feature timestamp;
- numeric value;
- data snapshot ID;
- creation timestamp.

## Versioning

The development Feature Store is append-only JSONL.

A definition is immutable for the pair feature_id + version. Changing methodology requires a new feature version.

## Temporal safety

Feature writes can receive an as_of cutoff. Records newer than that cutoff are rejected.

Feature reads also support as_of, so historical workflows cannot retrieve future feature records.

## Provenance

Every feature record references a data_snapshot_id. This makes downstream model inputs traceable to a specific data snapshot.

## Architecture

The production implementation can later replace the local JSONL store through the FeatureStore protocol with a database or columnar/object-store backend.

## F12 completion

- [x] Missing-data checks
- [x] Duplicate detection
- [x] Invalid numeric detection
- [x] Future-data/leakage detection
- [x] Robust outlier detection
- [x] Stale-data detection
- [x] Large-jump detection
- [x] Survivorship-bias risk flag
- [x] Data-quality report
- [x] Feature definition contract
- [x] Feature record contract
- [x] Versioned feature store
- [x] Point-in-time feature filtering
- [x] Feature provenance via data snapshot ID
- [x] Unit tests
- [x] Integration test

**Status: COMPLETE**

Next phase: F13 — Mathematical Core.
