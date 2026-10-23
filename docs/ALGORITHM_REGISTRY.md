# F33 — Algorithm Registry

## Purpose

F33 provides a versioned metadata catalog for quantitative algorithms, with discovery and compatibility checks. The registry is metadata-only: it does not import or execute implementations, dynamically evaluate expressions, load plugins, or bypass risk controls.

## Metadata contract

Each entry declares a stable lowercase ID, semantic version, name, category, summary, method/formula, typed input/output ports, parameters and numeric bounds, units, frequencies, data requirements, limitations, tags, implementation reference, lifecycle and test-evidence status. Metadata is immutable and rejects unknown fields, malformed identifiers, invalid semantic versions, and duplicate port/parameter names.

## Registry API

    from quant.registry import AlgorithmRegistry, builtin_algorithms, discover_algorithms
    registry = AlgorithmRegistry(builtin_algorithms())
    rsi = registry.get("technical.rsi", "1.0.0")
    technical = registry.discover(category="technical")
    rolling = discover_algorithms(tag="rolling")

Registration rejects duplicate (ID, version) pairs unless replacement is explicit. Batch registration is atomic. get(ID) returns the highest registered semantic version; get(ID, version) returns an exact version. Discovery is deterministic and excludes deprecated entries by default.

Compatibility checking compares output/input names, types, and declared units. It reports incompatibilities; it does not connect or execute pipelines.

## Lifecycle and evidence

- experimental: research-only candidate.
- implemented: implementation exists; this is not a claim of investment suitability.
- validated: project-defined validation gates have been met.
- deprecated: retained for lookup/migration, excluded from discovery by default.

Test status is independent: not_run, passing, or failing. CI passing for the registry does not automatically change each algorithm's evidence status. No seed item is marked validated without explicit evidence.

## Seed catalog and scope

The manifest currently contains 304 typed metadata entries spanning the repository's quantitative APIs. The registry validates metadata shape and identifier uniqueness; it does not prove that every referenced implementation has independent behavioral evidence. Each entry keeps `test_status=not_run` until evidence is explicitly recorded. Implementation references are declarative strings: the registry never imports or executes them. Placeholders and invented references are prohibited, and future additions should be tied to source implementations and contracts.

The catalog is descriptive metadata, not proof of correctness, causality, market alpha or future performance. Units, frequency, data availability and temporal validation remain caller responsibilities.
