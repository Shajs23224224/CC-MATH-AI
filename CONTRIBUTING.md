# Contributing to C-MATH-AI

## Principles

1. Preserve the architecture defined in `docs/ARCHITECTURE.md`.
2. Keep financial formulas deterministic and independently testable.
3. Do not introduce provider-specific logic into domain contracts.
4. Do not bypass the Risk Governor.
5. Never commit secrets.
6. Add tests when changing numerical behavior.
7. Keep changes small and traceable.

## Commit convention

Use concise imperative messages with the phase when relevant:

```text
F03: initialize repository structure
F13: add return calculations
F21: add risk metrics
```

## Pull requests

A change should explain:
- what changed;
- why it changed;
- affected architecture/contracts;
- tests executed;
- known limitations.
