# Differential report contract

Status: schema version 1, frozen on 2026-11-03.

The canonical machine-readable contract is
[`schemas/correctness-report-v1.schema.json`](../schemas/correctness-report-v1.schema.json).
It uses JSON Schema Draft 2020-12 and covers the stable stage envelope plus the metric,
mask, named-set, optimizer, and HVG records consumed by downstream checks.

Version 1 requires six stages in order-independent JSON objects:

1. `input`
2. `selection`
3. `raw_fitting`
4. `regularization`
5. `residual`
6. `hvg`

Writers validate the stable envelope before persisting a report and reject NaN/Infinity
JSON literals. `benchmarks.report_schema.load_correctness_report()` reads version 1,
rejects unknown versions and missing stages, and deliberately tolerates additive fields.
This lets a version-1 writer add diagnostics without breaking version-1 consumers.

Removing or renaming a required field, changing its meaning, or changing its type requires
a new schema version and an explicit reader branch. Readers must not silently interpret an
unknown version as version 1. The JSON Schema itself is checked in the normal test suite;
`jsonschema` is a test-only dependency and does not enter the package runtime.

The selection stage records both named-set agreement and exact order for fit cells. This
distinction matters when an external oracle returns a random selection order: the set
proves that the same cells were used, while order can expose a replay difference.

This contract formalizes report transport. Numerical acceptance thresholds remain defined
separately in [CORRECTNESS.md](CORRECTNESS.md) and `benchmarks/correctness.py`.
