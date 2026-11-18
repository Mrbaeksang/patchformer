# Architecture decision record

## ADR-001: Keep patch execution independent of data generation and prediction

**Status:** accepted, 2026-11-18.

Target states and evaluated predictions are dry-run through `patchformer.patch.oracle`. The synthetic generator does not directly mutate a final-state label, and neither a future model nor the baseline is trusted to validate its own output. This makes invalid paths, type changes, required-field removals, and pointer escaping visible at the safety boundary.

## ADR-002: Use a narrow V0 JSON Patch subset

**Status:** accepted, 2026-11-18.

Phase 0 supports object states and at most one `add`, `replace`, or `remove`. Arrays, root operations, and multi-operation ordering remain out of scope. A narrow contract allows unambiguous tests and error analysis before learning or expanding semantics. It is not a claim of complete RFC 6902 support.

## ADR-003: Split by schema/domain/template family, not random rows

**Status:** accepted, 2026-11-18.

Schema IDs, domains, and instruction-template families are disjoint across train, validation, and test. Non-boolean/non-null scalar values are also held out by split. This avoids obvious synthetic leakage, but does not make the benchmark realistic; future data needs more independent paraphrases and structures.

## ADR-004: Abstention is semantically distinct from a no-op

**Status:** accepted, 2026-11-18.

Both decisions retain the state, but a no-op means the request was understood and requires no change; abstention means it was unsupported, invalid, or ambiguous. Evaluation requires abstention for abstention-labelled rows, so a model cannot receive credit merely by emitting an empty patch.

## ADR-005: Do not add a learned model before an executable baseline exists

**Status:** accepted, 2026-11-18.

The initial deliverable is a reproducible CPU pipeline, safety oracle, leakage checks, deterministic baseline, and tests. A neural model will be introduced only with a tiny-data overfit test, checkpoint round-trip, seed reporting, and held-out evaluation. This prevents unsupported performance claims and keeps the first ML change reviewable.
