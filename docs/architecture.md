# Phase 0 architecture

## Implemented boundary

Patchformer currently has **no neural network, tokenizer, checkpoint, API, or training loop**. Phase 0 implements the executable boundary that a future model must satisfy:

```text
synthetic schema + state + instruction
                  |
                  v
     target operation / decision sampler
                  |
                  v
 independent dry-run patch oracle  <--- deterministic baseline proposals
                  |
                  v
      JSONL row + final-state evaluation
```

The separation is intentional. The generator chooses a target operation, while `patchformer.patch.oracle` validates and applies the target independently before a row is persisted. Evaluation also uses that oracle rather than trusting a generated patch string or a model-side validator.

## V0 mutation contract

`src/patchformer/patch/oracle.py` implements a deliberately narrow, schema-aware subset of RFC 6902:

- Object-only JSON states and closed object schemas.
- At most one `add`, `replace`, or `remove` operation per proposal.
- `add` only on an absent declared field; `replace` only on an existing field; `remove` only on an existing optional field.
- Typed values, enums, required fields, JSON Pointer decoding/escaping, and parent-path existence are checked.
- Root replacement, arrays, arbitrary extra properties, `move`, `copy`, and `test` are outside the V0 contract.
- `apply_patch` deep-copies its input, so it is always a dry run. It never executes a database, shell, or network action.

An empty patch is the V0 representation of `noop`; abstention is represented separately by `target_decision: "abstain"`. Treating an empty patch as an abstention is an evaluation error and has a dedicated unit test.

## Data row and prediction interface

Each generated JSONL row contains at least:

```json
{
  "state": {"...": "..."},
  "schema": {"type": "object", "...": "..."},
  "instruction": "For verification flag, use true.",
  "target_decision": "patch",
  "target_patch": [{"op": "replace", "path": "/profile/verified", "value": true}],
  "target_state": {"...": "..."},
  "split": "test"
}
```

A baseline or future model returns this in-memory shape:

```json
{"decision": "patch", "patch": [{"op": "replace", "path": "/...", "value": "..."}]}
```

Allowed decisions are `patch`, `noop`, and `abstain`. `patchformer.evaluation.evaluate_examples` reports final-state accuracy by independently dry-running every non-abstaining prediction. It also reports exact patch match, operation/path/value accuracy, mutation validity, and safe `noop`/`abstain` decision accuracy.

## Existing deterministic baseline

`patchformer.baseline.deterministic` is a transparent schema-title parser for the controlled Phase 0 templates. It recognizes a deliberately limited grammar and JSON literals, then revalidates any proposed patch with the oracle. Unknown targets, malformed values, required-field removals, and unsupported language lead to abstention.

This is plumbing, **not a learned baseline and not a general natural-language interpreter**. Its score on the generated template grammar must not be described as model accuracy.

## Next architecture gate

Before adding a Transformer, retain the input/output contract above and add an intentional tiny-data overfit test. A learned V1 candidate should predict operation, schema path, typed-value source/value, and abstention as structured outputs; serialization and patch application remain deterministic external components. Any pretrained model comparison must be isolated from the randomly initialized experiment and report its preprocessing, decoding, hardware, and cost.
