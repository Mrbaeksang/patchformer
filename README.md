# Patchformer

**Direct State Mutation Modeling — predicting structured changes instead of generating a textual answer.**

> **Status: Research proposal / pre-implementation.** This repository currently defines the research direction, experiments, and handoff plan. No trained model, benchmark result, working API, or setup command is claimed to exist yet.

## 1. Why Patchformer?

A large class of software tasks does not need a prose response. Given an existing state and an instruction or event, the useful answer is often **a small, precise change to that state**:

- Update an order's delivery date from a customer email.
- Modify selected fields in CRM, project-management, or configuration data.
- Synchronize records after receiving a new event.
- Propose a safe, reviewable change rather than rewriting an entire object.

Language models commonly solve these tasks by autoregressively generating JSON or code. **Patchformer investigates whether a smaller task-oriented model can predict mutation operations, target paths, and values more directly.**

Research hypothesis (not a proven claim):

> Given the current state, a schema, and a natural-language instruction, a purpose-built model can predict valid structured state mutations with a favorable accuracy–latency–compute tradeoff against appropriate baselines.

This project explores **structured mutation as an AI primitive**, not another chatbot.

## 2. Formal task

Inputs:

- `state`: the current structured data, initially JSON.
- `schema`: allowed fields, types, and relevant constraints.
- `instruction`: the change request in natural language.
- Optional: context, expected state version, and task-specific constraints.

Output:

- A **proposed mutation**, initially a limited subset of [JSON Patch (RFC 6902)](https://www.rfc-editor.org/rfc/rfc6902).
- An optional **abstain / needs-review** decision.
- Separately calibrated confidence and validation metadata where available.

### Example

Input:

~~~json
{
  "state": {
    "orders": {
      "1024": {
        "delivery_date": "2026-11-15",
        "status": "preparing"
      }
    }
  },
  "instruction": "Change the delivery date for order 1024 to November 19, 2026."
}
~~~

Target mutation:

~~~json
[
  {
    "op": "replace",
    "path": "/orders/1024/delivery_date",
    "value": "2026-11-19"
  }
]
~~~

Applying the mutation to the input state yields the intended final state while preserving all unrelated fields.

### Important distinction: model output vs API format

**The model should not be required to generate the JSON characters above token by token.** A candidate architecture predicts structured decisions such as operation type, a pointer to an existing path, and a typed value (possibly copied from the input). A deterministic serializer then emits RFC 6902 JSON Patch.

The HTTP response may wrap the patch in an envelope with validation and confidence metadata. **Do not insert custom metadata fields inside RFC 6902 operations**, and do not treat raw model probabilities as calibrated confidence.

## 3. Scope

### V0 — minimum research task

1. JSON object states with bounded size and depth.
2. One mutation per example: `add`, `remove`, or `replace`.
3. A finite set of typed values (strings, numbers, booleans, null) with explicit value-source handling.
4. Existing-path selection for `replace` / `remove`; valid parent-path selection for `add`.
5. No-op and ambiguous/unsupported instructions should be represented explicitly, not converted into invented edits.
6. Deterministic validation and patch application outside the neural network.
7. A held-out evaluation suite before scaling model size.

### V1+ — only after V0 results

- Multiple operations, dependencies, and operation ordering.
- Larger and unseen schemas, multilingual requests, and nested arrays.
- Richer value prediction (copy spans, typed extraction, constrained generation where necessary).
- `test` operations and optimistic concurrency / expected-version enforcement.
- Adapters for database changes, UI state, configuration, graph edits, and AST changes.
- Optional relationship matching (a future **Join Model** research track).

### Non-goals

- Building a general-purpose LLM or conversational assistant.
- Automatically executing predicted mutations in production databases.
- Claiming fully general text understanding from synthetic data alone.
- Treating a new model name or API wrapper as evidence of a novel architecture.
- Claiming universal superiority over LLMs, classical code, or existing structured-prediction models.

## 4. Architecture hypotheses

Do **not** assume the architecture below has been validated. Test it against simpler alternatives.

~~~text
                  Current JSON state
                         +
                  Schema / field types
                         +
                Natural-language request
                         |
                  Input representation
                         |
              Small Transformer encoder
                         |
             +-----------+-----------+
             |           |           |
       Operation head  Path pointer  Typed value head
       ADD/REMOVE/     Select valid  Copy/extract/
       REPLACE/NOOP    schema path   classify value
             |           |           |
             +-----------+-----------+
                         |
              Deterministic validator
                         |
                 JSON Patch serializer
                         |
             Proposed patch or ABSTAIN
~~~

Initial proposal:

- **Framework:** PyTorch; implement core model modules directly.
- **Tokenizer:** Hugging Face Tokenizers for an independently trained tokenizer, or a simpler baseline representation.
- **Data pipeline:** Hugging Face Datasets / Parquet / JSONL.
- **Training:** random initialization. No pretrained LLM weights in the *from-scratch* experiment.
- **Scale:** validate with approximately 1–5M parameters; test 10–30M only after the small experiment works.
- **Hardware target:** one NVIDIA RTX 5090 (32 GB VRAM), 64 GB system RAM. No multi-GPU requirement.
- **Optimization:** BF16 where supported, gradient accumulation, appropriate attention kernels, deterministic seeds for reproducibility.
- **Value prediction limitation:** selecting a path is easier than producing an arbitrary novel string. Start with values copied from instructions or from a constrained typed vocabulary; benchmark unsupported-value cases separately.

A hybrid or pretrained-encoder baseline is allowed **as a comparison**, but must be identified separately from the scratch-trained model.

## 5. Dataset plan

### Primary data: synthetic, executable, verifiable

Generate training examples programmatically:

1. Sample a JSON schema, domain, and valid initial state.
2. Sample a valid operation (including `no-op` / invalid requests as appropriate).
3. Execute the operation with an independent, deterministic patch implementation.
4. Generate several natural-language requests describing the same intended change.
5. Save the original state, schema, instruction, target operations, and resulting state.
6. Reject inconsistent or ambiguous examples by validation.

Proposed sample shape:

~~~json
{
  "id": "synthetic-example-000001",
  "domain": "orders",
  "schema_id": "schema-42",
  "state": { "order": { "status": "pending" } },
  "instruction": "Mark the order as shipped.",
  "target_patch": [
    { "op": "replace", "path": "/order/status", "value": "shipped" }
  ],
  "target_state": { "order": { "status": "shipped" } },
  "source": "synthetic",
  "split": "train"
}
~~~

Initial target: approximately **100k generated examples**, subject to pilot results; the sample count is an experiment setting, not a proven requirement. Keep small deterministic fixtures in Git and large generated datasets out of the repository.

### External data candidates

- [Schema-Guided Dialogue (SGD)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue): dialogue-state transitions that may be mapped to state updates; conversion requires careful validation.
- [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html): related structured-call tasks, primarily a reference for evaluation and task design; **do not contaminate held-out benchmark sets by training on them**.
- Other properly licensed edit, state-tracking, or structured extraction datasets, if their labels can be reliably converted.

Record provenance, transformation steps, and dataset licenses in `docs/data.md` once implemented. Never assume external datasets are already in Patchformer format.

### Split design — critical

A random row split is insufficient: it rewards memorizing templates. Report distinct test sets for:

- Unseen state **values and entity IDs**.
- Unseen **language templates / paraphrases**.
- Unseen **schema paths, field names, and structures**.
- Unseen **domains**.
- Adversarial and negative cases: conflicting instructions, invalid path, wrong type, missing reference, no-op, and requests requiring abstention.

Keep gold target generation independent from the model under test. Prevent near-duplicate examples, templates, or schemas from leaking across designated generalization splits.

## 6. Baselines and evaluation

Compare **end-to-end final-state correctness**, not just whether the output looks like valid JSON.

At minimum implement:

1. **Deterministic baseline:** simple rules / selectors with known limitations.
2. **Small learned baseline:** e.g., conventional encoder + classification heads, to test whether architectural novelty helps.
3. **LLM baseline:** a documented small pretrained model with constrained structured output, evaluated separately with its preprocessing, decoding, validation, and hardware cost included.
4. **Patchformer candidate:** scratch-trained direct mutation predictor.

Recommended metrics:

| Metric | Meaning |
| --- | --- |
| Final-state accuracy | Does applying the predicted patch produce exactly the expected resulting state? |
| Patch exact match | Does the canonicalized predicted patch equal the target? Note that distinct patches can yield the same final state. |
| Path accuracy | Was the correct field / parent pointer selected? |
| Operation accuracy | Was the correct mutation operation selected? |
| Typed-value accuracy | Was the correct value and type produced? |
| Validity rate | Does the proposal satisfy JSON Patch and schema constraints? |
| No-op / abstention quality | Does the model avoid unsafe or unjustified edits? |
| Generalization | Same metrics on unseen schemas, domains, and paraphrases. |
| Latency / throughput | End-to-end p50/p95 latency, requests/sec, and memory usage on stated hardware. |
| Confidence calibration | ECE / Brier score when calibrated confidence is exposed. |

**Never evaluate just on training-like synthetic templates.** Report dataset sizes, splits, random seeds, hardware, library versions, and confidence intervals or repeated runs where practical.

Benchmark success must be established empirically. There is **no claimed accuracy, speedup, or trained checkpoint** at repository creation.

## 7. Safety and correctness boundaries

A model proposes changes; an independent application decides whether to apply them.

- Validate operation whitelist, JSON Pointer escaping, path existence rules, and schema types.
- Guard immutable/sensitive fields and authorization outside the model.
- Validate against the correct state version; support atomic apply or reject-on-conflict.
- Ensure the executor is a separate component with no implicit side effects.
- Default to **dry-run** and human review when ambiguous or potentially destructive.
- Log redacted inputs, predictions, validation results, and model versions for auditability.
- Never run predicted SQL, shell commands, or arbitrary code directly.

## 8. Proposed project layout (not yet created)

~~~text
patchformer/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data/
│   ├── model/
│   └── train/
├── src/patchformer/
│   ├── data/            # synthetic generator, datasets, splits
│   ├── tokenizer/       # input representation
│   ├── model/           # encoder and prediction heads
│   ├── training/        # train/eval/checkpointing
│   ├── patch/           # JSON Patch serializer and validator
│   ├── inference/       # predict and dry-run apply
│   └── api/             # optional API after research MVP
├── scripts/
├── tests/
├── benchmarks/
├── docs/
│   ├── architecture.md
│   ├── data.md
│   ├── experiments.md
│   └── decisions.md
└── .github/workflows/
~~~

This tree is a **target layout**, not a statement that files already exist. The initial implementation currently includes only `src/patchformer/patch.py`, a dependency-free JSON-object patch oracle for `add`, `remove`, and `replace`; arrays and root-pointer edits are intentionally unsupported. It validates JSON Pointer escaping, returns a deep-copied result, and does not mutate the input state. Tests cover supported operations, invalid paths, escaping, and input preservation. This is an oracle foundation, not yet a full Phase 0 data generator or benchmark.

Run the focused test suite from a fresh checkout with:

~~~sh
python -m pip install -e '.[test]'
python -m pytest
~~~

## 9. Research roadmap and gates

### Phase 0 — reproducible foundation

- Set up Python environment, dependency locking, format/lint/type checks, pytest, and CI.
- Detect CUDA and verify GPU operation.
- Implement schema-aware JSON Patch oracle, JSON Pointer tests, and dry-run application.
- Implement reproducible synthetic generator and isolated evaluation splits.
- Produce a documented deterministic baseline.

**Exit gate:** a fresh clone can reproduce a tiny dataset and baseline evaluation with one documented command each.

### Phase 1 — first trainable model

- Implement a minimal encoder and single-operation/path/value prediction heads.
- Train a ~1–5M parameter randomly initialized model.
- Run an intentional overfit test on a tiny dataset before full training.
- Save and reload checkpoints; reproduce predictions.
- Compare with the baselines on held-out tests.

**Exit gate:** reproducible nontrivial performance above chance, with error analysis and no split leakage.

### Phase 2 — research iteration

- Try improved input representations, typed-copy heads, schema-path encodings, no-op/abstention.
- Run controlled ablations, repeated seeds, and unseen-schema experiments.
- Profile training/inference, then consider 10–30M parameters.

**Exit gate:** a defensible result showing when direct mutation prediction helps and when it fails.

### Phase 3 — optional productization

- Stable typed input/output API, batch inference, versioned schemas, audit logs, strict validation.
- Adapters for other mutation formats **only if semantics can be preserved**.
- Publish model/data cards, benchmark scripts, limitations, and licensing decisions.

No dates or completion promises are attached to these gates.

## 10. Environment and setup expectations

Target development environment: Linux or WSL2 with a CUDA-capable NVIDIA GPU, ideally RTX 5090 32 GB.

Suggested stack: Python 3.11+, PyTorch with a CUDA build compatible with the installed driver/GPU, `uv` or equivalent dependency manager, `pytest`, `ruff`, Hugging Face `datasets` and `tokenizers`.

**The repository is not runnable yet.** After Phase 0, replace this section with *verified* fresh-clone steps including:

- OS / GPU / driver checks and compatibility notes.
- Actual dependency installation commands and pinned versions.
- Commands to generate a small fixture dataset, run tests, train a tiny model, evaluate, and inspect predictions.
- CPU smoke-test mode and GPU performance mode.
- Paths for checkpoints, logs, experiment configs, and cached datasets.
- Troubleshooting for CUDA/VRAM issues.

Do not add plausible-looking commands without implementing and testing them.

## 11. Instructions for the next AI coding agent

If an AI assistant or coding agent is asked to continue this repository, **read this section first**.

**Mission:** turn the design into a reproducible, from-scratch research baseline. Prefer a minimal working experiment to a large unverified system.

Operating rules:

1. Inspect the actual repository and report what exists before planning changes. The layout and features above are aspirational.
2. Preserve the central research hypothesis: **predict a structured mutation directly rather than autoregressively printing a JSON patch**.
3. Implement and test the **data generator + independent patch oracle + held-out benchmark before model scaling**.
4. Start with single-operation JSON object changes. Explicitly handle no-op, invalid requests, and abstention; do not silently expand scope.
5. Randomly initialize the scratch model. If using pretrained weights for a baseline, clearly label and isolate that experiment.
6. Use deterministic seeds and versioned configs. Save per-run metadata and metrics. Treat measured results separately from assumptions.
7. Avoid data leakage. Prefer disjoint schemas/domains and new paraphrase families for generalization tests.
8. Keep state mutation execution separate and default to dry-run. Never wire a model prediction to a live database automatically.
9. Every completed feature needs tests and an executable command. Do not mark a roadmap item complete because a file or stub exists.
10. Update this README and `docs/experiments.md` with **what is implemented, what was measured, what failed, and what comes next**. Record significant design choices in `docs/decisions.md`.
11. Work in small, reviewable changes. Do not fabricate dataset rights, performance results, references, or successful runs.

### Recommended first development request

> Inspect the repository, then implement **Phase 0 only**. Create a minimal Python project with pinned dependencies; deterministic synthetic JSON-state/instruction/patch examples; an independent JSON Patch validation/application oracle; train/validation/test splits with schema- and template-held-out tests; a trivial deterministic baseline; unit tests; and verified CPU and CUDA smoke-test instructions. Do not implement the large neural network or API yet. Document actual commands, results, and limitations in README.md.

### What to report at the end of every AI work session

- Files added/changed and why.
- Exact commands run and whether they passed.
- Dataset provenance and split integrity.
- Measured metrics (or explicitly “not measured”).
- Remaining risks, decisions, and the next small milestone.

## 12. Related resources

- [JSON Patch — RFC 6902](https://www.rfc-editor.org/rfc/rfc6902)
- [JSON Pointer — RFC 6901](https://www.rfc-editor.org/rfc/rfc6901)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers)
- [Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- [BFCL / Gorilla](https://gorilla.cs.berkeley.edu/leaderboard.html)

---

**Project principle:** *Predict the change, not the explanation.*

**License:** Not selected yet. No open-source license should be assumed until one is added.
