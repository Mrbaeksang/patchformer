# Experiment log

## 2026-11-18 — Phase 0 CPU plumbing smoke run

**Purpose:** verify deterministic synthetic generation, split integrity, oracle application, and the controlled deterministic baseline. This was **not** a neural-network training run.

### Environment

- Python 3.11.2
- Patchformer 0.1.0 editable install
- `pytest==9.2.0`, `ruff==0.17.5`
- CPU-only sandbox; PyTorch was not installed and no CUDA hardware was available

### Commands run

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m patchformer smoke \
  --output data/generated/phase0-smoke.jsonl \
  --examples-per-split 10 \
  --seed 20261118
python -m patchformer evaluate \
  --data data/generated/phase0-smoke.jsonl \
  --split test
python -m patchformer cuda-smoke
```

### Results

- Tests: **19 passed**.
- Lint and formatting checks: passed.
- Smoke dataset: 30 rows total, 10 each for train/validation/test.
- Dataset SHA-256: `f7f970e3450fef7a2d3f6d4fd45dbd2e349060020fff433de33b3519e0f426be`.
- The output manifest had disjoint train `{orders-v1, catalog-v1}`, validation `{support-tickets-v1}`, and test `{profiles-v1}` schema memberships, with their respective distinct template families.
- Controlled baseline on the 10-row test split: final-state accuracy 1.0, exact patch match 1.0, operation accuracy 1.0, path accuracy 1.0 (6 evaluable rows), typed-value accuracy 1.0 (4 evaluable rows), mutation validity 1.0 (8 non-abstaining/no-op proposals), and `noop`/abstain accuracy 1.0 (4 rows).
- `cuda-smoke` returned the expected unavailable status because PyTorch was not installed in this CPU sandbox. It did not verify CUDA execution.

### Interpretation and limitations

The 1.0 baseline score is expected because the baseline parses the same constrained templates used by the synthetic generator. It is a test of benchmark wiring, oracle validation, and metric semantics only. It is **not** a Patchformer score, a trained-model result, or evidence of natural-language generalization. No checkpoint, parameter count, train/validation loss, latency measurement, confidence estimate, or GPU measurement exists yet.

### Next small milestone

Add a first randomly initialized structured predictor only after preserving the existing oracle and split tests. Start with a tiny overfit fixture, persist/reload a checkpoint, and report held-out results separately from the rule baseline. Add a CUDA run only on a machine where the pinned PyTorch wheel, driver, and GPU are verified compatible.
