"""Deterministic, executable Phase 0 synthetic-data generator.

Rows are created from state and operation choices, then independently dry-run
through :mod:`patchformer.patch.oracle` before they are emitted.  Generated data is
for pipeline testing only: natural-language templates are deliberately simple and
must not be mistaken for a real-world training corpus.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

from patchformer.patch import apply_patch
from patchformer.patch.oracle import encode_pointer

from .schemas import all_split_names, schemas_for_split
from .splits import TEMPLATE_FAMILIES, assert_split_integrity

JsonObject = dict[str, Any]
_MISSING = object()
_TASK_TYPES = ("replace", "add", "remove", "noop", "abstain")


def _leaf_specs(schema: JsonObject) -> list[tuple[list[str], JsonObject, bool]]:
    """Return (path, leaf-schema, required) triples for a closed object schema."""
    leaves: list[tuple[list[str], JsonObject, bool]] = []

    def visit(node: JsonObject, path: list[str], is_required: bool) -> None:
        if "object" not in _types(node):
            leaves.append((path, node, is_required))
            return
        properties = node["properties"]
        required = set(node.get("required", []))
        for name, child in properties.items():
            visit(child, [*path, name], name in required)

    visit(schema, [], True)
    return leaves


def _types(schema: JsonObject) -> set[str]:
    declared = schema["type"]
    return {declared} if isinstance(declared, str) else set(declared)


def _sample_value(
    schema: JsonObject,
    rng: random.Random,
    token: str,
    namespace: str,
) -> Any:
    if "enum" in schema:
        return rng.choice(schema["enum"])
    types = _types(schema)
    if "string" in types and "null" in types:
        if rng.random() < 0.25:
            return None
        return f"{namespace}-{token}-{rng.randrange(10_000):04d}"
    if "string" in types:
        if "date" in token:
            return f"2026-11-{rng.randrange(1, 29):02d}"
        return f"{namespace}-{token}-{rng.randrange(10_000):04d}"
    numeric_offset = {"train": 0, "validation": 10_000, "test": 20_000}[namespace]
    if "number" in types:
        return numeric_offset + rng.randrange(1, 1000) / 10
    if "integer" in types:
        return numeric_offset + rng.randrange(1, 100)
    if "boolean" in types:
        return bool(rng.randrange(2))
    raise ValueError(f"unsupported leaf types: {types!r}")


def _make_state(schema: JsonObject, rng: random.Random, namespace: str) -> JsonObject:
    def build(node: JsonObject, token: str) -> Any:
        if "object" not in _types(node):
            return _sample_value(node, rng, token, namespace)
        result: JsonObject = {}
        required = set(node.get("required", []))
        for name, child in node["properties"].items():
            if name in required or rng.random() < 0.5:
                result[name] = build(child, name)
        return result

    return build(schema, "root")


def _lookup(state: JsonObject, tokens: list[str]) -> Any:
    current: Any = state
    for token in tokens:
        if not isinstance(current, dict) or token not in current:
            return _MISSING
        current = current[token]
    return current


def _set_value(state: JsonObject, tokens: list[str], value: Any) -> None:
    current = state
    for token in tokens[:-1]:
        current = current[token]
    current[tokens[-1]] = value


def _delete_value(state: JsonObject, tokens: list[str]) -> None:
    current = state
    for token in tokens[:-1]:
        current = current[token]
    del current[tokens[-1]]


def _changed_value(
    schema: JsonObject,
    previous: Any,
    rng: random.Random,
    token: str,
    namespace: str,
) -> Any:
    for _ in range(20):
        candidate = _sample_value(schema, rng, token, namespace)
        if candidate != previous or type(candidate) is not type(previous):
            return candidate
    # Finite enums can be exhausted only by a malformed one-choice schema.
    raise ValueError(f"could not sample a changed value for {token!r}")


def _choose_template(split: str, index: int) -> str:
    return TEMPLATE_FAMILIES[split][index % len(TEMPLATE_FAMILIES[split])]


def _instruction(
    task_type: str,
    template_id: str,
    label: str,
    value: Any | None = None,
) -> str:
    value_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    templates: dict[str, dict[str, str]] = {
        "train-direct": {
            "replace": "Set {label} to {value}.",
            "add": "Add {label} with value {value}.",
            "remove": "Remove {label}.",
            "noop": "Keep {label} unchanged.",
            "abstain": "Set an unknown field to 1.",
        },
        "train-rephrase": {
            "replace": "Change {label} so it is {value}.",
            "add": "Create {label}; its value is {value}.",
            "remove": "Delete {label}.",
            "noop": "Leave {label} as it is.",
            "abstain": "This request is unclear, so do not change the record.",
        },
        "validation-explicit": {
            "replace": "Please update {label}; its new value is {value}.",
            "add": "Please add {label}; its value is {value}.",
            "remove": "Please take out {label}.",
            "noop": "No update is requested for {label}.",
            "abstain": "Update a nonexistent field to 1.",
        },
        "test-delegated": {
            "replace": "For {label}, use {value}.",
            "add": "For {label}, add the value {value}.",
            "remove": "For {label}, remove the current value.",
            "noop": "For {label}, make no changes.",
            "abstain": "A change may be needed, but the target is not specified.",
        },
    }
    return templates[template_id][task_type].format(label=label, value=value_text)


def _one_example(
    split: str,
    index: int,
    schema_record: JsonObject,
    rng: random.Random,
) -> JsonObject:
    schema = schema_record["schema"]
    state = _make_state(schema, rng, split)
    leaves = _leaf_specs(schema)
    task_type = _TASK_TYPES[index % len(_TASK_TYPES)]
    template_id = _choose_template(split, index)

    existing = [spec for spec in leaves if _lookup(state, spec[0]) is not _MISSING]
    optional = [spec for spec in leaves if not spec[2]]
    label = existing[0][1]["title"]
    patch: list[JsonObject] = []
    decision = "patch"

    if task_type == "replace":
        tokens, field_schema, _ = rng.choice(existing)
        prior = _lookup(state, tokens)
        value = _changed_value(field_schema, prior, rng, tokens[-1], split)
        patch = [{"op": "replace", "path": encode_pointer(tokens), "value": value}]
        label = field_schema["title"]
        instruction = _instruction(task_type, template_id, label, value)
    elif task_type == "add":
        candidates = [spec for spec in optional if _lookup(state, spec[0]) is _MISSING]
        if not candidates:
            tokens, field_schema, _ = optional[0]
            _delete_value(state, tokens)
            candidates = [(tokens, field_schema, False)]
        tokens, field_schema, _ = rng.choice(candidates)
        value = _sample_value(field_schema, rng, tokens[-1], split)
        patch = [{"op": "add", "path": encode_pointer(tokens), "value": value}]
        label = field_schema["title"]
        instruction = _instruction(task_type, template_id, label, value)
    elif task_type == "remove":
        candidates = [spec for spec in optional if _lookup(state, spec[0]) is not _MISSING]
        if not candidates:
            tokens, field_schema, _ = optional[0]
            _set_value(state, tokens, _sample_value(field_schema, rng, tokens[-1], split))
            candidates = [(tokens, field_schema, False)]
        tokens, field_schema, _ = rng.choice(candidates)
        label = field_schema["title"]
        patch = [{"op": "remove", "path": encode_pointer(tokens)}]
        instruction = _instruction(task_type, template_id, label)
    elif task_type == "noop":
        _, field_schema, _ = rng.choice(existing)
        label = field_schema["title"]
        decision = "noop"
        instruction = _instruction(task_type, template_id, label)
    else:
        decision = "abstain"
        instruction = _instruction(task_type, template_id, label)

    # The independent oracle is the sole source of the persisted target state.
    target_state = apply_patch(state, schema, patch)
    return {
        "id": f"synthetic-{split}-{index:06d}",
        "domain": schema_record["domain"],
        "schema_id": schema_record["schema_id"],
        "schema": deepcopy(schema),
        "state": state,
        "instruction": instruction,
        "target_decision": decision,
        "target_patch": patch,
        "target_state": target_state,
        "task_type": task_type,
        "template_id": template_id,
        "source": "synthetic-phase0",
        "split": split,
    }


def generate_examples(examples_per_split: int = 100, seed: int = 20261118) -> list[JsonObject]:
    """Generate split-isolated, reproducible executable examples.

    ``examples_per_split`` must be at least five so every operation / decision type
    appears in each split.  The same inputs always yield byte-equivalent row order.
    """
    if examples_per_split < len(_TASK_TYPES):
        raise ValueError(f"examples_per_split must be at least {len(_TASK_TYPES)}")
    rng = random.Random(seed)
    examples: list[JsonObject] = []
    for split in all_split_names():
        records = schemas_for_split(split)
        for index in range(examples_per_split):
            record = records[index % len(records)]
            examples.append(_one_example(split, index, record, rng))
    assert_split_integrity(examples)
    return examples


def write_jsonl(examples: list[JsonObject], output_path: str | Path) -> str:
    """Write canonical JSONL and return its SHA-256 fingerprint."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            encoded = json.dumps(
                example,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write(encoded)
            handle.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(input_path: str | Path) -> Iterator[JsonObject]:
    """Yield rows from a generated JSONL file, rejecting blank lines."""
    with Path(input_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line in dataset at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"dataset row {line_number} is not an object")
            yield row
