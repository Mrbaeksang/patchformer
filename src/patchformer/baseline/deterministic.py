"""Transparent rule baseline for validating the Phase 0 benchmark plumbing.

It only understands the generator's controlled language families and schema titles.
It is deliberately conservative: an unrecognised target, malformed literal, or
invalid mutation becomes ``abstain`` rather than an edit.  It is not an ML model and
must not be used as a production language interpreter.
"""

from __future__ import annotations

import json
import re
from typing import Any

from patchformer.patch import PatchValidationError, apply_patch
from patchformer.patch.oracle import encode_pointer

JsonObject = dict[str, Any]
_MISSING = object()


def _types(schema: JsonObject) -> set[str]:
    declared = schema["type"]
    return {declared} if isinstance(declared, str) else set(declared)


def _labelled_leaves(schema: JsonObject) -> list[tuple[str, list[str]]]:
    output: list[tuple[str, list[str]]] = []

    def visit(node: JsonObject, path: list[str]) -> None:
        if "object" not in _types(node):
            title = node.get("title")
            if isinstance(title, str):
                output.append((title, path))
            return
        for name, child in node["properties"].items():
            visit(child, [*path, name])

    visit(schema, [])
    return sorted(output, key=lambda item: len(item[0]), reverse=True)


def _lookup(state: JsonObject, tokens: list[str]) -> Any:
    current: Any = state
    for token in tokens:
        if not isinstance(current, dict) or token not in current:
            return _MISSING
        current = current[token]
    return current


def _required_at_path(schema: JsonObject, tokens: list[str]) -> bool:
    node = schema
    for index, token in enumerate(tokens):
        if index == len(tokens) - 1:
            return token in node.get("required", [])
        node = node["properties"][token]
    return False


def _find_label(instruction: str, schema: JsonObject) -> tuple[str, list[str]] | None:
    normalised = instruction.lower()
    for label, tokens in _labelled_leaves(schema):
        if label.lower() in normalised:
            return label, tokens
    return None


def _extract_json_literal(instruction: str) -> Any:
    lower = instruction.lower()
    patterns = (
        r"\bvalue is\s+(.+?)\.\s*$",
        r"\bwith value\s+(.+?)\.\s*$",
        r"\bthe value\s+(.+?)\.\s*$",
        r"\buse\s+(.+?)\.\s*$",
        r"\bto\s+(.+?)\.\s*$",
        r"\bis\s+(.+?)\.\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, lower, flags=re.IGNORECASE)
        if not match:
            continue
        literal = instruction[match.start(1) : match.end(1)]
        try:
            value, end = json.JSONDecoder().raw_decode(literal)
        except json.JSONDecodeError:
            continue
        if literal[end:].strip():
            continue
        return value
    return _MISSING


def _abstain() -> JsonObject:
    return {"decision": "abstain", "patch": []}


def predict_example(example: JsonObject) -> JsonObject:
    """Return a schema-checked patch proposal or a conservative abstention."""
    instruction = example["instruction"]
    state = example["state"]
    schema = example["schema"]
    inputs_are_objects = all(isinstance(value, dict) for value in (state, schema))
    if not inputs_are_objects or not isinstance(instruction, str):
        return _abstain()

    text = instruction.lower()
    abstention_markers = ("unknown field", "nonexistent", "unclear", "ambiguous", "not specified")
    if any(marker in text for marker in abstention_markers):
        return _abstain()

    identified = _find_label(instruction, schema)
    if identified is None:
        return _abstain()
    _, tokens = identified

    if any(marker in text for marker in ("unchanged", "as it is", "no update", "no changes")):
        return {"decision": "noop", "patch": []}

    path = encode_pointer(tokens)
    if any(marker in text for marker in ("remove", "delete", "take out")):
        if _lookup(state, tokens) is _MISSING or _required_at_path(schema, tokens):
            return _abstain()
        prediction = {"decision": "patch", "patch": [{"op": "remove", "path": path}]}
    else:
        value = _extract_json_literal(instruction)
        if value is _MISSING:
            return _abstain()
        is_add = any(marker in text for marker in ("add", "create"))
        exists = _lookup(state, tokens) is not _MISSING
        if is_add and exists:
            return _abstain()
        if not is_add and not exists:
            return _abstain()
        op = "add" if is_add else "replace"
        prediction = {"decision": "patch", "patch": [{"op": op, "path": path, "value": value}]}

    try:
        apply_patch(state, schema, prediction["patch"])
    except PatchValidationError:
        return _abstain()
    return prediction
