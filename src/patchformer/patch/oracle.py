"""A deliberately small, independent JSON Patch oracle for the Phase 0 task.

This module is the safety boundary for generated labels and baseline evaluation.  It
implements only the V0 contract: object-only documents, at most one RFC 6902
``add``, ``replace``, or ``remove`` operation, and a closed JSON-schema-like schema.
It is intentionally separate from the synthetic instruction generator.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

JsonObject = dict[str, Any]


class PatchValidationError(ValueError):
    """Raised when a proposed V0 patch violates document or schema rules."""


def _fail(message: str) -> None:
    raise PatchValidationError(message)


def decode_pointer(pointer: str) -> list[str]:
    """Decode a non-root RFC 6901 JSON Pointer into object-member tokens."""
    if not isinstance(pointer, str):
        _fail("patch path must be a string")
    if pointer == "":
        _fail("root replacement is outside the Phase 0 task")
    if not pointer.startswith("/"):
        _fail("JSON Pointer must start with '/'")

    tokens: list[str] = []
    for raw_token in pointer[1:].split("/"):
        token_chars: list[str] = []
        index = 0
        while index < len(raw_token):
            character = raw_token[index]
            if character != "~":
                token_chars.append(character)
                index += 1
                continue
            if index + 1 >= len(raw_token) or raw_token[index + 1] not in {"0", "1"}:
                _fail(f"invalid JSON Pointer escape in {pointer!r}")
            token_chars.append("~" if raw_token[index + 1] == "0" else "/")
            index += 2
        tokens.append("".join(token_chars))
    return tokens


def encode_pointer(tokens: list[str]) -> str:
    """Encode object-member tokens as an RFC 6901 JSON Pointer."""
    if not tokens:
        return ""
    return "/" + "/".join(token.replace("~", "~0").replace("/", "~1") for token in tokens)


def _allowed_types(schema: JsonObject) -> set[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return {declared}
    if isinstance(declared, list) and all(isinstance(item, str) for item in declared):
        return set(declared)
    _fail("every schema node must declare a string type or a list of string types")


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _value_matches_schema(value: Any, schema: JsonObject) -> bool:
    allowed_types = _allowed_types(schema)
    if not any(_matches_type(value, type_name) for type_name in allowed_types):
        return False

    if "enum" in schema and value not in schema["enum"]:
        return False

    if isinstance(value, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return False
        if set(value) - set(properties):
            return False
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            return False
        if any(name not in value for name in required):
            return False
        return all(_value_matches_schema(item, properties[name]) for name, item in value.items())
    return True


def _schema_at(schema: JsonObject, tokens: list[str]) -> JsonObject:
    current = schema
    for token in tokens:
        if "object" not in _allowed_types(current):
            _fail(f"schema does not permit nested path segment {token!r}")
        properties = current.get("properties")
        if not isinstance(properties, dict) or token not in properties:
            _fail(f"path is not declared by the schema: {encode_pointer(tokens)!r}")
        child = properties[token]
        if not isinstance(child, dict):
            _fail(f"schema node for {token!r} must be an object")
        current = child
    return current


def _object_at(document: JsonObject, tokens: list[str]) -> JsonObject:
    current: Any = document
    for token in tokens:
        if not isinstance(current, dict):
            _fail(f"path parent is not an object at {token!r}")
        if token not in current:
            _fail(f"path parent does not exist at {token!r}")
        current = current[token]
    if not isinstance(current, dict):
        _fail("Phase 0 patch targets must have an object parent")
    return current


def _validate_operation(
    document: JsonObject,
    schema: JsonObject,
    operation: Any,
) -> tuple[str, list[str]]:
    if not isinstance(operation, dict):
        _fail("each patch operation must be an object")
    op = operation.get("op")
    if op not in {"add", "replace", "remove"}:
        _fail("Phase 0 supports only add, replace, and remove")

    allowed_keys = {"op", "path"} | ({"value"} if op != "remove" else set())
    if set(operation) != allowed_keys:
        _fail(f"{op!r} operation must contain exactly {sorted(allowed_keys)!r}")

    tokens = decode_pointer(operation["path"])
    parent = _object_at(document, tokens[:-1])
    field = tokens[-1]
    field_schema = _schema_at(schema, tokens)
    exists = field in parent

    if op == "add":
        if exists:
            _fail("Phase 0 add must target an absent field; use replace for an existing field")
        if not _value_matches_schema(operation["value"], field_schema):
            _fail("add value does not match the target schema")
    elif op == "replace":
        if not exists:
            _fail("replace target must already exist")
        if not _value_matches_schema(operation["value"], field_schema):
            _fail("replace value does not match the target schema")
    else:
        if not exists:
            _fail("remove target must already exist")
        parent_schema = _schema_at(schema, tokens[:-1])
        required = parent_schema.get("required", [])
        if field in required:
            _fail("required schema fields cannot be removed in Phase 0")

    return op, tokens


def _apply_validated_operation(
    document: JsonObject,
    operation: JsonObject,
    op: str,
    tokens: list[str],
) -> None:
    parent = _object_at(document, tokens[:-1])
    field = tokens[-1]
    if op == "remove":
        del parent[field]
    else:
        parent[field] = deepcopy(operation["value"])


def _validate_and_apply(state: JsonObject, schema: JsonObject, patch: Any) -> JsonObject:
    if not isinstance(state, dict):
        _fail("Phase 0 state must be a JSON object")
    if not isinstance(schema, dict) or not _value_matches_schema(state, schema):
        _fail("input state does not satisfy the supplied closed schema")
    if not isinstance(patch, list):
        _fail("patch must be a list of operations")
    if len(patch) > 1:
        _fail("Phase 0 permits at most one mutation per proposal")

    result = deepcopy(state)
    for raw_operation in patch:
        op, tokens = _validate_operation(result, schema, raw_operation)
        _apply_validated_operation(result, raw_operation, op, tokens)

    if not _value_matches_schema(result, schema):
        _fail("patch result does not satisfy the supplied schema")
    return result


def validate_patch(state: JsonObject, schema: JsonObject, patch: Any) -> None:
    """Raise :class:`PatchValidationError` unless ``patch`` is a valid V0 dry run."""
    _validate_and_apply(state, schema, patch)


def apply_patch(state: JsonObject, schema: JsonObject, patch: Any) -> JsonObject:
    """Validate then return a deep-copied dry-run result without mutating ``state``."""
    return _validate_and_apply(state, schema, patch)
