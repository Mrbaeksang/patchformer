"""Small, dependency-free JSON Patch oracle for Patchformer's V0 scope.

Supports RFC 6902 add, remove, and replace operations over JSON objects.
Array operations and the root pointer are deliberately out of scope.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class PatchError(ValueError):
    """Raised when a patch is malformed or cannot be applied."""


def _decode_pointer(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise PatchError("path must be a non-root JSON Pointer beginning with '/'")
    tokens = pointer[1:].split("/")
    decoded: list[str] = []
    for token in tokens:
        # RFC 6901 only defines ~0 and ~1; reject malformed escapes.
        i = 0
        out = ""
        while i < len(token):
            if token[i] != "~":
                out += token[i]
                i += 1
                continue
            if i + 1 >= len(token) or token[i + 1] not in "01":
                raise PatchError("invalid JSON Pointer escape")
            out += "~" if token[i + 1] == "0" else "/"
            i += 2
        decoded.append(out)
    return decoded


def apply_patch(state: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply supported operations atomically, returning a copied state.

    Only object-member paths are supported. The input is never mutated, including
    when a later operation fails.
    """
    if not isinstance(state, dict) or not isinstance(operations, list):
        raise PatchError("state must be an object and operations must be a list")
    result = deepcopy(state)
    for operation in operations:
        if not isinstance(operation, dict):
            raise PatchError("each operation must be an object")
        op = operation.get("op")
        if op not in {"add", "remove", "replace"}:
            raise PatchError(f"unsupported operation: {op!r}")
        if "path" not in operation:
            raise PatchError("operation is missing path")
        tokens = _decode_pointer(operation["path"])
        parent: Any = result
        for token in tokens[:-1]:
            if not isinstance(parent, dict) or token not in parent:
                raise PatchError("parent path does not exist")
            parent = parent[token]
        key = tokens[-1]
        if not isinstance(parent, dict):
            raise PatchError("only object-member paths are supported")
        if op == "add":
            if "value" not in operation:
                raise PatchError("add operation is missing value")
            parent[key] = deepcopy(operation["value"])
        elif op == "remove":
            if key not in parent:
                raise PatchError("remove path does not exist")
            del parent[key]
        else:
            if "value" not in operation:
                raise PatchError("replace operation is missing value")
            if key not in parent:
                raise PatchError("replace path does not exist")
            parent[key] = deepcopy(operation["value"])
    return result
