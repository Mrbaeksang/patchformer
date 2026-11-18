from __future__ import annotations

from copy import deepcopy

import pytest

from patchformer.patch import PatchValidationError, apply_patch, validate_patch
from patchformer.patch.oracle import decode_pointer, encode_pointer

SCHEMA = {
    "type": "object",
    "properties": {
        "account": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "active": {"type": "boolean"},
                "memo": {"type": ["string", "null"]},
                "a/b~c": {"type": "number"},
            },
            "required": ["name", "active"],
            "additionalProperties": False,
        }
    },
    "required": ["account"],
    "additionalProperties": False,
}

STATE = {"account": {"name": "Ada", "active": True, "a/b~c": 1}}


def test_pointer_round_trip_handles_rfc_6901_escaping() -> None:
    tokens = ["account", "a/b~c"]
    pointer = encode_pointer(tokens)
    assert pointer == "/account/a~1b~0c"
    assert decode_pointer(pointer) == tokens


@pytest.mark.parametrize("pointer", ["account/name", "/account/~2", "/account/~"])
def test_invalid_pointers_are_rejected(pointer: str) -> None:
    with pytest.raises(PatchValidationError):
        validate_patch(STATE, SCHEMA, [{"op": "replace", "path": pointer, "value": "Grace"}])


def test_apply_is_a_dry_run_and_accepts_valid_add_replace_remove() -> None:
    before = deepcopy(STATE)
    replaced = apply_patch(
        STATE,
        SCHEMA,
        [{"op": "replace", "path": "/account/name", "value": "Grace"}],
    )
    added = apply_patch(
        STATE,
        SCHEMA,
        [{"op": "add", "path": "/account/memo", "value": None}],
    )
    removed = apply_patch(added, SCHEMA, [{"op": "remove", "path": "/account/memo"}])

    assert STATE == before
    assert replaced["account"]["name"] == "Grace"
    assert added["account"]["memo"] is None
    assert removed == STATE


@pytest.mark.parametrize(
    "patch",
    [
        [{"op": "add", "path": "/account/name", "value": "Grace"}],
        [{"op": "replace", "path": "/account/memo", "value": "Grace"}],
        [{"op": "replace", "path": "/account/active", "value": "true"}],
        [{"op": "remove", "path": "/account/name"}],
        [{"op": "copy", "path": "/account/name"}],
        [
            {"op": "replace", "path": "/account/name", "value": "Grace"},
            {"op": "replace", "path": "/account/active", "value": False},
        ],
    ],
)
def test_unsafe_or_out_of_scope_patches_are_rejected(patch: list[dict[str, object]]) -> None:
    with pytest.raises(PatchValidationError):
        validate_patch(STATE, SCHEMA, patch)


def test_empty_patch_is_a_valid_noop() -> None:
    assert apply_patch(STATE, SCHEMA, []) == STATE
