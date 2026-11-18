import pytest

from patchformer import PatchError, apply_patch


def test_add_remove_replace_and_preserve_unrelated_fields():
    state = {"order": {"status": "pending", "items": 2}, "keep": True}
    result = apply_patch(
        state,
        [
            {"op": "replace", "path": "/order/status", "value": "shipped"},
            {"op": "add", "path": "/order/note", "value": None},
            {"op": "remove", "path": "/order/items"},
        ],
    )
    assert result == {"order": {"status": "shipped", "note": None}, "keep": True}
    assert state["order"] == {"status": "pending", "items": 2}


def test_pointer_escape_and_empty_object_key():
    assert apply_patch({"a/b": {"~key": 1}, "": 0}, [
        {"op": "replace", "path": "/a~1b/~0key", "value": 2},
        {"op": "replace", "path": "/", "value": 3},
    ]) == {"a/b": {"~key": 2}, "": 3}


@pytest.mark.parametrize("operation", [
    {"op": "replace", "path": "/missing", "value": 1},
    {"op": "remove", "path": "/missing"},
    {"op": "add", "path": "/child/missing/key", "value": 1},
    {"op": "replace", "path": "/child/missing", "value": 1},
    {"op": "move", "path": "/x"},
    {"op": "add", "path": "/x~2y", "value": 1},
    {"op": "add", "path": "", "value": 1},
])
def test_invalid_operations_fail_without_mutating_input(operation):
    state = {"child": {}, "x": 0}
    original = {"child": {}, "x": 0}
    with pytest.raises(PatchError):
        apply_patch(state, [operation])
    assert state == original


def test_whole_sequence_is_atomic_from_callers_perspective():
    state = {"x": 1}
    with pytest.raises(PatchError):
        apply_patch(state, [
            {"op": "replace", "path": "/x", "value": 2},
            {"op": "remove", "path": "/absent"},
        ])
    assert state == {"x": 1}


def test_non_object_nested_parent_is_rejected():
    with pytest.raises(PatchError, match="only object-member"):
        apply_patch({"items": [1]}, [{"op": "replace", "path": "/items/0", "value": 2}])
