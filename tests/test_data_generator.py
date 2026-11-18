from __future__ import annotations

from patchformer.data import assert_split_integrity, generate_examples, split_manifest
from patchformer.patch import apply_patch


def test_generation_is_deterministic_and_oracle_executable() -> None:
    first = generate_examples(examples_per_split=10, seed=73)
    second = generate_examples(examples_per_split=10, seed=73)

    assert first == second
    assert len(first) == 30
    assert_split_integrity(first)
    for row in first:
        assert apply_patch(row["state"], row["schema"], row["target_patch"]) == row["target_state"]
        assert row["target_decision"] in {"patch", "noop", "abstain"}


def test_manifest_has_no_schema_template_or_domain_leakage() -> None:
    rows = generate_examples(examples_per_split=10)
    manifest = split_manifest(rows)

    splits = ("train", "validation", "test")
    for key in ("schema_ids", "template_ids", "domains"):
        train, validation, test = (set(manifest[split][key]) for split in splits)
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)


def _non_boolean_scalar_values(value: object) -> set[object]:
    if isinstance(value, dict):
        return set().union(*(_non_boolean_scalar_values(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_non_boolean_scalar_values(item) for item in value))
    if value is None or isinstance(value, bool):
        return set()
    return {value}


def test_non_boolean_values_are_held_out_between_splits() -> None:
    rows = generate_examples(examples_per_split=20, seed=31)
    split_values = {
        split: set().union(
            *(_non_boolean_scalar_values(row["state"]) for row in rows if row["split"] == split)
        )
        for split in ("train", "validation", "test")
    }
    assert split_values["train"].isdisjoint(split_values["validation"])
    assert split_values["train"].isdisjoint(split_values["test"])
    assert split_values["validation"].isdisjoint(split_values["test"])


def test_each_split_contains_all_phase0_decision_types() -> None:
    rows = generate_examples(examples_per_split=10)
    for split in ("train", "validation", "test"):
        task_types = {row["task_type"] for row in rows if row["split"] == split}
        assert task_types == {"replace", "add", "remove", "noop", "abstain"}
