"""Auditable split metadata for the deterministic synthetic benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .schemas import all_split_names

# Each listed family is exclusive to one split.  IDs are persisted in every row.
TEMPLATE_FAMILIES: dict[str, tuple[str, ...]] = {
    "train": ("train-direct", "train-rephrase"),
    "validation": ("validation-explicit",),
    "test": ("test-delegated",),
}


def split_manifest(examples: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Build the schema/template membership manifest from generated rows."""
    manifest: dict[str, dict[str, set[str]]] = {
        split: {"schema_ids": set(), "template_ids": set(), "domains": set()}
        for split in all_split_names()
    }
    for example in examples:
        split = example["split"]
        if split not in manifest:
            raise ValueError(f"row has unsupported split: {split!r}")
        manifest[split]["schema_ids"].add(example["schema_id"])
        manifest[split]["template_ids"].add(example["template_id"])
        manifest[split]["domains"].add(example["domain"])

    return {
        split: {key: sorted(values) for key, values in values_by_key.items()}
        for split, values_by_key in manifest.items()
    }


def assert_split_integrity(examples: Iterable[dict[str, Any]]) -> None:
    """Fail if any schema, template family, or domain leaks across splits."""
    manifest = split_manifest(examples)
    for membership_key in ("schema_ids", "template_ids", "domains"):
        observed: dict[str, str] = {}
        for split, split_values in manifest.items():
            for item in split_values[membership_key]:
                previous = observed.setdefault(item, split)
                if previous != split:
                    raise AssertionError(
                        f"{membership_key} item {item!r} leaks between {previous!r} and {split!r}"
                    )

    for split, template_ids in TEMPLATE_FAMILIES.items():
        expected = set(template_ids)
        actual = set(manifest[split]["template_ids"])
        if not actual <= expected:
            raise AssertionError(f"unexpected {split} template IDs: {sorted(actual - expected)!r}")
