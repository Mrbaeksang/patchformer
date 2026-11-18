"""Deterministic synthetic datasets and leakage-aware split definitions."""

from .generator import generate_examples, read_jsonl, write_jsonl
from .splits import assert_split_integrity, split_manifest

__all__ = [
    "assert_split_integrity",
    "generate_examples",
    "read_jsonl",
    "split_manifest",
    "write_jsonl",
]
