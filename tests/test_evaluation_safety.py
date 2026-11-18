from __future__ import annotations

from patchformer.data import generate_examples
from patchformer.evaluation import evaluate_examples


def test_empty_patch_is_not_counted_as_correct_for_abstention() -> None:
    abstention_row = next(
        row
        for row in generate_examples(examples_per_split=5)
        if row["target_decision"] == "abstain"
    )

    metrics = evaluate_examples([abstention_row], lambda _: {"decision": "noop", "patch": []})

    assert metrics["final_state_accuracy"] == 0.0
    assert metrics["noop_abstain_accuracy"] == 0.0
    assert metrics["patch_exact_match"] == 0.0
