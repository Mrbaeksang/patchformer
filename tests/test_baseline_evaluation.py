from __future__ import annotations

from pathlib import Path

from patchformer.baseline import predict_example
from patchformer.cli import main
from patchformer.data import generate_examples
from patchformer.evaluation import evaluate_examples


def test_deterministic_baseline_solves_the_controlled_fixture() -> None:
    rows = generate_examples(examples_per_split=10, seed=17)
    metrics = evaluate_examples(rows, predict_example)

    assert metrics["n_examples"] == 30
    assert metrics["final_state_accuracy"] == 1.0
    assert metrics["patch_exact_match"] == 1.0
    assert metrics["mutation_validity_rate"] == 1.0


def test_cli_can_generate_then_evaluate_a_fixture(tmp_path: Path) -> None:
    destination = tmp_path / "fixture.jsonl"
    assert main(["generate", "--output", str(destination), "--examples-per-split", "5"]) == 0
    assert destination.exists()
    assert main(["evaluate", "--data", str(destination), "--split", "test"]) == 0
