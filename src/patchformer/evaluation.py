"""End-to-end metrics for dry-run structured mutation proposals."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from patchformer.patch import PatchValidationError, apply_patch

JsonObject = dict[str, Any]
Predictor = Callable[[JsonObject], JsonObject]
_MISSING = object()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _label(decision: str, patch: list[JsonObject]) -> str:
    return patch[0]["op"] if decision == "patch" and patch else decision


def _path(patch: list[JsonObject]) -> Any:
    return patch[0].get("path") if patch else _MISSING


def _value(patch: list[JsonObject]) -> Any:
    return patch[0].get("value", _MISSING) if patch else _MISSING


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def evaluate_examples(examples: Iterable[JsonObject], predictor: Predictor) -> JsonObject:
    """Evaluate proposals by decision, patch, and independently applied final state.

    For abstention-labelled rows, a final state is counted correct only when the
    predictor abstains; silently returning an empty patch is not treated as a safe
    substitute.  Patches are always dry-run through the independent oracle.
    """
    totals = {
        "examples": 0,
        "final_state_correct": 0,
        "patch_exact_match": 0,
        "operation_correct": 0,
        "path_correct": 0,
        "path_evaluable": 0,
        "typed_value_correct": 0,
        "typed_value_evaluable": 0,
        "valid_mutation": 0,
        "mutation_evaluable": 0,
        "safe_decision_correct": 0,
        "safe_decision_evaluable": 0,
    }

    for example in examples:
        totals["examples"] += 1
        target_decision = example["target_decision"]
        target_patch = example["target_patch"]
        prediction = predictor(example)
        predicted_decision = prediction.get("decision")
        predicted_patch = prediction.get("patch")
        supported_decision = predicted_decision in {"patch", "noop", "abstain"}
        if not supported_decision or not isinstance(predicted_patch, list):
            predicted_decision, predicted_patch = "abstain", []

        target_label = _label(target_decision, target_patch)
        predicted_label = _label(predicted_decision, predicted_patch)
        totals["operation_correct"] += target_label == predicted_label

        if target_decision == "patch":
            totals["path_evaluable"] += 1
            totals["path_correct"] += _path(target_patch) == _path(predicted_patch)
            if "value" in target_patch[0]:
                totals["typed_value_evaluable"] += 1
                totals["typed_value_correct"] += _value(target_patch) == _value(predicted_patch)

        if target_decision in {"noop", "abstain"}:
            totals["safe_decision_evaluable"] += 1
            totals["safe_decision_correct"] += target_decision == predicted_decision

        same_decision = target_decision == predicted_decision
        same_patch = _canonical(target_patch) == _canonical(predicted_patch)
        totals["patch_exact_match"] += same_decision and same_patch

        if predicted_decision in {"patch", "noop"}:
            totals["mutation_evaluable"] += 1
            try:
                actual_state = apply_patch(example["state"], example["schema"], predicted_patch)
                is_valid = True
            except (PatchValidationError, TypeError, KeyError):
                actual_state = None
                is_valid = False
            totals["valid_mutation"] += is_valid
        else:
            actual_state = None
            is_valid = False

        if target_decision == "abstain":
            totals["final_state_correct"] += predicted_decision == "abstain"
        elif is_valid and actual_state == example["target_state"]:
            totals["final_state_correct"] += 1

    result = {
        "n_examples": totals["examples"],
        "final_state_accuracy": _ratio(totals["final_state_correct"], totals["examples"]),
        "patch_exact_match": _ratio(totals["patch_exact_match"], totals["examples"]),
        "operation_accuracy": _ratio(totals["operation_correct"], totals["examples"]),
        "path_accuracy": _ratio(totals["path_correct"], totals["path_evaluable"]),
        "path_evaluable_examples": totals["path_evaluable"],
        "typed_value_accuracy": _ratio(
            totals["typed_value_correct"], totals["typed_value_evaluable"]
        ),
        "typed_value_evaluable_examples": totals["typed_value_evaluable"],
        "mutation_validity_rate": _ratio(totals["valid_mutation"], totals["mutation_evaluable"]),
        "mutation_evaluable_examples": totals["mutation_evaluable"],
        "noop_abstain_accuracy": _ratio(
            totals["safe_decision_correct"], totals["safe_decision_evaluable"]
        ),
        "noop_abstain_evaluable_examples": totals["safe_decision_evaluable"],
    }
    return result
