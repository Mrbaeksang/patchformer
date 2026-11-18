"""Command-line entry points for the reproducible Phase 0 baseline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from patchformer.baseline import predict_example
from patchformer.data import (
    assert_split_integrity,
    generate_examples,
    read_jsonl,
    split_manifest,
    write_jsonl,
)
from patchformer.evaluation import evaluate_examples


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _generate(args: argparse.Namespace) -> int:
    examples = generate_examples(args.examples_per_split, args.seed)
    fingerprint = write_jsonl(examples, args.output)
    _print_json(
        {
            "command": "generate",
            "output": str(args.output),
            "seed": args.seed,
            "examples_per_split": args.examples_per_split,
            "n_examples": len(examples),
            "sha256": fingerprint,
            "split_manifest": split_manifest(examples),
        }
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    examples = list(read_jsonl(args.data))
    assert_split_integrity(examples)
    if args.split != "all":
        examples = [row for row in examples if row["split"] == args.split]
    if not examples:
        raise ValueError("no examples selected for evaluation")
    _print_json(
        {
            "command": "evaluate",
            "data": str(args.data),
            "split": args.split,
            "baseline": "deterministic-schema-title-parser",
            "metrics": evaluate_examples(examples, predict_example),
        }
    )
    return 0


def _smoke(args: argparse.Namespace) -> int:
    examples = generate_examples(args.examples_per_split, args.seed)
    assert_split_integrity(examples)
    fingerprint = write_jsonl(examples, args.output)
    metrics = evaluate_examples(examples, predict_example)
    _print_json(
        {
            "command": "smoke",
            "cpu_only": True,
            "output": str(args.output),
            "sha256": fingerprint,
            "split_manifest": split_manifest(examples),
            "metrics": metrics,
        }
    )
    return 0


def _cuda_smoke(_: argparse.Namespace) -> int:
    try:
        import torch
    except ModuleNotFoundError:
        _print_json(
            {
                "command": "cuda-smoke",
                "status": "unavailable",
                "reason": "PyTorch is not installed. Install the pinned cuda extra first.",
            }
        )
        return 2
    if not torch.cuda.is_available():
        _print_json(
            {
                "command": "cuda-smoke",
                "status": "unavailable",
                "torch_version": torch.__version__,
                "reason": "torch.cuda.is_available() returned false.",
            }
        )
        return 3

    device = torch.device("cuda")
    left = torch.arange(256, device=device, dtype=torch.float32).reshape(16, 16)
    product = left @ left.T
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(product).all().item()):
        raise RuntimeError("CUDA smoke tensor contains non-finite values")
    _print_json(
        {
            "command": "cuda-smoke",
            "status": "ok",
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
            "checksum": float(product.sum().item()),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patchformer Phase 0 research commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="create a deterministic synthetic JSONL dataset"
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--examples-per-split", type=int, default=100)
    generate.add_argument("--seed", type=int, default=20261118)
    generate.set_defaults(handler=_generate)

    evaluate = subparsers.add_parser("evaluate", help="evaluate the deterministic baseline")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--split", choices=("all", "train", "validation", "test"), default="test")
    evaluate.set_defaults(handler=_evaluate)

    smoke = subparsers.add_parser(
        "smoke", help="run CPU-only generation, integrity, and baseline smoke checks"
    )
    smoke.add_argument("--output", type=Path, default=Path("data/generated/phase0-smoke.jsonl"))
    smoke.add_argument("--examples-per-split", type=int, default=10)
    smoke.add_argument("--seed", type=int, default=20261118)
    smoke.set_defaults(handler=_smoke)

    cuda_smoke = subparsers.add_parser(
        "cuda-smoke", help="verify PyTorch can execute a CUDA tensor operation"
    )
    cuda_smoke.set_defaults(handler=_cuda_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, AssertionError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
