"""CLI script for running RAGAS evaluation on a JSON test set.

The test set JSON must be a list of objects with keys:
    question, answer, contexts (list of str), ground_truth

Usage::

    python scripts/evaluate.py --dataset data/evaluation/testset.json
    python scripts/evaluate.py --dataset data/evaluation/testset.json --output results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from evaluation.dataset_builder import EvalSample
from services.evaluation_service import EvaluationService

setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on a test set.")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the evaluation JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write scores as JSON (e.g. results.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()

    if not dataset_path.is_file():
        print(f"ERROR: '{dataset_path}' does not exist.")
        sys.exit(1)

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = [EvalSample(**item) for item in raw]
    print(f"Loaded {len(samples)} evaluation sample(s) from '{dataset_path.name}'.")

    t0 = time.perf_counter()
    service = EvaluationService()
    scores = service.run(samples)
    elapsed = round(time.perf_counter() - t0, 2)

    print(f"
RAGAS Evaluation Results ({elapsed}s):")
    print("-" * 40)
    for metric, score in scores.items():
        bar = "#" * int(score * 20)
        print(f"  {metric:<35} {score:.4f}  [{bar:<20}]")
    print("-" * 40)

    if args.output:
        args.output.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"
Scores written to '{args.output}'.")


if __name__ == "__main__":
    main()
