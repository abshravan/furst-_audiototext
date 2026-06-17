"""Standalone evaluation script.

Useful for checking a saved optimized program against a fresh held-out
set, or for re-running the baseline evaluation without re-training.

Usage:

    # Evaluate the optimized program saved by `optimize.py`:
    python -m dspy_optimize.evaluate \\
        --model /path/to/MOSS-Audio-4B-Instruct \\
        --labels dspy_optimize/labels.csv \\
        --program dspy_optimize/optimized_program.json

    # Or evaluate the unoptimized baseline:
    python -m dspy_optimize.evaluate \\
        --model /path/to/MOSS-Audio-4B-Instruct \\
        --labels dspy_optimize/labels.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dspy

from .backend import AudioInferenceBackend
from .dataset import load_examples, stratified_split
from .metrics import evaluate_program, format_report
from .module import FrustrationDetector
from .optimize import _NullLM  # reuse the placeholder LM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--program", default=None,
                   help="Path to a saved optimized program JSON. "
                        "Omit to evaluate the unoptimized baseline.")
    p.add_argument("--val-ratio", type=float, default=0.3,
                   help="If splitting, fraction reserved for validation. "
                        "Use 1.0 to evaluate on the whole file.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "float32", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dspy.settings.configure(lm=_NullLM())

    backend = AudioInferenceBackend(
        model_path=args.model, device=args.device, dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    detector = FrustrationDetector(backend)

    if args.program:
        prog_path = Path(args.program)
        if not prog_path.is_file():
            raise SystemExit(f"Program file not found: {prog_path}")
        detector.load(str(prog_path))
        label = f"OPTIMIZED ({prog_path.name})"
    else:
        label = "BASELINE"

    examples = load_examples(args.labels)
    if args.val_ratio >= 1.0:
        val = examples
    else:
        _, val = stratified_split(examples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[eval] Evaluating {label} on {len(val)} examples ...", file=sys.stderr)

    report = evaluate_program(detector, val, verbose=args.verbose)
    print()
    print(format_report(label, report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
