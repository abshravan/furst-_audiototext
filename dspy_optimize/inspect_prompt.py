"""Inspect the actual prompt DSPy ended up sending to the model.

After optimization, DSPy mutates `program.predict.signature.instructions`
(MIPROv2) and/or `program.predict.demos` (BootstrapFewShot). This script
loads a saved program and prints both.

Usage:

    python -m dspy_optimize.inspect_prompt \\
        --program dspy_optimize/optimized_program.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dspy

from .backend import AudioInferenceBackend  # noqa: F401  (for module init)
from .module import FrustrationDetector, build_prompt
from .optimize import _NullLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--program", required=True,
                   help="Path to the saved optimized program JSON.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prog_path = Path(args.program)
    if not prog_path.is_file():
        raise SystemExit(f"Program file not found: {prog_path}")

    dspy.settings.configure(lm=_NullLM())
    # Pass a None backend — we won't call forward(), only inspect state.
    detector = FrustrationDetector(backend=None)  # type: ignore[arg-type]
    detector.load(str(prog_path))

    instructions = (detector.predict.signature.instructions or "").strip()
    demos = list(detector.predict.demos or [])
    print("=" * 70)
    print(" OPTIMIZED INSTRUCTIONS")
    print("=" * 70)
    print(instructions if instructions else "(empty)")
    print()
    print("=" * 70)
    print(f" DEMOS ({len(demos)})")
    print("=" * 70)
    if not demos:
        print("(none)")
    for i, d in enumerate(demos, 1):
        print(f"\n--- demo {i} ---")
        for field in ("audio_path", "frustration", "confidence", "reason"):
            val = getattr(d, field, None)
            if val is not None:
                print(f"  {field}: {val}")
    print()
    print("=" * 70)
    print(" FULL PROMPT TEMPLATE (as sent to MOSS-Audio)")
    print("=" * 70)
    print(build_prompt(instructions, demos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
