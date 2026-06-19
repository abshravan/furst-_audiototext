"""Main training entrypoint: optimize the frustration-detection prompt.

Usage:

    python -m dspy_optimize.optimize \\
        --model /path/to/MOSS-Audio-4B-Instruct \\
        --labels dspy_optimize/labels.csv \\
        --optimizer bootstrap \\
        --output dspy_optimize/optimized_program.json

Defaults: BootstrapFewShot, val_ratio=0.3, max_bootstrapped_demos=3.

Why BootstrapFewShot is the default:
  * Needs no second LM (MIPROv2 and GEPA both require one).
  * Quickly tries small numbers of demos, evaluates with our metric,
    and keeps the best subset. Cheap enough to run on CPU.

Why you might switch to --optimizer mipro:
  * MIPROv2 *rewrites the instruction* (the part the audio model can
    actually use). For audio-input tasks where demos are inert (the
    model can't re-hear them), instruction optimization is the real
    win. Cost: needs a text LM (configure via DSPY_PROMPT_MODEL env
    var, e.g. `openai/gpt-4o-mini`) and runs many more trials.

Why you might switch to --optimizer gepa (recommended for small sets):
  * GEPA evolves the instruction by *reflecting* on failures with a
    reflection LM that reads natural-language feedback per example. It
    typically beats MIPROv2 on small datasets because it learns from
    why each prediction was wrong, not just whether it was wrong.
    Cost: needs DSPY_PROMPT_MODEL and the matching API key, and is
    the most expensive of the three (many reflection-LM calls).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import dspy

from .backend import AudioInferenceBackend
from .dataset import load_examples, stratified_split
from .metrics import (
    evaluate_program,
    format_comparison,
    format_report,
    frustration_metric,
    frustration_metric_with_feedback,
)
from .module import FrustrationDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Path to MOSS-Audio weights directory.")
    p.add_argument("--labels", required=True,
                   help="CSV with audio_path,frustration columns.")
    p.add_argument("--output", default="dspy_optimize/optimized_program.json",
                   help="Where to save the optimized program JSON.")
    p.add_argument("--optimizer", choices=["bootstrap", "mipro", "gepa"],
                   default="bootstrap",
                   help="Which DSPy optimizer to use.")
    p.add_argument("--max-demos", type=int, default=3,
                   help="Max few-shot demos to include (bootstrap only). "
                        "Keep low for audio because demos can't include audio data.")
    p.add_argument("--gepa-budget", default="light",
                   choices=["light", "medium", "heavy"],
                   help="GEPA search budget. 'light' = ~few dozen reflection "
                        "calls; 'heavy' = hundreds. Cost scales with reflection-LM "
                        "tokens.")
    p.add_argument("--val-ratio", type=float, default=0.3,
                   help="Fraction of dataset reserved for validation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "float32", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--verbose", action="store_true",
                   help="Print per-example results during evaluation.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Build the audio backend (loads model once, ~30s).
    print(f"[dspy] Loading MOSS-Audio from {args.model} ...", file=sys.stderr)
    backend = AudioInferenceBackend(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    # 2. Load and split the dataset.
    examples = load_examples(args.labels)
    if len(examples) < 4:
        raise SystemExit(f"Need at least 4 valid examples; got {len(examples)}.")
    train, val = stratified_split(examples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[dspy] Loaded {len(examples)} examples "
          f"({sum(1 for e in examples if e.frustration == 'yes')} yes / "
          f"{sum(1 for e in examples if e.frustration == 'no')} no). "
          f"Train={len(train)}  Val={len(val)}", file=sys.stderr)

    # 3. DSPy requires *some* LM in settings even if our Module bypasses it.
    #    Use a no-op for bootstrap; for mipro and gepa, swap in a real text LM.
    reflection_lm = None
    if args.optimizer == "bootstrap":
        dspy.settings.configure(lm=_NullLM())
    else:
        prompt_model_name = os.environ.get("DSPY_PROMPT_MODEL")
        if not prompt_model_name:
            raise SystemExit(
                f"{args.optimizer.upper()} needs a text LM. "
                "Set DSPY_PROMPT_MODEL (e.g. 'openai/gpt-4o-mini') and ensure "
                "the matching API key env var is set (OPENAI_API_KEY, etc.)."
            )
        prompt_lm = dspy.LM(prompt_model_name)
        dspy.settings.configure(lm=prompt_lm)
        # GEPA wants a separate handle for the reflection LM (it can be the
        # same model, but DSPy keeps them as distinct knobs). Allow override
        # via DSPY_REFLECTION_MODEL for users who want a stronger reflector.
        if args.optimizer == "gepa":
            reflection_model_name = os.environ.get(
                "DSPY_REFLECTION_MODEL", prompt_model_name,
            )
            reflection_lm = dspy.LM(
                reflection_model_name, temperature=1.0, max_tokens=8000,
            )

    detector = FrustrationDetector(backend)

    # 4. Baseline eval — the unoptimized program with the original docstring.
    print(f"\n[dspy] Evaluating BASELINE on val set ({len(val)} examples) ...",
          file=sys.stderr)
    t0 = time.time()
    baseline_report = evaluate_program(detector, val, verbose=args.verbose)
    print(f"[dspy] Baseline eval took {time.time() - t0:.1f}s", file=sys.stderr)
    print()
    print(format_report("BASELINE", baseline_report))

    # 5. Optimize.
    print(f"\n[dspy] Running {args.optimizer.upper()} optimization on "
          f"{len(train)} train examples ...", file=sys.stderr)
    t0 = time.time()
    if args.optimizer == "bootstrap":
        from dspy.teleprompt import BootstrapFewShot
        optimizer = BootstrapFewShot(
            metric=frustration_metric,
            max_bootstrapped_demos=args.max_demos,
            max_labeled_demos=args.max_demos,
        )
        optimized = optimizer.compile(detector, trainset=train)
    elif args.optimizer == "mipro":
        from dspy.teleprompt import MIPROv2
        optimizer = MIPROv2(
            metric=frustration_metric,
            auto="light",  # 'light' for small datasets; 'medium'/'heavy' cost more
        )
        optimized = optimizer.compile(
            detector,
            trainset=train,
            valset=val,
            requires_permission_to_run=False,
        )
    else:  # gepa
        # GEPA learns by reflecting on per-example feedback strings. It
        # tries instruction variants, scores them with our metric, keeps
        # a Pareto frontier of candidates, and asks the reflection LM to
        # propose new instructions based on failures it saw.
        try:
            from dspy.teleprompt import GEPA
        except ImportError as exc:
            raise SystemExit(
                "GEPA requires dspy>=2.6 (preferably >=3.0). "
                f"Current import failed: {exc}. Run: pip install -U 'dspy-ai>=3.0'"
            ) from exc
        optimizer = GEPA(
            metric=frustration_metric_with_feedback,
            auto=args.gepa_budget,
            reflection_lm=reflection_lm,
        )
        optimized = optimizer.compile(
            detector,
            trainset=train,
            valset=val,
        )
    print(f"[dspy] Optimization took {time.time() - t0:.1f}s", file=sys.stderr)

    # 6. Optimized eval.
    print(f"\n[dspy] Evaluating OPTIMIZED on val set ({len(val)} examples) ...",
          file=sys.stderr)
    t0 = time.time()
    optimized_report = evaluate_program(optimized, val, verbose=args.verbose)
    print(f"[dspy] Optimized eval took {time.time() - t0:.1f}s", file=sys.stderr)
    print()
    print(format_report("OPTIMIZED", optimized_report))

    # 7. Save the optimized program for later use.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(out_path))
    print(f"\n[dspy] Saved optimized program to {out_path}", file=sys.stderr)

    # 8. Comparison report.
    print()
    print("============================================================")
    print(" COMPARISON")
    print("============================================================")
    print(format_comparison(baseline_report, optimized_report))
    print()
    stats = backend.cache_stats()
    print(f"[dspy] Cache: {stats['disk_entries']} entries on disk at "
          f"{stats['cache_dir']}", file=sys.stderr)
    return 0


class _NullLM(dspy.LM):
    """Placeholder LM. Our Module overrides forward() and never invokes this.

    DSPy still requires *something* in `dspy.settings.lm` even when the
    module never calls through it (BootstrapFewShot validates the LM
    exists). Returning empty completions keeps DSPy happy.
    """

    def __init__(self) -> None:
        super().__init__(model="null")

    def __call__(self, prompt=None, messages=None, **kwargs):  # noqa: D401
        return [""]

    def basic_request(self, prompt: str, **kwargs):
        return ""


if __name__ == "__main__":
    sys.exit(main())
