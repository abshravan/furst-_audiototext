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
    p.add_argument("--no-full-eval", action="store_true",
                   help="Skip the extra evaluation on the FULL labels.csv "
                        "(by default we also report metrics on all 34 examples, "
                        "not just the 10-example val split — useful for spotting "
                        "overfitting to a particular val seed).")
    p.add_argument("--results-log", default=None,
                   help="Write a structured text report to this path: run "
                        "settings, per-example predictions for every eval "
                        "pass, summary tables, and the final optimized prompt. "
                        "Also writes a sibling .csv with one row per prediction.")
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
    baseline_val = evaluate_program(detector, val, verbose=args.verbose,
                                    collect_details=True)
    print(f"[dspy] Baseline eval took {time.time() - t0:.1f}s", file=sys.stderr)
    print()
    print(format_report("BASELINE", baseline_val))

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
    optimized_val = evaluate_program(optimized, val, verbose=args.verbose,
                                     collect_details=True)
    print(f"[dspy] Optimized eval took {time.time() - t0:.1f}s", file=sys.stderr)
    print()
    print(format_report("OPTIMIZED", optimized_val))

    # 7. Save the optimized program for later use.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(out_path))
    print(f"\n[dspy] Saved optimized program to {out_path}", file=sys.stderr)

    # 8. Comparison report on the held-out VAL split.
    print()
    print("============================================================")
    print(f" COMPARISON — VAL SPLIT ({len(val)} examples)")
    print("============================================================")
    print(format_comparison(baseline_val, optimized_val))

    # 9. Optional: also evaluate both programs on the FULL labels.csv.
    #    Train examples are cached after step 5, val examples after step 6,
    #    so this adds zero new GPU work for the OPTIMIZED program — and at
    #    most `len(train)` extra inference calls for the BASELINE.
    baseline_full = optimized_full = None
    if not args.no_full_eval:
        print(f"\n[dspy] Evaluating BASELINE on FULL set ({len(examples)} examples) ...",
              file=sys.stderr)
        t0 = time.time()
        baseline_full = evaluate_program(detector, examples, verbose=args.verbose,
                                         collect_details=True)
        print(f"[dspy] Baseline full-set eval took {time.time() - t0:.1f}s",
              file=sys.stderr)

        print(f"\n[dspy] Evaluating OPTIMIZED on FULL set ({len(examples)} examples) ...",
              file=sys.stderr)
        t0 = time.time()
        optimized_full = evaluate_program(optimized, examples, verbose=args.verbose,
                                          collect_details=True)
        print(f"[dspy] Optimized full-set eval took {time.time() - t0:.1f}s",
              file=sys.stderr)

        print()
        print(format_report("BASELINE  (full)", baseline_full))
        print()
        print(format_report("OPTIMIZED (full)", optimized_full))
        print()
        print("============================================================")
        print(f" COMPARISON — FULL DATASET ({len(examples)} examples)")
        print("============================================================")
        print(format_comparison(baseline_full, optimized_full))
        print()
        print("NOTE: full-dataset metrics include the train split the optimizer")
        print("saw during compilation, so they're optimistic. The VAL-SPLIT")
        print("comparison above is the honest estimate of generalization.")

    # 10. Optional: write a structured report file for analysis.
    if args.results_log:
        _write_results_log(
            log_path=Path(args.results_log),
            args=args,
            optimized=optimized,
            train_size=len(train),
            baseline_val=baseline_val,
            optimized_val=optimized_val,
            baseline_full=baseline_full,
            optimized_full=optimized_full,
        )
        print(f"\n[dspy] Wrote results log to {args.results_log}", file=sys.stderr)
        print(f"[dspy] Wrote per-prediction CSV to "
              f"{Path(args.results_log).with_suffix('.csv')}", file=sys.stderr)

    print()
    stats = backend.cache_stats()
    print(f"[dspy] Cache: {stats['disk_entries']} entries on disk at "
          f"{stats['cache_dir']}", file=sys.stderr)
    return 0


def _write_results_log(
    log_path: Path,
    args,
    optimized,
    train_size: int,
    baseline_val,
    optimized_val,
    baseline_full,
    optimized_full,
) -> None:
    """Dump a human-readable report + a sibling CSV of per-example rows."""
    import csv
    import datetime as _dt

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- text report ----
    lines = []
    lines.append("=" * 78)
    lines.append(" DSPy optimization results")
    lines.append("=" * 78)
    lines.append(f"timestamp     : {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"model         : {args.model}")
    lines.append(f"labels        : {args.labels}")
    lines.append(f"optimizer     : {args.optimizer}")
    if args.optimizer == "gepa":
        lines.append(f"gepa_budget   : {args.gepa_budget}")
        lines.append(f"prompt_model  : {os.environ.get('DSPY_PROMPT_MODEL', '(unset)')}")
    lines.append(f"device/dtype  : {args.device} / {args.dtype}")
    lines.append(f"seed          : {args.seed}")
    lines.append(f"val_ratio     : {args.val_ratio}")
    lines.append(f"train size    : {train_size}")
    lines.append(f"val size      : {baseline_val.n}")
    lines.append("")

    def _section(title: str, result):
        out = [
            "=" * 78,
            f" {title}",
            "=" * 78,
            format_report(title, result),
            "",
            f"{'#':>3s}  {'gold':>4s}  {'pred':>4s}  {'ok':>3s}  {'conf':>4s}  {'audio'}",
        ]
        for d in result.details:
            mark = "✓" if d["correct"] else "✗"
            out.append(
                f"{d['index']:>3d}  {d['gold']:>4s}  {d['pred']:>4s}  "
                f"{mark:>3s}  {d['confidence']:>4d}  {d['audio_path']}"
            )
            if d["reason"]:
                out.append(f"      reason: {d['reason']}")
            if d["error"]:
                out.append(f"      ERROR : {d['error']}")
        out.append("")
        return out

    lines.extend(_section("BASELINE — VAL", baseline_val))
    lines.extend(_section("OPTIMIZED — VAL", optimized_val))
    lines.append("=" * 78)
    lines.append(f" COMPARISON — VAL SPLIT ({baseline_val.n} examples)")
    lines.append("=" * 78)
    lines.append(format_comparison(baseline_val, optimized_val))
    lines.append("")

    if baseline_full and optimized_full:
        lines.extend(_section("BASELINE — FULL", baseline_full))
        lines.extend(_section("OPTIMIZED — FULL", optimized_full))
        lines.append("=" * 78)
        lines.append(f" COMPARISON — FULL DATASET ({baseline_full.n} examples)")
        lines.append("=" * 78)
        lines.append(format_comparison(baseline_full, optimized_full))
        lines.append("")

    lines.append("=" * 78)
    lines.append(" OPTIMIZED PROMPT (as DSPy will reuse on subsequent runs)")
    lines.append("=" * 78)
    instructions = (optimized.predict.signature.instructions or "").strip()
    demos = list(optimized.predict.demos or [])
    lines.append("--- instructions ---")
    lines.append(instructions if instructions else "(empty)")
    lines.append("")
    lines.append(f"--- demos ({len(demos)}) ---")
    for i, d in enumerate(demos, 1):
        lines.append(f"  {i}. audio_path={getattr(d, 'audio_path', '')}  "
                     f"frustration={getattr(d, 'frustration', '')}")
    log_path.write_text("\n".join(lines), encoding="utf-8")

    # ---- CSV (one row per prediction, easy to load in Excel / pandas) ----
    csv_path = log_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "split", "index", "audio_path",
                         "gold", "pred", "correct", "confidence", "reason", "error"])
        passes = [("BASELINE", "val", baseline_val),
                  ("OPTIMIZED", "val", optimized_val)]
        if baseline_full and optimized_full:
            passes += [("BASELINE", "full", baseline_full),
                       ("OPTIMIZED", "full", optimized_full)]
        for phase, split, res in passes:
            for d in res.details:
                writer.writerow([phase, split, d["index"], d["audio_path"],
                                 d["gold"], d["pred"], int(d["correct"]),
                                 d["confidence"], d["reason"], d["error"]])


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
