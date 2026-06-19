"""Evaluation metrics.

`frustration_metric` is the DSPy-shaped metric (callable returning
True/False) that optimizers use to score predictions during compilation.
Returning a bool (rather than a 0..1 score) makes BootstrapFewShot's
demo filtering straightforward: kept demos are the ones where the model
got the answer right.

`frustration_metric_with_feedback` is GEPA-shaped: it returns a
`dspy.Prediction(score=..., feedback=...)`. GEPA hands the feedback
string to the reflection LM, which uses it to propose instruction
rewrites. Richer feedback => better prompt evolution.

`evaluate_program` computes accuracy / precision / recall / F1 on a
held-out validation set — these are the numbers we report when comparing
the baseline (unoptimized) vs the optimized program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import dspy


# ---------------------------------------------------------------------------
# Per-example metric (the one DSPy optimizers consume)
# ---------------------------------------------------------------------------

def frustration_metric(example, pred, trace=None) -> bool:
    """Strict equality on the binary frustration label.

    DSPy passes (gold_example, model_prediction, trace) and expects a
    bool or float. We return bool — exact match between gold and predicted
    "yes"/"no" after normalization.
    """
    gold = (example.frustration or "").strip().lower()
    actual = (pred.frustration or "").strip().lower()
    return gold == actual


def frustration_metric_with_feedback(
    example, pred, trace=None, pred_name=None, pred_trace=None,
):
    """GEPA-shaped metric. Returns score + natural-language feedback.

    The feedback string is what GEPA's reflection LM reads to reason
    about *why* the prediction was wrong. The more specific we are, the
    better the prompt rewrites become. We surface:
      * gold vs predicted label
      * the model's own reasoning (if it gave one)
      * concrete guidance about which direction the classifier missed
    """
    gold = (example.frustration or "").strip().lower()
    actual = (pred.frustration or "").strip().lower()
    correct = (gold == actual)
    score = 1.0 if correct else 0.0
    reason = getattr(pred, "reason", "") or "(no reason given)"
    confidence = getattr(pred, "confidence", "") or "(no confidence given)"
    audio_path = getattr(example, "audio_path", "(unknown audio)")

    if correct:
        feedback = (
            f"Correct: model labeled '{actual}' for {audio_path}. "
            f"Confidence={confidence}. Reasoning: {reason}"
        )
    elif gold == "yes" and actual == "no":
        feedback = (
            f"FALSE NEGATIVE on {audio_path}. The patient IS frustrated "
            f"but the model labeled 'no'. Model said: '{reason}'. "
            f"The classifier is being too conservative — it is missing "
            f"frustration cues like raised pitch, sighs, sharp word "
            f"choice, repeated complaints, or interruptions. The "
            f"instruction should push the model to weigh subtle acoustic "
            f"cues more heavily and not require explicit angry words."
        )
    elif gold == "no" and actual == "yes":
        feedback = (
            f"FALSE POSITIVE on {audio_path}. The patient is NOT "
            f"frustrated but the model labeled 'yes'. Model said: "
            f"'{reason}'. The classifier is over-triggering — it is "
            f"confusing normal questioning, concern, or assertiveness "
            f"with frustration. The instruction should clarify that "
            f"frustration requires sustained negative affect, not just "
            f"a single emphatic word or an inquisitive tone."
        )
    else:
        feedback = (
            f"Incorrect prediction on {audio_path}: gold={gold} "
            f"pred={actual}. Reasoning: {reason}"
        )
    return dspy.Prediction(score=score, feedback=feedback)


# ---------------------------------------------------------------------------
# Aggregate metrics for the comparison report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricsReport:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
        }


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(gold_labels: List[str], pred_labels: List[str]) -> MetricsReport:
    """Binary metrics treating "yes" (frustrated) as the positive class."""
    if len(gold_labels) != len(pred_labels):
        raise ValueError("gold and pred lists must be the same length")
    tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == "yes" and p == "yes")
    fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == "no" and p == "yes")
    tn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == "no" and p == "no")
    fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == "yes" and p == "no")
    n = len(gold_labels)
    accuracy = _safe_div(tp + tn, n)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return MetricsReport(n=n, accuracy=accuracy, precision=precision,
                         recall=recall, f1=f1,
                         tp=tp, fp=fp, tn=tn, fn=fn)


def evaluate_program(
    program: dspy.Module,
    examples: Iterable[dspy.Example],
    verbose: bool = False,
    collect_details: bool = False,
) -> MetricsReport:
    """Run the program on every example and aggregate metrics.

    Sequential — we don't parallelize because the underlying audio model
    is already eating the GPU/CPU fully, and our backend cache means
    re-evaluations are essentially free.

    If `collect_details=True`, the returned MetricsReport is wrapped in
    an `EvalResult` carrying per-example predictions (gold, pred,
    confidence, reason, audio_path) so the caller can dump them to a
    log file. We keep the default return type backward compatible.
    """
    gold: list[str] = []
    pred: list[str] = []
    details: list[dict] = []
    for i, ex in enumerate(examples, 1):
        predicted = "no"
        confidence = 0
        reason = ""
        error = ""
        try:
            out = program(audio_path=ex.audio_path)
            predicted = out.frustration
            confidence = getattr(out, "confidence", 0) or 0
            reason = getattr(out, "reason", "") or ""
        except Exception as exc:  # noqa: BLE001 - record errors but keep going
            error = repr(exc)
            if verbose:
                print(f"  [{i}] ERROR on {ex.audio_path}: {exc}")
        gold.append(ex.frustration)
        pred.append(predicted)
        details.append({
            "index": i,
            "audio_path": ex.audio_path,
            "gold": gold[-1],
            "pred": pred[-1],
            "correct": gold[-1] == pred[-1],
            "confidence": confidence,
            "reason": reason,
            "error": error,
        })
        if verbose:
            mark = "✓" if gold[-1] == pred[-1] else "✗"
            print(f"  [{i}] {mark} gold={gold[-1]:3s} pred={pred[-1]:3s}  {ex.audio_path}")
    report = compute_metrics(gold, pred)
    if collect_details:
        return EvalResult(report=report, details=details)
    return report


@dataclass(frozen=True)
class EvalResult:
    """Bundles aggregate MetricsReport with per-example prediction rows."""
    report: MetricsReport
    details: list

    # Forward common attributes so call sites can keep using EvalResult
    # interchangeably with MetricsReport.
    @property
    def n(self) -> int: return self.report.n
    @property
    def accuracy(self) -> float: return self.report.accuracy
    @property
    def precision(self) -> float: return self.report.precision
    @property
    def recall(self) -> float: return self.report.recall
    @property
    def f1(self) -> float: return self.report.f1
    @property
    def tp(self) -> int: return self.report.tp
    @property
    def fp(self) -> int: return self.report.fp
    @property
    def tn(self) -> int: return self.report.tn
    @property
    def fn(self) -> int: return self.report.fn
    def as_dict(self) -> dict: return self.report.as_dict()


def format_report(label: str, report: MetricsReport) -> str:
    d = report.as_dict()
    return (
        f"{label} (n={d['n']})\n"
        f"  Accuracy : {d['accuracy']:.3f}\n"
        f"  Precision: {d['precision']:.3f}\n"
        f"  Recall   : {d['recall']:.3f}\n"
        f"  F1       : {d['f1']:.3f}\n"
        f"  Confusion: TP={d['tp']}  FP={d['fp']}  TN={d['tn']}  FN={d['fn']}"
    )


def format_comparison(baseline: MetricsReport, optimized: MetricsReport) -> str:
    def _delta(a: float, b: float) -> str:
        diff = b - a
        pct = (diff / a * 100) if a else float("inf")
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.3f} ({sign}{pct:.1f}%)" if a else f"{sign}{diff:.3f}"
    return (
        "Metric     | Baseline | Optimized | Delta\n"
        "-----------+----------+-----------+----------\n"
        f"Accuracy   | {baseline.accuracy:.3f}    | {optimized.accuracy:.3f}     | "
        f"{_delta(baseline.accuracy, optimized.accuracy)}\n"
        f"Precision  | {baseline.precision:.3f}    | {optimized.precision:.3f}     | "
        f"{_delta(baseline.precision, optimized.precision)}\n"
        f"Recall     | {baseline.recall:.3f}    | {optimized.recall:.3f}     | "
        f"{_delta(baseline.recall, optimized.recall)}\n"
        f"F1         | {baseline.f1:.3f}    | {optimized.f1:.3f}     | "
        f"{_delta(baseline.f1, optimized.f1)}"
    )
