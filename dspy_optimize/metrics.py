"""Evaluation metrics.

`frustration_metric` is the DSPy-shaped metric (callable returning
True/False) that optimizers use to score predictions during compilation.
Returning a bool (rather than a 0..1 score) makes BootstrapFewShot's
demo filtering straightforward: kept demos are the ones where the model
got the answer right.

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
) -> MetricsReport:
    """Run the program on every example and aggregate metrics.

    Sequential — we don't parallelize because the underlying audio model
    is already eating the GPU/CPU fully, and our backend cache means
    re-evaluations are essentially free.
    """
    gold: list[str] = []
    pred: list[str] = []
    for i, ex in enumerate(examples, 1):
        try:
            out = program(audio_path=ex.audio_path)
            predicted = out.frustration
        except Exception as exc:  # noqa: BLE001 - record errors but keep going
            predicted = "no"
            if verbose:
                print(f"  [{i}] ERROR on {ex.audio_path}: {exc}")
        gold.append(ex.frustration)
        pred.append(predicted)
        if verbose:
            mark = "✓" if gold[-1] == pred[-1] else "✗"
            print(f"  [{i}] {mark} gold={gold[-1]:3s} pred={pred[-1]:3s}  {ex.audio_path}")
    return compute_metrics(gold, pred)


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
