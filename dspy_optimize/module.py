"""The DSPy Module that bridges DSPy's optimization machinery with our
audio inference function.

Why a custom Module instead of `dspy.Predict(FrustrationSignature)`?
Because `dspy.Predict` would route through `dspy.settings.lm` (a text-
in/text-out LM), but MOSS-Audio is audio-in/text-out. We override
`forward()` to:

  1. Read the *currently active* instructions + demos from
     `self.predict.signature` and `self.predict.demos`. DSPy mutates
     these during optimization — that's how it propagates the
     "optimized prompt" into our forward pass.
  2. Compose a single prompt string that includes the optimized
     instruction and any text-form demos.
  3. Call our `AudioInferenceBackend` with `(audio_path, prompt)`.
  4. Parse the raw model output into structured fields.
  5. Wrap them in `dspy.Prediction` so DSPy can compute the metric.

Caveat about demos with audio inputs: BootstrapFewShot inserts past
predictions as demos. The model sees their TEXT (audio_path, label,
reason) but not the audio itself. So demos mainly help the model lock
onto the *output format*, not learn from audio→label mappings. If you
want true instruction-level optimization, use MIPROv2.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import dspy

from .backend import AudioInferenceBackend
from .signature import FrustrationSignature


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_frustration_output(raw: str) -> dict:
    """Pull frustration/confidence/reason out of a messy model output.

    Strategy:
      1. Try `json.loads` on the whole string.
      2. If that fails, search for the first {...} block and parse it.
      3. If that fails, scan the raw text for yes/no keywords.

    Always returns a dict with the three keys present and sane types.
    `frustration` is normalized to "yes" or "no".
    """
    raw = (raw or "").strip()
    data: dict[str, Any] = {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = {}

    frustration = str(data.get("frustration", "")).lower().strip()
    if frustration not in ("yes", "no"):
        rl = raw.lower()
        if "not frustrated" in rl or "no frustration" in rl:
            frustration = "no"
        elif "frustrat" in rl and re.search(r"\byes\b|\btrue\b", rl):
            frustration = "yes"
        else:
            frustration = "no"  # conservative default when ambiguous

    try:
        confidence = int(float(data.get("confidence", 0)))
        confidence = max(0, min(100, confidence))
    except (TypeError, ValueError):
        confidence = 0

    reason = str(data.get("reason", "")).strip()[:1000]

    return {"frustration": frustration, "confidence": confidence, "reason": reason}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _format_demos(demos: list) -> str:
    """Render past predictions as text examples in the prompt.

    The model can't hear demo audios, so we surface only the labels and
    reasoning. This still helps lock in the JSON output format.
    """
    if not demos:
        return ""
    lines = ["", "Reference examples of past judgments (audio not shown):"]
    for i, d in enumerate(demos, 1):
        label = getattr(d, "frustration", "")
        conf = getattr(d, "confidence", "")
        reason = getattr(d, "reason", "")
        lines.append(f'  {i}. {{"frustration": "{label}", "confidence": {conf}, '
                     f'"reason": "{reason}"}}')
    return "\n".join(lines)


def build_prompt(instructions: str, demos: list) -> str:
    """Compose the full text prompt sent to MOSS-Audio."""
    pieces = [instructions.strip()]
    demo_block = _format_demos(demos)
    if demo_block:
        pieces.append(demo_block)
    pieces.append(
        "\nNow analyze the provided audio and return strict JSON with "
        'fields frustration ("yes" or "no"), confidence (0-100), and reason.'
    )
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# The DSPy module
# ---------------------------------------------------------------------------

# Module-level call counter so trace lines have a monotonically
# increasing call number across the whole optimization run.
_CALL_COUNTER = [0]


class FrustrationDetector(dspy.Module):
    """Audio-in, structured-out classifier wrapping MOSS-Audio."""

    def __init__(self, backend: AudioInferenceBackend) -> None:
        super().__init__()
        self.backend = backend
        # dspy.Predict holds the signature (instructions can be optimized)
        # and the demos list (populated by BootstrapFewShot). We never
        # actually call self.predict() — we read its state and build the
        # prompt ourselves so we can route through our audio backend.
        self.predict = dspy.Predict(FrustrationSignature)

    def forward(self, audio_path: str) -> dspy.Prediction:
        instructions = (self.predict.signature.instructions or "").strip()
        demos = list(self.predict.demos or [])
        prompt = build_prompt(instructions, demos)
        raw = self.backend(audio_path, prompt)
        parsed = parse_frustration_output(raw)

        # Live trace — opt-in via env var so it works during eval AND
        # during the optimizer's internal trials, which our --verbose flag
        # can't reach (those happen inside DSPy's own loops).
        if os.environ.get("DSPY_TRACE_CALLS"):
            _CALL_COUNTER[0] += 1
            n = _CALL_COUNTER[0]
            name = os.path.basename(audio_path)
            reason_snip = (parsed["reason"][:80] + "…") if len(parsed["reason"]) > 80 \
                else parsed["reason"]
            print(
                f"[trace #{n:04d}] {name:<48s} "
                f"pred={parsed['frustration']:3s} "
                f"conf={parsed['confidence']:3d} "
                f"demos={len(demos)} "
                f"instr_len={len(instructions)} "
                f"reason='{reason_snip}'",
                file=sys.stderr, flush=True,
            )

        return dspy.Prediction(
            frustration=parsed["frustration"],
            confidence=parsed["confidence"],
            reason=parsed["reason"],
            raw_output=raw,
        )
