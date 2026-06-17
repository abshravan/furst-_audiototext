"""DSPy Signatures = typed schemas describing model inputs and outputs.

The signature's *docstring* doubles as the initial instruction string
that DSPy sends to the model. Optimizers like MIPROv2 rewrite this
docstring — and BootstrapFewShot prepends demos to it — to find the
phrasing that maximizes our metric on the validation set.

We define one signature: `FrustrationSignature`. Field descriptions
also feed into the prompt template DSPy builds, so they should be
clear and concise.
"""

from __future__ import annotations

import dspy


class FrustrationSignature(dspy.Signature):
    """Detect whether the patient speaker in a clinical call is frustrated.

    Listen to the audio and decide if the patient is expressing emotional
    frustration. Use only acoustic and linguistic evidence: tone, pace,
    sighs, raised pitch, sharp word choice, repeated complaints,
    interruptions. Avoid assumptions when the signal is unclear — return
    "no" with low confidence rather than guessing "yes".

    Output strict JSON with fields: frustration ("yes" or "no"),
    confidence (integer 0-100), and reason (short evidence-based
    explanation citing the specific acoustic or verbal cues).
    """

    audio_path: str = dspy.InputField(
        desc="Absolute or relative path to the audio file on disk."
    )
    frustration: str = dspy.OutputField(
        desc='Either "yes" or "no". Lowercase. No other values allowed.'
    )
    confidence: int = dspy.OutputField(
        desc="Integer 0-100 representing how confident the classification is."
    )
    reason: str = dspy.OutputField(
        desc="One or two sentences citing specific cues that justify the label."
    )
