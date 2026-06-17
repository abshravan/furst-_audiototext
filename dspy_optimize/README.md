# DSPy prompt optimization for MOSS-Audio frustration detection

This package wraps the existing `infer.py` (unchanged) with a DSPy
optimization layer that tunes the *prompt* used for binary frustration
classification.

## What gets optimized

| Optimizer | What it changes | When to use |
|---|---|---|
| **BootstrapFewShot** (default) | Inserts up to N few-shot demos drawn from your training set | Small datasets, no extra LM needed. Demos with audio inputs are inert (the model can't re-hear them) so the win is mostly *output-format consistency*. |
| **MIPROv2** | Rewrites the instruction text itself, optionally adds demos | Real prompt-text optimization. Needs a second text LM (e.g. `openai/gpt-4o-mini`) to propose instruction candidates. Higher compute cost. |

## Setup

```bash
cd /path/to/furst-_audiototext
source env/bin/activate
pip install 'dspy-ai>=2.5'

# Prepare a labels CSV (audio_path,frustration columns):
cp dspy_optimize/labels.example.csv dspy_optimize/labels.csv
# ...edit to point at your real audio files...
```

## Train (BootstrapFewShot default)

```bash
python -m dspy_optimize.optimize \
    --model ./MOSS-Audio/weights/MOSS-Audio-4B-Instruct \
    --labels dspy_optimize/labels.csv \
    --device cuda --dtype float16 \
    --max-new-tokens 256
```

Outputs `dspy_optimize/optimized_program.json` plus a comparison
report (baseline vs optimized) on the held-out 30% validation split.

## Train with MIPROv2 (instruction rewriting)

```bash
export DSPY_PROMPT_MODEL='openai/gpt-4o-mini'
export OPENAI_API_KEY='sk-...'

python -m dspy_optimize.optimize \
    --model ./MOSS-Audio/weights/MOSS-Audio-4B-Instruct \
    --labels dspy_optimize/labels.csv \
    --optimizer mipro
```

## Evaluate a saved program

```bash
python -m dspy_optimize.evaluate \
    --model ./MOSS-Audio/weights/MOSS-Audio-4B-Instruct \
    --labels dspy_optimize/labels.csv \
    --program dspy_optimize/optimized_program.json
```

## Inspect the optimized prompt

```bash
python -m dspy_optimize.inspect_prompt \
    --program dspy_optimize/optimized_program.json
```

Prints the final instructions, every demo DSPy decided to keep, and
the full prompt template as it gets sent to MOSS-Audio.

## How it fits together

```
labels.csv
   │
   ▼
dataset.load_examples + stratified_split
   │
   ▼
FrustrationDetector(dspy.Module)         ┐
   forward(audio_path) →                  │   <- DSPy mutates
   reads predict.signature.instructions   │      these during
   reads predict.demos                    │      optimization
   builds prompt (module.build_prompt)    │
   calls AudioInferenceBackend(           │
       audio_path, prompt)                ┘
   parses output → dspy.Prediction
```

`AudioInferenceBackend` (in `backend.py`) loads MOSS-Audio once and
caches every `(audio_path, prompt)` -> output to
`<repo>/.dspy_cache/`. That cache survives across runs and is the
reason re-evaluations are essentially free.

## Tips for tiny datasets (<50 examples)

* **Stratified split is non-negotiable.** With 34 examples and
  random splitting you'll regularly get unbalanced val sets, making
  F1 meaningless.
* **Keep `max_bootstrapped_demos` low (2-4).** Each demo eats prompt
  budget and brings diminishing returns when the model can't re-hear
  the demo audio anyway.
* **Trust the seed.** Re-runs with a different `--seed` will give
  different val sets; the only honest comparison is baseline-vs-
  optimized on the *same* seed.
* **Watch for overfitting.** With only ~10 val examples a single
  flipped label is 10 percentage points of accuracy. Report metrics
  on multiple seeds if you want to claim a real improvement.
