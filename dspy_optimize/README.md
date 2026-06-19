# DSPy prompt optimization for MOSS-Audio frustration detection

This package wraps the existing `infer.py` (unchanged) with a DSPy
optimization layer that tunes the *prompt* used for binary frustration
classification.

## What gets optimized

| Optimizer | What it changes | When to use |
|---|---|---|
| **BootstrapFewShot** (default) | Inserts up to N few-shot demos drawn from your training set | Small datasets, no extra LM needed. Demos with audio inputs are inert (the model can't re-hear them) so the win is mostly *output-format consistency*. |
| **MIPROv2** | Rewrites the instruction text itself, optionally adds demos | Real prompt-text optimization. Needs a second text LM (e.g. `openai/gpt-4o-mini`) to propose instruction candidates. Higher compute cost. |
| **GEPA** | Evolves the instruction by *reflecting* on per-example failures, then proposing rewrites; keeps a Pareto frontier of variants | Strongest results on small (<50) datasets. The reflection LM reads natural-language feedback ("FALSE NEGATIVE: model missed sighs and raised pitch...") and proposes targeted instruction edits. Needs `DSPY_PROMPT_MODEL`; most expensive of the three. |

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

## Train with GEPA (reflection-based prompt evolution)

GEPA reads natural-language feedback for each failed prediction and
asks a reflection LM to rewrite the instruction. It typically beats
MIPROv2 on small datasets because it learns from *why* something was
wrong, not just *whether*.

```bash
export DSPY_PROMPT_MODEL='openai/gpt-4o-mini'
export OPENAI_API_KEY='sk-...'

# Optional: use a stronger model just for reflection (rewrites)
# while keeping the cheaper one for everything else.
# export DSPY_REFLECTION_MODEL='openai/gpt-4o'

python -m dspy_optimize.optimize \
    --model ./MOSS-Audio/weights/MOSS-Audio-4B-Instruct \
    --labels dspy_optimize/labels.csv \
    --optimizer gepa \
    --gepa-budget light \
    --device cuda --dtype float16
```

`--gepa-budget` controls how many reflection rounds GEPA runs:
* `light` — ~few dozen reflection calls. Best starting point.
* `medium` — ~hundreds.
* `heavy` — full search; can run for hours on a small dataset.

Each reflection call hits your text LM (paid API tokens). The audio
model itself is still cached by `AudioInferenceBackend`, so GEPA only
pays for new `(audio, prompt)` combinations it hasn't tried yet.

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

## Watch the model run live during optimization

Two layers of visibility, depending on how much detail you want:

**1. Per-example eval prints** (covers the baseline + final-optimized
passes, not the optimizer's internal trials):

```bash
python -m dspy_optimize.optimize --verbose ... 
```
Prints `[i] ✓ gold=yes pred=yes  /path/to/file.mp3` per example.

**2. Every single backend call, including during optimization** (set
the env var, no other changes):

```bash
DSPY_TRACE_CALLS=1 python -m dspy_optimize.optimize \
    --model ./MOSS-Audio/weights/MOSS-Audio-4B-Thinking \
    --labels dspy_optimize/labels.csv \
    --optimizer gepa --gepa-budget light --device cuda --dtype float16
```

Each line on stderr looks like:
```
[trace #0042] PC0cLeFW5xLvAXVytstTT3s.mp3   pred=yes conf= 85 demos=3 instr_len=512 reason='Patient repeatedly raised...'
```
This covers GEPA/MIPRO's internal trials too — useful for catching
"the model is returning gibberish" or "the optimizer is hammering one
file over and over."

## Audit the cache after the run

Every `(audio_path, prompt) → output` is saved to `.dspy_cache/`. To
review what actually happened:

```bash
# Summary table — one row per audio file, how many calls, label agreement.
python -m dspy_optimize.inspect_cache

# Full prompt + raw model output for one specific file.
python -m dspy_optimize.inspect_cache --audio /path/to/PC0cLeFW5xLvAXVytstTT3s_patient.mp3

# Only show the most-evolved prompt's predictions (filter to longest prompt).
python -m dspy_optimize.inspect_cache --latest-prompt-only
```

Use the summary view to spot files where different prompt variants
gave different answers (`agreement < 100%` flagged "inconsistent"). Those
are the files the optimizer is uncertain about and probably the ones
to label-check first.

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
