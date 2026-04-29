# MOSS-Audio Local Runner (stable, minimal-deps, CPU-friendly)

Run any of the four MOSS-Audio variants on a local machine with a fresh,
modern Python environment — no torchaudio, no torchcodec, no system
FFmpeg ABI to fight.

## Supported variants

| `--variant`   | HF repo                                          | Params | RAM (fp16) | RAM (fp32) |
| ------------- | ------------------------------------------------ | ------ | ---------- | ---------- |
| `4b-instruct` | [OpenMOSS-Team/MOSS-Audio-4B-Instruct](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Instruct) | ~4.6B  | ~9 GB      | ~18 GB     |
| `4b-thinking` | [OpenMOSS-Team/MOSS-Audio-4B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking) | ~4.6B  | ~9 GB      | ~18 GB     |
| `8b-instruct` | [OpenMOSS-Team/MOSS-Audio-8B-Instruct](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct) | ~8.6B  | ~17 GB     | ~34 GB     |
| `8b-thinking` | [OpenMOSS-Team/MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking) | ~8.6B  | ~17 GB     | ~34 GB     |

Recommendation for 32 GB RAM, no GPU: **`--variant 4b-instruct`** — the
fastest and smallest variant, instruction-tuned for direct answers.

## What's different from the upstream setup

The upstream `pip install -e ".[torch-runtime]"` pulls
`torchaudio` → `torchcodec`, which then needs a system `libavutil.so.*`
matching one of FFmpeg 4–8. That chain breaks constantly on stock
machines. This wrapper:

- **Removes torchaudio / torchcodec entirely.**
- **Loads audio with librosa** (which uses libsndfile via soundfile and
  falls back to audioread for mp3/m4a — both pure-Python distribution
  paths).
- **Pins a minimal, modern stack** in `requirements.txt`
  (torch ≥ 2.4, transformers ≥ 4.45, numpy < 2).
- **Skips the upstream `[torch-runtime]` extra** so torchcodec never
  comes back in.

## Files

| File                 | Purpose                                                          |
| -------------------- | ---------------------------------------------------------------- |
| `infer.py`           | CLI inference script. librosa-based audio loader.                |
| `setup.sh`           | Clones MOSS-Audio, installs minimal deps, downloads 8B-Thinking. |
| `download_models.sh` | Download any subset of the 4 variants on demand.                 |
| `patch_existing.sh`  | Re-apply patches to an existing checkout.                        |
| `requirements.txt`   | Pinned, stable dependency set.                                   |

## Quick start

```bash
# Fresh Python 3.10–3.12 env recommended:
python -m venv env && source env/bin/activate

# 1. Install deps + clone upstream MOSS-Audio repo (does NOT download weights)
bash setup.sh

# 2. Download whichever variant(s) you want
bash download_models.sh 4b-instruct          # smallest/fastest
# or:
bash download_models.sh 8b-thinking          # best reasoning
# or:
bash download_models.sh all                  # ~52 GB total

# 3. Run inference (use --variant shorthand)
cd MOSS-Audio
python infer_local.py \
    --variant 4b-instruct \
    --audio path/to/clip.mp3 \
    --device cpu \
    --prompt "Describe this audio."
```

## Manual install

```bash
git clone https://github.com/OpenMOSS/MOSS-Audio.git
pip install --upgrade pip
pip install -r requirements.txt          # NOT pip install -e ".[torch-runtime]"
pip install -U "huggingface_hub[cli]"
hf download OpenMOSS-Team/MOSS-Audio-8B-Thinking \
    --local-dir MOSS-Audio/weights/MOSS-Audio-8B-Thinking
cp infer.py MOSS-Audio/infer_local.py
cd MOSS-Audio
python infer_local.py --audio sample.wav --model ./weights/MOSS-Audio-8B-Thinking
```

If you previously attempted the upstream install, purge the bad bits:

```bash
pip uninstall -y torchaudio torchcodec
# If torch was the CUDA build but you have no GPU, swap to CPU:
pip uninstall -y torch
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
```

### Why we don't install torchaudio

`src/processing_moss_audio.py` from upstream does `import torchaudio`
at module level but never actually calls anything from it. `infer.py`
inserts an empty stub into `sys.modules['torchaudio']` before importing
that file, so the import succeeds without loading torchaudio's native
extension (which is what was failing with `libcudart.so.13: cannot
open shared object file` on CPU-only machines).

## Batch mode + dashboard

Process every audio file in a folder and watch progress in a browser:

```bash
cd MOSS-Audio
python infer_local.py \
    --variant 4b-instruct \
    --batch-dir /path/to/audio_folder \
    --device cpu \
    --max-new-tokens 256 \
    --prompt "Transcribe this audio."

# Open the dashboard (auto-refreshes every 5 s while the job runs)
xdg-open /path/to/audio_folder/_batch_output/dashboard.html
```

The model loads **once** and then streams through every file. Output:

```
audio_folder/_batch_output/
├── dashboard.html    # self-contained, auto-refreshing dashboard
└── results.json      # machine-readable results (also used to resume)
```

### What you get

- **Stats panel** — total / done / failed / pending / avg duration / ETA
- **Live progress bar**
- **Per-file table** — status badge, duration, transcript (collapsible
  for long outputs)
- **Auto-refresh** every 5 s while running; static once finished
- **Dark-mode aware** (uses your OS preference)

### Resume

If you Ctrl-C halfway through and re-run the same command, already-done
files are skipped. Pass `--no-resume` to start over.

### Batch options

```
--batch-dir DIR     Folder of audio files (mutually exclusive with --audio).
--output-dir DIR    Where to write dashboard + results.json.
                    Default: <batch-dir>/_batch_output/.
--no-recursive      Don't recurse into subdirectories.
--no-resume         Ignore prior results.json and start fresh.
```

Supported audio extensions: `.wav .mp3 .flac .ogg .m4a .opus .webm
.aac .wma .aiff`.

## CLI options

```
--variant         4b-instruct | 4b-thinking | 8b-instruct | 8b-thinking
                  Shorthand. Resolves to ./weights/MOSS-Audio-<variant>/.
--model           Custom local weights dir. Overrides --variant.
--audio           Single audio file (mutually exclusive with --batch-dir).
--batch-dir       Folder of audio files; processes all and writes dashboard.html.
--output-dir      Where to write batch outputs. Default: <batch-dir>/_batch_output/.
--no-recursive    Don't recurse into subdirectories of --batch-dir.
--no-resume       Ignore prior results.json and start fresh.
--prompt          Instruction passed to the model (default: "Describe this audio.")
--device          auto | cpu | cuda | cuda:N (default: auto)
--dtype           auto | float32 | float16 | bfloat16 (default: auto)
                  auto = bfloat16 on GPU, float16 on CPU.
--max-new-tokens  Generation length cap (default: 512). Use 256 for faster CPU runs.
--temperature     Sampling temperature (default: 1.0)
--top-p           Nucleus sampling (default: 1.0)
--top-k           Top-k sampling (default: 50)
--no-time-marker  Disable time-marker tokens
```

### Choosing a variant

- **`4b-instruct`** — smallest, fastest, most direct. Best for ASR,
  captioning, and short Q&A on CPU.
- **`4b-thinking`** — same size as 4B-Instruct but emits chain-of-thought
  reasoning before the final answer. Slower because the response is longer.
- **`8b-instruct`** — higher answer quality at the cost of ~2× RAM and
  inference time. Needs float16 to fit in 32 GB RAM.
- **`8b-thinking`** — strongest reasoning but the slowest on CPU.

## Audio loading

`infer.py` uses exactly this:

```python
import librosa
import numpy as np

def load_audio(path, sample_rate=16000):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if isinstance(audio, np.ndarray):
        audio = audio.astype("float32")
    return audio
```

librosa decodes wav/flac/ogg via libsndfile (bundled in the `soundfile`
wheel — no system install needed) and mp3/m4a via audioread. If you
still hit a decode error on an exotic codec, install ffmpeg system-wide
once: `sudo apt install ffmpeg` or `conda install -c conda-forge ffmpeg`.

## CPU-only mode (32 GB RAM, no GPU)

The wrapper auto-detects no-GPU and falls back to CPU. Recommended:

```bash
bash download_models.sh 4b-instruct
cd MOSS-Audio
python infer_local.py \
    --variant 4b-instruct \
    --audio clip.mp3 \
    --device cpu \
    --max-new-tokens 256 \
    --prompt "Transcribe this audio."
```

On startup you'll see a memory estimate so you can confirm it fits, e.g.:

```
[infer] model=./weights/MOSS-Audio-4B-Instruct  params=~4.6B
[infer] device=cpu  dtype=torch.float16  threads=20
[infer] estimated model RAM: ~8.6 GB (system has 31.2 GB)
```

If you really want 8B-Thinking on a 32 GB machine, keep float16 (the
default on CPU) — float32 will OOM.

## Example prompts

- `"Transcribe this audio."` — speech-to-text
- `"Describe this audio."` — captioning
- `"What language is being spoken? Reason step by step."` — reasoning
- `"Summarize the conversation."` — dialogue summarization

## License

Apache 2.0 (matches upstream).
