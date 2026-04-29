# MOSS-Audio-8B-Thinking — Local Runner (stable, minimal-deps)

Run [`OpenMOSS-Team/MOSS-Audio-8B-Thinking`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking)
on a local machine with a fresh, modern Python environment — no
torchaudio, no torchcodec, no system FFmpeg ABI to fight.

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

| File              | Purpose                                                   |
| ----------------- | --------------------------------------------------------- |
| `infer.py`        | CLI inference script. librosa-based audio loader.         |
| `setup.sh`        | Clones MOSS-Audio, installs minimal deps, downloads weights. |
| `requirements.txt`| Pinned, stable dependency set.                            |

## Quick start

```bash
# Fresh Python 3.10–3.12 env recommended:
python -m venv env && source env/bin/activate

# One-shot install (clone + deps + weights + copy infer.py in)
bash setup.sh

# Run inference
cd MOSS-Audio
python infer_local.py \
    --audio path/to/clip.mp3 \
    --model ./weights/MOSS-Audio-8B-Thinking \
    --device auto \
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
```

## CLI options

```
--model           Path to local weights dir (default: ./weights/MOSS-Audio-8B-Thinking)
--audio           Input audio file (required)
--prompt          Instruction passed to the model (default: "Describe this audio.")
--device          auto | cpu | cuda | cuda:N (default: auto - GPU if present, else CPU)
--dtype           auto | float32 | float16 | bfloat16 (default: auto)
--max-new-tokens  Generation length cap (default: 1024)
--temperature     Sampling temperature (default: 1.0)
--top-p           Nucleus sampling (default: 1.0)
--top-k           Top-k sampling (default: 50)
--no-time-marker  Disable time-marker tokens
```

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

## CPU-only mode

```bash
python infer_local.py \
    --audio clip.mp3 \
    --model ./weights/MOSS-Audio-8B-Thinking \
    --device cpu \
    --prompt "Describe this audio."
```

8B-Thinking on CPU takes several minutes per response and needs ~32 GB
of RAM. For faster CPU runs use `OpenMOSS-Team/MOSS-Audio-4B-Instruct`.

## Example prompts

- `"Transcribe this audio."` — speech-to-text
- `"Describe this audio."` — captioning
- `"What language is being spoken? Reason step by step."` — reasoning
- `"Summarize the conversation."` — dialogue summarization

## License

Apache 2.0 (matches upstream).
