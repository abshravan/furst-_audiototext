# MOSS-Audio-8B-Thinking — Local Runner

A small wrapper that runs
[`OpenMOSS-Team/MOSS-Audio-8B-Thinking`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking)
locally. Point it at an audio file and a prompt; it prints the model's
response (transcription, description, captioning, QA, reasoning, etc.).

The model is ~8.6B parameters and the weights are ~17 GB on disk, so a
GPU with ≥24 GB VRAM (e.g. RTX 3090 / 4090 / A100) is recommended. CPU
inference is possible but very slow.

## Files

| File              | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `infer.py`        | CLI inference script (loads model, runs one prompt).        |
| `setup.sh`        | Clones MOSS-Audio, installs deps, downloads 8B weights.     |
| `requirements.txt`| Pip deps the wrapper touches directly.                      |

## Quick start

```bash
# 1. One-shot install (clones MOSS-Audio, installs torch+ffmpeg, downloads weights)
bash setup.sh

# 2. Run inference
cd MOSS-Audio
python infer_local.py \
    --audio path/to/clip.mp3 \
    --model ./weights/MOSS-Audio-8B-Thinking \
    --prompt "Describe this audio."
```

## Manual install

If you'd rather install things yourself:

```bash
git clone https://github.com/OpenMOSS/MOSS-Audio.git
cd MOSS-Audio

conda create -n moss-audio python=3.12 -y
conda activate moss-audio
conda install -c conda-forge "ffmpeg=7" -y
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"

# Download weights
pip install -U "huggingface_hub[cli]"
hf download OpenMOSS-Team/MOSS-Audio-8B-Thinking \
    --local-dir ./weights/MOSS-Audio-8B-Thinking

# Drop infer.py from this repo into the MOSS-Audio dir (it imports `src.*`)
cp ../infer.py ./infer_local.py
python infer_local.py --audio sample.wav --model ./weights/MOSS-Audio-8B-Thinking
```

## CLI options

```
--model           Path to local weights dir (default: ./weights/MOSS-Audio-8B-Thinking)
--audio           Input audio file (required)
--prompt          Instruction passed to the model (default: "Describe this audio.")
--device          Device map, e.g. "cuda:0" or "cpu" (default: cuda:0)
--max-new-tokens  Generation length cap (default: 1024)
--temperature     Sampling temperature (default: 1.0)
--top-p           Nucleus sampling (default: 1.0)
--top-k           Top-k sampling (default: 50)
--no-time-marker  Disable time-marker tokens
```

## Example prompts

- `"Transcribe this audio."` — speech-to-text
- `"Describe this audio."` — captioning
- `"What language is being spoken? Reason step by step."` — reasoning
- `"Summarize the conversation."` — dialogue summarization

## Notes

- `infer.py` imports the `src.*` package shipped inside the upstream
  MOSS-Audio repo, so it must live alongside that repo (`setup.sh` copies
  it in for you as `infer_local.py`).
- For a chat-style web UI, run `python app.py` from inside the MOSS-Audio
  repo after weights are downloaded.
- License: Apache 2.0 (matches upstream).
