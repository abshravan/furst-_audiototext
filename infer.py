"""Local inference for OpenMOSS-Team/MOSS-Audio-8B-Thinking (CPU edition).

Stable, minimal-dependency wrapper. Loads audio via librosa and runs the
model with stock PyTorch + Transformers. No torchaudio, no torchcodec,
no GPU required.

Layout (after running setup.sh):
    MOSS-Audio/
        weights/MOSS-Audio-8B-Thinking/
        infer_local.py   <- this file, copied in by setup.sh

Run:
    cd MOSS-Audio
    python infer_local.py \\
        --audio path/to/clip.mp3 \\
        --model ./weights/MOSS-Audio-8B-Thinking \\
        --device cpu \\
        --prompt "Describe this audio."

RAM guide (no GPU):
    8B model, float16  ~17 GB  <- default, fits in 32 GB
    8B model, float32  ~34 GB  <- will OOM on 32 GB, don't use
    4B model, float32  ~18 GB  <- safer, download MOSS-Audio-4B-Thinking
    4B model, float16   ~9 GB  <- most headroom

Speed: expect 5-30 min per response on i7 12th gen. Set --max-new-tokens
lower (e.g. 256) to get answers faster.
"""

import argparse
import os
import sys
import types

# src/processing_moss_audio.py has a dead `import torchaudio` at module
# level that breaks on CPU-only hosts (tries to dlopen libcudart).
# Insert an empty stub BEFORE any src.* import so the real torchaudio
# package is never loaded.
sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))

import librosa
import numpy as np
import torch

from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor


# ---------------------------------------------------------------------------
# Model variant registry
# ---------------------------------------------------------------------------

# Maps shorthand --variant values to the corresponding HF repo and a rough
# parameter count used for the RAM estimate. The 4B-Instruct is the most
# CPU-friendly choice (lowest RAM, fastest, instruction-tuned for direct
# answers); 8B-Thinking is the slowest but produces step-by-step reasoning.
VARIANTS = {
    "4b-instruct": {
        "repo": "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
        "dirname": "MOSS-Audio-4B-Instruct",
        "params": 4.6e9,
    },
    "4b-thinking": {
        "repo": "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        "dirname": "MOSS-Audio-4B-Thinking",
        "params": 4.6e9,
    },
    "8b-instruct": {
        "repo": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
        "dirname": "MOSS-Audio-8B-Instruct",
        "params": 8.6e9,
    },
    "8b-thinking": {
        "repo": "OpenMOSS-Team/MOSS-Audio-8B-Thinking",
        "dirname": "MOSS-Audio-8B-Thinking",
        "params": 8.6e9,
    },
}


def resolve_model_path(variant: str | None, model: str | None) -> tuple[str, float]:
    """Return (model_path, estimated_param_count). --model wins over --variant."""
    if model:
        # Try to infer params from the directory name; fall back to 8B.
        params = 8.6e9
        for v in VARIANTS.values():
            if v["dirname"] in model:
                params = v["params"]
                break
        return model, params
    if variant:
        info = VARIANTS[variant]
        return f"./weights/{info['dirname']}", info["params"]
    info = VARIANTS["8b-thinking"]
    return f"./weights/{info['dirname']}", info["params"]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Mono float32 numpy array at sample_rate. No torchaudio/torchcodec."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Audio file not found: {path!r}. "
            "Pass an absolute path with --audio."
        )
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if isinstance(audio, np.ndarray):
        audio = audio.astype("float32")
    return audio


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but no NVIDIA driver/GPU visible. "
            "Re-run with --device cpu."
        )
    return requested


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16
        # CPU: float16 halves memory vs float32.
        # 8B in float32 needs ~34 GB (OOM on 32 GB); float16 needs ~17 GB.
        return torch.float16
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]


def configure_cpu_threads() -> int:
    """Pin PyTorch to all available logical CPUs for max throughput."""
    n = os.cpu_count() or 4
    torch.set_num_threads(n)
    torch.set_num_interop_threads(2)
    return n


def ram_estimate_gb(dtype: torch.dtype, params: float) -> float:
    """Rough weight-only RAM estimate for a given param count."""
    bytes_per_param = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(dtype, 4)
    return round(params * bytes_per_param / 1024 ** 3, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MOSS-Audio on an audio file (CPU-friendly).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Variants (use --variant for shorthand, or --model for a custom path):
  4b-instruct  ~9 GB float16 / ~18 GB float32 - fastest, direct answers
  4b-thinking  ~9 GB float16 / ~18 GB float32 - direct + step-by-step reasoning
  8b-instruct  ~17 GB float16 / OOM float32 on 32 GB - higher quality answers
  8b-thinking  ~17 GB float16 / OOM float32 on 32 GB - best reasoning, slowest

For a 32 GB RAM no-GPU machine: --variant 4b-instruct is the recommended
starting point (smallest, fastest, instruction-tuned).
""",
    )
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument(
        "--variant",
        choices=sorted(VARIANTS.keys()),
        help="Shorthand for one of the four MOSS-Audio variants. "
             "Resolves to ./weights/MOSS-Audio-<variant>/.",
    )
    model_group.add_argument(
        "--model",
        help="Custom local model directory. Overrides --variant.",
    )
    p.add_argument("--audio", required=True,
                   help="Input audio file (wav, mp3, flac, ogg, m4a, ...).")
    p.add_argument("--prompt", default="Describe this audio.",
                   help="Instruction for the model.")
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', 'cuda', or 'cuda:N'. Default: auto.")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "float32", "bfloat16"],
                   help="Model dtype. Default: float16 on CPU, bfloat16 on GPU.")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens to generate. Lower = faster. Default: 512.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--no-time-marker", action="store_true",
                   help="Disable time-marker tokens.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    model_path, params = resolve_model_path(args.variant, args.model)
    if not os.path.isdir(model_path):
        raise SystemExit(
            f"Model directory not found: {model_path!r}\n"
            f"Download with:\n"
            f"    hf download OpenMOSS-Team/{os.path.basename(model_path)} "
            f"--local-dir {model_path}\n"
            f"Or run: bash download_models.sh 4b-instruct"
        )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    if device == "cpu":
        n_threads = configure_cpu_threads()
        est_gb = ram_estimate_gb(dtype, params)
        sys_gb = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3, 1)
        print(f"[infer] model={model_path}  params=~{round(params/1e9, 1)}B",
              file=sys.stderr)
        print(f"[infer] device=cpu  dtype={dtype}  threads={n_threads}", file=sys.stderr)
        print(f"[infer] estimated model RAM: ~{est_gb} GB (system has {sys_gb} GB)",
              file=sys.stderr)
        if est_gb > sys_gb - 4:
            print("[infer] WARNING: model may not fit in available RAM at this dtype. "
                  "Try --dtype float16, or switch to a smaller --variant "
                  "(4b-instruct is the lightest).",
                  file=sys.stderr)
        print("[infer] NOTE: CPU inference is slow (minutes per response). "
              "Use --max-new-tokens 256 for quicker results.", file=sys.stderr)
    else:
        print(f"[infer] model={model_path}", file=sys.stderr)
        print(f"[infer] device={device}  dtype={dtype}", file=sys.stderr)

    print("[infer] Loading model weights ...", file=sys.stderr)
    model = MossAudioModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,   # stream weights in; avoids doubling peak RAM
    )
    model.eval()

    processor = MossAudioProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        enable_time_marker=not args.no_time_marker,
    )

    print(f"[infer] Loading audio: {args.audio}", file=sys.stderr)
    raw_audio = load_audio(args.audio, sample_rate=processor.config.mel_sr)

    inputs = processor(text=args.prompt, audios=[raw_audio], return_tensors="pt")
    inputs = inputs.to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

    print(f"[infer] Generating (max_new_tokens={args.max_new_tokens}) ...", file=sys.stderr)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            num_beams=1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    output = processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
