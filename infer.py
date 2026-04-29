"""Local inference for OpenMOSS-Team/MOSS-Audio-8B-Thinking.

Stable, minimal-dependency wrapper. Loads audio via librosa (NumPy) and
runs the model with stock PyTorch + Transformers. Does NOT use
torchaudio or torchcodec.

Layout (after running setup.sh):
    MOSS-Audio/                  <- upstream repo, provides `src` package
        weights/
            MOSS-Audio-8B-Thinking/
        infer_local.py           <- this file, copied in by setup.sh

Run:
    cd MOSS-Audio
    python infer_local.py \
        --audio path/to/clip.mp3 \
        --model ./weights/MOSS-Audio-8B-Thinking \
        --device auto \
        --prompt "Describe this audio."
"""

import argparse
import sys
import types

# MOSS-Audio's src/processing_moss_audio.py has a module-level `import
# torchaudio` that is never actually used. Real torchaudio links to
# libtorchaudio + libcudart, both of which fail on machines without a
# matching CUDA runtime. Stub it out before src.* gets imported below.
sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))

import librosa
import numpy as np
import torch

from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Mono float32 numpy array at ``sample_rate``. No torchaudio/torchcodec."""
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if isinstance(audio, np.ndarray):
        audio = audio.astype("float32")
    return audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MOSS-Audio locally on an audio file.")
    parser.add_argument(
        "--model",
        default="./weights/MOSS-Audio-8B-Thinking",
        help="Path to the local MOSS-Audio model directory.",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the input audio file (wav, mp3, flac, ogg, m4a, ...).",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this audio.",
        help="Instruction for the model. Use 'Transcribe this audio.' for ASR.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' (GPU if present, else CPU), 'cpu', 'cuda', or 'cuda:N'.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype. 'auto' = bfloat16 on GPU, float32 on CPU.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--no-time-marker",
        action="store_true",
        help="Disable time-marker tokens in the processor.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA was requested but no NVIDIA driver / GPU is visible. "
                "Re-run with --device cpu, or install a CUDA-capable driver."
            )
        return "cuda:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but no NVIDIA driver / GPU is visible. "
            "Re-run with --device cpu, or install a CUDA-capable driver."
        )
    return requested


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]


def main() -> int:
    args = parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"[infer] device={device} dtype={dtype}", file=sys.stderr)
    if device == "cpu":
        print(
            "[infer] running on CPU - 8B model inference will take minutes "
            "per response and needs ~32GB RAM.",
            file=sys.stderr,
        )

    model = MossAudioModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    processor = MossAudioProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        enable_time_marker=not args.no_time_marker,
    )

    raw_audio = load_audio(args.audio, sample_rate=processor.config.mel_sr)

    inputs = processor(text=args.prompt, audios=[raw_audio], return_tensors="pt")
    inputs = inputs.to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

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
