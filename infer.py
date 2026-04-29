"""Local inference for OpenMOSS-Team/MOSS-Audio-8B-Thinking.

Run an audio file through the MOSS-Audio model on a local machine and print the
generated response. Wraps the upstream MOSS-Audio loading code with a small CLI
so you can point it at any audio file and instruction.

Prerequisites:
    1. Clone the upstream MOSS-Audio repo (it provides the `src` package this
       script imports) and install its dependencies:

           git clone https://github.com/OpenMOSS/MOSS-Audio.git
           cd MOSS-Audio
           conda create -n moss-audio python=3.12 -y
           conda activate moss-audio
           conda install -c conda-forge "ffmpeg=7" -y
           pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
               -e ".[torch-runtime]"

    2. Download the model weights:

           hf download OpenMOSS-Team/MOSS-Audio-8B-Thinking \
               --local-dir ./weights/MOSS-Audio-8B-Thinking

    3. Copy this file into the cloned MOSS-Audio directory (so the `src`
       imports resolve), then run:

           python infer.py --audio path/to/clip.mp3 \
               --model ./weights/MOSS-Audio-8B-Thinking \
               --prompt "Describe this audio."
"""

import argparse
import sys

import numpy as np
import torch

from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor


def load_audio_safe(path: str, sample_rate: int) -> np.ndarray:
    """Load audio as a mono float32 numpy array at ``sample_rate``.

    Bypasses torchaudio/torchcodec (which needs system FFmpeg shared libs and
    often breaks on stock installs) by trying soundfile first, then librosa,
    then ffmpeg via subprocess as a last resort. Returns a 1-D ndarray in [-1, 1].
    """
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != sample_rate:
            data = _resample(data, sr, sample_rate)
        return data.astype(np.float32, copy=False)
    except Exception as sf_err:  # noqa: BLE001 - intentional broad catch
        last_err = sf_err

    try:
        import librosa  # type: ignore
        data, _ = librosa.load(path, sr=sample_rate, mono=True)
        return data.astype(np.float32, copy=False)
    except Exception as lr_err:  # noqa: BLE001
        last_err = lr_err

    # Last resort: shell out to ffmpeg if it exists
    try:
        return _load_via_ffmpeg(path, sample_rate)
    except Exception as ff_err:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load {path}. Tried soundfile, librosa, and ffmpeg.\n"
            f"  soundfile/librosa error: {last_err}\n"
            f"  ffmpeg error: {ff_err}\n"
            "Fix: pip install librosa soundfile, or install ffmpeg "
            "(e.g. 'sudo apt install ffmpeg' / 'conda install -c conda-forge ffmpeg=7')."
        ) from ff_err


def _resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return data
    try:
        import librosa  # type: ignore
        return librosa.resample(data, orig_sr=src_sr, target_sr=dst_sr)
    except Exception:  # noqa: BLE001
        # Linear interpolation fallback - lower quality but no extra deps.
        ratio = dst_sr / src_sr
        new_len = int(round(len(data) * ratio))
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        return np.interp(x_new, x_old, data).astype(np.float32)


def _load_via_ffmpeg(path: str, sample_rate: int) -> np.ndarray:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg binary not found on PATH")

    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


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
        help="Path to the input audio file (wav, mp3, flac, etc.).",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this audio.",
        help="Instruction passed to the model. Use 'Transcribe this audio.' for ASR.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' (pick GPU if available, else CPU), 'cpu', 'cuda', or 'cuda:N'.",
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
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but no NVIDIA driver / GPU is visible to PyTorch. "
            "Re-run with --device cpu, or install a CUDA-capable driver."
        )
    return requested


def resolve_dtype(requested: str, device: str):
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
            "[infer] running on CPU - 8B model inference will be slow "
            "(many minutes per response) and needs ~32GB RAM.",
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

    raw_audio = load_audio_safe(args.audio, sample_rate=processor.config.mel_sr)

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
