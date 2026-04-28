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

import torch

from src.audio_io import load_audio
from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor


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
    parser.add_argument("--device", default="cuda:0", help="Device map, e.g. 'cuda:0' or 'cpu'.")
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


def main() -> int:
    args = parse_args()

    model = MossAudioModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=args.device,
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
