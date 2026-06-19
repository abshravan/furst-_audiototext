"""Inspect the disk cache built by AudioInferenceBackend.

Every (audio_path, prompt) -> output combination ever tried during
optimization or evaluation is saved as a JSON file under .dspy_cache/.
This script summarizes them so you can verify the prompt was actually
delivered and the model produced sensible outputs.

Usage:

    # Summary table — one row per cached call.
    python -m dspy_optimize.inspect_cache

    # Dump the full prompt + raw output for one specific audio file.
    python -m dspy_optimize.inspect_cache --audio /path/to/file.mp3

    # Filter to only the LATEST prompt variant (longest one), useful
    # when GEPA has tried many instruction rewrites and you want to see
    # what the final winner actually predicted on each file.
    python -m dspy_optimize.inspect_cache --latest-prompt-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from .module import parse_frustration_output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=".dspy_cache",
                   help="Where the cache lives. Default: <repo>/.dspy_cache/")
    p.add_argument("--audio", default=None,
                   help="If set, dump the full prompt + raw output for "
                        "every cached call against this audio file.")
    p.add_argument("--latest-prompt-only", action="store_true",
                   help="Only show calls made with the longest prompt "
                        "variant (proxy for 'most recently evolved').")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        raise SystemExit(f"Cache dir not found: {cache_dir.resolve()}")

    entries = []
    for p in sorted(cache_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        parsed = parse_frustration_output(obj.get("output", ""))
        entries.append({
            "file": p.name,
            "audio_path": obj.get("audio_path", ""),
            "prompt": obj.get("prompt", ""),
            "output": obj.get("output", ""),
            "frustration": parsed["frustration"],
            "confidence": parsed["confidence"],
            "reason": parsed["reason"],
        })

    if not entries:
        print(f"(no cache entries in {cache_dir.resolve()})")
        return 0

    if args.latest_prompt_only:
        longest = max(len(e["prompt"]) for e in entries)
        entries = [e for e in entries if len(e["prompt"]) == longest]

    # Detail view for a single audio file.
    if args.audio:
        target = os.path.abspath(args.audio)
        matching = [e for e in entries if os.path.abspath(e["audio_path"]) == target]
        if not matching:
            raise SystemExit(f"No cached calls for: {target}")
        for i, e in enumerate(matching, 1):
            print("=" * 78)
            print(f" CALL {i}/{len(matching)} — {e['file']}")
            print("=" * 78)
            print(f"audio_path : {e['audio_path']}")
            print(f"predicted  : {e['frustration']} (conf={e['confidence']})")
            print(f"reason     : {e['reason']}")
            print()
            print("--- PROMPT ---")
            print(e["prompt"])
            print()
            print("--- RAW MODEL OUTPUT ---")
            print(e["output"])
            print()
        return 0

    # Summary view across the whole cache.
    by_audio = defaultdict(list)
    for e in entries:
        by_audio[e["audio_path"]].append(e)

    print(f"Cache: {cache_dir.resolve()}")
    print(f"Total cached calls: {len(entries)}")
    print(f"Unique audio files: {len(by_audio)}")
    print(f"Unique prompts    : {len({e['prompt'] for e in entries})}")
    print()
    print(f"{'audio_file':<55s} {'calls':>5s}  {'yes':>3s} {'no':>3s}  "
          f"{'agreement'}")
    print("-" * 90)
    for audio, calls in sorted(by_audio.items()):
        name = os.path.basename(audio)[:55]
        yes = sum(1 for c in calls if c["frustration"] == "yes")
        no = sum(1 for c in calls if c["frustration"] == "no")
        # "agreement" = fraction of calls that picked the majority label.
        # Low agreement means the prompt variants disagreed on this audio.
        majority = max(yes, no)
        agree = majority / len(calls) if calls else 0.0
        flag = "" if agree == 1.0 else "  <- inconsistent"
        print(f"{name:<55s} {len(calls):>5d}  {yes:>3d} {no:>3d}  "
              f"{agree:.0%}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
