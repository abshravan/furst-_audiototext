"""Audio inference backend: loads MOSS-Audio once, caches results.

DSPy will invoke our model dozens to hundreds of times during
optimization and evaluation. Loading the 4B model takes ~30s and each
inference call takes seconds (GPU) to minutes (CPU). So we:

  1. Load model + processor ONCE at startup (`AudioInferenceBackend.__init__`).
  2. Cache `(audio_path, prompt)` -> output on disk, so re-running an
     identical call is instant. This is critical because BootstrapFewShot
     and MIPROv2 re-evaluate the same validation examples after each
     optimization round.

We intentionally do NOT modify `infer.py` — we import and call its
public helpers as-is.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Make the repo root importable so `import infer` works regardless of
# where this package is launched from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also make MOSS-Audio's src/ importable when running outside that dir.
_MOSS_DIR = _REPO_ROOT / "MOSS-Audio"
if _MOSS_DIR.is_dir() and str(_MOSS_DIR) not in sys.path:
    sys.path.insert(0, str(_MOSS_DIR))

import torch  # noqa: E402  (must come after sys.path massaging)

from infer import (  # noqa: E402
    configure_cpu_threads,
    load_model_and_processor,
    resolve_device,
    resolve_dtype,
    run_inference,
)


class AudioInferenceBackend:
    """Singleton-ish wrapper. Construct once, call many times."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        cache_dir: Optional[str] = None,
        enable_time_marker: bool = True,
    ) -> None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path!r}")
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        if self.device == "cpu":
            configure_cpu_threads()
        self.model, self.processor = load_model_and_processor(
            model_path, self.dtype, self.device,
            enable_time_marker=enable_time_marker,
        )
        # Generation kwargs are fixed across calls — only the prompt varies.
        self.gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_cache=True,
        )
        self._cache_dir = Path(cache_dir) if cache_dir else _REPO_ROOT / ".dspy_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, str] = {}

    def _cache_key(self, audio_path: str, prompt: str) -> str:
        """Stable hash over (absolute audio path, prompt, generation kwargs)."""
        h = hashlib.sha256()
        h.update(os.path.abspath(audio_path).encode())
        h.update(b"\n--PROMPT--\n")
        h.update(prompt.encode())
        h.update(b"\n--GEN--\n")
        h.update(json.dumps(self.gen_kwargs, sort_keys=True).encode())
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def __call__(self, audio_path: str, prompt: str) -> str:
        """Return the raw model output for `(audio_path, prompt)`."""
        key = self._cache_key(audio_path, prompt)
        if key in self._mem_cache:
            return self._mem_cache[key]
        disk = self._cache_path(key)
        if disk.is_file():
            try:
                obj = json.loads(disk.read_text(encoding="utf-8"))
                self._mem_cache[key] = obj["output"]
                return obj["output"]
            except (json.JSONDecodeError, KeyError):
                pass  # corrupted cache entry — recompute
        with torch.no_grad():
            output = run_inference(
                self.model, self.processor, audio_path, prompt, self.gen_kwargs,
            )
        disk.write_text(json.dumps({
            "audio_path": audio_path,
            "prompt": prompt,
            "output": output,
        }, indent=2), encoding="utf-8")
        self._mem_cache[key] = output
        return output

    def cache_stats(self) -> dict:
        return {
            "memory_entries": len(self._mem_cache),
            "disk_entries": len(list(self._cache_dir.glob("*.json"))),
            "cache_dir": str(self._cache_dir),
        }
