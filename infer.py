"""Local inference for OpenMOSS-Team MOSS-Audio variants (CPU-friendly).

Single-file mode:
    python infer_local.py --variant 4b-instruct --audio clip.mp3 --device cpu

Batch mode:
    python infer_local.py --variant 4b-instruct --batch-dir ./audio/ --device cpu
    open ./batch_output/dashboard.html

The wrapper has no torchaudio / torchcodec dependency; all decoding goes
through librosa.

RAM guide (no GPU):
    8B model, float16  ~17 GB  (fits in 32 GB)
    8B model, float32  ~34 GB  (will OOM on 32 GB)
    4B model, float32  ~18 GB  (safer)
    4B model, float16   ~9 GB  (most headroom)
"""

import argparse
import datetime as _dt
import html
import json
import os
import sys
import time

# IMPORTANT: thread env vars MUST be set BEFORE `import torch`. PyTorch
# reads OMP_NUM_THREADS / MKL_NUM_THREADS at OpenMP init; calling
# torch.set_num_threads() afterwards only resizes the pool, it cannot
# undo a OMP_NUM_THREADS=1 that was already in the env. We default to
# the physical core count (excluding hyperthreads). Hyperthreads
# typically *hurt* matmul-bound inference because the two siblings
# fight for the same FP units.

def _physical_core_count():
    """Return physical CPU cores; fall back to logical count."""
    try:
        cores = set()
        phys_id = core_id = None
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                elif not line.strip() and phys_id is not None:
                    cores.add((phys_id, core_id))
                    phys_id = core_id = None
        if cores:
            return len(cores)
    except (FileNotFoundError, OSError):
        pass
    return os.cpu_count() or 4

_DEFAULT_THREADS = str(_physical_core_count())
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _DEFAULT_THREADS)
# Bind threads to physical cores so the OS doesn't shuffle them across
# E-cores / hyperthread siblings mid-matmul.
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

# We do NOT stub torchaudio in sys.modules. transformers' import_utils
# calls importlib.util.find_spec("torchaudio") at import time; a stub
# with __spec__ = None makes that raise ValueError. Instead, the
# upstream's dead `import torchaudio` line is patched out by
# patch_existing.sh (and the real torchaudio is uninstalled), so
# nothing tries to import it.

import librosa
import numpy as np
import torch

# When this file is invoked via symlink (e.g. MOSS-Audio/infer_local.py
# -> ../infer.py), Python 3.11+ resolves sys.path[0] to the symlink
# *target*'s directory (this repo root), not the invocation directory.
# That breaks `from src.* import ...` because src/ lives next to the
# symlink, not the target. Add the cwd + the symlink's own directory
# to sys.path so the import works in both cases.
for _p in (
    os.getcwd(),
    os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else "",
):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# oneDNN / mkldnn handle most CPU matmul kernels — make sure it's on.
try:
    torch.backends.mkldnn.enabled = True
except AttributeError:
    pass

from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor


# ---------------------------------------------------------------------------
# Model variant registry
# ---------------------------------------------------------------------------

VARIANTS = {
    "4b-instruct": {"repo": "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
                    "dirname": "MOSS-Audio-4B-Instruct", "params": 4.6e9},
    "4b-thinking": {"repo": "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
                    "dirname": "MOSS-Audio-4B-Thinking", "params": 4.6e9},
    "8b-instruct": {"repo": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
                    "dirname": "MOSS-Audio-8B-Instruct", "params": 8.6e9},
    "8b-thinking": {"repo": "OpenMOSS-Team/MOSS-Audio-8B-Thinking",
                    "dirname": "MOSS-Audio-8B-Thinking", "params": 8.6e9},
}

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm",
              ".aac", ".wma", ".aiff", ".aif")


def resolve_model_path(variant, model):
    """Return (model_path, estimated_param_count). --model wins over --variant."""
    if model:
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

def load_audio(path, sample_rate=16000):
    """Mono float32 numpy array at sample_rate. No torchaudio/torchcodec."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path!r}")
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if isinstance(audio, np.ndarray):
        audio = audio.astype("float32")
    return audio


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def resolve_device(requested):
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but no NVIDIA driver/GPU visible. "
                         "Re-run with --device cpu.")
    return requested


def resolve_dtype(requested, device):
    if requested == "auto":
        # bfloat16 is the right CPU default: same memory as fp16 but with
        # better oneDNN kernel support. fp16 on CPU is usually emulated
        # via fp32 upcast and ends up slower despite the smaller weights.
        return torch.bfloat16 if device.startswith("cuda") else torch.bfloat16
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[requested]


def configure_cpu_threads(requested=None):
    """Set torch's thread pool. Defaults to physical cores."""
    n = requested if requested else _physical_core_count()
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # set_num_interop_threads can only be called once before any
        # parallel work happens; ignore if it's already locked in.
        pass
    return n


def ram_estimate_gb(dtype, params):
    bytes_per_param = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(dtype, 4)
    return round(params * bytes_per_param / 1024 ** 3, 1)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model_and_processor(model_path, dtype, device, enable_time_marker):
    print("[infer] Loading model weights ...", file=sys.stderr)
    model = MossAudioModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True,
    )
    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        model_path, trust_remote_code=True,
        enable_time_marker=enable_time_marker,
    )
    return model, processor


def run_inference(model, processor, audio_path, prompt, gen_kwargs):
    """Run one audio file through the model. Returns the decoded text.

    Pure sequential: no threads, no streaming. The model.generate() call
    blocks until done, then we decode and return.
    """
    sample_rate = getattr(getattr(processor, "config", None), "mel_sr", None) or 16000
    raw_audio = load_audio(audio_path, sample_rate=sample_rate)
    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
    inputs = inputs.to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    if hasattr(processor, "audio_token_id"):
        inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Batch mode + dashboard
# ---------------------------------------------------------------------------

def find_audio_files(input_dir, recursive=True):
    """Return sorted list of audio files under input_dir."""
    paths = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(AUDIO_EXTS):
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_dir):
            full = os.path.join(input_dir, f)
            if os.path.isfile(full) and f.lower().endswith(AUDIO_EXTS):
                paths.append(full)
    return sorted(paths)


def write_dashboard(output_dir, state):
    """Write a self-contained HTML dashboard from `state`. Auto-refreshes."""
    refresh_meta = '<meta http-equiv="refresh" content="5">' if state["status"] == "running" else ""
    started = state.get("started_at", "")
    finished = state.get("finished_at", "")
    rows_html = []
    for r in state["results"]:
        status = r["status"]
        badge_color = {"done": "#16a34a", "error": "#dc2626",
                       "running": "#2563eb", "pending": "#6b7280"}.get(status, "#6b7280")
        text = html.escape(r.get("output", "") or r.get("error", "") or "")
        if len(text) > 400:
            preview = text[:400] + "..."
            full = text
            text_html = (f'<details><summary>{html.escape(preview)}</summary>'
                         f'<pre>{html.escape(full)}</pre></details>')
        else:
            text_html = f'<pre>{text}</pre>' if text else ""
        rows_html.append(f"""
            <tr>
                <td class="filename">{html.escape(os.path.basename(r["path"]))}</td>
                <td><span class="badge" style="background:{badge_color}">{status}</span></td>
                <td>{r.get("duration_sec", "") and f"{r['duration_sec']:.1f}s" or ""}</td>
                <td class="output">{text_html}</td>
            </tr>""")

    total = len(state["results"])
    done = sum(1 for r in state["results"] if r["status"] == "done")
    failed = sum(1 for r in state["results"] if r["status"] == "error")
    running = sum(1 for r in state["results"] if r["status"] == "running")
    pending = total - done - failed - running
    pct = (done + failed) / total * 100 if total else 0
    avg = (sum(r.get("duration_sec", 0) for r in state["results"] if r["status"] == "done")
           / max(done, 1))
    eta_min = pending * avg / 60 if avg else 0

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MOSS-Audio Batch Dashboard</title>
{refresh_meta}
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
          margin: 0; padding: 24px; background:#f8fafc; color:#0f172a; }}
  h1 {{ margin-top:0; font-size:22px; }}
  .meta {{ color:#475569; font-size:13px; margin-bottom:18px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
            gap:12px; margin-bottom:18px; }}
  .stat {{ background:white; border:1px solid #e2e8f0; border-radius:8px;
           padding:12px 14px; }}
  .stat .label {{ font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:.05em; }}
  .stat .value {{ font-size:22px; font-weight:600; margin-top:4px; }}
  .progress {{ height:6px; background:#e2e8f0; border-radius:3px; overflow:hidden; margin-bottom:18px; }}
  .progress > div {{ height:100%; background:#16a34a; transition:width .4s; }}
  table {{ width:100%; border-collapse:collapse; background:white;
           border:1px solid #e2e8f0; border-radius:8px; overflow:hidden; }}
  th, td {{ text-align:left; padding:10px 12px; border-bottom:1px solid #e2e8f0;
            vertical-align:top; font-size:14px; }}
  th {{ background:#f1f5f9; font-weight:600; }}
  td.filename {{ font-family: ui-monospace, monospace; font-size:13px; word-break:break-all; max-width:260px; }}
  td.output {{ max-width: 720px; }}
  td.output pre {{ white-space:pre-wrap; word-break:break-word; margin:0;
                   font-family: ui-monospace, monospace; font-size:13px; }}
  .badge {{ display:inline-block; color:white; padding:2px 8px; border-radius:99px;
            font-size:12px; font-weight:600; }}
  details summary {{ cursor:pointer; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#0f172a; color:#e2e8f0; }}
    .stat, table {{ background:#1e293b; border-color:#334155; }}
    th {{ background:#334155; }}
    th, td {{ border-color:#334155; }}
    .meta {{ color:#94a3b8; }}
  }}
</style>
</head>
<body>
<h1>MOSS-Audio Batch Dashboard</h1>
<div class="meta">
  <strong>Status:</strong> {state["status"]} &middot;
  <strong>Model:</strong> {html.escape(state["model"])} &middot;
  <strong>Started:</strong> {started or "—"} &middot;
  <strong>Finished:</strong> {finished or "—"}
  {f' &middot; Auto-refresh every 5s' if state["status"] == "running" else ""}
</div>

<div class="stats">
  <div class="stat"><div class="label">Total</div><div class="value">{total}</div></div>
  <div class="stat"><div class="label">Done</div><div class="value" style="color:#16a34a">{done}</div></div>
  <div class="stat"><div class="label">Failed</div><div class="value" style="color:#dc2626">{failed}</div></div>
  <div class="stat"><div class="label">Pending</div><div class="value" style="color:#6b7280">{pending}</div></div>
  <div class="stat"><div class="label">Avg / file</div><div class="value">{avg:.1f}s</div></div>
  <div class="stat"><div class="label">ETA</div><div class="value">{eta_min:.1f} min</div></div>
</div>

<div class="progress"><div style="width:{pct:.1f}%"></div></div>

<table>
  <thead><tr><th>File</th><th>Status</th><th>Duration</th><th>Transcript / Output</th></tr></thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>
</body>
</html>"""
    out_html = os.path.join(output_dir, "dashboard.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)


def write_state(output_dir, state):
    """Persist results.json + dashboard.html atomically."""
    json_path = os.path.join(output_dir, "results.json")
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, json_path)
    write_dashboard(output_dir, state)


def load_existing_state(output_dir):
    """Return prior results.json if present (for resume), else None."""
    p = os.path.join(output_dir, "results.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def run_batch(args, model, processor, model_path, dtype):
    """Process all audio files in args.batch_dir; write dashboard + JSON."""
    files = find_audio_files(args.batch_dir, recursive=not args.no_recursive)
    if not files:
        print(f"[batch] No audio files found in {args.batch_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or os.path.join(args.batch_dir, "_batch_output")
    os.makedirs(output_dir, exist_ok=True)

    # Resume: load prior state if it exists.
    prior = load_existing_state(output_dir) if not args.no_resume else None
    prior_done = {r["path"]: r for r in (prior or {}).get("results", [])
                  if r.get("status") == "done"}

    state = {
        "status": "running",
        "model": model_path,
        "prompt": args.prompt,
        "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "finished_at": "",
        "results": [
            prior_done.get(p, {"path": p, "status": "pending",
                               "output": "", "error": "",
                               "duration_sec": 0.0,
                               "finished_at": ""})
            for p in files
        ],
    }
    write_state(output_dir, state)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens, do_sample=True, num_beams=1,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        use_cache=True,
    )

    print(f"[batch] {len(files)} file(s); {len(prior_done)} already done. "
          f"Output: {output_dir}/dashboard.html", file=sys.stderr)

    pending_indices = [i for i, r in enumerate(state["results"]) if r["status"] != "done"]
    last_pending = pending_indices[-1] if pending_indices else -1

    for idx, entry in enumerate(state["results"]):
        if entry["status"] == "done":
            continue
        entry["status"] = "running"
        entry["output"] = ""
        entry["error"] = ""
        write_state(output_dir, state)
        t0 = time.time()
        print(f"[batch] [{idx+1}/{len(files)}] {entry['path']}", file=sys.stderr)

        try:
            output = run_inference(model, processor, entry["path"],
                                   args.prompt, gen_kwargs)
            entry["status"] = "done"
            entry["output"] = output
            entry["error"] = ""
        except Exception as exc:  # noqa: BLE001 - keep going on per-file errors
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[batch]   ERROR: {entry['error']}", file=sys.stderr)

        entry["duration_sec"] = round(time.time() - t0, 2)
        entry["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        write_state(output_dir, state)
        print(f"[batch]   done in {entry['duration_sec']:.1f}s", file=sys.stderr)

        if args.sleep_between > 0 and idx != last_pending:
            print(f"[batch]   pausing {args.sleep_between}s before next file ...",
                  file=sys.stderr, flush=True)
            time.sleep(args.sleep_between)

    state["status"] = "finished"
    state["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    write_state(output_dir, state)
    print(f"[batch] Done. Open {output_dir}/dashboard.html in a browser.",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run MOSS-Audio on one file or a folder. CPU-friendly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument("--variant", choices=sorted(VARIANTS.keys()),
                             help="Shorthand for one of the four MOSS-Audio variants.")
    model_group.add_argument("--model",
                             help="Custom local model directory.")

    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio",
                             help="Single audio file to process.")
    input_group.add_argument("--batch-dir",
                             help="Folder of audio files; processes all of them "
                                  "and writes a dashboard.")

    p.add_argument("--output-dir", default=None,
                   help="Where to write batch results + dashboard. "
                        "Default: <batch-dir>/_batch_output/.")
    p.add_argument("--no-recursive", action="store_true",
                   help="Don't recurse into subdirectories of --batch-dir.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing results.json and start over.")
    p.add_argument("--sleep-between", type=float, default=10.0,
                   help="Seconds to pause between files in batch mode "
                        "(lets CPU/RAM cool down). Default: 10. Set to 0 to disable.")

    p.add_argument("--prompt", default="Describe this audio.")
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', 'cuda', or 'cuda:N'.")
    p.add_argument("--threads", type=int, default=None,
                   help=f"CPU threads for torch (default: physical cores = "
                        f"{_physical_core_count()}). Try the physical-core "
                        f"count, not logical — hyperthreads usually hurt "
                        f"matmul-bound inference.")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "float32", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Generation cap. Lower = faster. Default: 256.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--no-time-marker", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model_path, params = resolve_model_path(args.variant, args.model)
    if not os.path.isdir(model_path):
        raise SystemExit(
            f"Model directory not found: {model_path!r}\n"
            f"Download it with: bash download_models.sh "
            f"{(args.variant or '4b-instruct')}"
        )
    if args.batch_dir and not os.path.isdir(args.batch_dir):
        raise SystemExit(f"--batch-dir not found: {args.batch_dir!r}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    if device == "cpu":
        n_threads = configure_cpu_threads(args.threads)
        est_gb = ram_estimate_gb(dtype, params)
        sys_gb = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                       / 1024 ** 3, 1)
        logical = os.cpu_count() or n_threads
        print(f"[infer] model={model_path}  params=~{round(params/1e9, 1)}B",
              file=sys.stderr)
        print(f"[infer] device=cpu  dtype={dtype}  "
              f"threads={n_threads} (physical cores; logical={logical})",
              file=sys.stderr)
        print(f"[infer] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
              f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
              f"mkldnn={getattr(torch.backends.mkldnn, 'enabled', '?')}",
              file=sys.stderr)
        print(f"[infer] estimated model RAM: ~{est_gb} GB (system has {sys_gb} GB)",
              file=sys.stderr)
        if est_gb > sys_gb - 4:
            print("[infer] WARNING: model may not fit. Try --variant 4b-instruct.",
                  file=sys.stderr)
        if dtype == torch.float16:
            print("[infer] NOTE: float16 on CPU is usually slower than bfloat16 "
                  "(same memory). Drop --dtype to use auto/bfloat16.",
                  file=sys.stderr)
    else:
        print(f"[infer] model={model_path}  device={device}  dtype={dtype}",
              file=sys.stderr)

    model, processor = load_model_and_processor(
        model_path, dtype, device, enable_time_marker=not args.no_time_marker)

    if args.batch_dir:
        return run_batch(args, model, processor, model_path, dtype)

    print(f"[infer] Loading audio: {args.audio}", file=sys.stderr)
    print(f"[infer] Generating (max_new_tokens={args.max_new_tokens}) ...",
          file=sys.stderr)
    output = run_inference(
        model, processor, args.audio, args.prompt,
        gen_kwargs=dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                        num_beams=1, temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k, use_cache=True),
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
