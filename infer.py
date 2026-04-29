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
        return torch.bfloat16 if device.startswith("cuda") else torch.float16
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[requested]


def configure_cpu_threads():
    n = os.cpu_count() or 4
    torch.set_num_threads(n)
    torch.set_num_interop_threads(2)
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
    """Run one audio file through the model. Returns the decoded text."""
    raw_audio = load_audio(audio_path, sample_rate=processor.config.mel_sr)
    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
    inputs = inputs.to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
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

    for idx, entry in enumerate(state["results"]):
        if entry["status"] == "done":
            continue
        entry["status"] = "running"
        write_state(output_dir, state)
        t0 = time.time()
        print(f"[batch] [{idx+1}/{len(files)}] {entry['path']}", file=sys.stderr)
        try:
            output = run_inference(model, processor, entry["path"],
                                   args.prompt, gen_kwargs)
            entry["status"] = "done"
            entry["output"] = output
            entry["error"] = ""
        except Exception as exc:  # noqa: BLE001 - we want to keep going
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["output"] = ""
            print(f"[batch]   ERROR: {entry['error']}", file=sys.stderr)
        entry["duration_sec"] = round(time.time() - t0, 2)
        entry["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        write_state(output_dir, state)

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

    p.add_argument("--prompt", default="Describe this audio.")
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', 'cuda', or 'cuda:N'.")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "float32", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=512)
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
        n_threads = configure_cpu_threads()
        est_gb = ram_estimate_gb(dtype, params)
        sys_gb = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                       / 1024 ** 3, 1)
        print(f"[infer] model={model_path}  params=~{round(params/1e9, 1)}B",
              file=sys.stderr)
        print(f"[infer] device=cpu  dtype={dtype}  threads={n_threads}",
              file=sys.stderr)
        print(f"[infer] estimated model RAM: ~{est_gb} GB (system has {sys_gb} GB)",
              file=sys.stderr)
        if est_gb > sys_gb - 4:
            print("[infer] WARNING: model may not fit. Try --variant 4b-instruct.",
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
