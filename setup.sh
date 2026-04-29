#!/usr/bin/env bash
# Stable setup for the MOSS-Audio model family.
#
# Avoids the upstream `[torch-runtime]` extra (which pulls torchaudio +
# torchcodec and breaks on FFmpeg ABI mismatches). Installs a pinned
# minimal stack from requirements.txt and uses librosa for I/O.
#
# Usage:
#     bash setup.sh                 # install deps only (no weights)
#     bash setup.sh 4b-instruct     # also download the 4B-Instruct variant
#     bash setup.sh 8b-thinking
#
# After setup.sh finishes, use download_models.sh to fetch additional
# variants. Idempotent - safe to re-run.

set -euo pipefail

REPO_DIR="MOSS-Audio"
DEFAULT_VARIANT="${1:-}"

# 1. Source tree (provides the `src` package imported by infer.py).
if [ ! -d "${REPO_DIR}" ]; then
    echo ">>> Cloning OpenMOSS/MOSS-Audio"
    git clone https://github.com/OpenMOSS/MOSS-Audio.git "${REPO_DIR}"
else
    echo ">>> Reusing existing clone at ${REPO_DIR}"
fi

# 1b. Patch out the dead `import torchaudio` in src/processing_moss_audio.py.
#     The upstream module imports torchaudio at module level but never uses
#     it; on CPU-only machines the import explodes trying to load libcudart.
PROC_FILE="${REPO_DIR}/src/processing_moss_audio.py"
if [ -f "${PROC_FILE}" ]; then
    if grep -qE '^import torchaudio$' "${PROC_FILE}"; then
        echo ">>> Patching out 'import torchaudio' in ${PROC_FILE}"
        sed -i 's/^import torchaudio$/# import torchaudio  # patched: unused, breaks on CPU-only hosts/' "${PROC_FILE}"
    else
        echo ">>> ${PROC_FILE} already patched (no torchaudio import found)"
    fi
fi

# 2. Defensively remove fragile audio deps if a previous attempt installed them.
#    torchaudio is unused by our wrapper (we stub it in infer.py) so it can go.
echo ">>> Purging torchaudio / torchcodec if present"
pip uninstall -y torchaudio torchcodec >/dev/null 2>&1 || true

# 3. Pick a torch wheel index that matches the host. CPU-only by default;
#    GPU users can re-run with FORCE_CUDA=1 to keep their existing CUDA wheels.
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA=1
fi

if [ "${FORCE_CUDA:-0}" = "1" ] || [ "${HAS_NVIDIA}" = "1" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo ">>> NVIDIA GPU detected - using CUDA torch wheels (${TORCH_INDEX})"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo ">>> No NVIDIA GPU detected - using CPU-only torch wheels"
    # If a CUDA torch is already installed it will keep dragging libcudart in.
    pip uninstall -y torch >/dev/null 2>&1 || true
fi

# 4. Install the pinned minimal stack. We deliberately do NOT run
#    `pip install -e ".[torch-runtime]"` from inside MOSS-Audio - that
#    extra pulls torchcodec back in.
echo ">>> Installing pinned minimal stack from requirements.txt"
pip install --upgrade pip
pip install --extra-index-url "${TORCH_INDEX}" -r requirements.txt

# 5. Optionally download a model variant if requested.
if [ -n "${DEFAULT_VARIANT}" ]; then
    echo ">>> Downloading variant: ${DEFAULT_VARIANT}"
    bash download_models.sh "${DEFAULT_VARIANT}"
else
    echo ">>> Skipping weight download (run download_models.sh later)."
fi

# 6. Place infer.py inside the MOSS-Audio dir so its `src.*` imports resolve.
echo ">>> Installing infer_local.py into ${REPO_DIR}"
cp infer.py "${REPO_DIR}/infer_local.py"

cat <<EOF

Setup complete.

Next steps:
    1. Download a variant if you haven't already:
           bash download_models.sh 4b-instruct        # smallest, fastest
           bash download_models.sh 8b-thinking        # best reasoning

    2. Run inference:
           cd ${REPO_DIR}
           python infer_local.py \\
               --variant 4b-instruct \\
               --audio path/to/clip.mp3 \\
               --device cpu \\
               --prompt "Describe this audio."
EOF
