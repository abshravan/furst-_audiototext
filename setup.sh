#!/usr/bin/env bash
# Stable setup for OpenMOSS-Team/MOSS-Audio-8B-Thinking.
#
# Avoids the upstream `[torch-runtime]` extra (which pulls torchaudio +
# torchcodec and breaks on FFmpeg ABI mismatches). Instead we install
# a pinned minimal stack from requirements.txt and use librosa for I/O.
#
# Usage:
#     bash setup.sh
#
# Idempotent: safe to re-run. Reuses any existing clone and weight dir.

set -euo pipefail

REPO_DIR="MOSS-Audio"
MODEL_REPO="OpenMOSS-Team/MOSS-Audio-8B-Thinking"
WEIGHTS_DIR="${REPO_DIR}/weights/MOSS-Audio-8B-Thinking"

# 1. Source tree (provides the `src` package imported by infer.py).
if [ ! -d "${REPO_DIR}" ]; then
    echo ">>> Cloning OpenMOSS/MOSS-Audio"
    git clone https://github.com/OpenMOSS/MOSS-Audio.git "${REPO_DIR}"
else
    echo ">>> Reusing existing clone at ${REPO_DIR}"
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

# 5. Download weights.
echo ">>> Ensuring weights at ${WEIGHTS_DIR}"
if [ ! -d "${WEIGHTS_DIR}" ] || [ -z "$(ls -A "${WEIGHTS_DIR}" 2>/dev/null)" ]; then
    hf download "${MODEL_REPO}" --local-dir "${WEIGHTS_DIR}"
else
    echo "    Already present - skipping download."
fi

# 6. Place infer.py inside the MOSS-Audio dir so its `src.*` imports resolve.
echo ">>> Installing infer_local.py into ${REPO_DIR}"
cp infer.py "${REPO_DIR}/infer_local.py"

cat <<EOF

Setup complete.

To run:
    cd ${REPO_DIR}
    python infer_local.py \\
        --audio path/to/clip.mp3 \\
        --model ./weights/MOSS-Audio-8B-Thinking \\
        --device auto \\
        --prompt "Describe this audio."

Force CPU on machines without an NVIDIA GPU:
    python infer_local.py ... --device cpu
EOF
