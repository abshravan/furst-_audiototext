#!/usr/bin/env bash
# Nuke the old venv, rebuild from scratch, repatch MOSS-Audio, and run.
# Defaults match the typical /home/admin/frust_text_audio/ layout.
# Override any path via env var before running, e.g.:
#     PROJ_DIR=/some/other/path DEVICE=cpu bash reset_and_run.sh
#
# Run this from inside the cloned furst-_audiototext repo:
#     bash reset_and_run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- configuration (override via env) ----------
PROJ_DIR="${PROJ_DIR:-/home/admin/frust_text_audio}"
MOSS_DIR="${MOSS_DIR:-${PROJ_DIR}/MOSS-Audio}"
WEIGHTS="${WEIGHTS:-${MOSS_DIR}/weights/MOSS-Audio-4B-Thinking}"
AUDIO_DIR="${AUDIO_DIR:-${PROJ_DIR}/copied_audio}"
VENV="${VENV:-${PROJ_DIR}/venv-moss}"
DEVICE="${DEVICE:-auto}"           # auto | cpu | cuda
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-1}"
PROMPT_DEFAULT='Analyze the provided audio clip and return strict JSON with fields: frustration ("yes" or "no"), confidence (0-100), reason (short evidence-based explanation using tone, speech patterns, pace, interruptions, sighs, pitch, or wording), and evidence (array of timestamps or conversation segments with detected frustration indicators and short supporting quotes/descriptions); detect emotional frustration only, avoid assumptions, and if unclear return the most probable result with lower confidence.'
PROMPT="${PROMPT:-${PROMPT_DEFAULT}}"
# ------------------------------------------------------

echo "=========================================="
echo " Configuration"
echo "=========================================="
echo "  SCRIPT_DIR:    ${SCRIPT_DIR}"
echo "  PROJ_DIR:      ${PROJ_DIR}"
echo "  MOSS_DIR:      ${MOSS_DIR}"
echo "  WEIGHTS:       ${WEIGHTS}"
echo "  AUDIO_DIR:     ${AUDIO_DIR}"
echo "  VENV:          ${VENV}"
echo "  DEVICE:        ${DEVICE}"
echo "  MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "  SLEEP_BETWEEN: ${SLEEP_BETWEEN}"
echo "=========================================="
echo ""

# ---------- 0. sanity checks ----------
if [ ! -f "${SCRIPT_DIR}/requirements.txt" ] || [ ! -f "${SCRIPT_DIR}/infer.py" ]; then
    echo "ERROR: run this from inside the furst-_audiototext repo." >&2
    echo "       Expected ${SCRIPT_DIR}/requirements.txt and infer.py to exist." >&2
    exit 1
fi
if [ ! -d "${PROJ_DIR}" ]; then
    echo "ERROR: PROJ_DIR does not exist: ${PROJ_DIR}" >&2
    exit 1
fi
if [ ! -d "${AUDIO_DIR}" ]; then
    echo "ERROR: AUDIO_DIR does not exist: ${AUDIO_DIR}" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found in PATH." >&2
    exit 1
fi

# ---------- 1. wipe old venv ----------
if [ -d "${VENV}" ]; then
    echo ">>> Removing old venv: ${VENV}"
    rm -rf "${VENV}"
fi

# ---------- 2. fresh venv ----------
echo ">>> Creating fresh venv at ${VENV}"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
echo "    python: $(which python)"
echo "    version: $(python --version)"

# ---------- 3. pick torch wheel ----------
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA=1
fi

WANT_CUDA=0
case "${DEVICE}" in
    cuda|cuda:*) WANT_CUDA=1 ;;
    auto)        WANT_CUDA=${HAS_NVIDIA} ;;
    cpu)         WANT_CUDA=0 ;;
esac

if [ "${WANT_CUDA}" = "1" ] && [ "${HAS_NVIDIA}" = "1" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo ">>> Using CUDA torch wheels (${TORCH_INDEX})"
    EFFECTIVE_DEVICE="${DEVICE}"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo ">>> Using CPU torch wheels (${TORCH_INDEX})"
    if [ "${DEVICE}" != "cpu" ] && [ "${WANT_CUDA}" = "1" ]; then
        echo "    NOTE: --device cuda requested but no GPU detected; falling back to cpu."
    fi
    EFFECTIVE_DEVICE="cpu"
fi

# ---------- 4. install deps ----------
echo ">>> Upgrading pip"
pip install --upgrade pip
echo ">>> Installing requirements"
pip install --extra-index-url "${TORCH_INDEX}" -r "${SCRIPT_DIR}/requirements.txt"

# Make sure no torchaudio / torchcodec leaked back in.
pip uninstall -y torchaudio torchcodec >/dev/null 2>&1 || true

# ---------- 5. ensure MOSS-Audio source exists ----------
if [ ! -d "${MOSS_DIR}" ]; then
    echo ">>> Cloning MOSS-Audio into ${MOSS_DIR}"
    git clone https://github.com/OpenMOSS/MOSS-Audio.git "${MOSS_DIR}"
else
    echo ">>> Reusing existing ${MOSS_DIR}"
fi

# ---------- 6. patch upstream + install symlink ----------
echo ">>> Running patch_existing.sh"
bash "${SCRIPT_DIR}/patch_existing.sh" "${MOSS_DIR}"

# ---------- 7. weights present? ----------
if [ ! -d "${WEIGHTS}" ]; then
    echo ""
    echo "ERROR: model weights not found at ${WEIGHTS}" >&2
    echo "       Download them, e.g.:" >&2
    echo "           cd ${SCRIPT_DIR} && bash download_models.sh 4b-thinking" >&2
    echo "       (Then re-run this script.)" >&2
    exit 1
fi

# ---------- 8. verify torch ----------
echo ">>> Verifying torch install"
python -c "import torch; print('  torch', torch.__version__, '  cuda available:', torch.cuda.is_available())"

# ---------- 9. run inference ----------
echo ""
echo "=========================================="
echo " Starting batch inference"
echo "=========================================="
cd "${MOSS_DIR}"
exec python -u infer_local.py \
    --model "${WEIGHTS}" \
    --batch-dir "${AUDIO_DIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt "${PROMPT}" \
    --device "${EFFECTIVE_DEVICE}" \
    --sleep-between "${SLEEP_BETWEEN}"
