#!/usr/bin/env bash
# Patch a stale MOSS-Audio checkout: comments out the unused
# `import torchaudio` in src/processing_moss_audio.py and refreshes
# infer_local.py from this repo's infer.py. Use this when setup.sh
# already ran but the upstream file or infer_local.py is stale.
#
# Usage:
#     bash patch_existing.sh [path-to-MOSS-Audio]    # default: ./MOSS-Audio

set -euo pipefail

REPO_DIR="${1:-MOSS-Audio}"
PROC_FILE="${REPO_DIR}/src/processing_moss_audio.py"

if [ ! -d "${REPO_DIR}" ]; then
    echo "ERROR: ${REPO_DIR} not found. Pass the path to your MOSS-Audio checkout." >&2
    exit 1
fi

if [ -f "${PROC_FILE}" ] && grep -qE '^import torchaudio$' "${PROC_FILE}"; then
    echo ">>> Commenting out 'import torchaudio' in ${PROC_FILE}"
    sed -i 's/^import torchaudio$/# import torchaudio  # patched: unused, breaks on CPU-only hosts/' "${PROC_FILE}"
else
    echo ">>> ${PROC_FILE} already patched (or missing)"
fi

echo ">>> Refreshing ${REPO_DIR}/infer_local.py"
cp infer.py "${REPO_DIR}/infer_local.py"

echo ">>> Removing torchaudio / torchcodec if installed"
pip uninstall -y torchaudio torchcodec >/dev/null 2>&1 || true

echo "Done. Now run:"
echo "    cd ${REPO_DIR}"
echo "    python infer_local.py --audio your.mp3 --model ./weights/MOSS-Audio-8B-Thinking --device cpu --prompt \"Describe this audio.\""
