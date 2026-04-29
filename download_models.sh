#!/usr/bin/env bash
# Download MOSS-Audio model weights into ./MOSS-Audio/weights/.
#
# Usage:
#     bash download_models.sh 4b-instruct           # one variant
#     bash download_models.sh 4b-instruct 8b-thinking
#     bash download_models.sh all                   # all four
#
# Disk-size guide (download size is similar to on-disk size):
#     4B variants: ~9 GB each
#     8B variants: ~17 GB each

set -euo pipefail

declare -A REPOS=(
    ["4b-instruct"]="OpenMOSS-Team/MOSS-Audio-4B-Instruct"
    ["4b-thinking"]="OpenMOSS-Team/MOSS-Audio-4B-Thinking"
    ["8b-instruct"]="OpenMOSS-Team/MOSS-Audio-8B-Instruct"
    ["8b-thinking"]="OpenMOSS-Team/MOSS-Audio-8B-Thinking"
)

WEIGHTS_BASE="MOSS-Audio/weights"

if [ "$#" -eq 0 ]; then
    echo "Usage: bash download_models.sh <variant> [<variant> ...]" >&2
    echo "Variants: ${!REPOS[*]} | all" >&2
    exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "Installing huggingface_hub CLI..."
    pip install -U "huggingface_hub[cli]"
fi

mkdir -p "${WEIGHTS_BASE}"

REQUESTED=("$@")
if [ "${REQUESTED[0]}" = "all" ]; then
    REQUESTED=(4b-instruct 4b-thinking 8b-instruct 8b-thinking)
fi

for variant in "${REQUESTED[@]}"; do
    repo="${REPOS[${variant}]:-}"
    if [ -z "${repo}" ]; then
        echo "Unknown variant: '${variant}'" >&2
        echo "Valid: ${!REPOS[*]}" >&2
        exit 1
    fi
    target="${WEIGHTS_BASE}/$(basename "${repo}")"
    if [ -d "${target}" ] && [ -n "$(ls -A "${target}" 2>/dev/null)" ]; then
        echo ">>> ${variant} already at ${target} - skipping."
        continue
    fi
    echo ">>> Downloading ${repo} -> ${target}"
    hf download "${repo}" --local-dir "${target}"
done

echo ""
echo "Done. To run with the 4B-Instruct (recommended for 32 GB CPU machines):"
echo "    cd MOSS-Audio"
echo "    python infer_local.py \\"
echo "        --variant 4b-instruct \\"
echo "        --audio your_clip.mp3 \\"
echo "        --device cpu \\"
echo "        --prompt \"Transcribe this audio.\""
