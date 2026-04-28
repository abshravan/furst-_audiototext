#!/usr/bin/env bash
# One-shot setup for running MOSS-Audio-8B-Thinking locally.
#
# This clones the upstream MOSS-Audio repo (which contains the `src` package
# imported by infer.py), installs dependencies, downloads the 8B-Thinking
# weights, and copies infer.py into the repo so the imports resolve.
#
# Run from the directory that contains this script:
#     bash setup.sh

set -euo pipefail

REPO_DIR="MOSS-Audio"
MODEL_REPO="OpenMOSS-Team/MOSS-Audio-8B-Thinking"
WEIGHTS_DIR="${REPO_DIR}/weights/MOSS-Audio-8B-Thinking"

if [ ! -d "${REPO_DIR}" ]; then
    echo ">>> Cloning OpenMOSS/MOSS-Audio"
    git clone https://github.com/OpenMOSS/MOSS-Audio.git "${REPO_DIR}"
fi

echo ">>> Installing FFmpeg (needed for audio decoding)"
if command -v conda >/dev/null 2>&1; then
    conda install -c conda-forge "ffmpeg=7" -y
else
    echo "    conda not found - install FFmpeg manually (e.g. apt install ffmpeg)"
fi

echo ">>> Installing torch + MOSS-Audio runtime"
pushd "${REPO_DIR}" >/dev/null
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
popd >/dev/null

echo ">>> Installing huggingface_hub CLI"
pip install -U "huggingface_hub[cli]"

if [ ! -d "${WEIGHTS_DIR}" ]; then
    echo ">>> Downloading ${MODEL_REPO} (~17GB)"
    hf download "${MODEL_REPO}" --local-dir "${WEIGHTS_DIR}"
else
    echo ">>> Weights already present at ${WEIGHTS_DIR}"
fi

echo ">>> Copying infer.py into ${REPO_DIR}"
cp infer.py "${REPO_DIR}/infer_local.py"

cat <<EOF

Setup complete.

To run:
    cd ${REPO_DIR}
    python infer_local.py --audio path/to/clip.mp3 \\
        --model ./weights/MOSS-Audio-8B-Thinking \\
        --prompt "Describe this audio."
EOF
