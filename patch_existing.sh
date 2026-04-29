#!/usr/bin/env bash
# Apply all patches to an existing MOSS-Audio checkout.
# Safe to re-run. Pass the MOSS-Audio directory as $1 (default: ./MOSS-Audio).
#
# Usage:
#     bash patch_existing.sh [path-to-MOSS-Audio]

set -euo pipefail

REPO_DIR="${1:-MOSS-Audio}"

if [ ! -d "${REPO_DIR}" ]; then
    echo "ERROR: '${REPO_DIR}' not found." >&2
    exit 1
fi

# 1. Comment out the dead `import torchaudio` line in the processor.
PROC_FILE="${REPO_DIR}/src/processing_moss_audio.py"
if [ -f "${PROC_FILE}" ]; then
    if grep -qP '^import torchaudio' "${PROC_FILE}"; then
        echo ">>> Patching torchaudio import in ${PROC_FILE}"
        # Use python to handle any whitespace/encoding variation
        python3 - <<PYEOF
path = "${PROC_FILE}"
lines = open(path, encoding="utf-8").readlines()
patched = []
changed = 0
for line in lines:
    if line.strip() == "import torchaudio":
        patched.append("# " + line.lstrip())
        changed += 1
    else:
        patched.append(line)
open(path, "w", encoding="utf-8").writelines(patched)
print(f"  patched {changed} line(s)")
PYEOF
    else
        echo ">>> ${PROC_FILE} already patched."
    fi
else
    echo "WARNING: ${PROC_FILE} not found - check your MOSS-Audio path" >&2
fi

# 2. Verify no other src/ files import torchaudio at module level.
echo ">>> Checking for remaining torchaudio imports in src/..."
REMAINING=$(grep -rn "^import torchaudio" "${REPO_DIR}/src/" 2>/dev/null || true)
if [ -n "${REMAINING}" ]; then
    echo "WARNING: still found torchaudio imports:"
    echo "${REMAINING}"
else
    echo "    None found - all clear."
fi

# 3. Refresh infer_local.py.
echo ">>> Refreshing ${REPO_DIR}/infer_local.py"
cp infer.py "${REPO_DIR}/infer_local.py"
echo "    Done. Verify line 30 is the torchaudio stub:"
sed -n '28,32p' "${REPO_DIR}/infer_local.py"

# 4. Remove torchaudio / torchcodec.
echo ">>> Uninstalling torchaudio / torchcodec"
pip uninstall -y torchaudio torchcodec 2>/dev/null || true

echo ""
echo "All patches applied. Now run:"
echo "    cd ${REPO_DIR}"
echo "    python infer_local.py \\"
echo "        --audio your_file.mp3 \\"
echo "        --model ./weights/MOSS-Audio-8B-Thinking \\"
echo "        --device cpu \\"
echo "        --max-new-tokens 256 \\"
echo "        --prompt \"Describe this audio.\""
