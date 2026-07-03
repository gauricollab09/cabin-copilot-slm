#!/usr/bin/env bash
# Push a Kaggle kernel, substituting the username from ~/.kaggle/kaggle.json.
# Usage: scripts/kaggle_push.sh kaggle/teacher_gen
set -euo pipefail

KERNEL_DIR="${1:?usage: kaggle_push.sh <kernel dir>}"
USERNAME=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")

TMP=$(mktemp -d)
cp "$KERNEL_DIR"/* "$TMP/"
sed -i.bak "s/KAGGLE_USERNAME/$USERNAME/g" "$TMP/kernel-metadata.json" && rm "$TMP/kernel-metadata.json.bak"
kaggle kernels push -p "$TMP"
rm -rf "$TMP"
