#!/bin/bash
# Applies every server patch in the right order. Safe to re-run: each says "already patched" if done.
cd "$(dirname "$0")"
for p in patch_alerts patch_chat patch_visuals patch_evidence patch_faultphoto patch_manualfig patch_cutaway patch_assets patch_final; do
  echo "== $p"; python $p.py
done
python -m py_compile app/server.py && echo "server.py OK"
