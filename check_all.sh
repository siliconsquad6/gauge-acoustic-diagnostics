#!/bin/bash
# Final readiness check. Run from the gauge folder:  bash check_all.sh
cd "$(dirname "$0")"
ok(){ printf "  \033[32m✔\033[0m %s\n" "$1"; }
bad(){ printf "  \033[31m✘\033[0m %s\n" "$1"; }
chk(){ [ -e "$1" ] && ok "$1" || bad "$1  (missing)"; }
echo "App files";   for f in app/server.py app/train_asset.py; do chk $f; done
echo "Web files";   for f in web/index.html web/viewer.html web/chat.html; do chk $f; done
echo "Models";      for f in models/omni models/qwen-image models/omni-gauge-lora models/asset_id.joblib; do chk $f; done
echo "Manuals";     for f in manuals/sources.json manuals/chunks.json manuals/index.npz; do chk $f; done
echo "Server patches"
for k in "PeftModel.from_pretrained:fine-tuned Omni" "def send_alert:alerts" "def retrieve(:repair chat" "HINT = {:real MIMII parts" \
         "def evidence(:evidence + trend" "def fault_photo(:fault image" "def asset_id(:unit fingerprint" 'cache_key=f"{mm}_{aid}":final patch'; do
  grep -qF "${k%%:*}" app/server.py && ok "${k##*:}" || bad "${k##*:}  (patch not applied)"; done
echo "Visuals per unit"
for m in fan pump valve; do for i in 00 02 04 06; do d=web/cache/${m}_id_$i
  if [ -f $d/machine.glb ]; then ok "$d ($(ls $d/cutaway_*.png 2>/dev/null | wc -l) cutaways)"; else bad "$d  (not built)"; fi; done; done
