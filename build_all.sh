#!/bin/bash
# Build every remaining visual in one go (server must be stopped). ~3 h. Run inside tmux.
set -e
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate gauge
echo "== base visuals: pump, valve ==";   python -u app/rebuild_visuals.py pump valve
echo "== print files ==";                 python app/export_parts_print.py
echo "== units: fan, pump, valve ==";     rm -f web/cache/*_id_*/cutaway_*.png
     python -u app/build_assets.py fan pump valve
echo "ALL DONE $(date)"
