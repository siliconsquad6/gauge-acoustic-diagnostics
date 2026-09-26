#!/bin/bash
# Fast build (~70 min): 2 units per machine, 20 steps, 1024x768 cutaways.
# Run from the gauge folder with the server STOPPED:  bash build_fast.sh 2>&1 | tee build.log
set -e; cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate gauge

# fewer diffusion steps for the pump/valve rebuild
sed -i 's/num_inference_steps=40/num_inference_steps=20/; s/num_inference_steps=30/num_inference_steps=20/; s/num_inference_steps=25/num_inference_steps=20/' app/rebuild_visuals.py

python -u app/rebuild_visuals.py pump valve          # X-ray parts for pump + valve (~25 min)
python app/export_parts_print.py                     # STL/OBJ (~2 min)

UNITS="fan_id_00 fan_id_06 pump_id_00 pump_id_06 valve_id_00 valve_id_06"
for u in $UNITS; do rm -f web/cache/$u/cutaway_*.png; done
STEPS=20 CW=1024 CH=768 python -u app/build_assets.py $UNITS   # (~40 min)
echo "FAST BUILD DONE"
