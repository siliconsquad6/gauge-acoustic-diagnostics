"""2D cutaway diagnostic illustrations (catalog style) - one per machine part, with the fault callout drawn in.
Writes web/cache/<machine>/cutaway_<part>.png ; the dashboard shows the one matching Omni's diagnosis.

Run with the server STOPPED:
  python -u app/gen_cutaways.py fan          # ~15 min, check it first
  python -u app/gen_cutaways.py              # all three (~55 min)
  python -u app/gen_cutaways.py fan --redo   # regenerate even if files exist
"""
import sys
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

G = Path(__file__).resolve().parent.parent
CACHE = G / "web" / "cache"

STYLE = ("Clean 3D-rendered technical cutaway illustration of industrial equipment, product-catalog quality: a quarter-section is cut away "
         "to reveal the internal parts, cut faces painted red-brown, housings in glossy blue industrial paint, copper motor windings, "
         "polished silver shafts and ball bearings with visible balls, crisp edges, soft studio lighting with gentle shadows, "
         "pure white background, three-quarter side view, whole machine visible, centered, sharp and highly detailed.")
NEG = ("photo, photograph, blurry, messy, cluttered background, people, hands, watermark, logo, brand name, "
       "misspelled text, extra text, gibberish text, cartoon characters, low detail")

MACHINES = {
    "fan": ("an industrial centrifugal blower fan: cut-open steel spiral scroll housing revealing the backward-curved "
            "impeller wheel, the drive shaft, two ball bearings in bearing housings, and an electric motor cut open to "
            "show copper windings and rotor, all on a steel base frame"),
    "pump": ("an industrial end-suction centrifugal water pump: cut-open cast iron volute casing revealing the closed "
             "impeller, the mechanical seal behind it, the shaft, two ball bearings, and a close-coupled electric motor "
             "cut open to show copper windings and rotor, on a steel baseplate"),
    "valve": ("an industrial 2-way solenoid valve: cut-open brass valve body revealing the orifice valve seat and a "
              "rubber diaphragm, the steel plunger core and return spring inside the guide tube, and the black solenoid "
              "coil cut open to show copper windings, with threaded pipe ports"),
}
PARTS = {
    "fan":   {"impeller": ("impeller wheel", "imbalance at the rotating wheel"), "motor": ("electric motor windings", "irregular motor hum"),
              "bearing": ("drive-end ball bearing", "grinding near the rotating assembly"), "shaft": ("drive shaft", "vibration along the shaft")},
    "pump":  {"impeller": ("impeller", "clogging at the impeller"), "seal": ("mechanical seal", "leak noise at the seal"),
              "bearing": ("ball bearing", "grinding near the rotating assembly"), "shaft": ("shaft", "vibration along the shaft"),
              "motor": ("electric motor windings", "irregular motor hum")},
    "valve": {"seat": ("valve seat and orifice", "contamination at the seat"), "diaphragm": ("rubber diaphragm", "leak past the diaphragm"),
              "plunger": ("plunger core", "sticking plunger clicks"), "spring": ("return spring", "weak spring return"),
              "coil": ("solenoid coil", "irregular coil buzz")},
}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
redo = "--redo" in sys.argv
machines = args or list(MACHINES)
pipe = DiffusionPipeline.from_pretrained(G / "models" / "qwen-image", torch_dtype=torch.bfloat16).to("cuda")
for m in machines:
    for part, (what, hint) in PARTS[m].items():
        out = CACHE / m / f"cutaway_{part}.png"
        if out.exists() and not redo:
            continue
        label = f"SUSPECTED {part.upper()} FAULT"
        prompt = (f"{STYLE} The subject is {MACHINES[m]}. The {what} glows bright orange-red and is circled with a thick "
                  f"red ring; a red arrow points from it to a white callout box with a red border near the top right. "
                  f'The callout reads "{label}" in bold red capital letters, and below it in smaller black text '
                  f'"Acoustic anomaly: {hint}". No other text anywhere.')
        print(f"[cutaway] {m}/{part}", flush=True)
        pipe(prompt=prompt, negative_prompt=NEG, width=1472, height=1104, num_inference_steps=40,
             true_cfg_scale=4.0, generator=torch.Generator("cuda").manual_seed(5)).images[0].save(out)
print("DONE")
