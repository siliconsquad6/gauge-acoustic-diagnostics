"""Generate the INTERNAL parts of each machine (one-time, offline) for the X-ray view.

For every part: Qwen-Image renders it -> Hunyuan3D builds the mesh -> image colors painted on.
Output: web/cache/<machine>/parts/<part>.glb  +  web/cache/<machine>/parts.json

Run (stop the server first to free GPU memory):
  python app/build_parts.py              # all machines (~20 min)
  python app/build_parts.py pump         # just one
"""
import gc, json, sys
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image

G = Path(__file__).resolve().parent.parent
CACHE = G / "web" / "cache"
QWEN = G / "models" / "qwen-image"

# part -> (what to draw, words in Omni's "likely_component" that point to this part)
PARTS = {
    "pump": {
        "impeller": ("metal centrifugal pump impeller with curved vanes", ["impeller", "rotor", "vane", "blade"]),
        "shaft":    ("long polished steel drive shaft with keyway", ["shaft", "coupling", "misalign"]),
        "bearing":  ("steel ball bearing, rings and visible balls", ["bearing"]),
        "seal":     ("mechanical pump seal ring assembly with rubber gasket", ["seal", "packing", "gasket", "leak"]),
        "motor":    ("electric motor stator with copper windings", ["motor", "winding", "stator", "electrical"]),
    },
    "fan": {
        "blades":  ("industrial fan blade rotor with five blades and hub", ["blade", "impeller", "rotor", "fan", "imbalance"]),
        "motor":   ("small electric fan motor with copper windings", ["motor", "winding", "stator", "electrical"]),
        "bearing": ("steel ball bearing, rings and visible balls", ["bearing"]),
        "shaft":   ("short steel motor shaft", ["shaft", "hub", "coupling"]),
    },
    "valve": {
        "stem":    ("threaded steel valve stem spindle", ["stem", "spindle", "handwheel"]),
        "disc":    ("metal valve disc plug", ["disc", "plug", "gate", "ball"]),
        "seat":    ("metal valve seat ring", ["seat", "leak"]),
        "packing": ("valve packing seal rings stack, graphite and rubber", ["seal", "packing", "gasket"]),
        "spring":  ("steel compression spring", ["spring", "actuator"]),
    },
}

machines = sys.argv[1:] or list(PARTS)


def prompt(desc):
    return (f"Photorealistic 3D product render of a single {desc}, isolated industrial machine part, "
            f"whole object centered, three-quarter view, plain white background, soft studio lighting, no text.")


# ---------- 1. images (Qwen-Image) ----------
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(QWEN, torch_dtype=torch.bfloat16).to("cuda")
for m in machines:
    out = CACHE / m / "parts"; out.mkdir(parents=True, exist_ok=True)
    for name, (desc, _) in PARTS[m].items():
        p = out / f"{name}.png"
        if p.exists():
            continue
        print(f"[image] {m}/{name}", flush=True)
        pipe(prompt=prompt(desc), negative_prompt="text, labels, blurry, cropped, multiple objects, full machine",
             width=1024, height=1024, num_inference_steps=20, true_cfg_scale=4.0,
             generator=torch.Generator("cuda").manual_seed(11)).images[0].save(p)
del pipe; gc.collect(); torch.cuda.empty_cache()

# ---------- 2. meshes (Hunyuan3D) ----------
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
rembg = BackgroundRemover()
shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")

for m in machines:
    out = CACHE / m / "parts"
    manifest = []
    for name, (desc, keys) in PARTS[m].items():
        glb = out / f"{name}.glb"
        if not glb.exists():
            print(f"[mesh]  {m}/{name}", flush=True)
            rgba = rembg(Image.open(out / f"{name}.png").convert("RGB"))
            mesh = shape(image=rgba, num_inference_steps=30, octree_resolution=256,
                         generator=torch.manual_seed(7))[0]
            mesh = FaceReducer()(DegenerateFaceRemover()(FloaterRemover()(mesh)), max_facenum=20000)
            # paint image colors onto the mesh (front projection)
            a = np.array(rgba); alpha = a[..., 3] > 128
            ys, xs = np.where(alpha); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            v = mesh.vertices
            u = ((v[:, 0] - v[:, 0].min()) / np.ptp(v[:, 0]) * (x1 - x0) + x0).round().astype(int).clip(0, a.shape[1] - 1)
            w = ((v[:, 1].max() - v[:, 1]) / np.ptp(v[:, 1]) * (y1 - y0) + y0).round().astype(int).clip(0, a.shape[0] - 1)
            cols = a[w, u, :3]
            mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.c_[cols, np.full(len(cols), 255)])
            # normalise: centred at origin, largest side = 1
            mesh.apply_translation(-mesh.bounds.mean(0))
            mesh.apply_scale(1.0 / mesh.extents.max())
            mesh.export(glb)
        manifest.append({"name": name, "file": f"parts/{name}.glb", "keywords": keys})
    (CACHE / m / "parts.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {m}/parts.json ({len(manifest)} parts)", flush=True)
print("DONE")
