"""Give every machine UNIT (fan id_00, fan id_02, ...) its own look: its own 3D housing and its own cutaways.
Internal parts are shared with the machine type (web/cache/<machine>/parts, run rebuild_visuals.py first).

Output per unit: web/cache/<machine>_<id>/ {machine.png, machine.glb, diag.json, parts/, parts.json, cutaway_<part>.png}

Run with the server STOPPED (needs the GPU):
  python -u app/build_assets.py fan            # 4 fan units (~40 min)
  python -u app/build_assets.py                # all 12 units (~2 h)
  python -u app/build_assets.py fan_id_02      # a single unit
"""
import gc, json, os, shutil, sys
STEPS = int(os.environ.get("STEPS", 30)); CW, CH = int(os.environ.get("CW", 1328)), int(os.environ.get("CH", 1024))
from pathlib import Path
import numpy as np, torch, trimesh, cv2
from PIL import Image

G = Path(__file__).resolve().parent.parent
CACHE = G / "web" / "cache"

# four representative models per machine type (MIMII ids are different product models)
VARIANTS = {
    "fan": {
        "id_00": "belt-driven industrial centrifugal blower with a dark green painted steel scroll housing, belt guard and motor on a sliding base",
        "id_02": "direct-drive industrial centrifugal blower with a blue painted steel scroll housing and motor on a steel base frame",
        "id_04": "industrial axial duct fan in a long grey cylindrical steel housing with bolted flanges and an externally mounted motor",
        "id_06": "heavy-duty industrial radial blower with a safety-yellow cast housing, large square outlet and motor on a welded skid",
    },
    "pump": {
        "id_00": "industrial end-suction centrifugal water pump with a blue cast iron volute and close-coupled motor on a baseplate",
        "id_02": "vertical inline centrifugal water pump with a red cast iron body, flanged inlet and outlet on the same line and motor on top",
        "id_04": "self-priming centrifugal water pump with a green cast iron body, large priming chamber and flexible coupling to the motor",
        "id_06": "stainless steel multistage centrifugal water pump with stacked stages and a grey motor on a stainless baseplate",
    },
    "valve": {
        "id_00": "brass 2-way solenoid valve with a black epoxy coil and grey DIN connector, threaded ports",
        "id_02": "stainless steel 2-way solenoid valve with a square black coil and cable gland, threaded ports",
        "id_04": "compact plastic body 2-way solenoid valve with a blue coil and push-in fittings",
        "id_06": "large pilot-operated brass solenoid valve with flanged ports and a tall black coil housing",
    },
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
HOT = {"fan": "The bearing housing next to the motor", "pump": "The seal area between casing and motor", "valve": "The valve body around the seat"}
CUT_STYLE = ("Clean 3D-rendered technical cutaway illustration of industrial equipment, product-catalog quality: a quarter-section is cut away "
         "to reveal the internal parts, cut faces painted red-brown, housings in the machine's own glossy industrial paint color, copper motor windings, "
         "polished silver shafts and ball bearings with visible balls, crisp edges, soft studio lighting with gentle shadows, "
         "pure white background, three-quarter side view, whole machine visible, centered, sharp and highly detailed.")
NEG = "blurry, messy background, people, hands, watermark, logo, brand name, misspelled text, gibberish text, low detail"

args = sys.argv[1:] or list(VARIANTS)
units = []
for a in args:
    if "_id_" in a:
        mm, uid = a.split("_", 1); units.append((mm, uid))
    else:
        units += [(a, uid) for uid in VARIANTS[a]]

# ---------- images ----------
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(G / "models" / "qwen-image", torch_dtype=torch.bfloat16).to("cuda")
gen = lambda s: torch.Generator("cuda").manual_seed(s)
for m, uid in units:
    out = CACHE / f"{m}_{uid}"; out.mkdir(parents=True, exist_ok=True)
    desc = VARIANTS[m][uid]
    if not (out / "machine.png").exists():
        print(f"[studio]  {m}_{uid}", flush=True)
        pipe(prompt=(f"Photorealistic product photograph of a single {desc}, whole machine visible and centered, three-quarter "
                     f"view, plain white seamless background, soft studio lighting, realistic materials. {HOT[m]} glows bright "
                     f"orange-red as a fault highlight."),
             negative_prompt=NEG + ", multiple machines, cropped", width=CW, height=CW, num_inference_steps=STEPS,
             true_cfg_scale=4.0, generator=gen(7)).images[0].save(out / "machine.png")
    for part, (what, hint) in PARTS[m].items():
        f = out / f"cutaway_{part}.png"
        if f.exists():
            continue
        print(f"[cutaway] {m}_{uid}/{part}", flush=True)
        pipe(prompt=(f"{CUT_STYLE} The subject is a {desc}, cut open to show its internal parts. The {what} glows bright "
                     f"orange-red and is circled with a thick red ring; a red arrow points from it to a white callout box with a "
                     f'red border near the top right. The callout reads "SUSPECTED {part.upper()} FAULT" in bold red capital '
                     f'letters, and below it in smaller black text "Acoustic anomaly: {hint}". No other text anywhere.'),
             negative_prompt=NEG, width=CW, height=CH, num_inference_steps=STEPS, true_cfg_scale=4.0,
             generator=gen(5)).images[0].save(f)
del pipe; gc.collect(); torch.cuda.empty_cache()

# ---------- 3D housing per unit ----------
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
rembg = BackgroundRemover()
shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
for m, uid in units:
    out = CACHE / f"{m}_{uid}"
    print(f"[3d]      {m}_{uid}", flush=True)
    rgba = rembg(Image.open(out / "machine.png").convert("RGB"))
    mesh = shape(image=rgba, num_inference_steps=30, octree_resolution=256, generator=torch.manual_seed(7))[0]
    mesh = FaceReducer()(DegenerateFaceRemover()(FloaterRemover()(mesh)), max_facenum=60000)
    a = np.array(rgba); alpha = a[..., 3] > 128
    ys, xs = np.where(alpha); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    v = mesh.vertices
    u = ((v[:, 0] - v[:, 0].min()) / np.ptp(v[:, 0]) * (x1 - x0) + x0).round().astype(int).clip(0, a.shape[1] - 1)
    w = ((v[:, 1].max() - v[:, 1]) / np.ptp(v[:, 1]) * (y1 - y0) + y0).round().astype(int).clip(0, a.shape[0] - 1)
    cols = a[w, u, :3]
    hsv = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2HSV)
    hot = (((hsv[..., 0] < 20) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 150) & (hsv[..., 2] > 200) & alpha)[w, u]
    front = v[:, 2] > np.percentile(v[:, 2], 50)
    sel = hot & front if (hot & front).any() else hot
    if sel.any():
        pts = v[sel]; ctr = np.median(pts, 0); rad = float(np.clip(np.percentile(np.linalg.norm(pts - ctr, axis=1), 70), 0.04, 0.12))
    else:
        ctr, rad = v.mean(0), 0.1
    cols[hot] = [120, 124, 130]
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.c_[cols, np.full(len(cols), 255)])
    mesh.export(out / "machine.glb")
    base = CACHE / m
    diag = json.loads((base / "diag.json").read_text()) if (base / "diag.json").exists() else {"machine": m}
    diag.update(defect_xyz=ctr.tolist(), defect_radius=rad, unit=uid)
    (out / "diag.json").write_text(json.dumps(diag, indent=2))
    # share the machine type's internal parts (+ printable versions) with this unit
    if (base / "parts").exists():
        shutil.copytree(base / "parts", out / "parts", dirs_exist_ok=True)
        shutil.copy(base / "parts.json", out / "parts.json")
    print(f"done {m}_{uid}", flush=True)
print("DONE")
