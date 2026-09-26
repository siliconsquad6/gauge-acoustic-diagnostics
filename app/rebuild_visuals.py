"""Rebuild every machine visual so it matches the REAL MIMII machines and looks photographic.

MIMII (Purohit et al., 2019):
  fan   = industrial fans that keep gas/air flowing in factories  -> centrifugal blower
  pump  = water pumps that drain and return water to a pool       -> end-suction centrifugal pump
  valve = solenoid valves that are repeatedly opened and closed   -> 2-way solenoid valve

Per machine it writes to web/cache/<machine>/:
  photo.png          realistic factory photograph (shown in the dashboard)
  machine.png        studio shot with the fault zone highlighted (used to build the 3D model)
  machine.glb        3D housing + diag.json defect location
  parts/*.glb        internal parts for the X-ray view + parts.json

Run with the server STOPPED (needs the GPU):
  python -u app/rebuild_visuals.py              # all three (~50 min)
  python -u app/rebuild_visuals.py fan valve    # only some
Then: python app/export_parts_print.py
"""
import gc, json, shutil, sys
from pathlib import Path
import numpy as np, torch, trimesh, cv2
from PIL import Image

G = Path(__file__).resolve().parent.parent
CACHE = G / "web" / "cache"
QWEN = G / "models" / "qwen-image"

PHOTO_STYLE = ("Real photograph taken on a factory floor with a full-frame DSLR, 35mm lens, f/5.6, natural overhead "
               "industrial lighting, true-to-life materials, weathered paint, light dust, small scuffs and oil stains, "
               "sharp focus on the machine, background softly out of focus. Ultra HD, 4K, cinematic composition.")
PHOTO_NEG = ("cartoon, illustration, CGI, 3d render, toy, plastic, miniature, oversaturated, glowing, fire, lava, neon, "
             "text, watermark, logo, brand name, people, blurry, deformed")
STUDIO_NEG = "text, labels, logo, blurry, cropped, multiple machines, people, cartoon, toy, background clutter"

MACHINES = {
    "fan": {
        "photo": ("An industrial centrifugal blower fan installed in a factory: painted steel spiral scroll housing, "
                  "large electric motor on a welded steel base frame bolted to a concrete floor, round inlet duct, "
                  "rectangular outlet flange connected to ductwork, cable conduit to the motor."),
        "studio": ("industrial centrifugal blower fan with a painted steel spiral scroll housing, round inlet, "
                   "rectangular outlet flange and an electric motor on a steel base frame"),
        "hot": "The bearing housing between the motor and the fan scroll",
        "parts": {
            "impeller": ("backward-curved centrifugal fan wheel impeller with welded steel blades", ["impeller", "wheel", "blade", "rotor", "unbalanc", "imbalanc", "clog"]),
            "motor":    ("industrial electric motor stator with copper windings", ["motor", "winding", "stator", "electrical", "voltage"]),
            "bearing":  ("steel pillow block ball bearing", ["bearing"]),
            "shaft":    ("long polished steel fan drive shaft with keyway", ["shaft", "coupling", "hub", "misalign"]),
        },
    },
    "pump": {
        "photo": ("An industrial end-suction centrifugal water pump in a factory pump room: cast iron volute casing, "
                  "close-coupled electric motor, mounted on a steel baseplate, flanged suction and discharge pipes "
                  "with pressure gauge, painted blue-grey, a little water on the floor."),
        "studio": ("industrial end-suction centrifugal water pump with cast iron volute casing, flanged inlet and "
                   "outlet and a close-coupled electric motor on a steel baseplate"),
        "hot": "The mechanical seal area between the pump casing and the motor",
        "parts": {
            "impeller": ("closed cast bronze centrifugal pump impeller with curved vanes", ["impeller", "vane", "rotor", "clog", "cavitation", "contamin"]),
            "seal":     ("pump mechanical seal assembly with carbon face ring and rubber bellows", ["seal", "leak", "gasket", "packing"]),
            "bearing":  ("steel deep groove ball bearing", ["bearing"]),
            "shaft":    ("polished stainless steel pump shaft with keyway", ["shaft", "coupling", "misalign"]),
            "motor":    ("industrial electric motor stator with copper windings", ["motor", "winding", "stator", "electrical", "voltage"]),
        },
    },
    "valve": {
        "photo": ("An industrial 2-way solenoid valve installed on a factory pipeline: brass valve body with threaded "
                  "pipe fittings, black epoxy solenoid coil on top with a grey DIN connector and cable, mounted on "
                  "stainless steel pipes next to other plumbing."),
        "studio": ("industrial 2-way solenoid valve with a brass valve body, threaded inlet and outlet ports and a "
                   "black cylindrical solenoid coil with a DIN connector on top"),
        "hot": "The brass valve body around the inner valve seat",
        "parts": {
            "seat":      ("brass solenoid valve body cut open showing the orifice valve seat", ["seat", "orifice", "contamin", "clog", "debris", "body", "leak"]),
            "diaphragm": ("black rubber solenoid valve diaphragm with brass center insert", ["diaphragm", "membrane", "seal", "gasket"]),
            "plunger":   ("stainless steel solenoid valve plunger armature with rubber tip", ["plunger", "armature", "stem", "stick", "core"]),
            "spring":    ("small stainless steel solenoid return spring", ["spring"]),
            "coil":      ("black epoxy-encapsulated solenoid valve coil with DIN connector", ["coil", "solenoid", "electrical", "voltage", "magnet"]),
        },
    },
}
machines = sys.argv[1:] or list(MACHINES)

# ---------------- 1. images ----------------
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(QWEN, torch_dtype=torch.bfloat16).to("cuda")
gen = lambda s: torch.Generator("cuda").manual_seed(s)
for m in machines:
    c, out = MACHINES[m], CACHE / m
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "parts", ignore_errors=True); (out / "parts").mkdir()
    print(f"[photo]  {m}", flush=True)
    pipe(prompt=c["photo"] + " " + PHOTO_STYLE, negative_prompt=PHOTO_NEG, width=1328, height=1328,
         num_inference_steps=20, true_cfg_scale=4.0, generator=gen(21)).images[0].save(out / "photo.png")
    print(f"[studio] {m}", flush=True)
    pipe(prompt=(f"Photorealistic product photograph of a single {c['studio']}, whole machine visible and centered, "
                 f"three-quarter view, plain white seamless background, soft studio lighting, realistic metal and paint. "
                 f"{c['hot']} glows bright orange-red as a fault highlight."),
         negative_prompt=STUDIO_NEG, width=1024, height=1024, num_inference_steps=20, true_cfg_scale=4.0,
         generator=gen(7)).images[0].save(out / "machine.png")
    for name, (desc, _) in c["parts"].items():
        print(f"[part]   {m}/{name}", flush=True)
        pipe(prompt=(f"Photorealistic product photograph of a single {desc}, isolated industrial machine part, whole "
                     f"object centered, three-quarter view, plain white background, soft studio lighting, realistic metal."),
             negative_prompt=STUDIO_NEG + ", full machine", width=1024, height=1024, num_inference_steps=20,
             true_cfg_scale=4.0, generator=gen(11)).images[0].save(out / "parts" / f"{name}.png")
del pipe; gc.collect(); torch.cuda.empty_cache()

# ---------------- 2. 3D ----------------
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
rembg = BackgroundRemover()
shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")


def mesh_from(png, faces):
    rgba = rembg(Image.open(png).convert("RGB"))
    mesh = shape(image=rgba, num_inference_steps=20, octree_resolution=256, generator=torch.manual_seed(7))[0]
    mesh = FaceReducer()(DegenerateFaceRemover()(FloaterRemover()(mesh)), max_facenum=faces)
    a = np.array(rgba); alpha = a[..., 3] > 128
    ys, xs = np.where(alpha); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    v = mesh.vertices
    u = ((v[:, 0] - v[:, 0].min()) / np.ptp(v[:, 0]) * (x1 - x0) + x0).round().astype(int).clip(0, a.shape[1] - 1)
    w = ((v[:, 1].max() - v[:, 1]) / np.ptp(v[:, 1]) * (y1 - y0) + y0).round().astype(int).clip(0, a.shape[0] - 1)
    cols = a[w, u, :3]
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.c_[cols, np.full(len(cols), 255)])
    return mesh, a, alpha, u, w


for m in machines:
    c, out = MACHINES[m], CACHE / m
    print(f"[3d]     {m}", flush=True)
    mesh, a, alpha, u, w = mesh_from(out / "machine.png", 60000)
    # locate the highlighted fault zone: hottest front-facing vertices
    hsv = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2HSV)
    hot = (((hsv[..., 0] < 20) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 150) & (hsv[..., 2] > 200) & alpha)[w, u]
    v = mesh.vertices
    front = v[:, 2] > np.percentile(v[:, 2], 50)
    sel = hot & front if (hot & front).any() else hot
    if sel.any():
        pts = v[sel]; ctr = np.median(pts, 0)
        rad = float(np.clip(np.percentile(np.linalg.norm(pts - ctr, axis=1), 70), 0.04, 0.12))
    else:
        ctr, rad = v.mean(0), 0.1
    # repaint the fault highlight back to metal so the 3D model looks real (the viewer adds its own glow)
    cols = mesh.visual.vertex_colors.copy(); cols[hot, :3] = [120, 124, 130]; mesh.visual.vertex_colors = cols
    mesh.export(out / "machine.glb")
    dpath = out / "diag.json"
    diag = json.loads(dpath.read_text()) if dpath.exists() else {"machine": m, "likely_component": "", "fault": "", "severity": "medium", "sound_evidence": ""}
    diag.update(defect_xyz=ctr.tolist(), defect_radius=rad)
    dpath.write_text(json.dumps(diag, indent=2))

    manifest = []
    for name, (desc, keys) in c["parts"].items():
        print(f"[3d]     {m}/{name}", flush=True)
        pm, *_ = mesh_from(out / "parts" / f"{name}.png", 20000)
        pm.apply_translation(-pm.bounds.mean(0)); pm.apply_scale(1.0 / pm.extents.max())
        pm.export(out / "parts" / f"{name}.glb")
        manifest.append({"name": name, "file": f"parts/{name}.glb", "keywords": keys})
    (out / "parts.json").write_text(json.dumps(manifest, indent=2))
    print(f"done {m}", flush=True)
print("DONE. Now run: python app/export_parts_print.py")
