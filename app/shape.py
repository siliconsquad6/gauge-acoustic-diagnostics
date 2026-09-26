import json, numpy as np, torch, trimesh, cv2, sys
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer

OUT = "/home/hp20/Desktop/edge-ai/gauge/out"
FRONT = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # flip to -1 if marker lands on the back

rgba = BackgroundRemover()(Image.open(f"{OUT}/machine.png").convert("RGB"))
rgba.save(f"{OUT}/machine_rgba.png")
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
mesh = pipe(image=rgba, num_inference_steps=30, octree_resolution=256, generator=torch.manual_seed(7))[0]
mesh = FaceReducer()(DegenerateFaceRemover()(FloaterRemover()(mesh)), max_facenum=60000)

# project image colors onto mesh (front view)
a = np.array(rgba); alpha = a[..., 3] > 128
ys, xs = np.where(alpha); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
v = mesh.vertices
u = ((v[:, 0] - v[:, 0].min()) / np.ptp(v[:, 0]) * (x1 - x0) + x0).round().astype(int).clip(0, a.shape[1] - 1)
w = ((v[:, 1].max() - v[:, 1]) / np.ptp(v[:, 1]) * (y1 - y0) + y0).round().astype(int).clip(0, a.shape[0] - 1)
cols = a[w, u, :3]
mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.c_[cols, np.full(len(cols), 255)])

# find glowing defect region
hsv = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2HSV)
glow = ((hsv[..., 0] < 20) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 150) & alpha
hit = glow[w, u]
front = FRONT * v[:, 2] > np.percentile(FRONT * v[:, 2], 50)
sel = hit & front if (hit & front).any() else hit
pt = v[sel].mean(0) if sel.any() else v.mean(0)

mesh.export(f"{OUT}/machine.glb")
d = json.load(open(f"{OUT}/diag.json"))
d["defect_xyz"] = pt.tolist(); d["defect_radius"] = float(np.ptp(v[sel], 0).max() / 2 + 0.03) if sel.any() else 0.1
json.dump(d, open(f"{OUT}/diag.json", "w"), indent=2)
print("saved machine.glb, defect at", pt.round(3), "glow verts:", int(sel.sum()))
