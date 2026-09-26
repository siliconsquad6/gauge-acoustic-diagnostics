import trimesh, numpy as np, json, cv2
O = "/home/hp20/Desktop/edge-ai/gauge/out"
m = trimesh.load(f"{O}/machine.glb", force="mesh")
d = json.load(open(f"{O}/diag.json"))
c = np.ascontiguousarray(m.visual.vertex_colors[:, :3]).astype(np.uint8)
h = cv2.cvtColor(c.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
hot = ((h[:, 0] < 20) | (h[:, 0] > 165)) & (h[:, 1] > 150) & (h[:, 2] > 200)
v = m.vertices[hot]
v = v[np.linalg.norm(v - d["defect_xyz"], axis=1) < 0.25]
ctr = np.median(v, 0)
r = float(np.clip(np.percentile(np.linalg.norm(v - ctr, axis=1), 70), 0.04, 0.12))
d["defect_xyz"], d["defect_radius"] = ctr.tolist(), r
json.dump(d, open(f"{O}/diag.json", "w"), indent=2)
print("hot verts", len(v), "center", ctr.round(3), "radius", round(r, 3))
