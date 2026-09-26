"""Make 3D-printable DEFECTIVE and FIXED versions of every internal part.

For each part in web/cache/<machine>/parts/:
  print/<part>_fixed.stl / .obj    healthy replacement part (watertight, in mm)
  print/<part>_defect.stl / .obj   same part with wear damage (dents + chipped edge)
  print/<part>_defect.glb          damaged version with colours, used by the viewer

CPU only, a few seconds per part. Run from the gauge folder:
  python app/export_parts_print.py            # all machines
  python app/export_parts_print.py pump 80    # one machine, 80 mm tall prints
"""
import sys
from pathlib import Path
import numpy as np, trimesh
from scipy import ndimage
from skimage import measure

G = Path(__file__).resolve().parent.parent
CACHE = G / "web" / "cache"
args = sys.argv[1:]
SIZE_MM = float(args.pop()) if args and args[-1].replace(".", "").isdigit() else 60.0
machines = args or ["fan", "pump", "valve"]


def solidify(mesh, size_mm):
    """Voxelize -> thicken -> fill -> marching cubes: always watertight and printable."""
    pitch = mesh.extents.max() / 160
    vox = mesh.voxelized(pitch)
    mat = ndimage.binary_fill_holes(ndimage.binary_dilation(vox.matrix, iterations=2))
    mat = np.pad(mat, 1)
    verts, faces, _, _ = measure.marching_cubes(mat.astype(np.float32), 0.5)
    s = trimesh.Trimesh((verts - 1) * pitch + vox.transform[:3, 3], faces)
    s.fix_normals()
    trimesh.smoothing.filter_taubin(s, iterations=8)
    s = max(s.split(only_watertight=False), key=lambda x: len(x.faces))
    s.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # y-up -> z-up
    s.apply_translation(-s.bounds[0])
    s.apply_scale(size_mm / s.extents.max())
    return s


def damage(mesh, seed=3):
    """Simulated wear: a few dents pushed into the surface plus one chipped corner."""
    m = mesh.copy()
    rng = np.random.default_rng(seed)
    v, n, ext = m.vertices.copy(), m.vertex_normals, m.extents.max()
    for _ in range(4):                                   # dents
        c = v[rng.integers(len(v))]
        w = np.exp(-(np.linalg.norm(v - c, axis=1) / (0.10 * ext)) ** 2)
        v -= (w * 0.07 * ext)[:, None] * n
    c = v[np.argmax(v @ rng.normal(size=3))]            # chipped corner: flatten one extreme region
    far = np.linalg.norm(v - c, axis=1) < 0.12 * ext
    v[far] = v[far] + (m.centroid - v[far]) * 0.25
    m.vertices = v
    return m


for mach in machines:
    pdir = CACHE / mach / "parts"
    out = pdir / "print"
    out.mkdir(parents=True, exist_ok=True)
    for glb in sorted(pdir.glob("*.glb")):
        name = glb.stem
        mesh = trimesh.load(glb, force="mesh")
        bad = damage(mesh)
        bad.export(out / f"{name}_defect.glb")
        for tag, src in (("fixed", mesh), ("defect", bad)):
            s = solidify(src, SIZE_MM)
            for ext in ("stl", "obj"):
                s.export(out / f"{name}_{tag}.{ext}")
        print(f"{mach}/{name}: fixed + defect saved (watertight {s.is_watertight}, {SIZE_MM:.0f} mm)", flush=True)
print("DONE")
