"""Calibrate which part an abnormal sound points to, from where its energy sits in frequency.
Low energy -> slow rotating parts (shaft, motor), high energy -> bearings, seals, seats.
Reads real MIMII abnormal clips, splits each machine's spectral-centroid range into one bin per part.

Run from the gauge folder (CPU, ~2 min):  python app/calib_parts.py
Saves models/part_bins.json
"""
import json, random, re
from pathlib import Path
import numpy as np, soundfile as sf, librosa
from joblib import Parallel, delayed

G = Path(__file__).resolve().parent.parent
OUT = G / "models" / "part_bins.json"
# parts ordered from low-frequency to high-frequency sources
ORDER = {"fan": ["shaft", "motor", "impeller", "bearing"],
         "pump": ["shaft", "motor", "impeller", "bearing", "seal"],
         "valve": ["coil", "spring", "plunger", "diaphragm", "seat"]}
PER = 300


def centroid(y, sr=16000):
    if y.ndim == 2:
        y = y.mean(1)
    return float(np.median(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))


def job(p):
    try:
        y, sr = sf.read(p, dtype="float32"); return centroid(y, sr)
    except Exception:
        return None


if __name__ == "__main__":
    random.seed(0)
    roots = list((G.parent.parent / "data").glob("combined_data_sh*")) + list((Path.home() / "Downloads").glob("*valve*"))
    files = {m: [] for m in ORDER}
    for r in roots:
        for p in r.rglob("*.wav"):
            s = str(p).lower()
            mt = re.search(r"/(fan|pump|valve)/", s)
            if mt and "/abnormal/" in s:
                files[mt.group(1)].append(p)
    out = {}
    for m, ps in files.items():
        if not ps:
            print(f"{m}: no abnormal clips found, skipping"); continue
        random.shuffle(ps); ps = ps[:PER]
        c = np.array([v for v in Parallel(n_jobs=-1)(delayed(job)(p) for p in ps) if v is not None])
        k = len(ORDER[m])
        edges = [float(np.percentile(c, 100 * i / k)) for i in range(1, k)]
        out[m] = {"order": ORDER[m], "edges": edges}
        print(f"{m}: {len(c)} clips, centroid {c.min():.0f}-{c.max():.0f} Hz, edges {[round(e) for e in edges]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("saved", OUT)
