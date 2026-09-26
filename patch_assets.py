"""Recognise the machine UNIT (fan id_02 ...) from its sound and show that unit's own 2D/3D.
Run once from the gauge folder after train_asset.py (needs patch_faultphoto + patch_evidence applied)."""
p = "app/server.py"; s = open(p).read()
if "def asset_id(" in s:
    print("already patched"); raise SystemExit
ok = []

s = s.replace("# ---------- API ----------", '''# ---------- Unit fingerprint (which fan / pump / valve) ----------
import joblib
from train_asset import mfeat as _afeat
try:
    ASSET = joblib.load(G / "models" / "asset_id.joblib"); print("Unit fingerprint loaded:", list(ASSET.classes_))
except Exception as e:
    ASSET = None; print("Unit fingerprint not loaded (run app/train_asset.py):", e)


def asset_id(y, sr, m):
    """Most likely unit id (e.g. 'id_02') of machine type m, from the sound alone."""
    if ASSET is None or not m:
        return None, 0.0
    pr = ASSET.predict_proba([_afeat(y, sr)])[0]
    cand = [(float(q), c) for q, c in zip(pr, ASSET.classes_) if c.startswith(m + "_")]
    if not cand:
        return None, 0.0
    q, c = max(cand); tot = sum(x for x, _ in cand) or 1
    return c.split("_", 1)[1], q / tot


# ---------- API ----------''', 1); ok.append("def asset_id(" in s)

a = '        m = m or pick_machine(d.get("machine")) or "fan"\n        d["machine"] = m\n'
ok.append(a in s)
s = s.replace(a, a + '''        aid, aconf = asset_id(y, sr, m)
        key = f"{m}_{aid}" if aid and (WEB / "cache" / f"{m}_{aid}").exists() else m
        res.update(asset=aid, asset_conf=round(aconf, 3), cache_key=key)
''', 1)
for old, new in [('cache = WEB / "cache" / m / "diag.json"', 'cache = WEB / "cache" / key / "diag.json"'),
                 ('image_url=fault_photo(m, d.get("likely_component"))', 'image_url=fault_photo(key, d.get("likely_component"))'),
                 ('viewer_url=f"/viewer.html?m={m}&score={p:.3f}"', 'viewer_url=f"/viewer.html?m={key}&score={p:.3f}"')]:
    ok.append(old in s); s = s.replace(old, new, 1)

# normal clips: still identify the unit so the health trend is per unit
b = '    res["evidence"] = EVID\n'
ok.append(b in s)
s = s.replace(b, '''    if "asset" not in res:
        aid, aconf = asset_id(y, sr, res.get("machine"))
        res.update(asset=aid, asset_conf=round(aconf, 3))
''' + b, 1)
s = s.replace('HIST.setdefault(res.get("machine") or "unknown", [])', 'HIST.setdefault(" ".join(x for x in (res.get("machine"), res.get("asset")) if x) or "unknown", [])', 1)
s = s.replace('res["history"] = HIST[res.get("machine") or "unknown"][-20:]', 'res["history"] = HIST[" ".join(x for x in (res.get("machine"), res.get("asset")) if x) or "unknown"][-20:]', 1)
ok.append('" ".join(x for x in (res.get("machine"), res.get("asset"))' in s)
open(p, "w").write(s)
names = ["fingerprint model", "unit lookup", "diag per unit", "2D per unit", "3D per unit", "unit on normal clips", "trend per unit"]
for n, g in zip(names, ok): print(("OK   " if g else "MISS ") + n)
