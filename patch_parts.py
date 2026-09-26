"""Pick the faulty part from the sound itself (not always the first part Omni sees in the prompt).
Run once from the gauge folder after app/calib_parts.py, then restart the server."""
p = "app/server.py"; s = open(p).read()
if "def pick_part(" in s:
    print("already patched"); raise SystemExit

FN = '''# ---------- Which part: from where the sound energy sits ----------
import librosa as _lr
try:
    PBINS = json.loads((G / "models" / "part_bins.json").read_text()); print("Part bins loaded:", list(PBINS))
except Exception as e:
    PBINS = {}; print("Part bins not loaded (run app/calib_parts.py):", e)
PART_FAULT = {
    "fan": {"shaft": "vibration along the shaft (misalignment or imbalance)", "motor": "irregular motor hum (voltage change)",
            "impeller": "imbalance or clogging at the impeller wheel", "bearing": "grinding from a worn drive-end bearing"},
    "pump": {"shaft": "vibration along the shaft (misalignment)", "motor": "irregular motor hum",
             "impeller": "clogging or contamination at the impeller", "bearing": "grinding from a worn bearing",
             "seal": "leak noise at the mechanical seal"},
    "valve": {"coil": "irregular coil buzz", "spring": "weak spring return", "plunger": "sticking plunger clicks",
              "diaphragm": "leak past the diaphragm", "seat": "contamination at the valve seat"},
}


def pick_part(d, m, y, sr):
    """Map the clip's spectral centroid to a part using bins calibrated on real MIMII abnormal clips."""
    b = PBINS.get(m)
    if not b:
        return d
    try:
        yy = y.mean(1) if y.ndim == 2 else y
        c = float(np.median(_lr.feature.spectral_centroid(y=yy, sr=sr)[0]))
    except Exception as e:
        print("pick_part failed:", e); return d
    part = b["order"][int(np.searchsorted(b["edges"], c))]
    said = str(d.get("likely_component", ""))
    d["omni_component"] = said
    d["likely_component"] = part
    if part not in str(d.get("fault", "")).lower():
        d["fault"] = PART_FAULT[m][part]
    d["part_reason"] = f"Sound energy centred near {c:.0f} Hz, typical of the {part}"
    print(f"[part] {m}: centroid {c:.0f} Hz -> {part} (Omni said: {said})", flush=True)
    return d


# ---------- API ----------'''
s = s.replace("# ---------- API ----------", FN, 1)
a = '        d["machine"] = m\n'
ok = [a in s]
s = s.replace(a, a + '        d = pick_part(d, m, y, sr)\n', 1)
open(p, "w").write(s)
print("OK   part picker" if all(ok) and "def pick_part(" in s else "MISS: anchor not found, send me app/server.py")
