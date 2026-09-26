"""Show the photo that matches the diagnosed part (run once from the gauge folder)."""
p = "app/server.py"; s = open(p).read()
if "def fault_photo(" in s:
    print("already patched"); raise SystemExit
FN = '''def fault_photo(m, comp):
    """photo_<part>.png for the part Omni named, else the general machine photo."""
    base = WEB / "cache" / m
    try:
        parts = json.loads((base / "parts.json").read_text())
        c = str(comp or "").lower()
        hit = next((p["name"] for p in parts if p["name"] in c or any(k in c for k in p["keywords"])), None)
        if hit and (base / f"photo_{hit}.png").exists():
            return f"/cache/{m}/photo_{hit}.png"
    except Exception:
        pass
    return f"/cache/{m}/photo.png" if (base / "photo.png").exists() else f"/cache/{m}/machine.png"


# ---------- API ----------'''
s = s.replace("# ---------- API ----------", FN, 1)
old = 'image_url=f"/cache/{m}/photo.png"'
if old not in s:
    old = 'image_url=f"/cache/{m}/machine.png"'
s = s.replace(old, 'image_url=fault_photo(m, d.get("likely_component"))', 1)
open(p, "w").write(s)
print("OK" if "image_url=fault_photo(" in s else "MISS: image_url line not found")
