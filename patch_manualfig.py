"""Show the manual's real drawing (with the part heat-marked) as the 2D image. Run once, gauge folder.
Needs patch_faultphoto.py applied first."""
p = "app/server.py"; s = open(p).read()
if "manual_{hit}.png" in s:
    print("already patched"); raise SystemExit
ok = []
a = '        if hit and (base / f"photo_{hit}.png").exists():'
ok.append(a in s)
s = s.replace(a, '        if hit and (base / f"manual_{hit}.png").exists():\n            return f"/cache/{m}/manual_{hit}.png"\n' + a, 1)
b = '    res["latency_ms"] = '
ok.append(b in s)
s = s.replace(b, '''    iu = res.get("image_url") or ""
    if "/manual_" in iu:
        try:
            res["image_info"] = json.loads((WEB / iu.lstrip("/").replace(".png", ".json")).read_text())
        except Exception:
            pass
''' + b, 1)
open(p, "w").write(s)
print("OK" if all(ok) else f"MISS {ok} (apply patch_faultphoto.py first)")
