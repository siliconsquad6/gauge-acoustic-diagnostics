"""Make the dashboard show the cutaway illustration first (run once, gauge folder; needs patch_faultphoto.py)."""
p = "app/server.py"; s = open(p).read()
if "cutaway_{hit}.png" in s:
    print("already patched"); raise SystemExit
anchor = '        if hit and (base / f"manual_{hit}.png").exists():'
if anchor not in s:
    anchor = '        if hit and (base / f"photo_{hit}.png").exists():'
if anchor not in s:
    print("MISS: run patch_faultphoto.py first"); raise SystemExit
s = s.replace(anchor, '        if hit and (base / f"cutaway_{hit}.png").exists():\n            return f"/cache/{m}/cutaway_{hit}.png"\n' + anchor, 1)
open(p, "w").write(s)
print("OK")
