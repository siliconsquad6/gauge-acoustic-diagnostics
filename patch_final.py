"""Final server touch: normal recordings also get their unit's 3D model (cache_key). Run once, gauge folder."""
p = "app/server.py"; s = open(p).read()
if "cache_key=f\"{mm}_{aid}\"" in s:
    print("already patched"); raise SystemExit
a = '''    if "asset" not in res:
        aid, aconf = asset_id(y, sr, res.get("machine"))
        res.update(asset=aid, asset_conf=round(aconf, 3))
'''
b = '''    if "asset" not in res:
        aid, aconf = asset_id(y, sr, res.get("machine"))
        mm = res.get("machine")
        res.update(asset=aid, asset_conf=round(aconf, 3),
                   cache_key=f"{mm}_{aid}" if aid and (WEB / "cache" / f"{mm}_{aid}").exists() else mm)
'''
if a not in s:
    print("MISS: run patch_assets.py first"); raise SystemExit
s = s.replace(a, b, 1); open(p, "w").write(s); print("OK")
