"""Add per-clip evidence (frequency bands, mic weights, spectrogram), severity level and machine
health history to /analyze. Run once from the gauge folder, then restart the server."""
p = "app/server.py"; s = open(p).read()
if "def evidence(" in s:
    print("already patched"); raise SystemExit
ok = []

EVID_CODE = '''import base64, math as _m
EVID = {}
HIST = {}


def band_ranges(f_dim):
    mel = lambda f: 1127 * _m.log(1 + f / 700)
    hz = lambda v: 700 * (_m.exp(v / 1127) - 1)
    lo, hi = mel(20), mel(8000)
    e = [hz(lo + i * (hi - lo) / 129) for i in range(130)]
    return [(e[10 * b], e[min(10 * b + 17, 129)]) for b in range(f_dim)]


def evidence(x, mic_w, band_w):
    """What the classifier listened to: band attention, mic attention, fused spectrogram."""
    bands = band_w.float()[0].cpu().numpy()
    mics = mic_w.float()[0].mean(-1).cpu().numpy()
    fused = (x[0].float() * mic_w.float()[0].unsqueeze(1)).sum(0).cpu().numpy()   # time x mel
    fused = fused[: (fused.shape[0] // 4) * 4].reshape(-1, 4, fused.shape[1]).mean(1)  # 256 x 128
    lo, hi = np.percentile(fused, 2), np.percentile(fused, 99.5)
    img = (np.clip((fused - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8).T[::-1]  # mel rows, high freq on top
    rng = band_ranges(len(bands))
    return {
        "bands": [{"lo": round(a), "hi": round(b), "w": round(float(w), 4)} for (a, b), w in zip(rng, bands)],
        "top": [int(i) for i in np.argsort(bands)[::-1][:3]],
        "mics": [round(float(v), 4) for v in mics],
        "spec": {"w": int(img.shape[1]), "h": int(img.shape[0]), "b64": base64.b64encode(img.tobytes()).decode()},
    }


'''
s = s.replace("def classify(y):", EVID_CODE + "def classify(y):", 1); ok.append("def evidence(" in s)

old = "        logits, _, _ = clf(x)\n    return torch.softmax(logits.float(), -1)[0, 1].item()"
new = ("        logits, mic_w, band_w = clf(x)\n    global EVID\n"
       "    try:\n        EVID = evidence(x, mic_w, band_w)\n    except Exception as e:\n        EVID = {}; print('evidence failed:', e)\n"
       "    return torch.softmax(logits.float(), -1)[0, 1].item()")
ok.append(old in s); s = s.replace(old, new, 1)

hook = '''    res["evidence"] = EVID
    sc = res["score"]
    res["level"] = "normal" if res["status"] == "normal" else ("critical" if sc >= 0.95 else "warning" if sc >= 0.75 else "watch")
    HIST.setdefault(res.get("machine") or "unknown", []).append(
        {"t": round(time.time()), "score": sc, "status": res["status"], "file": file.filename})
    res["history"] = HIST[res.get("machine") or "unknown"][-20:]
    res["latency_ms"] = '''
ok.append('    res["latency_ms"] = ' in s); s = s.replace('    res["latency_ms"] = ', hook, 1)
s = s.replace('viewer_url=f"/viewer.html?m={m}"', 'viewer_url=f"/viewer.html?m={m}&score={p:.3f}"', 1); ok.append("&score={p:.3f}" in s)
open(p, "w").write(s)
for n, g in zip(["evidence helpers", "classifier returns attention", "level + history", "viewer gets score"], ok):
    print(("OK   " if g else "MISS ") + n)
