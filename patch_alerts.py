"""Add ntfy phone-push alerts to server.py (run from the gauge folder, once)."""
p = "app/server.py"; s = open(p).read()
if "def send_alert" in s:
    print("already patched"); raise SystemExit

s = s.replace("# ---------- API ----------", '''# ---------- Alerts (ntfy phone push) ----------
import os, threading, requests
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "gauge-zgx-alerts-7f3k")
DASH_URL = os.environ.get("DASH_URL", "http://100.68.23.2:8000/")
PRIORITY = {"low": "3", "medium": "4", "high": "5"}


def send_alert(res):
    """Push an alert with the fault image to the phone. Text + picture only, never raw audio."""
    def _go():
        d, m = res.get("diag", {}), res.get("machine", "machine")
        sev = str(d.get("severity", "medium")).lower()
        msg = (f"{m.upper()} abnormal ({res['score']*100:.0f}% confidence). "
               f"Suspected {d.get('likely_component', '?')}: {d.get('fault', '?')}. "
               f"Heard: {d.get('sound_evidence', '?')}.")
        headers = {"Title": f"Gauge alert: {m} fault ({sev})", "Message": msg.encode("ascii", "ignore").decode(),
                   "Priority": PRIORITY.get(sev, "4"), "Tags": "rotating_light,factory", "Click": DASH_URL,
                   "Filename": f"{m}_fault.png"}
        img = WEB / "cache" / m / "machine.png"
        try:
            r = requests.put(f"https://ntfy.sh/{NTFY_TOPIC}", data=img.read_bytes() if img.exists() else msg.encode(),
                             headers=headers if img.exists() else {k: v for k, v in headers.items() if k not in ("Filename", "Message")},
                             timeout=15)
            print("ntfy alert:", r.status_code, flush=True)
        except Exception as e:
            print("ntfy alert failed:", e, flush=True)
    threading.Thread(target=_go, daemon=True).start()


# ---------- API ----------''', 1)

s = s.replace('    res["latency_ms"] = ', '''    if res["status"] == "abnormal":
        send_alert(res)
        res["alert"] = f"ntfy:{NTFY_TOPIC}"
    res["latency_ms"] = ''', 1)
open(p, "w").write(s)
print("OK send_alert" if "def send_alert" in s else "MISS send_alert")
print("OK hook" if "send_alert(res)" in s else "MISS hook")
