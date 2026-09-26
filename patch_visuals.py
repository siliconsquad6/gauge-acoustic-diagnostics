"""Point the dashboard at the realistic photos and tell Omni the real MIMII parts/faults (run once, gauge folder)."""
p = "app/server.py"; s = open(p).read()
ok = []
s = s.replace('image_url=f"/cache/{m}/machine.png"', 'image_url=f"/cache/{m}/photo.png"'); ok.append("photo.png" in s)
if "HINT = {" not in s:
    s = s.replace('MACHINES = ["fan", "pump", "valve"]\n', '''MACHINES = ["fan", "pump", "valve"]
HINT = {  # what the MIMII machines really are, and their documented fault types
    "fan": "It is an industrial centrifugal blower fan. Typical faults: imbalance, voltage change, clogging. Pick likely_component from: impeller, motor, bearing, shaft.",
    "pump": "It is a centrifugal water pump. Typical faults: leakage, contamination, clogging. Pick likely_component from: impeller, seal, bearing, shaft, motor.",
    "valve": "It is a solenoid valve that opens and closes repeatedly. Typical faults: contamination of the seat or diaphragm, sticking plunger, weak coil. Pick likely_component from: seat, diaphragm, plunger, spring, coil.",
}
''', 1)
s = s.replace('flagged as abnormal. Reply ONLY with JSON: ', "flagged as abnormal. {HINT.get(machine or '', '')} Reply ONLY with JSON: ", 1)
ok.append("HINT.get(machine" in s)
open(p, "w").write(s)
for n, g in zip(["dashboard uses photo.png", "Omni knows real parts"], ok): print(("OK   " if g else "MISS ") + n)
