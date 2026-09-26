"""Gauge backend: loads models ONCE, serves the UI and POST /analyze.
Run:  cd ~/Desktop/edge-ai/gauge && python app/server.py
"""
import io, json, math, re, tempfile, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn as nn, torchaudio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import ASTModel, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

ROOT = Path.home() / "Desktop" / "edge-ai"
G = Path(__file__).resolve().parent.parent
WEB = G / "web"
RUN_DIR = ROOT / "runs" / "gauge_combined"
AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
OMNI = G / "models" / "omni"
SR, N_MICS, N_MELS, N_FRAMES = 16000, 8, 128, 1024
MEAN, STD = -4.2677393, 4.5689974
MACHINES = ["fan", "pump", "valve"]
HINT = {  # what the MIMII machines really are, and their documented fault types
    "fan": "It is an industrial centrifugal blower fan. Typical faults: imbalance, voltage change, clogging. Pick likely_component from: impeller, motor, bearing, shaft.",
    "pump": "It is a centrifugal water pump. Typical faults: leakage, contamination, clogging. Pick likely_component from: impeller, seal, bearing, shaft, motor.",
    "valve": "It is a solenoid valve that opens and closes repeatedly. Typical faults: contamination of the seat or diaphragm, sticking plunger, weak coil. Pick likely_component from: seat, diaphragm, plunger, spring, coil.",
}
DEV = "cuda"


# ---------- Classifier (same as Scripts/predict.py) ----------
def channel_to_spec(y):
    wav = torch.from_numpy(y).unsqueeze(0)
    wav = wav - wav.mean()
    spec = torchaudio.compliance.kaldi.fbank(
        wav, sample_frequency=SR, num_mel_bins=N_MELS, frame_length=25, frame_shift=10,
        window_type="hanning", htk_compat=True, use_energy=False, dither=0.0)
    if spec.shape[0] < N_FRAMES:
        spec = torch.nn.functional.pad(spec, (0, 0, 0, N_FRAMES - spec.shape[0]))
    return (spec[:N_FRAMES] - MEAN) / (STD * 2)


class GaugeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = ASTModel.from_pretrained(AST)
        cfg = self.ast.config
        H = cfg.hidden_size
        self.f_dim = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
        self.t_dim = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
        self.mic_scorer = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 1))
        self.band_scorer = nn.Sequential(nn.Linear(H, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(2 * H), nn.Dropout(0.1), nn.Linear(2 * H, 2))

    def forward(self, x):
        B = x.shape[0]
        stats = torch.stack([x.mean(2), x.std(2)], dim=-1)
        mic_w = torch.softmax(self.mic_scorer(stats).squeeze(-1), dim=1)
        fused = (x * mic_w.unsqueeze(2)).sum(1)
        h = self.ast(input_values=fused).last_hidden_state
        cls = (h[:, 0] + h[:, 1]) / 2
        bands = h[:, 2:].reshape(B, self.f_dim, self.t_dim, -1).mean(2)
        band_w = torch.softmax(self.band_scorer(bands).squeeze(-1), dim=1)
        band_vec = (bands * band_w.unsqueeze(-1)).sum(1)
        return self.head(torch.cat([cls, band_vec], dim=-1)), mic_w, band_w


# ---------- Load everything once ----------
print("Loading classifier...")
clf = GaugeNet().to(DEV).eval()
sd = torch.load(RUN_DIR / "best_model.pt", map_location=DEV)
def _rk(k):
    for a, b in [(".attention.attention.query.", ".attention.q_proj."), (".attention.attention.key.", ".attention.k_proj."),
                 (".attention.attention.value.", ".attention.v_proj."), (".attention.output.dense.", ".attention.o_proj."),
                 (".intermediate.dense.", ".mlp.fc1."), (".output.dense.", ".mlp.fc2.")]:
        k = k.replace(a, b)
    return k.replace("ast.encoder.layer.", "ast.layers.")
clf.load_state_dict({_rk(k): v for k, v in sd.items()})
print("Loading Omni (a few minutes, only once)...")
omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(OMNI, torch_dtype=torch.bfloat16, device_map="cuda")
proc = Qwen2_5OmniProcessor.from_pretrained(OMNI)
from peft import PeftModel
omni.thinker = PeftModel.from_pretrained(omni.thinker, G / "models" / "omni-gauge-lora")
omni.thinker.eval()
print("Loaded fine-tuned adapter: omni-gauge-lora")
MPROMPT = ("You are a factory acoustic monitor. Listen to this machine recording. "
           'Reply ONLY with JSON: {"machine":"fan|pump|valve","status":"normal|abnormal"}')


def machine_type(data):
    """Fine-tuned Omni: which machine is this?"""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(data); f.flush()
        conv = [{"role": "user", "content": [{"type": "audio", "audio": f.name}, {"type": "text", "text": MPROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, _, _ = process_mm_info(conv, use_audio_in_video=False)
        inp = proc(text=text, audio=audios, return_tensors="pt", padding=True).to("cuda").to(torch.bfloat16)
        with torch.no_grad():
            out = omni.thinker.generate(**inp, max_new_tokens=30, do_sample=False)
    txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print("OMNI MACHINE RAW:", txt, flush=True)
    return pick_machine(txt.replace("fan|pump|valve", ""))
from peft import PeftModel
omni.thinker.eval()
print("Loaded fine-tuned adapter: omni-gauge-lora")
MPROMPT = ("You are a factory acoustic monitor. Listen to this machine recording. "
           'Reply ONLY with JSON: {"machine":"fan|pump|valve","status":"normal|abnormal"}')


def machine_type(data):
    """Fine-tuned Omni: which machine is this?"""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(data); f.flush()
        conv = [{"role": "user", "content": [{"type": "audio", "audio": f.name}, {"type": "text", "text": MPROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, _, _ = process_mm_info(conv, use_audio_in_video=False)
        inp = proc(text=text, audio=audios, return_tensors="pt", padding=True).to("cuda").to(torch.bfloat16)
        with torch.no_grad():
            out = omni.thinker.generate(**inp, max_new_tokens=30, do_sample=False)
    txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print("OMNI MACHINE RAW:", txt, flush=True)
    return pick_machine(txt.replace("fan|pump|valve", ""))
print("READY on http://0.0.0.0:8000")


import base64, math as _m
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


def classify(y):
    x = torch.stack([channel_to_spec(np.ascontiguousarray(y[:, c])) for c in range(N_MICS)]).unsqueeze(0).to(DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, mic_w, band_w = clf(x)
    global EVID
    try:
        EVID = evidence(x, mic_w, band_w)
    except Exception as e:
        EVID = {}; print('evidence failed:', e)
    return torch.softmax(logits.float(), -1)[0, 1].item()


def describe(wav_path, machine):
    who = f"a {machine}" if machine else "an industrial machine (fan, pump, or valve)"
    ask = (f"This is audio from {who} flagged as abnormal. {HINT.get(machine or '', '')} Reply ONLY with JSON: "
           '{"machine":"fan|pump|valve","likely_component":...,"fault":...,'
           '"severity":"low|medium|high","sound_evidence":...}')
    if VLLM:
        try:
            return json.loads(re.search(r"\{.*\}", describe_vllm(wav_path, ask), re.S).group())
        except Exception as e:
            print("LLM endpoint describe failed, using PyTorch:", e, flush=True)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": str(wav_path)}, {"type": "text", "text": ask}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, _, _ = process_mm_info(conv, use_audio_in_video=False)
    inp = proc(text=text, audio=audios, return_tensors="pt", padding=True).to("cuda").to(omni.dtype)
    with torch.no_grad(), omni.thinker.disable_adapter():
        out = omni.generate(**inp, max_new_tokens=200, return_audio=False)
    txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return json.loads(re.search(r"\{.*\}", txt, re.S).group())


def pick_machine(*hints):
    for h in hints:
        m = re.search(r"fan|pump|valve", str(h or ""), re.I)
        if m:
            return m.group().lower()
    return None


# ---------- Alerts (ntfy phone push) ----------
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
                   "Priority": PRIORITY.get(sev, "4"), "Tags": "rotating_light,factory", "Click": DASH_URL + (f"chat.html?incident={res['incident']}" if res.get("incident") else ""),
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


# ---------- Repair chatbot (answers from the manuals) ----------
import uuid, threading as _th
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer
from pydantic import BaseModel
INCIDENTS = {}
GEN_LOCK = _th.Lock()
MAN = G / "manuals"; MAN.mkdir(exist_ok=True)
try:
    from sentence_transformers import SentenceTransformer
    EMBED = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    CH = json.loads((MAN / "chunks.json").read_text())
    EMB = np.load(MAN / "index.npz")["emb"]
    print(f"Manuals loaded: {len(CH)} chunks", flush=True)
except Exception as e:
    EMBED, CH, EMB = None, [], None
    print("Manuals not loaded (run get_manuals.sh + ingest_manuals.py):", e, flush=True)


def retrieve(query, machine, k=5):
    if EMBED is None or not CH:
        return []
    q = EMBED.encode([query], normalize_embeddings=True)[0]
    sc = EMB @ q
    sc = np.where(np.array([c["machine"] in (machine, "all") for c in CH]), sc, -1)
    return [dict(CH[i], score=float(sc[i])) for i in np.argsort(-sc)[:k] if sc[i] > 0]


SYSTEM = ("You are Gauge, a maintenance assistant for factory technicians. Answer ONLY from the numbered manual "
          "excerpts provided and cite them inline like [1] or [2]. Structure every repair answer as: "
          "1) Safety first (lockout/tagout), 2) Likely cause, 3) Step-by-step fix, 4) Parts and tools. Be concise. "
          "If the excerpts do not cover something, say so plainly and recommend contacting the manufacturer. "
          "Never invent torque values, part numbers or specifications.")


class ChatReq(BaseModel):
    incident: str | None = None
    messages: list[dict]


def fault_photo(m, comp):
    """photo_<part>.png for the part Omni named, else the general machine photo."""
    base = WEB / "cache" / m
    try:
        parts = json.loads((base / "parts.json").read_text())
        c = str(comp or "").lower()
        hit = next((p["name"] for p in parts if p["name"] in c or any(k in c for k in p["keywords"])), None)
        if hit and (base / f"cutaway_{hit}.png").exists():
            return f"/cache/{m}/cutaway_{hit}.png"
        if hit and (base / f"manual_{hit}.png").exists():
            return f"/cache/{m}/manual_{hit}.png"
        if hit and (base / f"photo_{hit}.png").exists():
            return f"/cache/{m}/photo_{hit}.png"
    except Exception:
        pass
    return f"/cache/{m}/photo.png" if (base / "photo.png").exists() else f"/cache/{m}/machine.png"


# ---------- Unit fingerprint (which fan / pump / valve) ----------
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


# ---------- Which part: from where the sound energy sits ----------
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


# ---------- vLLM endpoint (optional) ----------
from vllm_client import VLLM, URL as VLLM_URL, describe_vllm, chat_with_tools, health as vllm_health
print("Omni engine:", f"vLLM at {VLLM_URL}" if VLLM else "PyTorch (in-process)", flush=True)


# ---------- API ----------
app = FastAPI()


@app.post("/analyze")
def analyze(file: UploadFile = File(...), machine: str = Form(None)):
    t0 = time.perf_counter()
    data = file.file.read()
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        return JSONResponse({"error": "Could not read audio file"}, 400)
    if sr != SR or y.ndim != 2 or y.shape[1] != N_MICS:
        return JSONResponse({"error": "Need an 8-channel 16 kHz MIMII wav"}, 400)

    p = classify(y)
    m = pick_machine(machine) or machine_type(data)
    res = {"status": "abnormal" if p >= 0.5 else "normal", "score": round(p, 4), "machine": m}

    if res["status"] == "abnormal":
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(data)
            f.flush()
            try:
                d = describe(f.name, m)
            except Exception as e:
                d = {"likely_component": "unknown", "fault": "unrecognized anomaly",
                     "severity": "medium", "sound_evidence": f"description failed: {e}"}
        m = m or pick_machine(d.get("machine")) or "fan"
        d["machine"] = m
        d = pick_part(d, m, y, sr)
        aid, aconf = asset_id(y, sr, m)
        key = f"{m}_{aid}" if aid and (WEB / "cache" / f"{m}_{aid}").exists() else m
        res.update(asset=aid, asset_conf=round(aconf, 3), cache_key=key)
        # keep cached 3D defect location, refresh the text for the viewer label
        cache = WEB / "cache" / key / "diag.json"
        if cache.exists():
            c = json.loads(cache.read_text())
            c.update({k: d[k] for k in ("machine", "likely_component", "fault", "severity", "sound_evidence") if k in d})
            cache.write_text(json.dumps(c, indent=2))
        res.update(machine=m, diag=d,
                   image_url=fault_photo(key, d.get("likely_component")), viewer_url=f"/viewer.html?m={key}&score={p:.3f}")

    if res["status"] == "abnormal":
        iid = uuid.uuid4().hex[:8]
        res["incident"] = iid
        res["chat_url"] = f"/chat.html?incident={iid}"
        INCIDENTS[iid] = dict(res)
        send_alert(res)
        res["alert"] = f"ntfy:{NTFY_TOPIC}"
    if "asset" not in res:
        aid, aconf = asset_id(y, sr, res.get("machine"))
        mm = res.get("machine")
        res.update(asset=aid, asset_conf=round(aconf, 3),
                   cache_key=f"{mm}_{aid}" if aid and (WEB / "cache" / f"{mm}_{aid}").exists() else mm)
    res["evidence"] = EVID
    sc = res["score"]
    res["level"] = "normal" if res["status"] == "normal" else ("critical" if sc >= 0.95 else "warning" if sc >= 0.75 else "watch")
    HIST.setdefault(" ".join(x for x in (res.get("machine"), res.get("asset")) if x) or "unknown", []).append(
        {"t": round(time.time()), "score": sc, "status": res["status"], "file": file.filename})
    res["history"] = HIST[" ".join(x for x in (res.get("machine"), res.get("asset")) if x) or "unknown"][-20:]
    iu = res.get("image_url") or ""
    if "/manual_" in iu:
        try:
            res["image_info"] = json.loads((WEB / iu.lstrip("/").replace(".png", ".json")).read_text())
        except Exception:
            pass
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return res


@app.get("/incident/{iid}")
def get_incident(iid: str):
    return INCIDENTS.get(iid) or JSONResponse({"error": "unknown incident"}, 404)


@app.post("/chat")
def chat(req: ChatReq):
    inc = INCIDENTS.get(req.incident or "", {})
    d = inc.get("diag", {})
    q = req.messages[-1]["content"]
    m = inc.get("machine") or pick_machine(q) or "pump"
    hits = retrieve(f"{m} {d.get('likely_component', '')} {d.get('fault', '')} {q}", m)
    ctx = "\n\n".join(f"[{i + 1}] {h['title']}, page {h['page']}:\n{h['text']}" for i, h in enumerate(hits)) \
        or "(no manual excerpts found)"
    about = (f"Incident: {m} flagged ABNORMAL ({inc.get('score', 0) * 100:.0f}% anomaly probability). "
             f"Suspected component: {d.get('likely_component', '?')}. Fault: {d.get('fault', '?')}. "
             f"Severity: {d.get('severity', '?')}. Heard: {d.get('sound_evidence', '?')}.") if inc else f"Machine: {m}."
    if VLLM:
        r = vllm_chat(req, inc, d, m, q, hits, about)
        if r is not None:
            return r
    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    for x in req.messages[:-1][-6:]:
        conv.append({"role": x["role"], "content": [{"type": "text", "text": x["content"]}]})
    conv.append({"role": "user", "content": [{"type": "text", "text": f"{about}\n\nManual excerpts:\n{ctx}\n\nQuestion: {q}"}]})
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, return_tensors="pt", padding=True).to("cuda")
    streamer = TextIteratorStreamer(proc.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def run():
        with GEN_LOCK, torch.no_grad(), omni.thinker.disable_adapter():
            omni.thinker.generate(**inp, max_new_tokens=500, do_sample=False, streamer=streamer)
    _th.Thread(target=run, daemon=True).start()
    src = [{"n": i + 1, "title": h["title"], "page": h["page"], "url": f"/manuals/{h['file']}#page={h['page']}"}
           for i, h in enumerate(hits)]

    def stream():
        yield json.dumps({"sources": src, "machine": m}) + "\n"
        for t in streamer:
            yield t
    return StreamingResponse(stream(), media_type="text/plain")


TOOL_NOTE = (" You can call tools: search_manuals for more excerpts, get_incident for the current fault, "
             "get_part_visual to show where a part is, get_machine_history to check if it is getting worse. "
             "Cite every manual fact with its number like [3].")


def vllm_chat(req, inc, d, m, q, hits, about):
    allh = list(hits)

    def numbered(hs, start):
        return [{"n": start + i + 1, "title": h["title"], "page": h["page"], "text": h["text"]} for i, h in enumerate(hs)]

    def search_manuals(query, machine=None):
        new = retrieve(query, machine or m); start = len(allh); allh.extend(new)
        return numbered(new, start) or {"result": "no matching excerpts"}

    def get_incident():
        if not inc:
            return {"result": "no active incident"}
        return {k: inc.get(k) for k in ("machine", "asset", "status", "score", "level", "diag", "image_url", "viewer_url")}

    def get_part_visual(part):
        key = inc.get("cache_key") or m
        return {"part": part, "image_url": fault_photo(key, part), "viewer_url": f"/viewer.html?m={key}"}

    def get_machine_history(machine=None):
        mm = machine or m
        return {k: v[-10:] for k, v in HIST.items() if k.startswith(mm)} or {"result": "no history yet"}

    funcs = dict(search_manuals=search_manuals, get_incident=get_incident,
                 get_part_visual=get_part_visual, get_machine_history=get_machine_history)
    ctx = "\n\n".join(f"[{i + 1}] {h['title']}, page {h['page']}:\n{h['text']}" for i, h in enumerate(hits)) \
        or "(no manual excerpts found)"
    msgs = [{"role": "system", "content": SYSTEM + TOOL_NOTE}]
    msgs += [{"role": x["role"], "content": x["content"]} for x in req.messages[:-1][-6:]]
    msgs.append({"role": "user", "content": f"{about}\n\nManual excerpts:\n{ctx}\n\nQuestion: {q}"})
    try:
        answer, used = chat_with_tools(msgs, funcs)
    except Exception as e:
        print("LLM endpoint chat failed, using PyTorch:", e, flush=True)
        return None
    src = [{"n": i + 1, "title": h["title"], "page": h["page"], "url": f"/manuals/{h['file']}#page={h['page']}"}
           for i, h in enumerate(allh)]

    def stream():
        yield json.dumps({"sources": src, "machine": m, "engine": "vLLM", "tools_used": used}) + "\n"
        for i in range(0, len(answer), 40):
            yield answer[i:i + 40]
    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/engine")
def engine():
    return {"omni_engine": "vLLM" if VLLM else "PyTorch", "vllm_url": VLLM_URL if VLLM else None,
            "vllm": vllm_health() if VLLM else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


app.mount("/manuals", StaticFiles(directory=MAN), name="manuals")

app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)