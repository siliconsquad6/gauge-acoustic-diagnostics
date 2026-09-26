"""Add the manual-grounded repair chatbot to server.py (run once, from the gauge folder).
Needs patch_alerts.py applied first.
"""
p = "app/server.py"; s = open(p).read()
if "def retrieve(" in s:
    print("already patched"); raise SystemExit

CHAT_CORE = '''# ---------- Repair chatbot (answers from the manuals) ----------
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


'''

CHAT_ROUTES = '''@app.get("/incident/{iid}")
def get_incident(iid: str):
    return INCIDENTS.get(iid) or JSONResponse({"error": "unknown incident"}, 404)


@app.post("/chat")
def chat(req: ChatReq):
    inc = INCIDENTS.get(req.incident or "", {})
    d = inc.get("diag", {})
    q = req.messages[-1]["content"]
    m = inc.get("machine") or pick_machine(q) or "pump"
    hits = retrieve(f"{m} {d.get('likely_component', '')} {d.get('fault', '')} {q}", m)
    ctx = "\\n\\n".join(f"[{i + 1}] {h['title']}, page {h['page']}:\\n{h['text']}" for i, h in enumerate(hits)) \\
        or "(no manual excerpts found)"
    about = (f"Incident: {m} flagged ABNORMAL ({inc.get('score', 0) * 100:.0f}% anomaly probability). "
             f"Suspected component: {d.get('likely_component', '?')}. Fault: {d.get('fault', '?')}. "
             f"Severity: {d.get('severity', '?')}. Heard: {d.get('sound_evidence', '?')}.") if inc else f"Machine: {m}."
    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    for x in req.messages[:-1][-6:]:
        conv.append({"role": x["role"], "content": [{"type": "text", "text": x["content"]}]})
    conv.append({"role": "user", "content": [{"type": "text", "text": f"{about}\\n\\nManual excerpts:\\n{ctx}\\n\\nQuestion: {q}"}]})
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
        yield json.dumps({"sources": src, "machine": m}) + "\\n"
        for t in streamer:
            yield t
    return StreamingResponse(stream(), media_type="text/plain")


app.mount("/manuals", StaticFiles(directory=MAN), name="manuals")
'''

ok = []
s = s.replace("# ---------- API ----------", CHAT_CORE + "# ---------- API ----------", 1); ok.append("def retrieve(" in s)
s = s.replace("        send_alert(res)\n",
              "        iid = uuid.uuid4().hex[:8]\n"
              "        res[\"incident\"] = iid\n"
              "        res[\"chat_url\"] = f\"/chat.html?incident={iid}\"\n"
              "        INCIDENTS[iid] = dict(res)\n"
              "        send_alert(res)\n", 1); ok.append("INCIDENTS[iid]" in s)
s = s.replace('"Click": DASH_URL,', '"Click": DASH_URL + (f"chat.html?incident={res[\'incident\']}" if res.get("incident") else ""),', 1)
ok.append("chat.html?incident=" in s)
s = s.replace('app.mount("/", StaticFiles', CHAT_ROUTES + '\napp.mount("/", StaticFiles', 1); ok.append('@app.post("/chat")' in s)
open(p, "w").write(s)
for name, good in zip(["chat core", "incident ids", "alert opens chat", "chat routes"], ok):
    print(("OK   " if good else "MISS ") + name)
