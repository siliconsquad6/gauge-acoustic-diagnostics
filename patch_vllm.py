"""Serve Omni through the local vLLM endpoint and give the repair chatbot tool calling.
Run once from the gauge folder, after the other patches:  python patch_vllm.py

Off by default. Turn on with:  GAUGE_VLLM=1 python app/server.py   (start bash serve_vllm.sh first)
  - fault description (/analyze, abnormal clips) -> vLLM
  - repair chat (/chat) -> vLLM with tools: search_manuals, get_incident, get_part_visual, get_machine_history
  - GET /engine shows which engine is serving and what vLLM reports
The fine-tuned machine-type LoRA still runs in PyTorch inside the server.
"""
p = "app/server.py"; s = open(p).read()
if "def vllm_chat(" in s:
    print("already patched"); raise SystemExit


def put(anchor, text, before=True):
    global s
    assert s.count(anchor) == 1, f"anchor not found exactly once: {anchor[:60]}"
    s = s.replace(anchor, text + anchor if before else anchor + text)


# 1. import the client
put("# ---------- API ----------", '''# ---------- vLLM endpoint (optional) ----------
from vllm_client import VLLM, URL as VLLM_URL, describe_vllm, chat_with_tools, health as vllm_health
print("Omni engine:", f"vLLM at {VLLM_URL}" if VLLM else "PyTorch (in-process)", flush=True)


''')

# 2. fault description through vLLM
put('    conv = [{"role": "user", "content": [{"type": "audio", "audio": str(wav_path)}, {"type": "text", "text": ask}]}]',
    '    if VLLM:\n'
    '        try:\n'
    '            return json.loads(re.search(r"\\{.*\\}", describe_vllm(wav_path, ask), re.S).group())\n'
    '        except Exception as e:\n'
    '            print("LLM endpoint describe failed, using PyTorch:", e, flush=True)\n')

# 3. chat through vLLM with tools
put('    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]',
    '    if VLLM:\n'
    '        r = vllm_chat(req, inc, d, m, q, hits, about)\n'
    '        if r is not None:\n'
    '            return r\n')

# 4. the tool-calling chat + /engine endpoint
put('app.mount("/manuals"', '''TOOL_NOTE = (" You can call tools: search_manuals for more excerpts, get_incident for the current fault, "
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
    ctx = "\\n\\n".join(f"[{i + 1}] {h['title']}, page {h['page']}:\\n{h['text']}" for i, h in enumerate(hits)) \\
        or "(no manual excerpts found)"
    msgs = [{"role": "system", "content": SYSTEM + TOOL_NOTE}]
    msgs += [{"role": x["role"], "content": x["content"]} for x in req.messages[:-1][-6:]]
    msgs.append({"role": "user", "content": f"{about}\\n\\nManual excerpts:\\n{ctx}\\n\\nQuestion: {q}"})
    try:
        answer, used = chat_with_tools(msgs, funcs)
    except Exception as e:
        print("LLM endpoint chat failed, using PyTorch:", e, flush=True)
        return None
    src = [{"n": i + 1, "title": h["title"], "page": h["page"], "url": f"/manuals/{h['file']}#page={h['page']}"}
           for i, h in enumerate(allh)]

    def stream():
        yield json.dumps({"sources": src, "machine": m, "engine": "vLLM", "tools_used": used}) + "\\n"
        for i in range(0, len(answer), 40):
            yield answer[i:i + 40]
    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/engine")
def engine():
    return {"omni_engine": "vLLM" if VLLM else "PyTorch", "vllm_url": VLLM_URL if VLLM else None,
            "vllm": vllm_health() if VLLM else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


''')

open(p, "w").write(s)
print("patched: vLLM describe + tool-calling chat + /engine")

# 5. show engine and tools in the chat header
h = "web/chat.html"; t = open(h).read()
old = "$('mtag').textContent = meta.machine ? 'Machine: ' + meta.machine : '';"
if old in t:
    t = t.replace(old, "$('mtag').textContent = (meta.machine ? 'Machine: ' + meta.machine : '')"
                       " + (meta.engine ? ' · ' + meta.engine : '')"
                       " + (meta.tools_used && meta.tools_used.length ? ' · tools: ' + meta.tools_used.join(', ') : '');")
    open(h, "w").write(t); print("patched: chat.html shows engine + tools used")
else:
    print("chat.html: header already patched or changed, skipped")