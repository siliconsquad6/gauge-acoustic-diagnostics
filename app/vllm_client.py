"""Client for the local vLLM endpoint (OpenAI-compatible) running Qwen2.5-Omni on the ZGX Nano.

Start the endpoint first:  bash serve_vllm.sh
Turn it on in the server:   GAUGE_VLLM=1 python app/server.py

Used for:
  describe_vllm()    abnormal clip -> fault description JSON (audio in, text out)
  chat_with_tools()  repair chatbot with tool calling (the model decides when to search manuals etc.)
"""
import base64, io, json, os
import numpy as np, requests, soundfile as sf

VLLM = os.environ.get("GAUGE_VLLM", "0") not in ("", "0", "false", "False")
URL = os.environ.get("GAUGE_VLLM_URL", "http://127.0.0.1:8001/v1")
MODEL = os.environ.get("GAUGE_VLLM_MODEL", "gauge-omni")
TIMEOUT = 180

# Tools the repair assistant may call. The server supplies the Python function for each name.
TOOLS = [
    {"type": "function", "function": {
        "name": "search_manuals",
        "description": "Search the maintenance manuals for repair steps, causes, safety or specifications. "
                       "Returns numbered excerpts to cite like [n].",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look up, e.g. 'replace mechanical seal'"},
            "machine": {"type": "string", "enum": ["fan", "pump", "valve"]}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_incident",
        "description": "Get the current incident: machine, unit, anomaly score, severity, suspected part and fault.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_part_visual",
        "description": "Get the image and 3D viewer link that show where a part is on this machine.",
        "parameters": {"type": "object", "properties": {
            "part": {"type": "string", "description": "e.g. bearing, seal, impeller, plunger, seat"}},
            "required": ["part"]}}},
    {"type": "function", "function": {
        "name": "get_machine_history",
        "description": "Recent health scores for a machine, to judge whether it is getting worse.",
        "parameters": {"type": "object", "properties": {
            "machine": {"type": "string", "enum": ["fan", "pump", "valve"]}}}}},
]


def _post(payload):
    r = requests.post(f"{URL}/chat/completions", json={"model": MODEL, **payload}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def health():
    try:
        r = requests.get(f"{URL}/models", timeout=5)
        return {"ok": r.ok, "models": [m["id"] for m in r.json().get("data", [])]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _audio_data_url(wav_path):
    """8-mic MIMII clip -> mono 16-bit wav as a base64 data URL (small and what Omni expects)."""
    y, sr = sf.read(wav_path, dtype="float32")
    if y.ndim == 2:
        y = y.mean(1)
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def describe_vllm(wav_path, ask):
    """Audio + instruction -> raw text answer from Omni served by vLLM."""
    out = _post({
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": _audio_data_url(wav_path)}},
            {"type": "text", "text": ask}]}],
        "temperature": 0, "max_tokens": 200})
    return out["choices"][0]["message"]["content"] or ""


def chat_with_tools(messages, funcs, max_rounds=4):
    """Let the model call tools until it writes a final answer.
    Returns (final_text, tools_used). Falls back to a plain call if the endpoint rejects tools."""
    used, use_tools = [], True
    for _ in range(max_rounds):
        payload = {"messages": messages, "temperature": 0, "max_tokens": 600}
        if use_tools:
            payload.update(tools=TOOLS, tool_choice="auto")
        try:
            msg = _post(payload)["choices"][0]["message"]
        except requests.HTTPError as e:
            if use_tools:
                print("vLLM rejected tools, retrying without:", e, flush=True)
                use_tools = False
                continue
            raise
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip(), used
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
                result = funcs[name](**args)
            except Exception as e:
                result = {"error": f"{name} failed: {e}"}
            used.append(name)
            print(f"[tool] {name} -> {str(result)[:120]}", flush=True)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "name": name,
                             "content": json.dumps(result)[:6000]})
    # out of rounds: ask for an answer with what it has
    msg = _post({"messages": messages, "temperature": 0, "max_tokens": 600})["choices"][0]["message"]
    return (msg.get("content") or "").strip(), used