"""Wire the fine-tuned Omni adapter into server.py (run from the gauge folder)."""
import re
p = "app/server.py"; s = open(p).read()

# 1. gauge folder = wherever this file lives (it moved into Sreeram/)
s = s.replace('G = ROOT / "gauge"', 'G = Path(__file__).resolve().parent.parent')

# 2. drop the old logistic-regression machine classifier
s = re.sub(r'import joblib\nfrom train_machine import mfeat\nMT = joblib\.load\([^\n]*\)\n', '', s)

# 3. load the LoRA adapter on top of Omni
s = s.replace('proc = Qwen2_5OmniProcessor.from_pretrained(OMNI)\n',
 '''proc = Qwen2_5OmniProcessor.from_pretrained(OMNI)
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
    return pick_machine(txt)
''', 1)

# 4. fault description uses the ORIGINAL Omni (adapter switched off)
s = s.replace('''    with torch.no_grad():
        out = omni.generate(**inp, max_new_tokens=200, return_audio=False)''',
'''    with torch.no_grad(), omni.thinker.disable_adapter():
        out = omni.generate(**inp, max_new_tokens=200, return_audio=False)''')

# 5. use it in /analyze
s = re.sub(r'    m = pick_machine\(machine[^\n]*\n', '    m = pick_machine(machine) or machine_type(data)\n', s)

open(p, "w").write(s)
for k in ["Path(__file__).resolve().parent.parent", "PeftModel.from_pretrained", "disable_adapter", "machine_type(data)"]:
    print(("OK   " if k in s else "MISS ") + k)
print("leftover MT:", "MT." in s)