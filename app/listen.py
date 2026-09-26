import sys, json, torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

M = "/home/hp20/Desktop/edge-ai/gauge/models/omni"
wav = sys.argv[1]
machine = sys.argv[2] if len(sys.argv) > 2 else "industrial machine"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda")
proc = Qwen2_5OmniProcessor.from_pretrained(M)

ask = (f"This is audio from a {machine} flagged as abnormal. "
       "Reply ONLY with JSON: {\"machine\":..., \"likely_component\":..., "
       "\"fault\":..., \"severity\":\"low|medium|high\", \"sound_evidence\":...}")
conv = [{"role": "user", "content": [{"type": "audio", "audio": wav}, {"type": "text", "text": ask}]}]

text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
inp = proc(text=text, audio=audios, return_tensors="pt", padding=True).to("cuda").to(model.dtype)
out = model.generate(**inp, max_new_tokens=200, return_audio=False)
print(proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])
import re
txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
json.dump(json.loads(re.search(r"\{.*\}", txt, re.S).group()), open("/home/hp20/Desktop/edge-ai/gauge/out/diag.json","w"), indent=2)
