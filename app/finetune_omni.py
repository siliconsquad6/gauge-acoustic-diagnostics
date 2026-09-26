"""LoRA fine-tune Qwen2.5-Omni (thinker) on MIMII so it answers
{"machine": fan|pump|valve, "status": normal|abnormal} from audio alone.

- Train: all machine ids except id_06, balanced per (machine, status)
- Test : id_06 only (machines it has never heard), scored BEFORE and AFTER training
- Saves LoRA adapter to gauge/models/omni-gauge-lora

Run:  python app/finetune_omni.py            # 400 clips per (machine,status)
      python app/finetune_omni.py 800        # more data, longer
"""
import json, random, re, sys, time
from pathlib import Path
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from peft import LoraConfig, get_peft_model

H = Path.home()
G = Path(__file__).resolve().parent.parent
OMNI = G / "models" / "omni"
OUT = G / "models" / "omni-gauge-lora-machine"
ROOTS = list((G.parent.parent / "data").glob("combined_data_sh*")) + list((H / "Downloads").glob("*valve*"))
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N_TEST = 30
ACC, LR = 8, 1e-4
PROMPT = ("You are a factory acoustic monitor. Listen to this machine recording. "
          'Reply ONLY with JSON: {"machine":"fan|pump|valve",}')


def label(p):
    s = str(p).lower()
    m = next((k for k in ("fan", "pump", "valve") if f"/{k}/" in s), None)
    st = "abnormal" if "/abnormal/" in s else ("normal" if "/normal/" in s else None)
    return m, st


# ---------- data ----------
random.seed(0)
print("Roots:", ROOTS)
buckets = {}
for r in ROOTS:
    for p in r.rglob("*.wav"):
        m, st = label(p)
        if m and st:
            split = "test" if "id_06" in str(p).lower() else "train"
            buckets.setdefault((split, m, st), []).append(p)
for k in sorted(buckets):
    print(k, len(buckets[k]))


def take(split, n):
    out = []
    for (s, m, st), v in buckets.items():
        if s == split:
            random.shuffle(v)
            out += [(p, m, st) for p in v[:n]]
    random.shuffle(out)
    return out


train, test = take("train", N_TRAIN), take("test", N_TEST)
print(f"train {len(train)} | test {len(test)}")

# ---------- model ----------
print("Loading Omni...")
full = Qwen2_5OmniForConditionalGeneration.from_pretrained(OMNI, torch_dtype=torch.bfloat16, device_map="cuda")
proc = Qwen2_5OmniProcessor.from_pretrained(OMNI)
model = full.thinker          # the part that listens and writes text
del full
torch.cuda.empty_cache()


def build(path, target=None):
    conv = [{"role": "user", "content": [{"type": "audio", "audio": str(path)}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if target:
        text += target + "<|im_end|>"
    audios, _, _ = process_mm_info(conv, use_audio_in_video=False)
    return proc(text=text, audio=audios, return_tensors="pt", padding=True).to("cuda").to(torch.bfloat16)


@torch.no_grad()
def evaluate(tag):
    model.eval()
    ok_m = ok_s = bad = 0
    t0 = time.time()
    for p, m, st in test:
        inp = build(p)
        out = model.generate(**inp, max_new_tokens=30, do_sample=False)
        txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        try:
            d = json.loads(re.search(r"\{.*?\}", txt, re.S).group())
        except Exception:
            d, bad = {}, bad + 1
        ok_m += str(d.get("machine", "")).lower() == m
        ok_s += str(d.get("status", "")).lower() == st
    n = len(test)
    print(f"[{tag}] machine acc {ok_m / n:.1%} | status acc {ok_s / n:.1%} | unparsable {bad}/{n} | {time.time() - t0:.0f}s")


evaluate("BEFORE fine-tune")

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=r"^(?!.*(audio_tower|visual)).*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

# ---------- train (1 epoch) ----------
model.train()
t0, run = time.time(), 0.0
for i, (p, m, st) in enumerate(train):
    target = json.dumps({"machine": m})
    inp = build(p, target)
    n = len(proc.tokenizer(target + "<|im_end|>").input_ids)
    labels = torch.full_like(inp["input_ids"], -100)
    labels[:, -n:] = inp["input_ids"][:, -n:]          # learn only the answer
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**inp, labels=labels).loss / ACC
    loss.backward()
    run += loss.item() * ACC
    if (i + 1) % ACC == 0:
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step()
        opt.zero_grad()
    if (i + 1) % 50 == 0:
        el = time.time() - t0
        print(f"{i + 1}/{len(train)} loss {run / 50:.4f} | {el / (i + 1):.2f}s/clip | ETA {el / (i + 1) * (len(train) - i - 1) / 60:.0f} min")
        run = 0.0

model.save_pretrained(OUT)
print("saved adapter to", OUT)
evaluate("AFTER fine-tune")