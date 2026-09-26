"""
Gauge: test the trained model on ONE audio clip.

Usage:
  python Scripts/predict.py                       # random clip from the TEST split
  python Scripts/predict.py --abnormal            # random abnormal test clip
  python Scripts/predict.py --normal              # random normal test clip
  python Scripts/predict.py path/to/clip.wav      # any 8-channel MIMII wav
"""
import sys, json, math, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from transformers import ASTModel

ROOT = Path.home() / "Desktop" / "edge-ai"
RUN_DIR = ROOT / "runs" / "gauge_combined"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR, N_MICS, N_MELS, N_FRAMES = 16000, 8, 128, 1024
MEAN, STD = -4.2677393, 4.5689974   # must match preprocessing


# ---------- Same preprocessing as preprocess_8mic.py ----------
def channel_to_spec(y):
    wav = torch.from_numpy(y).unsqueeze(0)
    wav = wav - wav.mean()
    spec = torchaudio.compliance.kaldi.fbank(
        wav, sample_frequency=SR, num_mel_bins=N_MELS, frame_length=25, frame_shift=10,
        window_type="hanning", htk_compat=True, use_energy=False, dither=0.0)
    if spec.shape[0] < N_FRAMES:
        spec = torch.nn.functional.pad(spec, (0, 0, 0, N_FRAMES - spec.shape[0]))
    return (spec[:N_FRAMES] - MEAN) / (STD * 2)


def load_clip(path):
    y, sr = sf.read(path, dtype="float32")
    assert sr == SR and y.ndim == 2 and y.shape[1] == N_MICS, "Need an 8-channel 16 kHz MIMII clip"
    return torch.stack([channel_to_spec(np.ascontiguousarray(y[:, c])) for c in range(N_MICS)])


# ---------- Same architecture as training ----------
class GaugeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = ASTModel.from_pretrained(MODEL)
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


def band_ranges(f_dim):
    mel = lambda f: 1127 * math.log(1 + f / 700)
    hz = lambda m: 700 * (math.exp(m / 1127) - 1)
    lo, hi = mel(20), mel(8000)
    e = [hz(lo + i * (hi - lo) / 129) for i in range(130)]
    return [f"{e[10*b]:.0f}-{e[min(10*b+17,129)]:.0f} Hz" for b in range(f_dim)]


# ---------- Pick the clip ----------
args = sys.argv[1:]
truth = None
if args and not args[0].startswith("--"):
    clip = args[0]
else:
    meta = pd.read_csv(RUN_DIR / "split_meta.csv")
    test = meta[meta.split == "test"]          # never seen during training
    if "--abnormal" in args: test = test[test.label == "abnormal"]
    if "--normal" in args: test = test[test.label == "normal"]
    row = test.sample(1, random_state=random.randint(0, 10**6)).iloc[0]
    clip, truth = row.audio_path, row.label
    print(f"Test clip: {row.machine_type} {row.machine_id} | {row.noise} | true label: {truth}")

# ---------- Run ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GaugeNet().to(device).eval()
model.load_state_dict(torch.load(RUN_DIR / "best_model.pt", map_location=device))

t0 = time.perf_counter()
x = load_clip(clip).unsqueeze(0).to(device)
t1 = time.perf_counter()
with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
    logits, mic_w, band_w = model(x)
if device == "cuda": torch.cuda.synchronize()
t2 = time.perf_counter()

p = torch.softmax(logits.float(), -1)[0, 1].item()
bands = band_w.float()[0].cpu().numpy()
mics = mic_w.float()[0].mean(-1).cpu().numpy()
ranges = band_ranges(model.f_dim)
top = np.argsort(bands)[::-1][:3]

result = {
    "clip": str(clip),
    "true_label": truth,
    "prob_abnormal": round(p, 4),
    "prediction": "abnormal" if p >= 0.5 else "normal",
    "correct": (None if truth is None else ("abnormal" if p >= 0.5 else "normal") == truth),
    "top_bands": {ranges[i]: round(float(bands[i]), 4) for i in top},
    "mic_weights": [round(float(m), 4) for m in mics],
    "latency_ms": {"preprocess": round((t1 - t0) * 1000, 1), "model": round((t2 - t1) * 1000, 1)},
}
print(json.dumps(result, indent=2))