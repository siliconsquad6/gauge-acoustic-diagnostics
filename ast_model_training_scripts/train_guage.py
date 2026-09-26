"""
GaugeNet (combined): ONE model for BOTH fan and pump.
Same architecture: 8-mic attention fusion + AST + frequency attention head.
Classifies machine audio as normal (0) or abnormal (1), regardless of machine type.

Usage:  python train_guage_combined.py
"""
import sys, json, random, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from transformers import ASTModel

# ======================= Settings =======================
MACHINES = ["fan", "pump"]
FEAT_DIR = Path.home() / "Desktop" / "edge-ai" / "data" / "features_8mic"
OUT_DIR = Path.home() / "Desktop" / "edge-ai" / "runs" / "gauge_combined"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
EPOCHS, BATCH, SEED = 100, 4, 42
PATIENCE = 10   # early stopping: stop if val AUC doesn't improve for 10 epochs
LR_BACKBONE, LR_NEW = 1e-5, 1e-3   # pretrained AST moves slowly, new layers learn fast
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

# ======================= 1. Load data =======================
# features: (N, 8 mics, 1024 time frames, 128 mel bands). Row i matches meta row i.
# Load both machine types. Each keeps its own feature file;
# "src_row" remembers which row inside that file a clip lives in.
X = {}
metas = []
for m in MACHINES:
    X[m] = np.load(FEAT_DIR / f"{m}_features.npy", mmap_mode="r")
    mm = pd.read_csv(FEAT_DIR / f"{m}_meta.csv")
    assert len(X[m]) == len(mm)
    mm["src_row"] = np.arange(len(mm))
    metas.append(mm)
meta = pd.concat(metas, ignore_index=True)   # one table, fan rows then pump rows
meta["machine_key"] = meta.machine_type + "_" + meta.machine_id   # e.g. fan_id_00

# ======================= 2. Leak-free split =======================
# One group = one original recording (its 0 dB and 6 dB copies share a group).
# Groups are shuffled and dealt 70/15/15 within each (machine_id, label),
# so twins never cross splits and every split has every ID and both labels.
meta["group"] = meta.machine_type + "/" + meta.machine_id + "/" + meta.label + "/" + meta.original_file
split_of = {}
for _, g in meta.groupby(["machine_type", "machine_id", "label"]):
    groups = sorted(g.group.unique())
    random.Random(SEED).shuffle(groups)
    a, b = int(0.70 * len(groups)), int(0.85 * len(groups))
    for i, gr in enumerate(groups):
        split_of[gr] = "train" if i < a else "val" if i < b else "test"
meta["split"] = meta.group.map(split_of)
assert meta.groupby("group").split.nunique().max() == 1, "Leakage detected"
meta.to_csv(OUT_DIR / "split_meta.csv", index=False)
print(meta.groupby(["machine_type", "split", "label"]).size(), "\n")


# ======================= 3. Dataset =======================
class SpecDataset(Dataset):
    def __init__(self, rows):
        self.rows = np.array(rows)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        row = meta.iloc[r]
        x = torch.from_numpy(np.array(X[row.machine_type][row.src_row], dtype=np.float32))  # (8, 1024, 128)
        return x, int(meta.label_id[r]), r

loaders = {
    s: DataLoader(SpecDataset(meta.index[meta.split == s]), batch_size=BATCH,
                  shuffle=(s == "train"), num_workers=2, pin_memory=False)
    for s in ["train", "val", "test"]
}


# ======================= 4. Model =======================
class GaugeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = ASTModel.from_pretrained(MODEL)  # pretrained audio backbone
        cfg = self.ast.config
        H = cfg.hidden_size  # 768
        # AST cuts the spectrogram into a grid of patches: f_dim bands x t_dim time steps
        self.f_dim = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1  # 12
        self.t_dim = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1         # 101

        # --- Idea 1: 8-mic attention fusion ---
        # For every mic and every mel band, summarize the sound (mean, std over time),
        # score it, and softmax across the 8 mics. Result: per-band mic weights that
        # blend 8 spectrograms into 1. Last layer starts at zero = plain average at first.
        self.mic_scorer = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.mic_scorer[-1].weight); nn.init.zeros_(self.mic_scorer[-1].bias)

        # --- Idea 2: frequency attention head ---
        # Average AST patch outputs over time per band, score each band,
        # softmax across bands. The weights say which pitch range drove the decision.
        self.band_scorer = nn.Sequential(nn.Linear(H, 128), nn.Tanh(), nn.Linear(128, 1))

        # Final decision uses AST's global summary + the band-attended summary
        self.head = nn.Sequential(nn.LayerNorm(2 * H), nn.Dropout(0.1), nn.Linear(2 * H, 2))

    def forward(self, x):                                   # x: (B, 8, T, F)
        B = x.shape[0]
        # Idea 1: fuse microphones
        stats = torch.stack([x.mean(2), x.std(2)], dim=-1)  # (B, 8, F, 2)
        mic_w = torch.softmax(self.mic_scorer(stats).squeeze(-1), dim=1)  # (B, 8, F)
        fused = (x * mic_w.unsqueeze(2)).sum(1)             # (B, T, F)

        # Pretrained AST encoder
        h = self.ast(input_values=fused).last_hidden_state  # (B, 2 + f*t, H)
        cls = (h[:, 0] + h[:, 1]) / 2                       # global summary
        patches = h[:, 2:].reshape(B, self.f_dim, self.t_dim, -1)

        # Idea 2: attend over frequency bands
        bands = patches.mean(2)                              # (B, f_dim, H)
        band_w = torch.softmax(self.band_scorer(bands).squeeze(-1), dim=1)  # (B, f_dim)
        band_vec = (bands * band_w.unsqueeze(-1)).sum(1)     # (B, H)

        logits = self.head(torch.cat([cls, band_vec], dim=-1))
        return logits, mic_w, band_w


model = GaugeNet().cuda()

# Two learning rates: gentle for pretrained AST, faster for our new layers
backbone = list(model.ast.parameters())
new = [p for n, p in model.named_parameters() if not n.startswith("ast.")]
optim = torch.optim.AdamW(
    [{"params": backbone, "lr": LR_BACKBONE}, {"params": new, "lr": LR_NEW}],
    weight_decay=0.01,
)

# Class weights from TRAIN only: abnormal clips are rarer, so they count more
train_labels = meta.label_id[meta.split == "train"].values
counts = np.bincount(train_labels, minlength=2)
weights = torch.tensor(len(train_labels) / (2 * counts), dtype=torch.float32).cuda()
loss_fn = nn.CrossEntropyLoss(weight=weights)
print("Class weights (normal, abnormal):", [round(w, 3) for w in weights.tolist()])


# ======================= 5. Evaluation =======================
@torch.no_grad()
def evaluate(split):
    model.eval()
    probs, labels, rows, mics, bands = [], [], [], [], []
    for x, y, r in loaders[split]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, mic_w, band_w = model(x.cuda(non_blocking=True))
        probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
        mics.append(mic_w.float().mean(-1).cpu())   # avg over bands -> (B, 8)
        bands.append(band_w.float().cpu())            # (B, f_dim)
        labels += y.tolist(); rows += r.tolist()
    return (torch.cat(probs).numpy(), np.array(labels), np.array(rows),
            torch.cat(mics).numpy(), torch.cat(bands).numpy())


# ======================= 6. Train =======================
VAL_EVERY = 500   # steps between mid-epoch validation checks
best_auc, bad_epochs = -1, 0
history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    total, correct, seen = 0.0, 0, 0
    for step, (x, y, _) in enumerate(loaders["train"]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = model(x.cuda(non_blocking=True))
        loss = loss_fn(logits.float(), y.cuda())
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item()
        correct += (logits.argmax(-1).cpu() == y).sum().item()
        seen += len(y)

        # Every 100 steps: loss + running train accuracy
        if step % 100 == 0:
            print(f"epoch {epoch} step {step}/{len(loaders['train'])} "
                  f"loss {loss.item():.4f} | train acc {correct / seen:.4f}")

        # Every 500 steps: quick check on the validation set
        if step > 0 and step % VAL_EVERY == 0:
            vp, vy, *_ = evaluate("val")
            print(f"   >> step {step} | val acc {accuracy_score(vy, (vp >= 0.5).astype(int)):.4f} "
                  f"| val AUC {roc_auc_score(vy, vp):.4f}")
            model.train()  # evaluate() switched to eval mode, switch back

    # End of epoch: full validation. The test split is never touched here.
    p, y, *_ = evaluate("val")
    auc = roc_auc_score(y, p)
    val_acc = accuracy_score(y, (p >= 0.5).astype(int))
    train_acc = correct / seen
    print(f"== epoch {epoch}: train loss {total / len(loaders['train']):.4f} | "
          f"train acc {train_acc:.4f} | val acc {val_acc:.4f} | val AUC {auc:.4f}")
    history.append({"epoch": epoch, "train_loss": total / len(loaders['train']),
                    "train_acc": train_acc, "val_acc": val_acc, "val_auc": auc})
    pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)

    # Save best model + early stopping
    if auc > best_auc:
        best_auc, bad_epochs = auc, 0
        torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        print("   saved best model")
    else:
        bad_epochs += 1
        print(f"   no improvement ({bad_epochs}/{PATIENCE})")
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best val AUC {best_auc:.4f}")
            break


# ======================= 7. Test + explanations =======================
# First and only time the test split is used: after training has fully finished.
model.load_state_dict(torch.load(OUT_DIR / "best_model.pt"))
p, y, rows, mic_w, band_w = evaluate("test")
pred = (p >= 0.5).astype(int)

# Hz range of each AST frequency band (kaldi mel scale, 20 Hz to 8 kHz)
def mel(f): return 1127 * math.log(1 + f / 700)
def hz(m): return 700 * (math.exp(m / 1127) - 1)
lo, hi = mel(20), mel(8000)
edges = [hz(lo + i * (hi - lo) / 129) for i in range(130)]
band_ranges = [f"{edges[10 * b]:.0f}-{edges[min(10 * b + 17, 129)]:.0f} Hz"
               for b in range(model.f_dim)]

results = {
    "machines": MACHINES,
    "test_auc": round(roc_auc_score(y, p), 4),
    "test_accuracy": round(accuracy_score(y, pred), 4),
    "test_f1_abnormal": round(f1_score(y, pred), 4),
    "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y, pred).tolist(),
}
test_df = meta.loc[rows].assign(prob_abnormal=p, pred=pred)
for col in ["machine_type", "noise", "machine_key"]:
    results[f"auc_by_{col}"] = {
        k: round(roc_auc_score(g.label_id, g.prob_abnormal), 4)
        for k, g in test_df.groupby(col) if g.label_id.nunique() == 2
    }

# Explainability summaries for the dashboard and image generation
results["mic_weight_avg"] = {f"mic_{i}": round(float(w), 4) for i, w in enumerate(mic_w.mean(0))}
results["band_attention_abnormal"] = dict(zip(band_ranges, np.round(band_w[y == 1].mean(0), 4).tolist()))
results["band_attention_normal"] = dict(zip(band_ranges, np.round(band_w[y == 0].mean(0), 4).tolist()))

np.savez(OUT_DIR / "test_explanations.npz", rows=rows, prob=p, mic_w=mic_w,
         band_w=band_w, band_ranges=np.array(band_ranges))
test_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)
json.dump(results, open(OUT_DIR / "results.json", "w"), indent=2)
print(json.dumps(results, indent=2))