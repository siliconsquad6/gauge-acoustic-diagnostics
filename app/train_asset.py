"""Acoustic fingerprint: learn WHICH machine unit (e.g. fan id_02) a recording comes from.
MIMII ids are different product models, so each sounds different even when healthy.

Run from the gauge folder (CPU, ~10 min):  python app/train_asset.py
Saves models/asset_id.joblib (classes like "fan_id_02") and prints per-unit accuracy on held-out clips.
"""
import os, random, re
from pathlib import Path
import numpy as np, soundfile as sf, librosa, joblib
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

G = Path(__file__).resolve().parent.parent
OUT = G / "models" / "asset_id.joblib"
PER_UNIT = 700


def mfeat(y, sr=16000):
    """Mean + std of a 64-band log-mel spectrogram, plus spectral shape stats."""
    if y.ndim == 2:
        y = y.mean(1)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    L = librosa.power_to_db(S)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    return np.r_[L.mean(1), L.std(1), cent.mean(), cent.std(), flat.mean(), flat.std()]


def job(p):
    try:
        y, sr = sf.read(p, dtype="float32")
        return mfeat(y, sr)
    except Exception:
        return None


if __name__ == "__main__":
    random.seed(0)
    roots = list((G.parent.parent / "data").glob("combined_data_sh*")) + list((Path.home() / "Downloads").glob("*valve*"))
    print("Roots:", roots)
    units = {}
    for r in roots:
        for p in r.rglob("*.wav"):
            mt = re.search(r"/(fan|pump|valve)/(id_\d\d)/", str(p), re.I)
            if mt:
                units.setdefault(f"{mt.group(1).lower()}_{mt.group(2).lower()}", []).append(p)
    items = []
    for u, ps in sorted(units.items()):
        random.shuffle(ps)
        items += [(p, u) for p in ps[:PER_UNIT]]
        print(f"{u}: {len(ps)} clips (using {min(len(ps), PER_UNIT)})")
    print(f"extracting {len(items)} clips on {os.cpu_count()} cores...")
    feats = Parallel(n_jobs=-1, verbose=5)(delayed(job)(p) for p, _ in items)
    X = np.array([f for f in feats if f is not None]); Y = [u for (p, u), f in zip(items, feats) if f is not None]
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=2.0))
    clf.fit(Xtr, Ytr)
    print(classification_report(Yte, clf.predict(Xte), digits=3))
    clf.fit(X, Y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT)
    print("saved", OUT, "classes:", list(clf.classes_))
