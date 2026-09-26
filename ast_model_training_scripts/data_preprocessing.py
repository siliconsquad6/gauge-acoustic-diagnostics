"""
8-mic preprocessing for MIMII. No model involved.
- Keeps ALL 8 microphone channels
- Each channel -> log-mel spectrogram (1024 frames x 128 mel bands)
- Saves (N, 8, 1024, 128) float16 per machine type (~40 GB total)
"""
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

DATA_ROOT = Path.home() / "Desktop" / "edge-ai" / "data"
MANIFEST = DATA_ROOT / "combined_data_shuffled" / "merge_manifest.csv"
OUT_DIR = DATA_ROOT / "features_8mic"

SR = 16000
N_MICS = 8
N_MELS = 128
N_FRAMES = 1024
MEAN, STD = -4.2677393, 4.5689974
WORKERS = 12


def channel_to_spec(y):
    wav = torch.from_numpy(y).unsqueeze(0)
    wav = wav - wav.mean()
    spec = torchaudio.compliance.kaldi.fbank(
        wav, sample_frequency=SR, num_mel_bins=N_MELS,
        frame_length=25, frame_shift=10, window_type="hanning",
        htk_compat=True, use_energy=False, dither=0.0,
    )
    if spec.shape[0] < N_FRAMES:
        spec = torch.nn.functional.pad(spec, (0, 0, 0, N_FRAMES - spec.shape[0]))
    else:
        spec = spec[:N_FRAMES]
    return (spec - MEAN) / (STD * 2)


def to_8mic_spectrogram(path):
    y, sr = sf.read(path, dtype="float32")   # shape (samples, 8)
    assert sr == SR and y.ndim == 2 and y.shape[1] == N_MICS, f"Unexpected format: {path}"
    specs = [channel_to_spec(np.ascontiguousarray(y[:, c])) for c in range(N_MICS)]
    return torch.stack(specs).numpy().astype(np.float16)   # (8, 1024, 128)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST)

    for machine, mdf in df.groupby("machine_type"):
        mdf = mdf.reset_index(drop=True)
        out_file = OUT_DIR / f"{machine}_features.npy"
        feats = np.lib.format.open_memmap(
            out_file, mode="w+", dtype=np.float16,
            shape=(len(mdf), N_MICS, N_FRAMES, N_MELS),
        )
        with Pool(WORKERS) as pool:
            for i, spec in enumerate(pool.imap(to_8mic_spectrogram, mdf["audio_path"], chunksize=8)):
                feats[i] = spec
                if i % 500 == 0:
                    print(f"{machine}: {i}/{len(mdf)}")
        feats.flush()

        mdf["label_id"] = (mdf["label"] == "abnormal").astype(int)
        mdf.to_csv(OUT_DIR / f"{machine}_meta.csv", index=False)
        print(f"{machine}: saved {len(mdf)} clips -> {out_file}")

    print("8-mic preprocessing done.")