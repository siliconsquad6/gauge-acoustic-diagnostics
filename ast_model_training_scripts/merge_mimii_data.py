from pathlib import Path
import shutil
import csv

DATA_ROOT = Path.home() / "Desktop" / "edge-ai" / "data"
OUTPUT_ROOT = DATA_ROOT / "combined_data_shuffled"

MACHINE_TYPES = ["fan", "pump"]
LABELS = ["normal", "abnormal"]
NOISE_LEVELS = ["0_dB", "6_dB"]

if OUTPUT_ROOT.exists():
    raise FileExistsError(f"{OUTPUT_ROOT} already exists. Remove it first.")

manifest_rows = []

for machine in MACHINE_TYPES:
    machine_ids = sorted({
        folder.name
        for noise in NOISE_LEVELS
        for folder in (DATA_ROOT / f"{noise}_{machine}" / machine).glob("id_*")
        if folder.is_dir()
    })

    for machine_id in machine_ids:
        for label in LABELS:
            destination_folder = OUTPUT_ROOT / machine / machine_id / label
            destination_folder.mkdir(parents=True, exist_ok=True)
            count = 0

            for noise in NOISE_LEVELS:
                source_folder = DATA_ROOT / f"{noise}_{machine}" / machine / machine_id / label
                if not source_folder.exists():
                    continue

                for source_file in sorted(source_folder.glob("*.wav")):
                    # e.g. 0_dB_00000123.wav or 6_dB_00000123.wav
                    new_name = f"{noise}_{source_file.name}"
                    destination_file = destination_folder / new_name
                    shutil.copy2(source_file, destination_file)
                    count += 1

                    manifest_rows.append({
                        "audio_path": str(destination_file),
                        "machine_type": machine,
                        "machine_id": machine_id,
                        "label": label,
                        "noise": noise,
                        "original_file": source_file.name,
                    })

            print(f"{machine} | {machine_id} | {label}: {count} files")

manifest_file = OUTPUT_ROOT / "merge_manifest.csv"
with open(manifest_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"\nDone. {len(manifest_rows)} files in {OUTPUT_ROOT}")
print(f"Manifest: {manifest_file}")