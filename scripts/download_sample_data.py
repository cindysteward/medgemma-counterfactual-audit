"""Downloads a small stratified sample from NIH ChestX-ray14.

NIH hosts the dataset in 12 batch archives (~2GB each) at
https://nihcc.app.box.com/v/ChestXray-NIHCC. This script expects you've
already downloaded images_001.tar.gz and Data_Entry_2017.csv manually
into data/raw/ (the box.com links require a browser, not scriptable), then
extracts a stratified sample of ~30 images across a few pathology labels
plus No Finding, to keep case count manageable for a first run.
"""

import csv
import random
import shutil
import tarfile
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/sample_images")
N_PER_LABEL = 5
TARGET_LABELS = ["No Finding", "Effusion", "Cardiomegaly", "Consolidation", "Atelectasis", "Mass"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = {}
    with open(RAW_DIR / "Data_Entry_2017.csv") as f:
        for row in csv.DictReader(f):
            label = row["Finding Labels"].split("|")[0]
            entries.setdefault(label, []).append(row["Image Index"])

    random.seed(42)
    selected = []
    for label in TARGET_LABELS:
        candidates = entries.get(label, [])
        selected += random.sample(candidates, min(N_PER_LABEL, len(candidates)))

    with tarfile.open(RAW_DIR / "images_001.tar.gz") as tar:
        members = {m.name.split("/")[-1]: m for m in tar.getmembers()}
        for filename in selected:
            if filename in members:
                tar.extract(members[filename], path="data/_extract_tmp")
                src = Path("data/_extract_tmp/images") / filename
                shutil.move(str(src), OUT_DIR / filename)

    shutil.rmtree("data/_extract_tmp", ignore_errors=True)
    print(f"extracted {len(list(OUT_DIR.glob('*.png')))} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
