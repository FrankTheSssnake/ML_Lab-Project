import pandas as pd
from PIL import Image
import os
import glob

DATASET_ROOT = "archive/"        # change to your unzip path if needed
OUTPUT_DIR   = "dataset"  # Red/, Yellow/, Green/ will be created here

# map annotation tags to folder names
LABEL_MAP = {
    "go"      : "Green",
    "stop"    : "Red",
    "warning" : "Yellow",
    "goLeft"  : "Green",
    "stopLeft": "Red",
}

# find every frameAnnotationsBOX.csv in the entire tree
csv_files = glob.glob(f"{DATASET_ROOT}/**/frameAnnotationsBOX.csv", recursive=True)
print(f"Found {len(csv_files)} annotation files")

count = {"Green": 0, "Red": 0, "Yellow": 0, "skipped": 0}

for csv_path in csv_files:
    clip_dir = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path, sep=";")

    for _, row in df.iterrows():
        tag = row["Annotation tag"].strip().lower()

        if tag not in LABEL_MAP:
            count["skipped"] += 1
            continue

        label    = LABEL_MAP[tag]
        img_path = os.path.join(clip_dir, "frames", os.path.basename(row["Filename"]))

        if not os.path.exists(img_path):
            count["skipped"] += 1
            continue

        x1, y1 = int(row["Upper left corner X"]),  int(row["Upper left corner Y"])
        x2, y2 = int(row["Lower right corner X"]), int(row["Lower right corner Y"])

        # skip boxes that are too small to be useful
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            count["skipped"] += 1
            continue

        img  = Image.open(img_path)
        crop = img.crop((x1, y1, x2, y2))

        out_folder = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_folder, exist_ok=True)

        # unique filename: clipname_framename
        clip_name  = os.path.basename(clip_dir)
        frame_name = os.path.basename(row["Filename"])
        crop.save(os.path.join(out_folder, f"{clip_name}_{frame_name}"))
        count[label] += 1

print("\nDone!")
print(f"  Green  : {count['Green']}")
print(f"  Red    : {count['Red']}")
print(f"  Yellow : {count['Yellow']}")
print(f"  Skipped: {count['skipped']}")

