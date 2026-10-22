"""data/manifest.json + data/{eval,calib}/*.jpg -> data/{eval,calib}.npz (uint8 crops + labels)."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as T

DATA = Path(__file__).resolve().parents[2] / "data"
tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True), T.CenterCrop(224)])
man = json.loads((DATA / "manifest.json").read_text())
for split in ("eval", "calib"):
    xs, ys = [], []
    for m in man[split]:
        img = Image.open(DATA / split / m["file"]).convert("RGB")
        xs.append(np.asarray(tf(img), dtype=np.uint8).transpose(2, 0, 1))
        ys.append(m["label"])
    np.savez(DATA / f"{split}.npz", x=np.stack(xs), y=np.array(ys, dtype=np.int64))
    print(split, len(ys), "images; distinct classes:", len(set(ys)))
