"""PyTorch FX graph-mode static PTQ, 'x86' backend (torch.ao; deprecated API). Calibration: data/calib.npz.
build() is called by accuracy.py and the timing script (the quantized GraphModule does not unpickle, so it is rebuilt;
calibration is deterministic)."""
import json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from common import get_model, get_input, load_split, normalize, TEMP


def build():
    torch.backends.quantized.engine = "x86"
    x_u8, _ = load_split("calib")
    qmap = get_default_qconfig_mapping("x86")
    p = prepare_fx(get_model(pretrained=True), qmap, (torch.from_numpy(get_input(1)),))
    with torch.inference_mode():
        for i in range(0, len(x_u8), 25):
            p(torch.from_numpy(normalize(x_u8[i:i + 25])))
    return convert_fx(p), qmap, len(x_u8)


if __name__ == "__main__":
    q, qmap, n = build()
    g = qmap.global_qconfig
    (TEMP / "resnet18_pretrained_int8_pt.json").write_text(json.dumps({
        "torch": torch.__version__, "api": "torch.ao.quantization.quantize_fx (deprecated)", "backend": "x86",
        "qconfig_activation": str(g.activation) if g else None, "qconfig_weight": str(g.weight) if g else None,
        "calib_images": n}, indent=1, default=str))
    print("quantized OK; config saved")
