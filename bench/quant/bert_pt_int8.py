"""PyTorch dynamic INT8: torch.ao.quantization.quantize_dynamic on nn.Linear, qint8, engine x86, default qconfig (per-tensor weights)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings
warnings.simplefilter("ignore")
import torch
from bert_common import MODELS, load_model, LogitsOnly


def build(name):
    torch.backends.quantized.engine = "x86"
    w = LogitsOnly(load_model(name), MODELS[name]["inputs"])
    return torch.ao.quantization.quantize_dynamic(w, {torch.nn.Linear}, dtype=torch.qint8)


def fp32(name):
    return LogitsOnly(load_model(name), MODELS[name]["inputs"])
