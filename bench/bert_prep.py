"""Stage 3b prep (main venv): convert B .bin -> safetensors (weights_only load), tokenize val/calib, export FP32 ONNX, check vs PyTorch."""
import json, sys
import numpy as np, torch
from bert_common import *

# 1) B: pickle -> safetensors, loaded with weights_only=True (tensors only); no other code path touches the .bin
from safetensors.torch import save_file
src = MODELS["B"]["dir"] / "pytorch_model.bin"
dst = MODELS["B"]["dir"] / "model.safetensors"
if not dst.exists():
    sd = torch.load(src, map_location="cpu", weights_only=True)
    assert isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()), "unexpected pickle content"
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, str(dst), metadata={"format": "pt"})
    print("B: converted", len(sd), "tensors to safetensors")

# 2) tokenized data (val 872 + calib 300) for seq 32 and 128
vs, vy = read_split("validation"); cs = calib_sentences()
assert len(vs) == 872
info = {}
for n in MODELS:
    for L in (32, 128):
        for tag, sents in (("val", vs), ("calib", cs)):
            enc, ntr = encode(n, sents, L)
            extra = {"labels": vy} if tag == "val" else {}
            np.savez(BERT / f"{n}_{tag}_L{L}.npz", **enc, **extra)
            info[f"{n}_{tag}_L{L}"] = {"n": len(sents), "truncated": ntr}
print(json.dumps(info, indent=1))
(BERT / "data_info.json").write_text(json.dumps(info, indent=1))

# 3) FP32 ONNX export (dynamic batch/seq) + agreement with PyTorch
TEMP.mkdir(exist_ok=True)
import onnxruntime as ort
for n in MODELS:
    hf = load_model(n)
    attn = getattr(hf.config, "_attn_implementation", None)
    w = LogitsOnly(hf, MODELS[n]["inputs"])
    enc, _ = encode(n, vs[:4], 32)
    args = tuple(torch.from_numpy(enc[k]) for k in MODELS[n]["inputs"])
    dyn = {k: {0: "batch", 1: "seq"} for k in MODELS[n]["inputs"]} | {"logits": {0: "batch"}}
    torch.onnx.export(w, args, str(onnx_fp32_path(n)), input_names=MODELS[n]["inputs"], output_names=["logits"],
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    s = ort.InferenceSession(str(onnx_fp32_path(n)), providers=["CPUExecutionProvider"])
    enc8, _ = encode(n, vs[:8], 128)
    with torch.inference_mode():
        ref = w(*[torch.from_numpy(enc8[k]) for k in MODELS[n]["inputs"]]).numpy()
    got = s.run(None, enc8)[0]
    print(f"{n}: attn_impl={attn} onnx exported; max abs logit diff ORT vs torch (8 sents, seq128) = {np.abs(ref-got).max():.2e}")
