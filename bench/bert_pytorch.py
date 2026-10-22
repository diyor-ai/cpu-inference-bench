"""Stage 3b: PyTorch FP32 / dynamic INT8 (quantize_dynamic on nn.Linear, x86), BERT SST-2. One process = one repeat."""
import sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent / "quant"))
from bert_common import bert_args, make_inputs, reference_logits, MODELS
from common import run_repeat, save, max_abs_diff
from bert_pt_int8 import build, fp32

args = bert_args()
torch.set_num_threads(args.threads)
x_np = make_inputs(args.model, args.seq, args.batch)
x = [torch.from_numpy(x_np[k]) for k in MODELS[args.model]["inputs"]]
extra = {"threads_set": args.threads, "torch_num_threads": torch.get_num_threads(), "inputs": MODELS[args.model]["inputs"]}
if args.precision == "fp32":
    model = fp32(args.model)
else:
    model = build(args.model)                      # quantize_dynamic: no calibration; done BEFORE cooldown/warmup/timing
    extra["quantized_engine"] = torch.backends.quantized.engine
    assert extra["quantized_engine"] == "x86", extra["quantized_engine"]
    mods = Counter(f"{type(m).__module__}.{type(m).__name__}" for m in model.modules())
    extra["dynamic_quantized_linear"] = mods.get("torch.ao.nn.quantized.dynamic.modules.linear.Linear", 0)
    extra["float_linear_left"] = mods.get("torch.nn.modules.linear.Linear", 0)
with torch.inference_mode():
    row = run_repeat(lambda: model(*x), args.batch, iters=args.iters)
    row["max_abs_diff_vs_pytorch_fp32"] = max_abs_diff(model(*x).numpy(), reference_logits(args.model, x_np))
print(f"pytorch {args.model} {args.precision} L{args.seq} b{args.batch} r{args.repeat}: p50 {row['p50_ms']} ms, "
      f"{row['telemetry']['freq_mean_mhz']} MHz, {row['telemetry']['temp_end_c']} C, throttled={row['telemetry']['throttled']}")
save(f"pytorch_{args.model}_{args.precision}_L{args.seq}_t{args.threads}_b{args.batch}_r{args.repeat}", {
    "framework": "pytorch", "version": torch.__version__, "precision": args.precision, "model": args.model, "seq": args.seq,
    "threads": args.threads, "repeat": args.repeat, "execution_check": extra, "result": row}, out=f"{args.out}/raw")
