"""Stage 2: PyTorch FP32 / INT8 (FX x86), pretrained IMAGENET1K_V1 weights. One process = one repeat."""
import sys
from collections import Counter
from pathlib import Path
import torch
from common import common_args, run_repeat, save, get_model, get_input, quant_info, reference_output, max_abs_diff

args = common_args()
torch.set_num_threads(args.threads)
x_np = get_input(args.batch)
x = torch.from_numpy(x_np)
extra = {"threads_set": args.threads, "torch_num_threads": torch.get_num_threads()}
if args.precision == "fp32":
    model = get_model(pretrained=True)
else:
    sys.path.insert(0, str(Path(__file__).parent / "quant"))
    from pt_int8 import build            # calibration (300 imgs) happens here, BEFORE cooldown/warmup/timing
    model, _, _ = build()
    extra["quantized_engine"] = torch.backends.quantized.engine
    assert extra["quantized_engine"] == "x86", extra["quantized_engine"]
    mods = Counter(f"{type(m).__module__}.{type(m).__name__}" for m in model.modules())
    extra["quantized_module_counts"] = {k: v for k, v in mods.items() if "quantized" in k}
    extra["n_quantized_conv_linear"] = sum(v for k, v in mods.items() if "quantized" in k and ("Conv" in k or "Linear" in k) and "Packed" not in k)
    extra["quantization"] = quant_info("resnet18_pretrained_int8_pt")

with torch.inference_mode():
    row = run_repeat(lambda: model(x), args.batch, iters=args.iters)
    row["max_abs_diff_vs_pytorch_fp32"] = max_abs_diff(model(x).numpy(), reference_output(get_model(pretrained=True), x_np))
print(f"pytorch {args.precision} t{args.threads} b{args.batch} r{args.repeat}: p50 {row['p50_ms']} ms, "
      f"{row['telemetry']['freq_mean_mhz']} MHz, {row['telemetry']['temp_end_c']} C, throttled={row['telemetry']['throttled']}")
save(f"pytorch_{args.precision}_t{args.threads}_b{args.batch}_r{args.repeat}", {
    "framework": "pytorch", "version": torch.__version__, "precision": args.precision, "weights": "IMAGENET1K_V1",
    "model": "resnet18", "threads": args.threads, "repeat": args.repeat, "execution_check": extra, "result": row,
}, out=f"{args.out}/raw")
