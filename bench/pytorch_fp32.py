import torch
from common import common_args, run_repeat, save, get_model, get_input

args = common_args()
torch.set_num_threads(args.threads)
# Random (seeded) weights: latency does not depend on weight values.
model = get_model()
x = torch.from_numpy(get_input(args.batch))

with torch.inference_mode():
    row = run_repeat(lambda: model(x), args.batch, iters=args.iters)
print(f"pytorch t{args.threads} b{args.batch} r{args.repeat}: p50 {row['p50_ms']} ms, "
      f"{row['telemetry']['freq_mean_mhz']} MHz, {row['telemetry']['temp_end_c']} C, "
      f"throttled={row['telemetry']['throttled']}")

save(f"pytorch_fp32_t{args.threads}_b{args.batch}_r{args.repeat}", {
    "framework": "pytorch", "version": torch.__version__, "precision": "fp32",
    "model": "resnet18", "threads": args.threads, "repeat": args.repeat, "result": row,
}, out=f"{args.out}/raw")
