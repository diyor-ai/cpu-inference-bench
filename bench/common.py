import json, os, platform, statistics, time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def benchmark(fn, warmup=20, iters=200):
    """Run fn() warmup times (discarded), then iters times; return ms per call."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return times


def summarize(times_ms, batch):
    s = sorted(times_ms)
    p50 = statistics.median(s)
    return {
        "batch": batch,
        "iters": len(s),
        "p50_ms": round(p50, 3),
        "p95_ms": round(s[int(0.95 * (len(s) - 1))], 3),
        "mean_ms": round(statistics.fmean(s), 3),
        "std_ms": round(statistics.pstdev(s), 3),
        "min_ms": round(s[0], 3),
        "throughput_img_s": round(batch / (p50 / 1000), 2),
    }


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def env_info():
    cpu = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    base = "/sys/devices/system/cpu/cpu0/cpufreq/"
    cpus = sorted(os.sched_getaffinity(0))
    rapl = "/sys/class/powercap/intel-rapl:0/"
    return {
        "cpu": cpu,
        "python": platform.python_version(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "governor": _read(base + "scaling_governor"),
        "epp": _read(base + "energy_performance_preference"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "online_cpus": _read("/sys/devices/system/cpu/online"),
        "no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "epp_per_cpu": {c: _read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/energy_performance_preference") for c in cpus},
        "scaling_min_khz": {c: _read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_min_freq") for c in cpus},
        "scaling_max_khz": {c: _read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq") for c in cpus},
        "rapl_pl1_uw": _read(rapl + "constraint_0_power_limit_uw"),
        "rapl_pl2_uw": _read(rapl + "constraint_1_power_limit_uw"),
    }


def save(name, record, out=None):
    out = Path(out) if out else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    record["env"] = env_info()
    path = out / f"{name}.json"
    path.write_text(json.dumps(record, indent=2))
    print("saved", path.relative_to(RESULTS.parent) if path.is_relative_to(RESULTS.parent) else path)


SEED = 0
TEMP = RESULTS.parent / "temp"


DATA = RESULTS.parent / "data"
WEIGHTS = DATA / "resnet18-f37072fd.pth"   # torchvision IMAGENET1K_V1


def get_model(pretrained=False):
    """ResNet-18, identical for every framework.
    pretrained=False: seeded random weights (Phase 1 latency runs).
    pretrained=True: torchvision IMAGENET1K_V1 weights (Stage 2 timing + accuracy)."""
    import torch, torchvision
    torch.manual_seed(SEED)
    model = torchvision.models.resnet18(weights=None)
    if pretrained:
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model.eval()


def get_input(batch):
    """Seeded random input, NCHW float32; same values for every framework."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    return rng.standard_normal((batch, 3, 224, 224), dtype=np.float32)


def reference_output(model, x_np):
    """FP32 PyTorch output for numpy input x_np (single-process reference)."""
    import torch
    with torch.inference_mode():
        return model(torch.from_numpy(x_np)).numpy()


def max_abs_diff(a, b):
    import numpy as np
    return float(np.abs(a - b).max())


def onnx_path(pretrained=False):
    """Export the shared model to temp/ (git-ignored) once; dynamic batch axis."""
    import torch
    TEMP.mkdir(exist_ok=True)
    p = TEMP / ("resnet18_pretrained_fp32.onnx" if pretrained else "resnet18_fp32.onnx")
    if not p.exists():
        model = get_model(pretrained)
        torch.onnx.export(model, torch.from_numpy(get_input(1)), str(p),
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                          opset_version=17, dynamo=False)
    return p


# ---- stability harness: telemetry, cooldown, one measured repeat -------------

import argparse, threading


def common_args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--repeat", type=int, required=True, help="repeat index (one process = one repeat)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--out", default=str(RESULTS / "phase1"))
    return ap.parse_args()


def pkg_temp_c():
    base = Path("/sys/class/thermal")
    for z in sorted(base.glob("thermal_zone*")):
        if _read(z / "type") == "x86_pkg_temp":
            return int(_read(z / "temp")) / 1000
    return None


def throttle_counts(cpus):
    """Cumulative kernel thermal-throttle counters (package, summed core) for cpus."""
    d = "/sys/devices/system/cpu/cpu{}/thermal_throttle/{}_throttle_count"
    pkg = int(_read(d.format(cpus[0], "package")) or 0)
    core = sum(int(_read(d.format(c, "core")) or 0) for c in cpus)
    return pkg, core


def cooldown(target_c=60.0, max_wait_s=180.0):
    t0 = time.time()
    while True:
        t = pkg_temp_c()
        if t is None or t < target_c:
            return {"start_temp_c": t, "waited_s": round(time.time() - t0, 1), "timed_out": False}
        if time.time() - t0 > max_wait_s:
            return {"start_temp_c": t, "waited_s": round(time.time() - t0, 1), "timed_out": True}
        time.sleep(2)


class FreqSampler(threading.Thread):
    """Samples scaling_cur_freq of the pinned CPUs every 250 ms."""
    def __init__(self, cpus):
        super().__init__(daemon=True)
        self.cpus, self.vals, self._stop_ev = cpus, [], threading.Event()

    def run(self):
        while not self._stop_ev.is_set():
            for c in self.cpus:
                v = _read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")
                if v:
                    self.vals.append(int(v) / 1000)
            self._stop_ev.wait(0.25)

    def stop(self):
        self._stop_ev.set(); self.join()


def run_repeat(fn, batch, iters=100, warmup=10):
    """One independent repeat: cooldown, warmup, timed iters, with telemetry."""
    cpus = sorted(os.sched_getaffinity(0))
    cd = cooldown()
    benchmark(fn, warmup=warmup, iters=0)
    th0, t0 = throttle_counts(cpus), pkg_temp_c()
    fs = FreqSampler(cpus); fs.start()
    times = benchmark(fn, warmup=0, iters=iters)
    fs.stop()
    th1, t1 = throttle_counts(cpus), pkg_temp_c()
    row = summarize(times, batch)
    row["telemetry"] = {
        "cooldown": cd, "temp_start_c": t0, "temp_end_c": t1,
        "freq_mean_mhz": round(statistics.fmean(fs.vals), 0) if fs.vals else None,
        "freq_min_mhz": round(min(fs.vals), 0) if fs.vals else None,
        "pkg_throttle_delta": th1[0] - th0[0], "core_throttle_delta": th1[1] - th0[1],
        "throttled": (th1[0] - th0[0]) > 0 or (th1[1] - th0[1]) > 0,
    }
    return row


# ---- Stage 2: shared eval/calibration data (preprocessing defined once) ------
# Preprocessing (identical to torchvision's ResNet18_Weights.IMAGENET1K_V1 transform):
#   RGB -> Resize(256, bilinear, antialias) -> CenterCrop(224) -> /255 -> Normalize(mean, std) -> NCHW float32
# Stored as uint8 224x224 crops in data/{split}.npz; normalization is applied at load time for every framework.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(split):
    """Returns (uint8 NCHW crops, int64 labels) for 'eval' or 'calib'."""
    import numpy as np
    d = np.load(DATA / f"{split}.npz")
    return d["x"], d["y"]


def normalize(x_u8):
    import numpy as np
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return ((x_u8.astype(np.float32) / 255.0) - mean) / std


def quant_info(name):
    """Versions/settings recorded at quantization time (temp/<name>.json), or None."""
    p = TEMP / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None
