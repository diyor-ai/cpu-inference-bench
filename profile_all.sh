#!/usr/bin/env bash
# Stage 3a: profiling runs (NOT timing). Same pinning/threads as the benchmark: 6 threads on cpus 0,2,4,6,8,10.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python; CPUS="0,2,4,6,8,10"; CPU=/sys/devices/system/cpu
export CUDA_VISIBLE_DEVICES=""
[ "$(cat $CPU/cpu1/online)" = 0 ] || { echo "ABORT: siblings online"; exit 1; }
for c in ${CPUS//,/ }; do f=$CPU/cpu$c/cpufreq
  [ "$(cat $f/scaling_min_freq)" = 2600000 ] && [ "$(cat $f/scaling_max_freq)" = 2600000 ] && [ "$(cat $f/energy_performance_preference)" = performance ] || { echo "ABORT: cpu$c cap/EPP wrong"; exit 1; }
done
OUT=results/profile; mkdir -p $OUT/raw
for b in 1 32; do
  it=50; [ $b = 32 ] && it=20
  for fw in pytorch onnx openvino; do for p in fp32 int8; do
    taskset -c $CPUS $PY bench/profile_$fw.py --threads 6 --batch $b --repeat 0 --iters $it --precision $p --out $OUT 2>&1 | grep -v USDT
  done; done
done
