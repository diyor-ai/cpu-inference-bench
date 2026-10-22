#!/usr/bin/env bash
# Usage: ./run_all.sh <phase1|phase2|pilot|stage2|stage2_smoke>
#   phase1: FP32 main comparison, random weights, 6 threads on the 6 physical P-cores (SMT siblings offline)
#   phase2: FP32 SMT experiment, random weights, 12 threads on cpus 0-11 (siblings online)
#   pilot : short check of the phase1 setup (pytorch+onnx, batch 1 and 32)
#   stage2: pretrained weights, FP32 vs INT8 for all 3 frameworks, same conditions as phase1
#   stage2_smoke: 1 repeat, batch 1, all 6 stage2 configs (code-path check; not a result)
#   bert: BERT SST-2 FP32 vs dynamic/NNCF INT8, 3 frameworks; grid A seq32/128 x batch1/8, B seq128 x batch1/8; 5 repeats x 100 iters
#   bert_smoke: 1 repeat, 10 iters, all configs of the grid (code-path check; not a result; output results/bert_smoke)
# One process = one repeat of one config. Rounds are interleaved:
# for each batch: for each round: for each framework: [fp32, int8] (so drift hits everyone equally).
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:?usage: $0 phase1|phase2|pilot|stage2|stage2_smoke|bert|bert_smoke}"
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=""
CPU=/sys/devices/system/cpu
S2=0; PRECS="fp32"; BERT=0; ITERS=100
case "$MODE" in
  phase1) CPUS="0,2,4,6,8,10"; T=6;  FWS="pytorch onnx openvino"; BATCHES="1 8 32"; ROUNDS=5; SMT=off ;;
  pilot)  CPUS="0,2,4,6,8,10"; T=6;  FWS="pytorch onnx";          BATCHES="1 32";   ROUNDS=5; SMT=off ;;
  phase2) CPUS="0-11";         T=12; FWS="pytorch onnx openvino"; BATCHES="1 8 32"; ROUNDS=5; SMT=on ;;
  stage2) CPUS="0,2,4,6,8,10"; T=6;  FWS="pytorch onnx openvino"; BATCHES="1 8 32"; ROUNDS=5; SMT=off; S2=1; PRECS="fp32 int8" ;;
  stage2_smoke) CPUS="0,2,4,6,8,10"; T=6; FWS="pytorch onnx openvino"; BATCHES="1"; ROUNDS=1; SMT=off; S2=1; PRECS="fp32 int8" ;;
  bert)       CPUS="0,2,4,6,8,10"; T=6; FWS="pytorch onnx openvino"; ROUNDS=5; SMT=off; S2=0; PRECS="fp32 int8"; BERT=1 ;;
  bert_smoke) CPUS="0,2,4,6,8,10"; T=6; FWS="pytorch onnx openvino"; ROUNDS=1; SMT=off; S2=0; PRECS="fp32 int8"; BERT=1; ITERS=10 ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
OUT="results/$MODE"

# ---- preflight: refuse to run in the wrong machine state ----
sib_online=$(cat $CPU/cpu1/online 2>/dev/null || echo 0)
if [ "$SMT" = off ] && [ "$sib_online" = 1 ]; then echo "ABORT: SMT siblings are online; offline them first."; exit 1; fi
if [ "$SMT" = on ] && [ "$sib_online" != 1 ]; then echo "ABORT: bring SMT siblings online for phase2."; exit 1; fi
for c in $(taskset -c "$CPUS" $PY -c 'import os;print(*sorted(os.sched_getaffinity(0)))'); do
  f=$CPU/cpu$c/cpufreq
  mn=$(cat $f/scaling_min_freq); mx=$(cat $f/scaling_max_freq); epp=$(cat $f/energy_performance_preference)
  if [ "$mx" = "$(cat $f/cpuinfo_max_freq)" ] && [ "${ALLOW_UNCAPPED:-0}" != 1 ]; then
    echo "ABORT: cpu$c is not frequency-capped (scaling_max_freq=$mx). Apply the cap or set ALLOW_UNCAPPED=1."; exit 1; fi
  if [ "$mn" != "$mx" ]; then echo "ABORT: cpu$c min ($mn) != max ($mx): frequency is not fixed."; exit 1; fi
  if [ "$epp" != performance ]; then echo "ABORT: cpu$c EPP is '$epp', expected 'performance'."; exit 1; fi
  echo "  cpu$c epp=$epp min=$mn max=$mx"
done
if [ "$S2" = 1 ]; then
  for f in data/resnet18-f37072fd.pth data/calib.npz temp/resnet18_pretrained_int8_ort_minmax.onnx temp/resnet18_pretrained_int8_ov.xml; do
    [ -f "$f" ] || { echo "ABORT: missing $f (run the Stage 2 data/quantization steps)"; exit 1; }; done
fi
if [ "$BERT" = 1 ]; then
  for f in temp/bert_A_fp32.onnx temp/bert_B_fp32.onnx temp/bert_A_int8_ort_dyn.onnx temp/bert_B_int8_ort_dyn.onnx temp/bert_A_L32_int8_ov.xml temp/bert_A_L128_int8_ov.xml temp/bert_B_L128_int8_ov.xml data/bert/A_val_L128.npz; do
    [ -f "$f" ] || { echo "ABORT: missing $f (run the Stage 3b prep/quantization steps)"; exit 1; }; done
fi
echo "mode=$MODE threads=$T cpus=$CPUS cap_khz(cpu0)=$(cat $CPU/cpu0/cpufreq/scaling_max_freq) precisions=[$PRECS]"

mkdir -p "$OUT/raw"
if [ "$BERT" = 1 ]; then
  # grid: model seq batch (A: seq 32/128 x batch 1/8; B: seq 128 x batch 1/8); interleaved: config > round > framework > fp32,int8
  for cfg in "A 32 1" "A 32 8" "A 128 1" "A 128 8" "B 128 1" "B 128 8"; do
    set -- $cfg
    for r in $(seq 0 $((ROUNDS-1))); do
      for fw in $FWS; do for prec in $PRECS; do
        taskset -c "$CPUS" $PY bench/bert_${fw}.py --model "$1" --seq "$2" --threads "$T" --batch "$3" --repeat "$r" --iters "$ITERS" --precision "$prec" --out "$OUT"
      done; done
    done
  done
  $PY bench/bert_compare.py "$MODE" | tee "$OUT/summary.txt"
  case "$MODE" in bert_smoke) echo ">>> BERT SMOKE DONE (not a result). Next: ./run_all.sh bert" ;; bert) echo ">>> BERT TIMING DONE." ;; esac
  exit 0
fi
for b in $BATCHES; do
  for r in $(seq 0 $((ROUNDS-1))); do
    for fw in $FWS; do
      for prec in $PRECS; do
        if [ "$S2" = 1 ]; then script=bench/${fw}_stage2.py; else script=bench/${fw}_fp32.py; fi
        taskset -c "$CPUS" $PY $script --threads "$T" --batch "$b" --repeat "$r" --precision "$prec" --out "$OUT"
      done
    done
  done
done

$PY bench/compare.py "$MODE" | tee "$OUT/summary.txt"
case "$MODE" in
  phase1) echo; echo ">>> PHASE 1 DONE. For phase 2: bring siblings online, cap them, then run ./run_all.sh phase2" ;;
  pilot)  echo; echo ">>> PILOT DONE. Review spread before running ./run_all.sh phase1" ;;
  stage2_smoke) echo; echo ">>> SMOKE DONE (1 repeat per config: not a result). Next: ./run_all.sh stage2" ;;
  stage2) echo; echo ">>> STAGE 2 TIMING DONE." ;;
esac
