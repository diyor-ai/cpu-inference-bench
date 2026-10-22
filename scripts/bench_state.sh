#!/usr/bin/env bash
# Benchmark machine state for this repo (Linux, intel_pstate, systemd). Uses sudo for the privileged writes.
#   scripts/bench_state.sh on      stop tuned/thermald/localsearch, EPP performance, P-cores fixed at 2.6 GHz (min=max), SMT siblings offline
#   scripts/bench_state.sh off     restore: siblings online, clocks uncapped, powersave + balance_performance, services started, tuned 'balanced'
#   scripts/bench_state.sh status  read-only verification (printed after on/off as well)
# Adjust P_CORES / SIBLINGS / CAP_KHZ for another CPU: P_CORES are the first thread of each P-core, SIBLINGS their SMT partners (lscpu -e).
set -uo pipefail
P_CORES="0 2 4 6 8 10"; SIBLINGS="1 3 5 7 9 11"; CAP_KHZ=2600000
CPU=/sys/devices/system/cpu
w() { echo "$2" | sudo tee "$1" >/dev/null || echo "WARN: could not write $2 to $1"; }

status() {
  echo "--- state"
  echo "online CPUs: $(cat $CPU/online)"
  for c in $P_CORES; do f=$CPU/cpu$c/cpufreq
    echo "cpu$c min=$(cat $f/scaling_min_freq) max=$(cat $f/scaling_max_freq) gov=$(cat $f/scaling_governor) epp=$(cat $f/energy_performance_preference) cur=$(cat $f/scaling_cur_freq)"; done
  echo "siblings online: $(for c in $SIBLINGS; do cat $CPU/cpu$c/online 2>/dev/null || echo '?'; done | tr '\n' ' ')"
  echo "tuned=$(systemctl is-active tuned) thermald=$(systemctl is-active thermald) localsearch-3=$(systemctl --user is-active localsearch-3.service)"
  echo "platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null) AC online=$(cat /sys/class/power_supply/A*/online 2>/dev/null | tr '\n' ' ')"
  for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = x86_pkg_temp ] && echo "package temp: $(( $(cat $z/temp) / 1000 )) C"; done
  echo "heavy processes (>30% CPU):"; ps -eo pcpu,comm --sort=-pcpu | awk 'NR>1 && $1>30 {print "  " $0}'
}

on() {
  sudo systemctl stop tuned thermald
  systemctl --user stop localsearch-3.service
  for c in $(seq 0 11); do [ -w $CPU/cpu$c/cpufreq/energy_performance_preference ] && w $CPU/cpu$c/cpufreq/energy_performance_preference performance; done
  for c in $P_CORES; do w $CPU/cpu$c/cpufreq/scaling_min_freq $CAP_KHZ; w $CPU/cpu$c/cpufreq/scaling_max_freq $CAP_KHZ
    w $CPU/cpu$c/cpufreq/energy_performance_preference performance; done
  for c in $SIBLINGS; do w $CPU/cpu$c/online 0; done
  sleep 20   # let the clocks settle
}

off() {
  for c in $SIBLINGS; do w $CPU/cpu$c/online 1; done
  for c in $(seq 0 11); do f=$CPU/cpu$c/cpufreq
    w $f/scaling_max_freq "$(cat $f/cpuinfo_max_freq)"; w $f/scaling_min_freq "$(cat $f/cpuinfo_min_freq)"
    w $f/scaling_governor powersave; w $f/energy_performance_preference balance_performance; done
  w /sys/firmware/acpi/platform_profile balanced
  sudo systemctl start tuned thermald
  sudo tuned-adm profile balanced
  systemctl --user start localsearch-3.service
  sleep 5
}

case "${1:-}" in
  on) on; status ;;
  off) off; status ;;
  status) status ;;
  *) echo "usage: $0 on|off|status"; exit 2 ;;
esac
