#!/usr/bin/env bash
# Pinned thread-scaling benchmark for Theft Guard.
# Runs bench_single on different CPU sets (via taskset), rests between runs,
# samples the clock of the used CPUs, and prints one summary table.
#
# Usage:  bash pin_bench.sh [clip]        (default clip: data/clips/tool_attack_1.mp4)
#         REST=30 bash pin_bench.sh       (shorter rest between runs)

set -u
cd ~/theft-guard || { echo "~/theft-guard not found"; exit 1; }
CLIP=${1:-data/clips/tool_attack_1.mp4}
REST=${REST:-60}
OUT=bench/threads_pinned.txt
[ -x ./build/bench_single ] || { echo "./build/bench_single not found. Build first."; exit 1; }
[ -f "$CLIP" ] || { echo "Clip $CLIP not found."; exit 1; }

# CPU topology: one "first" logical CPU per physical core, and cpu0's Hyper-Threading sibling.
TOPO=$(lscpu -p=CPU,CORE | grep -v '^#')
PHYS=($(echo "$TOPO" | awk -F, '!seen[$2]++ {print $1}'))
ALL=$(echo "$TOPO" | awk -F, '{print $1}' | paste -sd,)
CORE0=$(echo "$TOPO" | awk -F, '$1==0 {print $2}')
SIB0=$(echo "$TOPO" | awk -F, -v c="$CORE0" '$2==c && $1!=0 {print $1; exit}')

CONFIGS=("1 thread, 1 core|${PHYS[0]}")
[ ${#PHYS[@]} -ge 2 ] && CONFIGS+=("2 threads, 2 cores|${PHYS[0]},${PHYS[1]}")
[ -n "$SIB0" ] && CONFIGS+=("2 threads, 1 core (HT)|0,$SIB0")
[ ${#PHYS[@]} -ge 4 ] && CONFIGS+=("4 threads, 4 cores|$(IFS=,; echo "${PHYS[*]:0:4}")")
CONFIGS+=("all threads|$ALL")

sample_mhz() {   # print the average MHz of the given CPU list once per second
  while :; do
    awk -v set="$1" 'BEGIN { n = split(set, a, ","); for (i = 1; i <= n; i++) w[a[i]] = 1 }
      /^processor/ { p = $3 }
      /^cpu MHz/   { if (p in w) { s += $4; k++ } }
      END          { if (k) printf "%.0f\n", s / k }' /proc/cpuinfo
    sleep 1
  done
}

TMP=$(mktemp -d)
trap 'kill $SAMPLER 2>/dev/null; rm -rf "$TMP"' EXIT
SAMPLER=""

{
  echo "# Pinned thread scaling, $(date '+%Y-%m-%d %H:%M')"
  echo "# CPU: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "# Clip: $CLIP, rest ${REST} s between runs"
  echo "# Topology (CPU,CORE): $(echo "$TOPO" | paste -sd' ')"
} > "$TMP/header"
cat "$TMP/header"
echo

i=0
for cfg in "${CONFIGS[@]}"; do
  name=${cfg%%|*}; set=${cfg#*|}; i=$((i + 1))
  echo "[$i/${#CONFIGS[@]}] $name on CPUs $set: resting ${REST} s to cool down..."
  sleep "$REST"
  echo "      running..."
  sample_mhz "$set" > "$TMP/mhz_$i" & SAMPLER=$!
  taskset -c "$set" ./build/bench_single "$CLIP" > "$TMP/out_$i" 2>&1
  kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; SAMPLER=""
  thr=$(sed -n 's/.*OpenCV threads: *\([0-9]*\).*/\1/p' "$TMP/out_$i")
  inf=$(awk '/inference/ {print $3; exit}' "$TMP/out_$i")
  tot=$(awk '/total/ {print $3; exit}' "$TMP/out_$i")
  mhz=$(awk '{ s += $1; n++ } END { if (n) printf "%.0f", s / n; else print "?" }' "$TMP/mhz_$i")
  echo "$name|$set|${thr:-?}|${inf:-?}|${tot:-?}|$mhz" >> "$TMP/rows"
  if [ -z "$inf" ]; then echo "      bench_single failed, its output:"; cat "$TMP/out_$i"; fi
done

{
  cat "$TMP/header"
  echo
  awk -F'|' 'BEGIN { printf "%-26s %-9s %7s %12s %8s %9s\n", "Config", "CPUs", "Threads", "Infer (ms)", "Speedup", "Avg MHz" }
    { if (NR == 1) base = $4
      sp = ($4 + 0 > 0) ? sprintf("%.2fx", base / $4) : "?"
      printf "%-26s %-9s %7s %12s %8s %9s\n", $1, $2, $3, $4, sp, $6 }' "$TMP/rows"
} | tee "$OUT"
echo
echo "Saved to $OUT"
