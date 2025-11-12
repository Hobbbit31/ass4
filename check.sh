#!/usr/bin/env bash
# autotune.sh
# Usage: ./autotune.sh ./gemm_opt_kernels N threads
# Example: ./autotune.sh ./gemm_opt_kernels 1024 8

BIN="$1"
N="$2"
T="$3"
if [[ -z "$BIN" || -z "$N" || -z "$T" ]]; then
  echo "Usage: $0 ./binary N threads"
  exit 1
fi

OUT="autotune_results.csv"
echo "KC,MC,NC,PREF,trial1,trial2,trial3,mean_gflops,stdev_gflops,checksum" > $OUT

# Grid (edit as needed)
KC_list=(64 96 128 192)
MC_list=(128 256 512)
NC_list=(256 512 1024)
PREF_list=(16 32 48 64)

for KC in "${KC_list[@]}"; do
  for MC in "${MC_list[@]}"; do
    for NC in "${NC_list[@]}"; do
      for PREF in "${PREF_list[@]}"; do
        gflops_arr=()
        checksum_val=""
        for trial in 1 2 3; do
          echo "Running: KC=$KC MC=$MC NC=$NC PREF=$PREF trial=$trial"
          # run and capture GFLOPs and checksum
          out=$($BIN $N $T $MC $NC $KC $PREF 2>&1)
          # expected last line like: elapsed_s=... GFLOPs=... checksum=...
          line=$(echo "$out" | tail -n 1)
          # parse GFLOPs and checksum
          gflops=$(echo "$line" | sed -n 's/.*GFLOPs=\([0-9.eE+-]*\).*/\1/p')
          checksum=$(echo "$line" | sed -n 's/.*checksum=\([0-9.eE+-]*\).*/\1/p')
          if [[ -z "$gflops" ]]; then
            gflops=0
          fi
          gflops_arr+=("$gflops")
          checksum_val="$checksum"
        done
        # compute mean and stdev
        # python small helper
        stats=$(python3 - <<PY
import sys,math
vals = [float(x) for x in ${gflops_arr[@]}]
mean = sum(vals)/len(vals)
var = sum((x-mean)**2 for x in vals)/len(vals)
import math
stdev = math.sqrt(var)
print("{:.6f},{:.6f}".format(mean,stdev))
PY
)
        mean=$(echo $stats | cut -d, -f1)
        stdev=$(echo $stats | cut -d, -f2)
        echo "$KC,$MC,$NC,$PREF,${gflops_arr[0]},${gflops_arr[1]},${gflops_arr[2]},$mean,$stdev,$checksum_val" >> $OUT
      done
    done
  done
done

echo "Done. Results in $OUT"
