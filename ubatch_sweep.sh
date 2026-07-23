#!/usr/bin/env bash
# Side-experiment: how prefill throughput / TTFT and intercard PCIe traffic scale with the
# physical forward-pass width (n_ubatch), at single-user (-np 1) tensor split. n_ubatch is the
# ONLY prefill lever for a single stream; it does not affect decode. Bigger ub -> bigger matmuls
# and bigger all-reduce messages -> should raise both prefill speed (until saturated) and
# intercard BW (until Gen3/no-P2P caps it). Pure prefill (-p D -n 0) = the "loud" phase.
set -uo pipefail
BIN=/root/llama-src/build/bin/llama-bench
MODEL="${MODEL:-/root/models/base/Qwen3.6-27B-UD-Q6_K_XL.gguf}"
RES=/root/3080bench/results/ubatch
mkdir -p "$RES"
CTXPOINTS="${CTXPOINTS:-98304 262144}"   # 98K (user pick) + 256K (worst-case TTFT)
UBS="${UBS:-256 512 1024 2048 4096}"

for D in $CTXPOINTS; do
  for UB in $UBS; do
    if [ "$UB" -gt 4096 ]; then B="$UB"; else B=4096; fi   # n_batch must be >= n_ubatch
    tag="d${D}_ub${UB}"
    echo ">>> $(date +%T) START $tag (b=$B ub=$UB)"
    # capture per-GPU PCIe rx/tx (MB/s) for the duration of this run
    nvidia-smi dmon -s t -d 1 -o T > "$RES/$tag.dmon" 2>/dev/null &
    DPID=$!
    "$BIN" -m "$MODEL" -sm tensor -fa 1 -ngl 999 -p "$D" -n 0 -b "$B" -ub "$UB" -r 2 -o json 2>"$RES/$tag.err" \
      | python3 -c "
import sys, json
try: rows = json.load(sys.stdin)
except Exception: rows = []
for r in rows:
    r.update({'depth': $D, 'ubatch': $UB, 'nbatch': $B}); print(json.dumps(r))" >> "$RES/ubatch.jsonl"
    kill "$DPID" 2>/dev/null
    echo ">>> $(date +%T) DONE $tag"
  done
done
echo ">>> UBATCH SWEEP DONE $(date +%T)"
