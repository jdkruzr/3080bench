#!/usr/bin/env bash
# Synthetic reference curves via llama-bench: PP at each prompt size, TG at each depth.
# Emits one JSON object per llama-bench result row (tagged with config) to $OUT (jsonl).
# Flag names (-fa 1 vs -fa on, -sm tensor, -p 0) verified against --help before the real run.
set -uo pipefail

MODEL="${1:?model.gguf}"
SPLIT="${2:?tensor|layer}"
OUT="${3:?out.jsonl}"

BIN="${LLAMA_BENCH:-/root/llama-src/build/bin/llama-bench}"
DEPTHS="${DEPTHS:-512 1024 2048 4096 8192 16384 32768 49152 65536 98304 131072 196608 262144}"
REPS="${REPS:-3}"
FA="${FA:-1}"
NGEN="${NGEN:-128}"
TAG_QUANT="${TAG_QUANT:-}"
TAG_POWER="${TAG_POWER:-}"

emit() { # kind depth  (reads llama-bench json array on stdin)
  python3 -c "
import sys, json
try:
    rows = json.load(sys.stdin)
except Exception as e:
    sys.stderr.write(f'parse fail: {e}\n'); sys.exit(0)
for r in rows:
    r.update({'kind':'$1','depth':$2,'split':'$SPLIT',
              'quant':'$TAG_QUANT','power':'$TAG_POWER'})
    print(json.dumps(r))
"
}

for d in $DEPTHS; do
  # PP: process a d-token prompt, no prior context (n_gen 0)
  if ! "$BIN" -m "$MODEL" -sm "$SPLIT" -fa "$FA" -ngl 999 -p "$d" -n 0 -r "$REPS" -o json 2>>"$OUT.err" \
       | emit pp "$d" >> "$OUT"; then echo "[synthetic] PP d=$d failed" >&2; fi
  # TG at depth d: NGEN gen tokens with d tokens already in KV
  if ! "$BIN" -m "$MODEL" -sm "$SPLIT" -fa "$FA" -ngl 999 -p 0 -n "$NGEN" -d "$d" -r "$REPS" -o json 2>>"$OUT.err" \
       | emit tg "$d" >> "$OUT"; then echo "[synthetic] TG d=$d failed" >&2; fi
  echo "[synthetic] done depth=$d ($SPLIT $TAG_QUANT ${TAG_POWER}W)" >&2
done
