#!/usr/bin/env bash
# Pre-flight checks before the full matrix. Verifies the risky unknowns:
#   1) -sm tensor loads qwen35 across 4 GPUs on no-P2P GeForce (row-split crashed on the 5060Ti box)
#   2) NCCL is actually used
#   3) coherent output (numeric sanity)
#   4) llama-bench flag shape (-fa 1, -sm tensor, -d)
#   5) ngram self-speculation engages (mtp tested separately once mtp gguf lands)
set -uo pipefail
BASE="${BASE:-/root/3080bench}"
BIN=/root/llama-src/build/bin
MODEL="${MODEL:-/root/models/base/Qwen3.6-27B-UD-Q6_K_XL.gguf}"
PORT="${PORT:-8081}"
SPLIT="${SPLIT:-tensor}"

echo "===== 0. versions / flags ====="
"$BIN/llama-server" --version 2>&1 | head -2
echo "-- llama-bench flags --"; "$BIN/llama-bench" --help 2>&1 | grep -E "\-fa|\-sm|\-d,|\-p,|\-n," | head
echo "-- server split help --"; "$BIN/llama-server" --help 2>&1 | grep -E "\-sm,|\-fa," | head

echo "===== 1. start server: $SPLIT, ctx 8192, fa on ====="
pkill -f "llama-server" 2>/dev/null; sleep 2
SERVERLOG=/root/smoke_server.log PIDFILE=/root/server.pid LLAMA_BIN="$BIN/llama-server" \
  bash "$BASE/serve.sh" "$MODEL" "$SPLIT" 8192 none
rc=$?
echo "serve rc=$rc"
if [ $rc -ne 0 ]; then echo "!!! server failed to come up — tensor split may be unsupported here. log:"; tail -40 /root/smoke_server.log; exit 1; fi

echo "===== 2. NCCL / tensor-split / arch in log ====="
grep -iE "nccl|tensor|split|qwen35|meta device|CUDA[0-9]" /root/smoke_server.log | head -20

echo "===== 3. coherence (Paris check) ====="
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":12,"temperature":0,"cache_prompt":false}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('OUT:',repr(r.get('content',''))); t=r.get('timings',{}); print('PP tok/s=%.1f  TG tok/s=%.2f  prompt_n=%s'%(t.get('prompt_per_second',0),t.get('predicted_per_second',0),t.get('prompt_n')))"

echo "===== 4. quick TG-at-depth sanity via /completion (2k prompt) ====="
python3 - <<'PY'
import json,urllib.request
p="def f(x):\n    return x*x\n"*400
body={"prompt":p,"n_predict":64,"temperature":0,"cache_prompt":False}
r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8081/completion",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=300))
t=r["timings"]; print("prompt_n=%s  PP=%.1f tok/s  TG=%.2f tok/s"%(t.get("prompt_n"),t.get("prompt_per_second",0),t.get("predicted_per_second",0)))
PY

pkill -f "llama-server" 2>/dev/null; sleep 2

echo "===== 5. llama-bench flag shape (tensor, small) ====="
"$BIN/llama-bench" -m "$MODEL" -sm "$SPLIT" -fa 1 -ngl 999 -p 512 -n 32 -d 0 -r 1 -o json 2>/tmp/lb.err \
  | python3 -c "import sys,json; rows=json.load(sys.stdin); [print(r.get('n_prompt'),r.get('n_gen'),r.get('n_depth'),round(r.get('avg_ts',0),1),'tok/s') for r in rows]" \
  || { echo 'llama-bench failed; err:'; tail -15 /tmp/lb.err; }
echo "===== SMOKE TEST DONE ====="
