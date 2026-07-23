#!/usr/bin/env bash
# Orchestrate the full matrix on the box. Resumable via per-cell done-markers, so it can
# be run while models are still downloading (missing models are skipped) and re-run later
# to fill gaps without redoing completed cells.
#
# Matrix: power{220,320} x quant{q6xl,q8} x split{tensor,layer}
#   - synthetic (llama-bench) PP+TG curves per cell (base model)
#   - realistic decode cross: tensor -> {none,ngram,mtp,mtp_ngram}; layer -> {none}
# KV is f16 (required by tensor; parity on layer). cache_prompt=false in the client.
set -uo pipefail

BASE="${BASE:-/root/3080bench}"
RES="$BASE/results"; TEL="$RES/telemetry"; MARK="$RES/.done"
mkdir -p "$RES" "$TEL" "$MARK"
export LLAMA_BIN=/root/llama-src/build/bin/llama-server
export LLAMA_BENCH=/root/llama-src/build/bin/llama-bench
PORT="${PORT:-8081}"; export PORT
PROMPTS="$BASE/prompts.json"

POWERS="${POWERS:-220 320}"
QUANTS="${QUANTS:-q6xl q8}"
SPLITS="${SPLITS:-tensor layer}"
DEPTHS_SYN="${DEPTHS_SYN:-512 1024 2048 4096 8192 16384 32768 49152 65536 98304 131072 196608 262144}"
REALISTIC_TARGETS="512,1024,2048,4096,8192,16384,32768,49152,65536,98304,131072,196608,258048"

mp(){ local q=$1 v=$2 f; [ "$q" = q6xl ] && f=Qwen3.6-27B-UD-Q6_K_XL.gguf || f=Qwen3.6-27B-Q8_0.gguf; echo "/root/models/$v/$f"; }
variant_for_spec(){ case "$1" in mtp|mtp_ngram) echo mtp;; *) echo base;; esac; }
have_model(){ [ -f "$1" ] && [ "$(stat -c %s "$1")" -gt 1000000000 ]; }  # >1GB = real, not a stub

start_tel(){ python3 "$BASE/gpu_telemetry.py" --out "$1" --interval 500 --tag "$2" >/dev/null 2>&1 & echo $! > /tmp/tel.pid; }
stop_tel(){ [ -f /tmp/tel.pid ] && kill "$(cat /tmp/tel.pid)" 2>/dev/null; rm -f /tmp/tel.pid; sleep 1; }
stop_server(){ [ -f /root/server.pid ] && { kill "$(cat /root/server.pid)" 2>/dev/null; for i in $(seq 1 20); do kill -0 "$(cat /root/server.pid)" 2>/dev/null || break; sleep 1; done; }; rm -f /root/server.pid; }

build_prompts_once(){
  [ -s "$PROMPTS" ] && { echo "[prompts] reuse existing"; return 0; }
  local m; m="$(mp q6xl base)"
  have_model "$m" || { echo "[prompts] base q6xl not ready yet; deferring"; return 1; }
  echo "[prompts] building..."
  bash "$BASE/serve.sh" "$m" tensor 262144 none || return 1
  python3 "$BASE/build_prompts.py" --server "http://127.0.0.1:$PORT" --corpus /root/llama-src \
     --out "$PROMPTS" --targets "$REALISTIC_TARGETS"
  local rc=$?; stop_server; return $rc
}

main(){
  build_prompts_once || echo "[warn] prompts not built; realistic phase will be skipped until models land"
  for P in $POWERS; do
    bash "$BASE/set_power.sh" "$P" || echo "[warn] set_power $P failed"
    for Q in $QUANTS; do
      for S in $SPLITS; do
        # ---- SYNTHETIC (base model) ----
        key="syn_${Q}_${S}_${P}w"; bm="$(mp "$Q" base)"
        if [ -f "$MARK/$key" ]; then echo "[skip] $key"; elif ! have_model "$bm"; then echo "[defer] $key (model missing)"; else
          echo "[run] $key"; start_tel "$TEL/$key.csv" "$key"
          TAG_QUANT="$Q" TAG_POWER="$P" REPS="${SYN_REPS:-2}" DEPTHS="$DEPTHS_SYN" \
            bash "$BASE/run_synthetic.sh" "$bm" "$S" "$RES/synthetic.jsonl" && touch "$MARK/$key"
          stop_tel
        fi
        # ---- REALISTIC decode cross ----
        [ -s "$PROMPTS" ] || { echo "[defer] realistic (no prompts yet)"; continue; }
        if [ "$S" = tensor ]; then SPECS="none ngram mtp mtp_ngram"; else SPECS="none"; fi
        for SP in $SPECS; do
          key="real_${Q}_${S}_${P}w_${SP}"; V="$(variant_for_spec "$SP")"; M="$(mp "$Q" "$V")"
          if [ -f "$MARK/$key" ]; then echo "[skip] $key"; continue; fi
          have_model "$M" || { echo "[defer] $key (model missing)"; continue; }
          echo "[run] $key"
          if ! bash "$BASE/serve.sh" "$M" "$S" 262144 "$SP"; then echo "[fail] serve $key"; stop_server; continue; fi
          start_tel "$TEL/$key.csv" "$key"
          # none + synthetic carry the full 13-depth PP/TG curve; spec configs sample a 6-pt subset
          # (they don't change PP, only decode) to avoid re-paying long-prefill cost 4x each.
          if [ "$SP" = none ]; then TGT=""; else TGT="--targets ${SPEC_TARGETS:-512,4096,32768,98304,196608,258048}"; fi
          python3 "$BASE/bench_realistic.py" --server "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
            --out "$RES/realistic.jsonl" --reps "${REAL_REPS:-2}" --n-predict 256 $TGT \
            --quant "$Q" --split "$S" --power "$P" --spec "$SP" --variant "$V" && touch "$MARK/$key"
          stop_tel; stop_server
        done
      done
    done
  done
  echo "[done] run_all complete $(date +%T)"
}
main
