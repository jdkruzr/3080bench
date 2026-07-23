#!/usr/bin/env bash
# Launch llama-server for one benchmark config and block until /health is ready.
# Writes PID to $PIDFILE. Server log -> $SERVERLOG.
#
# Usage: serve.sh <model.gguf> <tensor|layer> <ctx> <none|ngram|mtp|mtp_ngram>
# KV is f16 always (required by -sm tensor; kept f16 on layer for parity).
set -uo pipefail

MODEL="${1:?model path}"
SPLIT="${2:?tensor|layer}"
CTX="${3:?ctx size}"
SPEC="${4:-none}"

BIN="${LLAMA_BIN:-/root/llama-src/build/bin/llama-server}"
PORT="${PORT:-8081}"
PIDFILE="${PIDFILE:-/root/server.pid}"
SERVERLOG="${SERVERLOG:-/root/server.log}"
NGRAM_TYPE="${NGRAM_TYPE:-ngram-cache}"   # ngram-cache is robust; ngram-map-k4v is best for code repetition
DRAFT_NMAX="${DRAFT_NMAX:-16}"

case "$SPEC" in
  none)      SPECARGS=() ;;
  ngram)     SPECARGS=(--spec-type "$NGRAM_TYPE" --spec-draft-n-max "$DRAFT_NMAX") ;;
  mtp)       SPECARGS=(--spec-type draft-mtp) ;;
  mtp_ngram) SPECARGS=(--spec-type "draft-mtp,$NGRAM_TYPE" --spec-draft-n-max "$DRAFT_NMAX") ;;
  *) echo "unknown SPEC=$SPEC"; exit 2 ;;
esac

echo "[serve] model=$(basename "$MODEL") split=$SPLIT ctx=$CTX spec=$SPEC port=$PORT"
"$BIN" -m "$MODEL" -sm "$SPLIT" -fa on -ngl 999 -c "$CTX" -np 1 \
  -ctk f16 -ctv f16 --host 127.0.0.1 --port "$PORT" \
  "${SPECARGS[@]}" > "$SERVERLOG" 2>&1 &
SVPID=$!
echo "$SVPID" > "$PIDFILE"

# wait for readiness (generous for 256K KV alloc + warmup)
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[serve] ready after ${i}s (pid $SVPID)"; exit 0
  fi
  if ! kill -0 "$SVPID" 2>/dev/null; then
    echo "[serve] SERVER DIED during startup; tail of log:"; tail -30 "$SERVERLOG"; exit 1
  fi
  sleep 1
done
echo "[serve] TIMEOUT waiting for health; tail of log:"; tail -30 "$SERVERLOG"
kill "$SVPID" 2>/dev/null; exit 1
