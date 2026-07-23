#!/usr/bin/env bash
# Set a uniform power limit (W) on all 4 GPUs. Clamped to [100,320] (modded 3080 range).
# Usage: set_power.sh 220 | 320
set -euo pipefail
W="${1:?usage: set_power.sh <watts 100-320>}"
if (( W < 100 || W > 320 )); then echo "watts out of range [100,320]"; exit 1; fi
nvidia-smi -pm 1 >/dev/null
for g in 0 1 2 3; do nvidia-smi -i "$g" -pl "$W" >/dev/null; done
echo "power limit set to ${W}W on GPUs 0-3"
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
