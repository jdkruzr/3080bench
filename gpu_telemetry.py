#!/usr/bin/env python3
"""Background GPU telemetry sampler -> CSV.

Samples all GPUs every --interval ms until SIGTERM/SIGINT. One row per (timestamp, gpu).
Columns chosen because "utilization is a liar" (REPORT.md): we want power, clocks, and
PCIe traffic to explain latency-bound behavior, not just util%.

Usage: gpu_telemetry.py --out results/telemetry/<run>.csv --interval 500
"""
import argparse, csv, signal, subprocess, sys, time

FIELDS = [
    "index", "utilization.gpu", "memory.used", "power.draw", "temperature.gpu",
    "clocks.sm", "clocks.mem", "pcie.link.gen.current", "pcie.link.width.current",
]
QUERY = ",".join(FIELDS)

_stop = False
def _handle(*_):
    global _stop
    _stop = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=int, default=500, help="ms between samples")
    ap.add_argument("--tag", default="", help="free-form run tag stored in each row")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_unix", "tag"] + FIELDS)
        while not _stop:
            t = time.time()
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={QUERY}",
                     "--format=csv,noheader,nounits"], text=True)
            except Exception as e:
                w.writerow([t, args.tag, "ERROR", str(e)])
                f.flush()
                time.sleep(args.interval / 1000)
                continue
            for line in out.strip().splitlines():
                vals = [v.strip() for v in line.split(",")]
                w.writerow([f"{t:.3f}", args.tag] + vals)
            f.flush()
            # simple fixed-interval sleep; sampling jitter is acceptable for this use
            time.sleep(args.interval / 1000)

if __name__ == "__main__":
    main()
