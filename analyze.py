#!/usr/bin/env python3
"""Parse benchmark results -> tidy CSV summaries + charts.

Inputs (defaults under ./results):
  synthetic.jsonl   - llama-bench rows (kind pp|tg, depth, split, quant, power, avg_ts...)
  realistic.jsonl   - per-rep client rows (pp_tok_s, tg_tok_s, ttft_ms, draft{...}, spec...)
  telemetry/*.csv   - per-run GPU samples

Outputs (under ./analysis):
  synthetic_summary.csv, realistic_summary.csv, telemetry_summary.csv
  charts/*.png (best-effort; skipped if matplotlib missing)

Charts are stdlib-computed (medians); matplotlib only for rendering.
"""
import argparse, csv, glob, json, os, statistics as st
from collections import defaultdict

def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except Exception: pass
    return rows

def median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return st.median(xs) if xs else None

def summarize_synthetic(rows, outdir):
    # llama-bench json fields: avg_ts (tokens/s), n_prompt, n_gen, n_depth
    grp = defaultdict(list)
    for r in rows:
        key = (r.get("quant"), r.get("power"), r.get("split"), r.get("kind"), r.get("depth"))
        ts = r.get("avg_ts")
        if ts is not None:
            grp[key].append(ts)
    out = os.path.join(outdir, "synthetic_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quant", "power", "split", "kind", "depth", "tok_s_median", "n"])
        for (q, p, s, k, d), v in sorted(grp.items(), key=lambda x: (str(x[0][0]), str(x[0][1]), str(x[0][2]), str(x[0][3]), x[0][4] or 0)):
            w.writerow([q, p, s, k, d, round(median(v), 2) if median(v) else "", len(v)])
    print(f"wrote {out} ({len(grp)} cells)")
    return grp

def accept_rate(draft):
    if not isinstance(draft, dict):
        return None
    prop = draft.get("draft_n") or draft.get("n_draft")
    acc = (draft.get("draft_n_accepted") or draft.get("n_draft_accepted")
           or draft.get("n_accepted") or draft.get("draft_accepted"))
    if prop and acc is not None and prop > 0:
        return acc / prop
    return None

def summarize_realistic(rows, outdir):
    grp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r.get("quant"), r.get("power"), r.get("split"), r.get("spec"),
               r.get("variant"), r.get("target"))
        for m in ("pp_tok_s", "tg_tok_s", "ttft_ms", "prompt_n"):
            if r.get(m) is not None:
                grp[key][m].append(r[m])
        ar = accept_rate(r.get("draft"))
        if ar is not None:
            grp[key]["accept_rate"].append(ar)
    out = os.path.join(outdir, "realistic_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quant", "power", "split", "spec", "variant", "target",
                    "prompt_n_med", "pp_tok_s_med", "tg_tok_s_med", "ttft_ms_med",
                    "accept_rate_med", "n"])
        for key, mv in sorted(grp.items(), key=lambda x: tuple(str(v) for v in x[0])):
            q, p, s, sp, v, t = key
            w.writerow([q, p, s, sp, v, t,
                        round(median(mv["prompt_n"]) or 0),
                        round(median(mv["pp_tok_s"]) or 0, 1) if mv["pp_tok_s"] else "",
                        round(median(mv["tg_tok_s"]) or 0, 2) if mv["tg_tok_s"] else "",
                        round(median(mv["ttft_ms"]) or 0) if mv["ttft_ms"] else "",
                        round(median(mv["accept_rate"]), 3) if mv["accept_rate"] else "",
                        max(len(mv["tg_tok_s"]), len(mv["pp_tok_s"]))])
    print(f"wrote {out} ({len(grp)} cells)")
    return grp

def summarize_telemetry(outdir, teldir):
    out = os.path.join(outdir, "telemetry_summary.csv")
    files = sorted(glob.glob(os.path.join(teldir, "*.csv")))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "gpu", "power_W_med", "power_W_max", "temp_med", "temp_max",
                    "sm_clock_med", "util_med", "mem_used_max_MiB", "samples"])
        for fp in files:
            run = os.path.splitext(os.path.basename(fp))[0]
            per = defaultdict(lambda: defaultdict(list))
            try:
                with open(fp) as fh:
                    for row in csv.DictReader(fh):
                        g = row.get("index")
                        if g is None: continue
                        def fv(k):
                            try: return float(row.get(k, "").strip())
                            except Exception: return None
                        for k in ("power.draw", "temperature.gpu", "clocks.sm",
                                  "utilization.gpu", "memory.used"):
                            val = fv(k)
                            if val is not None: per[g][k].append(val)
            except Exception:
                continue
            for g, mv in sorted(per.items()):
                w.writerow([run, g,
                            round(median(mv["power.draw"]) or 0, 1),
                            round(max(mv["power.draw"]) if mv["power.draw"] else 0, 1),
                            round(median(mv["temperature.gpu"]) or 0, 1),
                            round(max(mv["temperature.gpu"]) if mv["temperature.gpu"] else 0, 1),
                            round(median(mv["clocks.sm"]) or 0),
                            round(median(mv["utilization.gpu"]) or 0),
                            round(max(mv["memory.used"]) if mv["memory.used"] else 0),
                            len(mv["power.draw"])])
    print(f"wrote {out} ({len(files)} runs)")

def make_charts(syn, real, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[charts] skipped (matplotlib unavailable: {e})")
        return
    cdir = os.path.join(outdir, "charts"); os.makedirs(cdir, exist_ok=True)

    def line(ax, xs, ys, label):
        pts = sorted([(x, y) for x, y in zip(xs, ys) if y is not None])
        if pts: ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, label=label)

    # Synthetic PP & TG vs depth, one figure per power, lines per split+quant
    powers = sorted({k[1] for k in syn})
    for kind, ylab in (("pp", "Prompt processing (tok/s)"), ("tg", "Token generation (tok/s)")):
        for pw in powers:
            fig, ax = plt.subplots(figsize=(8, 5))
            for q in ("q6xl", "q8"):
                for s in ("tensor", "layer"):
                    cells = [(d, median(v)) for (qq, pp, ss, kk, d), v in syn.items()
                             if qq == q and pp == pw and ss == s and kk == kind]
                    if cells:
                        cells.sort()
                        line(ax, [c[0] for c in cells], [c[1] for c in cells], f"{q}-{s}")
            ax.set_xscale("log", base=2); ax.set_xlabel("context depth (tokens)")
            ax.set_ylabel(ylab); ax.set_title(f"Synthetic {kind.upper()} @ {pw}W"); ax.grid(True, alpha=.3); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(cdir, f"synthetic_{kind}_{pw}w.png"), dpi=120); plt.close(fig)

    # Realistic TG spec-decode cross (tensor), per quant/power
    combos = sorted({(k[0], k[1]) for k in real if k[2] == "tensor"})
    for q, pw in combos:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sp in ("none", "ngram", "mtp", "mtp_ngram"):
            cells = [(t, median(mv["tg_tok_s"])) for (qq, pp, ss, s2, v, t), mv in real.items()
                     if qq == q and pp == pw and ss == "tensor" and s2 == sp]
            if cells:
                cells.sort(); line(ax, [c[0] for c in cells], [c[1] for c in cells], sp)
        ax.set_xscale("log", base=2); ax.set_xlabel("prompt tokens"); ax.set_ylabel("TG (tok/s)")
        ax.set_title(f"Realistic TG spec-decode cross — {q} tensor @ {pw}W"); ax.grid(True, alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(cdir, f"realistic_spec_{q}_{pw}w.png"), dpi=120); plt.close(fig)
    print(f"[charts] wrote PNGs to {cdir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    syn = summarize_synthetic(load_jsonl(os.path.join(args.results, "synthetic.jsonl")), args.out)
    real = summarize_realistic(load_jsonl(os.path.join(args.results, "realistic.jsonl")), args.out)
    summarize_telemetry(args.out, os.path.join(args.results, "telemetry"))
    make_charts(syn, real, args.out)

if __name__ == "__main__":
    main()
