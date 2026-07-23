#!/usr/bin/env python3
"""Realistic-coding benchmark client.

For each prompt (context bucket) in prompts.json, send N reps to the native
/completion endpoint with cache_prompt=false (forces full prompt reprocessing =>
true prompt-processing measurement, no cache reuse). Records PP/TG tok/s, TTFT,
and draft-acceptance stats (for MTP/n-gram runs). Saves the generation of the first
rep for a possible phase-2 code-quality eval.

Rows -> JSONL. Config identity comes from CLI flags and is stored in every row.

Usage: bench_realistic.py --server ... --prompts prompts.json --out results.jsonl \
        --quant q6xl --split tensor --power 220 --spec none --variant base --reps 3
"""
import argparse, json, time, urllib.request, sys

def post(server, path, obj, timeout=1800):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(server + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def draft_stats(timings, resp):
    """Pull draft accept counts across possible field names/locations."""
    d = {}
    for k in ("draft_n", "n_draft", "draft_n_accepted", "n_draft_accepted",
              "n_accepted", "draft_accepted"):
        if isinstance(timings, dict) and k in timings:
            d[k] = timings[k]
        if k in resp:
            d[k] = resp[k]
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=256)
    # config identity (stored per row)
    for k in ("quant", "split", "power", "spec", "variant"):
        ap.add_argument("--" + k, default="")
    ap.add_argument("--targets", default="", help="comma-sep target subset; empty=all depths")
    ap.add_argument("--save-gen", action="store_true", default=True,
                    help="store first-rep generation for phase-2 quality eval")
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    if args.targets:
        want = set(int(t) for t in args.targets.split(","))
        prompts = [p for p in prompts if p["target"] in want]
    fout = open(args.out, "a")
    cfg = {k: getattr(args, k) for k in ("quant", "split", "power", "spec", "variant")}

    # single global warmup (spin up GPU clocks / capture graphs) — smallest prompt, short gen
    if prompts:
        try:
            wp = min(prompts, key=lambda x: x["target"])
            post(args.server, "/completion", {"prompt": wp["prompt"], "n_predict": 16,
                 "cache_prompt": False, "temperature": 0.0, "top_k": 1, "stream": False})
        except Exception as e:
            print(f"[warn] warmup failed: {e}", file=sys.stderr)

    for p in prompts:
        body = {"prompt": p["prompt"], "n_predict": args.n_predict,
                "cache_prompt": False, "temperature": 0.0, "top_k": 1,
                "stream": False}
        for rep in range(args.reps):
            t0 = time.time()
            try:
                r = post(args.server, "/completion", dict(body))
            except Exception as e:
                print(f"[err] depth~{p['target']} rep{rep}: {e}", file=sys.stderr)
                continue
            wall = time.time() - t0
            tm = r.get("timings", {}) or {}
            prompt_n = tm.get("prompt_n") or r.get("tokens_evaluated")
            prompt_ms = tm.get("prompt_ms")
            pred_n = tm.get("predicted_n") or r.get("tokens_predicted")
            pred_ms = tm.get("predicted_ms")
            row = {
                **cfg,
                "target": p["target"], "prompt_tokens_built": p["tokens"],
                "rep": rep,
                "prompt_n": prompt_n, "prompt_ms": prompt_ms,
                "pp_tok_s": (prompt_n / prompt_ms * 1000) if prompt_n and prompt_ms else None,
                "predicted_n": pred_n, "predicted_ms": pred_ms,
                "tg_tok_s": (pred_n / pred_ms * 1000) if pred_n and pred_ms else None,
                "ttft_ms": prompt_ms,           # non-streamed: prompt_ms ~= time-to-first-token
                "wall_s": wall,
                "draft": draft_stats(tm, r),
                "timings_raw": tm,
            }
            if args.save_gen and rep == 0:
                row["generation"] = r.get("content", "")
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            ppn = row["pp_tok_s"]; tgn = row["tg_tok_s"]
            print(f"  d~{p['target']:>7} rep{rep}  PP={ppn and round(ppn,1)}  "
                  f"TG={tgn and round(tgn,2)} tok/s  prompt_n={prompt_n}", file=sys.stderr)
    fout.close()

if __name__ == "__main__":
    main()
