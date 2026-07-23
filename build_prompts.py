#!/usr/bin/env python3
"""Build realistic coding prompts bucketed to target context sizes.

Uses a *running* llama-server (any config) purely for its tokenizer + chat template,
so token counts are exact for THIS model. Output prompts.json is reused across every
benchmark config, so all configs see identical prompts (fair comparison).

Each prompt = a realistic coding instruction over real source code, sized so the
final templated prompt tokenizes to ~target tokens. We store the templated prompt
string; the bench sends it to /completion (raw, so we control tokens exactly and get
full timings). temperature 0 keeps generations reproducible for a later quality pass.

Usage: build_prompts.py --server http://127.0.0.1:8081 --corpus /root/llama-src \
                        --out prompts.json --targets 512,1024,...,262144
"""
import argparse, glob, json, os, sys, urllib.request

def post(server, path, obj, timeout=120):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(server + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def ntok(server, text):
    return len(post(server, "/tokenize", {"content": text})["tokens"])

def apply_template(server, messages):
    r = post(server, "/apply-template", {"messages": messages})
    # llama-server returns {"prompt": "..."}
    return r["prompt"] if isinstance(r, dict) and "prompt" in r else r

INSTRUCTION = (
    "You are a senior software engineer doing a code review. Carefully read the "
    "source code below. Identify the most significant bug or design flaw, explain "
    "it, and provide a corrected, complete implementation of the affected "
    "function(s). Think step by step and be concrete.\n\n"
)

def gather_corpus(root, exts, max_bytes):
    buf = []
    total = 0
    pats = [os.path.join(root, "**", "*" + e) for e in exts]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files.sort()
    for f in files:
        try:
            with open(f, "r", errors="ignore") as fh:
                txt = fh.read()
        except Exception:
            continue
        if len(txt) < 400:
            continue
        buf.append(f"// ===== FILE: {os.path.relpath(f, root)} =====\n{txt}\n")
        total += len(txt)
        if total >= max_bytes:
            break
    return "".join(buf)

def build_one(server, corpus, target, sys_msg):
    # converge code length so templated prompt ~= target tokens
    approx_cpt = 3.6  # chars per token for code (initial guess)
    lo_overhead = ntok(server, INSTRUCTION) + 64  # instruction + template slack
    want_code_tok = max(target - lo_overhead, 16)
    n_chars = int(want_code_tok * approx_cpt)
    prompt = None
    got = 0
    for _ in range(6):
        n_chars = max(1, min(n_chars, len(corpus)))
        code = corpus[:n_chars]
        user = INSTRUCTION + "```\n" + code + "\n```"
        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": user}]
        prompt = apply_template(server, messages)
        got = ntok(server, prompt)
        if abs(got - target) <= max(24, int(0.02 * target)):
            break
        # proportional adjust
        ratio = want_code_tok / max(got - lo_overhead, 1)
        n_chars = int(n_chars * ratio)
        if n_chars >= len(corpus):
            break
    return prompt, got

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--corpus", default="/root/llama-src")
    ap.add_argument("--out", default="prompts.json")
    ap.add_argument("--targets", default="512,1024,2048,4096,8192,16384,32768,49152,65536,98304,131072,196608,262144")
    ap.add_argument("--exts", default=".cpp,.c,.h,.hpp,.py,.cu")
    ap.add_argument("--sys", default="You are a helpful, precise senior software engineer.")
    args = ap.parse_args()

    targets = [int(t) for t in args.targets.split(",")]
    exts = args.exts.split(",")
    # corpus big enough for the largest target (~4 chars/token upper bound + margin)
    corpus = gather_corpus(args.corpus, exts, max_bytes=max(targets) * 6)
    if len(corpus) < 1000:
        print("ERROR: corpus too small; check --corpus path", file=sys.stderr)
        sys.exit(1)
    print(f"corpus: {len(corpus)} chars", file=sys.stderr)

    out = []
    for t in targets:
        prompt, got = build_one(args.server, corpus, t, args.sys)
        out.append({"target": t, "tokens": got, "prompt": prompt})
        print(f"  target={t:>7}  actual_tokens={got:>7}", file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out)} prompts -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
