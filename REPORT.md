# Cursed Inference Lab II — 4× *modded* 20GB RTX 3080, Qwen3.6-27B

**Purpose:** Characterize llama.cpp single-user inference on a 4× modded-3080 box, as a
companion / counterpoint to the 8× RTX 5060 Ti study (`~/REPORT.md`). Same model, same
tensor-parallel path, different silicon and a different country's firewall.

Date: 2026-07-22/23 · Operator: jarett (jdkruzr) · Box: Vast.ai, **in China** (4× RTX 3080 20GB, sm_86)

---

## TL;DR — Executive summary

Four **clamshell-modded 20GB RTX 3080s** (stock GA102 silicon, 8704 cores, 320-bit/760 GB/s,
double-density VRAM on the back of the board — *not* the pricier 24GB 3080 Ti bin) were
benchmarked on Qwen3.6-27B (`qwen35`, dense hybrid-SSM) via llama.cpp `-sm tensor`.

**The one lesson:** **run Q8_0 with the MTP head, tensor-parallel across all 4 cards.** That
config does **~93–95 tok/s decode** at low/mid context and **~70 tok/s at 256K**, with full-fat
Q8 quality — roughly **2.4× the fastest the 5060 Ti box ever managed (39.5 t/s)**. Everything
else follows from three facts:

- **`-sm tensor` works here** (unlike `-sm row`, which *crashed* on the 5060 Ti box). It loads
  `qwen35` across 4 no-P2P GeForce cards and 256K context fits at **~11 GB / 20 GB per GPU**.
- **MTP is a ~1.8× decode win; n-gram is a net loss.** The trained multi-token head drafts at
  **77–87% acceptance** → nearly doubles single-stream TG. Prompt-lookup n-gram drafts at only
  ~40% and runs **~2.7× *slower* than baseline** (caveat below).
- **This box beats the 5060 Ti box on every axis that matters for one user**: +20% prefill
  (compute/FLOPS), +39% decode (memory bandwidth), and no 8-way interconnect collapse to fall
  into (only 4 cards — permanently at the TP sweet spot).

**Two knobs turned out not to matter:** Q6_K_XL vs Q8_0 decode is a **dead heat** here (the
5060 Ti's ~6% Q6 edge vanishes — decode is latency-bound, not stream-size-bound), and
**`n_ubatch` tuning buys ≤4%** (default 512 is already near-optimal). **One knob was locked
away:** the modded 320W headroom — the container denies `nvidia-smi -pl`, so everything ran at
**220W**, where the cards sit **thermally bored at 67°C** but power-capped.

---

## 0. Hardware / platform (measured, via `cudaGetDeviceProperties`)

| thing | value | why it matters |
|---|---|---|
| GPU ×4 | **RTX 3080, 68 SM → 8704 CUDA cores**, 1710 MHz | stock 3080 GA102 bin (3080 Ti = 80 SM/10240) |
| VRAM | **21.0 GB**, 320-bit GDDR6X @ 9501 MHz | clamshell-doubled 10→20GB; same bus → same BW |
| Mem BW | **760 GB/s** (measured) | decode is bound by this |
| PCIe | **Gen3 ×16 ≈ 15.75 GB/s**, no NVLink | the multi-GPU wall |
| **P2P** | **disabled** (GeForce driver) | all-reduce hairpins through host DRAM → latency-bound decode |
| Topology | GPU0↔1, GPU2↔3 = PIX; across pairs = PHB (host bridge) | ring crosses one slow host hop |
| Host | 80 vCPU, 314 GB RAM, 128 GB disk, CUDA 12.9 | fine |
| Power | limit **220 W** (default/max 320 W) | **can't raise** — container lacks perms |

**Two walls, same as the 5060 Ti box:** (1) no P2P → decode is latency-bound on host-staged
all-reduce; (2) ~15.75 GB/s interconnect ceiling. The 3080's stronger silicon rides *on top* of
the same floor.

## 1. Software

- llama.cpp `master` b10076 (846e991ec), built from source for **sm_86**, `-DGGML_CUDA=ON
  -DGGML_NCCL=ON -DLLAMA_CURL=OFF`. NCCL used for the tensor-split all-reduce.
- Models: `unsloth/Qwen3.6-27B-GGUF` (UD-Q6_K_XL, Q8_0) + `-MTP-GGUF` variants, sha256-verified.
- **China networking note:** huggingface.co is GFW-sinkholed; models pulled from **hf-mirror.com**
  (4-parallel `curl -C -`), CUDA dev headers from nvidia.cn. Reproducibility detail, not a result.

## 2. Model — Qwen3.6-27B (arch `qwen35`)

Hybrid **state-space + attention**, 64 layers, native **262144 (256K)** ctx, reasoning model
(emits `<think>`). Most layers are O(1) SSM → tiny KV, graceful decode decay. **256K KV footprint
≈ 11 GB/GPU** in `-sm tensor` (f16 KV, required by tensor mode). Fits 4×20GB with ~9 GB to spare.

---

## 3. RESULTS — quant × context (tensor, 220W, synthetic `llama-bench`)

Clean PP/TG curves. Both quants; the Q6-vs-Q8 delta is the story.

| ctx | PP Q6 | PP Q8 | | TG Q6 | TG Q8 |
|---|---|---|---|---|---|
| 512 | 1392 | **1458** | | 54.8 | 53.7 |
| 4K | 1367 | **1388** | | 53.6 | 53.3 |
| 32K | 1294 | **1324** | | 52.0 | 51.0 |
| 131K | 1081 | **1104** | | 45.6 | 45.0 |
| 256K | 876 | **893** | | 38.3 | **39.0** |

- **Prefill: Q8 wins ~2–5%** everywhere (cheaper dequant, compute-bound) — matches the 5060 Ti.
- **Decode: dead heat** (±2%, noise). The 5060 Ti's ~6% Q6-decode edge **disappears** here: with
  760 GB/s of bandwidth headroom, decode is dominated by the **quant-independent no-P2P all-reduce
  latency**, not weight-streaming size. → **Just run Q8** (quality, no decode cost, faster prefill).
- **Context decay is graceful** (SSM doing its job): ~37% PP / ~30% TG falloff 512→256K, where a
  pure transformer would collapse.

## 4. RESULTS — speculative decode cross (realistic coding prompts, Q8, tensor, 220W)

The headline. `cache_prompt=false`, real code prompts, TG in tok/s (draft acceptance).

| ctx | none | **mtp** | ngram |
|---|---|---|---|
| 512 | 51.9 | **92.8** (0.77) | 20.9 (0.31) |
| 32K | 48.6 | **95.4** (0.86) | 18.4 (0.46) |
| 98K | 45.4 | **78.3** (0.87) | 17.x (0.46) |
| 256K | 37.8 | **69.3** (0.87) | 17.6 (0.43) |

- **MTP ≈ 1.8× across the board**, peaking **95 t/s at 32K** where acceptance is highest (0.86).
  The trained next-token head produces drafts good enough (~80% accepted) that even the no-P2P
  verify overhead pays off big. Decode actually gets *faster* with a little context before the
  long-context falloff. **This is the config to ship.**
- **n-gram is a net loss** (~2.7× slower, ~40% acceptance). **Caveat:** our methodology
  (`cache_prompt=false`, only 256 generated tokens) keeps the ngram-cache cold — it drafts from
  *history* we keep wiping, while MTP is history-independent. So MTP's win is robust; ngram's loss
  is partly workload. A warm-cache/long-gen retest would be fairer, but MTP dominates regardless.
- **MTP + n-gram stacked → `ggml_abort` at load** (unsupported). Moot given ngram alone hurts.

## 5. RESULTS — interconnect (PCIe telemetry, whole run, `dmon -s t`)

Reconciles the live ~3.1 GB/s observation with the saturation story:

| phase | intercard PCIe |
|---|---|
| decode | **~0 GB/s (median)** — tiny latency-bound messages |
| typical prefill | **~3.2–3.5 GB/s** (p99) ← the ~3.1 you saw live |
| high-context prefill burst | **~15–16 GB/s** (RX 14.7 / TX 16.4) → **saturates Gen3 ×16** |

The high-context PP/TG falloff **is** the interconnect saturating: at 128K–256K, prefill bursts
slam the ~15.75 GB/s ceiling. (In `-sm tensor` the KV is split across cards, so the attention
layers gather KV cross-card — traffic that grows with context, independent of `n_ubatch`.)

## 6. RESULTS — `n_ubatch` sweep (single-user prefill lever, Q6, tensor)

`n_ubatch` = physical forward-pass width during prefill (NOT concurrent users; that's `-np 1`).
The only prefill/TTFT knob for a single stream — and it barely moves the needle.

| ub | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| PP @98K | 1080 | 1144 | 1189 | **1194** | 1162 |
| PP @256K | 843 | 875 | **879** | 840 | 833 |

Default **512 is within ~4% of optimal**; sweet spot 1024–2048 gains +4% at 98K and **~nothing
at 256K** (bigger *hurts* — compute-buffer pressure). TTFT is bound by compute + interconnect,
not forward-pass width. Leave it at 512.

## 7. RESULTS — power / thermals (the "modded" angle we *could* see, at 220W)

| metric (per card, under load) | value |
|---|---|
| power draw | median ~205–211 W, **max ~220 W (pinned at the cap)** |
| temperature | median 63–65 °C, **max 67 °C** |
| SM clock | ~1755–1785 MHz (at/above stock boost — not thermal-throttled) |
| VRAM used | ~11–12 GB / 20 GB |

The cards are **power-limited, not thermal-limited** — bumping against 220W while loafing at
67°C. The locked-away **320W headroom would very likely help** (they have the thermal room), but
we can't test it in this container. MTP runs draw ~9W less median than `none` (lighter verify duty
cycle). This is the one place the "modded card" story is under-explored — needs a privileged/bare-
metal instance.

---

## 8. vs the 5060 Ti box (matched: 4-GPU tensor, Q6, short ctx)

| | 4×3080 (here) | 4×5060 Ti (prior) | driver |
|---|---|---|---|
| **PP** tok/s | ~1392 | 1119 | **+20%** ← ~1.26× FP32 FLOPS |
| **TG** tok/s | ~55 | 39.5 | **+39%** ← 1.70× mem BW |
| CUDA cores | 8704 | 4608 | 1.89× (misleading across gens) |
| mem BW | 760 GB/s | 448 GB/s | 1.70× |
| interconnect | Gen3×16, no P2P | Gen4×8, no P2P | ~same wall |

**Cores/FLOPS explain prefill, bandwidth explains decode, the shared no-P2P wall trims both.**
With MTP, the gap widens to **2.4×** (95 vs 39.5). And unlike the 5060 Ti box (TP peaks at 4,
*collapses* to 11 t/s at 8), this box has only 4 cards → no 8-way cliff to fall off.

## 9. Constraints / caveats (what this run does NOT cover)

- **No 320W pass** — container denies power-limit control. §7 is 220W-only; the modded headroom
  is untested. Needs bare-metal/privileged.
- **`layer` split dropped** — pipeline-parallel is single-active-GPU for single-stream decode;
  structurally ≪ tensor (5060 Ti data confirmed). Only wins high-concurrency serving.
- **n-gram handicapped** by cold-cache/short-gen methodology (§4).
- **Single-user only** (`-np 1`); no concurrency/throughput sweep.

## 10. Reproduce

Harness in `/root/3080bench/` (and `/home/jtd/3080bench/`). `bash run_all.sh` (resumable via
`results/.done/` markers). Pull + `python3 analyze.py`. Champion serve command:

```bash
llama-server -m Qwen3.6-27B-MTP-Q8_0.gguf -sm tensor -fa on -ngl 999 \
  -c 262144 -np 1 -ctk f16 -ctv f16 --spec-type draft-mtp --host 127.0.0.1 --port 8081
```
