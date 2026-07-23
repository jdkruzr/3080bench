# 3080bench — Qwen3.6-27B on 4× modded RTX 3080 20GB (llama.cpp)

Companion study to `~/REPORT.md` (8× RTX 5060 Ti). Characterizes prompt-processing (PP,
cache off) and token-generation (TG) vs context, on 4× Ampere sm_86, PCIe Gen3 ×16, no
NVLink/P2P — driven by realistic coding prompts plus a synthetic reference.

## Matrix
power {220W, 320W} × quant {UD-Q6_K_XL, Q8_0} × split {tensor, layer}
- synthetic (`llama-bench`): PP at each size, TG at each depth, to 256K
- realistic (`llama-server` + client, `cache_prompt=false`): decode cross on tensor =
  {none, ngram, mtp, mtp_ngram}; layer = {none}
- KV = f16 everywhere (required by `-sm tensor`; parity on layer)

## Files
| file | where it runs | role |
|---|---|---|
| `set_power.sh` | box | `nvidia-smi -pl` to 220/320 W |
| `serve.sh` | box | launch llama-server for one config, wait for /health |
| `gpu_telemetry.py` | box | sample power/clocks/temp/util/PCIe/mem @500ms → CSV |
| `build_prompts.py` | box | build real-code prompts sized to each ctx target via /tokenize+/apply-template |
| `bench_realistic.py` | box | drive /completion (cache off), record PP/TG/TTFT/draft-accept + save gens |
| `run_synthetic.sh` | box | llama-bench PP/TG-at-depth sweep |
| `run_all.sh` | box | orchestrate the whole matrix; resumable via `results/.done/` markers |
| `analyze.py` | local | JSONL+telemetry → summary CSVs + charts |

## Spec-decode flag mapping (llama.cpp b10076)
- ngram: `--spec-type ngram-cache` (self-speculative; great for repetitive code)
- mtp: `--spec-type draft-mtp` (uses the MTP-variant GGUF's nextn head)
- both: `--spec-type draft-mtp,ngram-cache`

## Run (on box, after build + models ready)
```bash
cd /root/3080bench
bash run_all.sh            # resumable; skips missing models, re-run to fill gaps
```
Pull results locally and analyze:
```bash
rsync -a -e 'ssh -p <port>' root@HOST:/root/3080bench/results/ ./results/
python3 analyze.py --results results --out analysis
```

## Environment notes / gotchas (this box)
- **No direct HF/GitHub egress**: huggingface.co is DNS-sinkholed; models pulled from
  **hf-mirror.com** (reachable), 4 parallel `curl -C -` streams, sha256-verified.
- **Persistence**: PID1 is bash + supervisord; sshd-session processes get reaped on
  disconnect. Long jobs MUST be launched `setsid nohup … &` (tmux server does not survive).
- **CUDA build**: toolkit at `/usr/local/cuda-12.9` had runtime cuBLAS but no dev headers;
  `apt-get install libcublas-dev-12-9` fixes `cublas_v2.h`. Built with `-DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NCCL=ON -DLLAMA_CURL=OFF`. Web-UI asset fetch fails
  offline but is non-fatal.
- **Port 8080** = Jupyter; llama-server uses **8081**.
