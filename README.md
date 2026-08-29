# vllm-pytorch

Educational project for comparing two ways to serve the same language model locally on **CPU** using Docker Compose:

- **[vLLM](https://github.com/vllm-project/vllm)** — optimized inference engine (`vllm-docker-compose.yml`)
- **PyTorch + Hugging Face Transformers** — `transformers serve` on top of PyTorch (`pt-docker-compose.yml`)

Both stacks expose an **OpenAI-compatible API** on port `8000`, so you can benchmark them with the same client.

> **Note:** This repository was created for **educational purposes** to explore and compare inference runtimes. It is not intended for production use.

## Model

Both compose files serve:

**`Qwen/Qwen2.5-7B-Instruct`**

This model fits reasonably on a **32 GB RAM** machine in `bfloat16` (~15–20 GB for weights plus server overhead).

## Requirements

- Docker and Docker Compose
- ~32 GB system RAM (more is better)
- CPU only — no GPU required
- Hugging Face token (if model download requires authentication): copy `.env.example` to `.env` and set `HF_TOKEN`

```bash
cp .env.example .env
```

## Project layout

```
vllm-docker-compose.yml   # vLLM CPU deployment
pt-docker-compose.yml     # PyTorch / Transformers deployment
.env.example              # Environment variable template (copy to .env)
main.py                   # Benchmark script (5 prompts + timing)
huggingface-cache/        # Shared model cache (created on first run)
vllm-cache/               # vLLM cache (created on first run)
```

## Quick start

### 1. Start a server

**vLLM:**

```bash
docker compose -f vllm-docker-compose.yml up
```

**PyTorch:**

```bash
docker compose -f pt-docker-compose.yml up
```

Wait until the server is ready (model download and load can take several minutes on first run).

### 2. Run the benchmark

In another terminal:

```bash
python main.py
```

The script sends 5 prompts to `http://localhost:8000/v1/chat/completions` and prints per-prompt latency plus total/average time.

Optional flags:

```bash
python main.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct --max-tokens 256
```

## Comparing vLLM vs PyTorch

| | vLLM | PyTorch (Transformers) |
|---|---|---|
| Compose file | `vllm-docker-compose.yml` | `pt-docker-compose.yml` |
| Image | `vllm/vllm-openai-cpu` | `pytorch/pytorch:2.6.0-cpu` |
| Server command | `vllm serve` | `transformers serve` |
| Typical CPU speed | Faster | Slower |
| First startup | Model loads at start | Installs deps + loads model |

Run **only one** stack at a time — both bind port `8000`.

## Compose files: main differences

Both files deploy the same model (`Qwen/Qwen2.5-7B-Instruct`) on CPU with an OpenAI-compatible API on port `8000`. The difference is *how* inference is run.

### Side-by-side

| Aspect | `vllm-docker-compose.yml` | `pt-docker-compose.yml` |
|---|---|---|
| **Runtime** | vLLM (dedicated inference engine) | PyTorch + Hugging Face `transformers serve` |
| **Docker image** | Pre-built `vllm/vllm-openai-cpu` | Generic `pytorch/pytorch:2.6.0-cpu` |
| **Dependencies** | Bundled in the image | Installed at startup via `pip install transformers[serving]` |
| **Startup time** | Faster after image pull | Slower (pip install on every start) |
| **Inference speed** | Optimized (PagedAttention, batching) | Baseline PyTorch; generally slower on CPU |
| **Config complexity** | Many tuning knobs (KV cache, batching, threads) | Minimal (device, dtype, continuous batching) |
| **Volumes** | `huggingface-cache` + `vllm-cache` | `huggingface-cache` only |
| **Container extras** | `SYS_NICE`, `seccomp=unconfined` | None |

### What they share

- Same model: `Qwen/Qwen2.5-7B-Instruct`
- Same API: `POST /v1/chat/completions`
- Same port: `8000`
- Same dtype: `bfloat16`
- Same Hugging Face cache: `./huggingface-cache` (model downloaded once, reused by both)
- Same benchmark client: `main.py` works with either stack without changes
- CPU-only — no GPU required

### vLLM compose — key points

- Uses a **purpose-built inference server** designed for production-scale throughput.
- **CPU-specific settings** control memory and parallelism:
  - `VLLM_CPU_KVCACHE_SPACE` — KV-cache size in GB
  - `VLLM_CPU_OMP_THREADS_BIND` — CPU thread binding
  - `MAX_NUM_BATCHED_TOKENS` / `MAX_NUM_SEQS` — batching limits
  - `GPU_MEMORY_UTILIZATION` — on CPU, sets the fraction of RAM used for the model
- Separate `vllm-cache` volume for engine-specific artifacts.
- Best choice when you want **lower latency and higher throughput** on CPU.

### PyTorch compose — key points

- Uses **vanilla PyTorch** with Hugging Face's built-in `transformers serve` CLI.
- **Simpler setup** — fewer environment variables, easier to understand the stack.
- Installs Python packages on each container start; no custom image build required.
- `CONTINUOUS_BATCHING` can be enabled for modest throughput gains (off by default).
- Best choice when you want to see **how inference works at the framework level** without an optimized serving layer.

### When to use which

| Goal | Recommended stack |
|---|---|
| Compare inference performance | Run both with `main.py` and compare timings |
| Learn how LLM serving is optimized | vLLM |
| Learn PyTorch / Transformers basics | PyTorch compose |
| Fastest responses on CPU | vLLM |
| Simplest mental model | PyTorch compose |

## Configuration

Copy `.env.example` to `.env` and edit values. Docker Compose loads `.env` automatically.

Edit environment variables to change:

- `MODEL` — Hugging Face model ID
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` — Hugging Face access token
- `DTYPE` — e.g. `bfloat16`
- vLLM-specific: KV cache size, batch limits, thread binding
- PyTorch-specific: `DEVICE`, `CONTINUOUS_BATCHING`

On Apple Silicon, switch the vLLM image tag from `latest-x86_64` to `latest-arm64`.

## License

Educational use. See upstream projects (vLLM, Transformers, PyTorch) for their respective licenses.
