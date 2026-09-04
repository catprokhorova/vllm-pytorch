# vllm-pytorch

Educational project for comparing two ways to serve the same language model on **GPU** using Docker Compose:

- **[vLLM](https://github.com/vllm-project/vllm)** — optimized inference engine (`vllm-docker-compose.yml`)
- **PyTorch + Hugging Face Transformers** — `transformers serve` on top of PyTorch (`pt-docker-compose.yml`)

Both stacks expose an **OpenAI-compatible API** on port `8000`, so you can benchmark them with the same client (`main.py`).

> **Note:** This repository was created for **educational purposes** to explore and compare inference runtimes. It is not intended for production use.

## Model

Both stacks are configured for:

**`Qwen/Qwen2.5-14B-Instruct-AWQ`**

Qwen2.5 14B Instruct, AWQ 4-bit quantized. Fits on a single **16 GB GPU** (e.g. NVIDIA T4) with `dtype=auto` / half precision.

## Requirements

- Docker and Docker Compose
- **NVIDIA GPU** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **VRAM:** ~16 GB+ for the default AWQ model
- Hugging Face token (recommended for higher rate limits): copy `.env.example` to `.env` and set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`

```bash
cp .env.example .env
```

Verify GPU access inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Project layout

```
vllm-docker-compose.yml   # vLLM GPU deployment (AWQ)
pt-docker-compose.yml     # PyTorch / Transformers GPU deployment
.env.example              # Environment variable template (copy to .env)
main.py                   # Benchmark script (5 prompts + timing)
model_cache/              # vLLM Hugging Face cache (created on first run)
huggingface-cache/        # PyTorch Hugging Face cache (created on first run)
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

Wait until the server is ready (model download and load can take several minutes on first run). Run **only one** stack at a time — both bind port `8000`.

### 2. Run the benchmark

In another terminal:

```bash
python main.py
```

The script sends 5 prompts to the OpenAI-compatible `/v1/chat/completions` endpoint and prints per-prompt latency plus total/average time.

Optional flags:

```bash
python main.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-14B-Instruct-AWQ --max-tokens 256
```

## Comparing vLLM vs PyTorch

| | vLLM | PyTorch (Transformers) |
|---|---|---|
| Compose file | `vllm-docker-compose.yml` | `pt-docker-compose.yml` |
| Image | `vllm/vllm-openai:v0.28.0` | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| Server command | `vllm serve` (via image entrypoint) | `transformers serve` |
| Quantization | AWQ (`--quantization awq`) | AWQ via model weights + `gptqmodel` |
| Typical GPU speed | Faster | Slower |
| First startup | Model loads at start | Installs deps + loads model |

## Compose files: main differences

Both deploy the same AWQ model on GPU with an OpenAI-compatible API on port `8000`. The difference is *how* inference is run.

### Side-by-side

| Aspect | `vllm-docker-compose.yml` | `pt-docker-compose.yml` |
|---|---|---|
| **Runtime** | vLLM (dedicated inference engine) | PyTorch + Hugging Face `transformers serve` |
| **Docker image** | Pre-built `vllm/vllm-openai` | Generic `pytorch/pytorch` CUDA runtime |
| **GPU access** | NVIDIA runtime + device reservation | `gpus: all`, `ipc: host` |
| **Dependencies** | Bundled in the image | Installed at startup via `pip install` |
| **Startup time** | Faster after image pull | Slower (pip install on every start) |
| **Inference speed** | Optimized (PagedAttention, AWQ kernels) | Baseline PyTorch; generally slower |
| **Config complexity** | Memory, context length, KV cache, API key | Device, dtype, continuous batching |
| **Volumes** | `./model_cache` | `./huggingface-cache` |

### What they share

- Same default model: `Qwen/Qwen2.5-14B-Instruct-AWQ`
- Same API shape: `POST /v1/chat/completions`
- Same port: `8000`
- Same benchmark client: `main.py` works with either stack
- GPU required — NVIDIA Container Toolkit must be installed

### vLLM compose — key points

- Image `vllm/vllm-openai:v0.28.0` with AWQ enabled (`--quantization awq`, `--dtype half`).
- **GPU / serving knobs** via `.env`:
  - `MODEL_NAME` — Hugging Face model ID
  - `GPU_UTILIZE` — fraction of VRAM used (default `0.9`)
  - `MODEL_NUM_CTX` — max model length / batched tokens (default `4096`)
  - `KV_CACHE_DTYPE` — KV cache dtype (default `auto`)
  - `API_KEY` — required by the vLLM OpenAI server
- Best choice for **lowest latency and highest throughput**.

### PyTorch compose — key points

- Uses **vanilla PyTorch + CUDA** with Hugging Face's `transformers serve` CLI.
- Installs `transformers[serving]`, `accelerate`, `kernels`, `triton`, `openai-harmony`, and **`gptqmodel`** (required to load AWQ models) at startup.
- `DEVICE=cuda:0` runs inference on the first GPU; `DTYPE=auto` follows the model's AWQ weights.
- Best choice to see **how inference works at the framework level** without an optimized serving layer.

### When to use which

| Goal | Recommended stack |
|---|---|
| Compare inference performance | Run both with `main.py` and compare timings |
| Learn how LLM serving is optimized | vLLM |
| Learn PyTorch / Transformers basics | PyTorch compose |
| Fastest responses on GPU | vLLM |
| Simplest mental model | PyTorch compose |

## Configuration

Copy `.env.example` to `.env` and edit values. Docker Compose loads `.env` automatically.

Key variables:

- `MODEL` / `MODEL_NAME` — Hugging Face model ID (`Qwen/Qwen2.5-14B-Instruct-AWQ`)
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` — Hugging Face access token (recommended)
- `DTYPE` — `auto` for AWQ (PyTorch); vLLM uses `half` in the compose file
- vLLM-specific: `API_KEY`, `GPU_UTILIZE`, `MODEL_NUM_CTX`, `KV_CACHE_DTYPE`
- PyTorch-specific: `DEVICE`, `CONTINUOUS_BATCHING`, `PYTORCH_IMAGE_TAG`, `TRUST_REMOTE_CODE`

## License

Educational use. See upstream projects (vLLM, Transformers, PyTorch, Qwen) for their respective licenses.
