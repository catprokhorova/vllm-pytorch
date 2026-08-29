# vllm-pytorch

Educational project for comparing two ways to serve the same language model on **GPU** using Docker Compose:

- **[vLLM](https://github.com/vllm-project/vllm)** — optimized inference engine (`vllm-docker-compose.yml`)
- **PyTorch + Hugging Face Transformers** — `transformers serve` on top of PyTorch (`pt-docker-compose.yml`)

Both stacks expose an **OpenAI-compatible API** on port `8000`, so you can benchmark them with the same client.

> **Note:** This repository was created for **educational purposes** to explore and compare inference runtimes. It is not intended for production use.

## Model

Both compose files serve:

**`openai/gpt-oss-20b`**

OpenAI's 20B-parameter MoE reasoning model with native MXFP4 quantization (~16 GB VRAM). With `bfloat16` it requires ~48 GB VRAM.

## Requirements

- Docker and Docker Compose
- **NVIDIA GPU** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **VRAM:** 16 GB+ (MXFP4 / `dtype=auto`) or 48 GB+ (`bfloat16`)
- Hugging Face token: copy `.env.example` to `.env` and set `HF_TOKEN`

```bash
cp .env.example .env
```

Verify GPU access inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Project layout

```
vllm-docker-compose.yml   # vLLM GPU deployment
pt-docker-compose.yml     # PyTorch / Transformers GPU deployment
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
python main.py --base-url http://localhost:8000/v1 --model openai/gpt-oss-20b --max-tokens 256
```

## Comparing vLLM vs PyTorch

| | vLLM | PyTorch (Transformers) |
|---|---|---|
| Compose file | `vllm-docker-compose.yml` | `pt-docker-compose.yml` |
| Image | `vllm/vllm-openai:gptoss` | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| Server command | `vllm serve` | `transformers serve` |
| Typical GPU speed | Faster | Slower |
| First startup | Model loads at start | Installs deps + loads model |

Run **only one** stack at a time — both bind port `8000`.

## Compose files: main differences

Both files deploy the same model (`openai/gpt-oss-20b`) on GPU with an OpenAI-compatible API on port `8000`. The difference is *how* inference is run.

### Side-by-side

| Aspect | `vllm-docker-compose.yml` | `pt-docker-compose.yml` |
|---|---|---|
| **Runtime** | vLLM (dedicated inference engine) | PyTorch + Hugging Face `transformers serve` |
| **Docker image** | Pre-built `vllm/vllm-openai:gptoss` | Generic `pytorch/pytorch` CUDA runtime |
| **GPU access** | `gpus: all`, `ipc: host` | `gpus: all`, `ipc: host` |
| **Dependencies** | Bundled in the image | Installed at startup via `pip install` |
| **Startup time** | Faster after image pull | Slower (pip install on every start) |
| **Inference speed** | Optimized (PagedAttention, MXFP4 MoE kernels) | Baseline PyTorch; generally slower |
| **Config complexity** | Many tuning knobs (batching, memory, tool calling) | Minimal (device, dtype, continuous batching) |
| **Volumes** | `huggingface-cache` + `vllm-cache` | `huggingface-cache` only |
| **gpt-oss extras** | Tool calling parser, MXFP4 kernels | `openai-harmony`, `kernels` packages |

### What they share

- Same model: `openai/gpt-oss-20b`
- Same API: `POST /v1/chat/completions`
- Same port: `8000`
- Same Hugging Face cache: `./huggingface-cache` (model downloaded once, reused by both)
- Same benchmark client: `main.py` works with either stack without changes
- GPU required — NVIDIA Container Toolkit must be installed

### vLLM compose — key points

- Uses the **`gptoss`-tagged image** with native support for gpt-oss MXFP4 MoE and tool calling.
- **GPU tuning** via environment variables:
  - `GPU_MEMORY_UTILIZATION` — fraction of VRAM used (default `0.9`)
  - `TENSOR_PARALLEL_SIZE` — multi-GPU sharding (default `1`)
  - `MAX_MODEL_LEN` / `MAX_NUM_BATCHED_TOKENS` / `MAX_NUM_SEQS` — batching limits
  - `TOOL_CALL_PARSER` / `ENABLE_AUTO_TOOL_CHOICE` — gpt-oss tool calling
- On **non-Hopper GPUs** (RTX 4090, 3090), set `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1` in `.env`.
- Best choice for **lowest latency and highest throughput**.

### PyTorch compose — key points

- Uses **vanilla PyTorch + CUDA** with Hugging Face's `transformers serve` CLI.
- Installs `transformers[serving]`, `accelerate`, `kernels`, `triton`, and `openai-harmony` at startup.
- `DEVICE=cuda:0` runs inference on the first GPU; `DTYPE=auto` uses the model's native MXFP4 weights.
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

- `MODEL` — Hugging Face model ID (`openai/gpt-oss-20b`)
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` — Hugging Face access token (required)
- `DTYPE` — `auto` (MXFP4, ~16 GB VRAM) or `bfloat16` (~48 GB VRAM)
- vLLM-specific: `GPU_MEMORY_UTILIZATION`, `TENSOR_PARALLEL_SIZE`, `VLLM_ATTENTION_BACKEND`
- PyTorch-specific: `DEVICE`, `CONTINUOUS_BATCHING`, `PYTORCH_IMAGE_TAG`

### Multi-GPU

Set `TENSOR_PARALLEL_SIZE=2` (or more) in `.env` for vLLM when a single GPU does not have enough VRAM.

## License

Educational use. See upstream projects (vLLM, Transformers, PyTorch, OpenAI gpt-oss) for their respective licenses.
