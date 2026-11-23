# vllm-pytorch

Industrial LLM serving stack for lecture "21. Промышленное обслуживание, развёртывание и API":

```
client ──► LiteLLM gateway :4000 ──► vLLM :8000 (GPU, chat-main)
                │                        ▲
                └──► Ollama :11434 ──────┘ fallback (CPU, chat-reserve)
                │
                └──► Postgres (virtual keys, quotas, spend)
```

Only **one public port: 4000** — the gateway. vLLM has no `ports:` section;
clients talk to aliases (`chat-main`, `chat-reserve`), never to model weights.

## Hardware

- Single GPU, **~4 GB VRAM**, Linux + NVIDIA Container Toolkit
- Model: `Qwen/Qwen2.5-1.5B-Instruct-AWQ` (AWQ 4-bit, ~1.1 GB weights)
- Reserve model runs on **CPU** — two GPU engines do not fit in 4 GB

## Layout

```
.env.example          # versions + secrets template (copy to .env, never commit)
.env                  # your copy (gitignored)
.env.previous         # last known-good config for rollback (gitignored)
compose.yaml          # vllm, litellm, db, ollama — all versions pinned
litellm.yaml          # gateway config: models, fallbacks, master key
keys.sh               # issue/list/revoke virtual keys, smoke-test 401/429
main.py               # benchmark client (non-streaming)
```

## Quick start

```bash
cp .env.example .env        # fill in secrets, keep pinned tags

docker compose up -d        # vllm downloads ~1.1GB weights on first run

docker compose exec ollama ollama pull qwen2.5:0.5b   # reserve model, once

# wait until vllm is healthy (model load takes a few minutes)
docker compose ps
```

Issue a virtual key and test the gateway:

```bash
./keys.sh new demo-user 30 20000     # prints a virtual key
export API_KEY=sk-...                # the virtual key
./keys.sh test "$API_KEY"            # 200 with key, 401 without
python main.py                       # 5 prompts through the gateway
```

## Architecture notes

- **Aliases, not models.** Clients request `chat-main`; the real model can be
  swapped by editing `litellm.yaml` + `.env` without touching clients.
- **Keys and quotas.** Every consumer gets its own virtual key via `keys.sh`
  (`rpm_limit`, `tpm_limit`, optional `budget`). One noisy client gets 429,
  others are unaffected. Logs attribute spend per key via Postgres.
- **Fallback.** `router_settings.fallbacks` sends `chat-main` traffic to
  `chat-reserve` (Ollama on CPU) after `num_retries`. A dead vLLM means slower
  answers, not errors. Streaming responses interrupted mid-flight are NOT
  resumed — clients must retry.
- **Only the gateway is public.** vLLM requires `VLLM_API_KEY` anyway, but its
  port is unreachable from outside the compose network.

## Updates and rollback

Everything is pinned: `VLLM_TAG`, `LITELLM_TAG`, `OLLAMA_TAG`, `MODEL_REVISION`.
No `latest` anywhere.

Before any change:

```bash
cp .env .env.previous           # remember last working config
docker compose pull             # fetch new images first if tags changed
```

Rollback:

```bash
cp .env.previous .env
docker compose up -d vllm       # recreate only what changed
```

Notes:

- `hf-cache` volume keeps old weights — do NOT prune it right after an update,
  rollback needs the previous revision on disk.
- vLLM restart is not instant: weights reload into GPU memory (minutes).
  That window is what the Ollama fallback covers.
- Agree on rollback criteria BEFORE shipping: e.g. error rate or p95 latency
  above a threshold, or quality regression on a reference prompt set.
- Rehearse the rollback once — an untested rollback fails when needed.

## Metrics and tuning (vLLM)

vLLM exposes Prometheus metrics on `:8000/metrics` (internal network only —
`docker compose exec` or publish temporarily for debugging). Watch three things:

- KV-cache usage near 100% + growing queue → memory-bound: lower
  `--max-model-len`, quantize harder, or raise `--gpu-memory-utilization`
  (leave headroom for activation spikes).
- Queue growing with empty cache → check `--max-num-seqs`.
- Low prefix-cache hit rate → something variable sits at the START of your
  prompts (timestamp, user id). Static content first, variable last.

## Mac / no-NVIDIA fallback

vLLM requires Linux + NVIDIA GPU. On macOS run Ollama (or llama-server / LM
Studio) on the host instead and drop the `vllm` service:

- In `litellm.yaml` point `chat-main` at `http://host.docker.internal:11434/v1`
  with `model: openai/<your-ollama-model>` and any non-empty `api_key`.
- Remove the `vllm` service and its `depends_on` from `compose.yaml`.
- Everything else (gateway, keys, quotas, fallbacks) works identically —
  throughput numbers will not match a GPU host, functionality will.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `litellm` never starts | `vllm` not healthy yet — model still loading (check `docker compose logs vllm`) |
| 401 on every request | Missing/expired virtual key, or wrong `API_KEY` env in client |
| 429 under load | Key's `rpm_limit`/`tpm_limit` hit — raise via `keys.sh` or issue a new key |
| Fallback never triggers | `ollama` model not pulled yet, or `RESERVE_MODEL` ≠ name in `litellm.yaml` |
| OOM in vLLM | Lower `--max-model-len` / `--gpu-memory-utilization`, or pick a smaller model |

## License

Educational use. See upstream projects (vLLM, LiteLLM, Ollama, Qwen) for their
respective licenses.
