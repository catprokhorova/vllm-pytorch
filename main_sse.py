#!/usr/bin/env python3
"""Benchmark client for OpenAI-compatible SSE (streaming) chat completions.

Use this against `transformers serve` (pt-docker-compose.yml), which always
returns text/event-stream regardless of the `stream` request field.
"""

import argparse
import json
import sys
import time
import traceback
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://89.169.165.97:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
API_KEY = "***"

PROMPTS = [
    "Explain what a binary search tree is in two sentences.",
    "Write a Python function that checks if a string is a palindrome.",
    "List three differences between TCP and UDP.",
    "Translate to French: 'The weather is nice today.'",
    "What is the capital of Japan? Answer in one word.",
]


def parse_sse_chat_completion(raw: str) -> str:
    """Assemble assistant text from OpenAI-style SSE chat.completion.chunk events."""
    parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue

        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid SSE JSON chunk: {payload[:500]}") from exc

        if "error" in chunk:
            raise RuntimeError(f"SSE error from server: {chunk['error']}")

        choices = chunk.get("choices") or []
        if not choices:
            continue

        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            parts.append(content)

    text = "".join(parts).strip()
    if not text:
        raise RuntimeError(
            "No content deltas found in SSE stream.\n"
            f"Raw body (truncated): {raw[:2000]}"
        )
    return text


def chat_completion(base_url: str, model: str, prompt: str, max_tokens: int) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} {exc.reason} from {request.full_url}\n"
            f"Response body: {error_body or '<empty>'}"
        ) from exc
    elapsed = time.perf_counter() - started

    if not raw.strip():
        raise RuntimeError(f"Empty response from {request.full_url}")

    content = parse_sse_chat_completion(raw)
    return content, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark local model inference over SSE streaming responses."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by the API")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response")
    args = parser.parse_args()

    print(f"Model:   {args.model}")
    print(f"API:     {args.base_url}")
    print(f"Mode:    SSE stream")
    print(f"Prompts: {len(PROMPTS)}\n")

    total_time = 0.0
    for index, prompt in enumerate(PROMPTS, start=1):
        print(f"--- Prompt {index}/{len(PROMPTS)} ---")
        print(f"Input: {prompt}")

        try:
            content, elapsed = chat_completion(args.base_url, args.model, prompt, args.max_tokens)
        except Exception:
            print(f"Request failed for prompt {index}/{len(PROMPTS)}:", file=sys.stderr)
            traceback.print_exc()
            return 1

        total_time += elapsed
        print(f"Time:  {elapsed:.2f}s")
        print(f"Reply: {content}\n")

    avg_time = total_time / len(PROMPTS)
    print("--- Summary ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time:   {avg_time:.2f}s per prompt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
