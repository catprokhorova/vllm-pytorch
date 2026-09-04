#!/usr/bin/env python3
"""Send test prompts to a local OpenAI-compatible server and measure latency."""

import argparse
import json
import sys
import time
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


def chat_completion(base_url: str, model: str, prompt: str, max_tokens: int) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {API_KEY}"
            },
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode())
    elapsed = time.perf_counter() - started

    content = body["choices"][0]["message"]["content"].strip()
    return content, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local model inference.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by the API")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response")
    args = parser.parse_args()

    print(f"Model:   {args.model}")
    print(f"API:     {args.base_url}")
    print(f"Prompts: {len(PROMPTS)}\n")

    total_time = 0.0
    for index, prompt in enumerate(PROMPTS, start=1):
        print(f"--- Prompt {index}/{len(PROMPTS)} ---")
        print(f"Input: {prompt}")

        try:
            content, elapsed = chat_completion(args.base_url, args.model, prompt, args.max_tokens)
        except urllib.error.URLError as exc:
            print(f"Request failed: {exc}", file=sys.stderr)
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
