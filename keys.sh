#!/usr/bin/env bash
# Key management for the LiteLLM gateway.
# Usage: ./keys.sh <command> [args]
# Requires .env in this directory (LITELLM_MASTER_KEY, optional GATEWAY_URL).

set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

GATEWAY_URL="${GATEWAY_URL:-http://localhost:4000}"
AUTH="Authorization: Bearer ${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY in .env}"

cmd="${1:-help}"
shift || true

case "$cmd" in
  new)
    # Issue a virtual key: ./keys.sh new <key-alias> [rpm] [tpm]
    alias="${1:?usage: keys.sh new <alias> [rpm] [tpm]}"
    rpm="${2:-30}"
    tpm="${3:-20000}"
    curl -s "$GATEWAY_URL/key/generate" \
      -H "$AUTH" -H "Content-Type: application/json" \
      -d "{\"key_alias\":\"$alias\",\"models\":[\"chat-main\",\"chat-reserve\"],\"rpm_limit\":$rpm,\"tpm_limit\":$tpm}"
    ;;
  list)
    curl -s "$GATEWAY_URL/key/list" -H "$AUTH"
    ;;
  revoke)
    # ./keys.sh revoke <key-or-token>
    curl -s "$GATEWAY_URL/key/delete" -H "$AUTH" -H "Content-Type: application/json" \
      -d "{\"keys\":[\"${1:?usage: keys.sh revoke <key>}\"]}"
    ;;
  test)
    # Smoke checks: valid key -> 200, no key -> 401, tiny rpm -> 429
    KEY="${1:?usage: keys.sh test <virtual-key>}"
    echo "== valid key =="
    curl -s -o /dev/null -w "%{http_code}\n" "$GATEWAY_URL/v1/chat/completions" \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d '{"model":"chat-main","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
    echo "== no key =="
    curl -s -o /dev/null -w "%{http_code}\n" "$GATEWAY_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"chat-main","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
    ;;
  *)
    cat <<EOF
keys.sh — LiteLLM virtual key helper

  ./keys.sh new <alias> [rpm] [tpm]   issue a virtual key (default 30 rpm / 20000 tpm)
  ./keys.sh list                      list keys
  ./keys.sh revoke <key>              revoke a key
  ./keys.sh test <key>                check 200 with key / 401 without

Env: GATEWAY_URL (default http://localhost:4000), LITELLM_MASTER_KEY from .env
EOF
    ;;
esac
