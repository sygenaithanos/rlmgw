#!/usr/bin/env bash
# Session start hook: check vLLM availability and report status
# This runs when Claude Code starts a new session

VLLM_URL="${RLMGW_UPSTREAM_BASE_URL:-http://localhost:8000/v1}"

# Try to discover the model from vLLM
RESPONSE=$(curl -s --connect-timeout 3 "${VLLM_URL}/models" 2>/dev/null)

if [ $? -eq 0 ] && echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null; then
    MODEL=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
    echo "RLMgw: vLLM is running at ${VLLM_URL} with model '${MODEL}'. Use /rlmgw to explore the codebase."
else
    echo "RLMgw: vLLM not detected at ${VLLM_URL}. Context selection will use keyword fallback. Use /rlmgw-config to update settings."
fi
