#!/usr/bin/env bash
# Session start hook: check vLLM availability and report status

VLLM_URL="${RLMGW_UPSTREAM_BASE_URL:-http://192.168.2.37:8000/v1}"

RESPONSE=$(curl -s --connect-timeout 3 "${VLLM_URL}/models" 2>/dev/null)

if [ $? -eq 0 ]; then
    MODEL=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
    if [ -n "$MODEL" ]; then
        echo "RLMgw: vLLM running at ${VLLM_URL} with model '${MODEL}'. Use /rlmgw:rlmgw to explore the codebase."
        exit 0
    fi
fi

echo "RLMgw: vLLM not detected at ${VLLM_URL}. Context selection will use keyword fallback. Use /rlmgw:rlmgw-config to configure."
