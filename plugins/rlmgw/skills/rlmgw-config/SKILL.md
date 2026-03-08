---
description: Configure RLMgw vLLM connection settings — set the upstream vLLM host, port, and context selection parameters.
disable-model-invocation: true
allowed-tools: Read, Edit, Bash(curl -s http://*)
---

# RLMgw Configuration

Help the user configure their vLLM and RLMgw settings.

## Where settings live

All RLMgw settings are environment variables stored in the `env` field of
**`.claude/settings.json`** (the project-level shared settings file).

**Important:** Do NOT put RLMgw env vars in `.claude/settings.local.json` —
the MCP server inherits env vars from the parent process, and only
`settings.json` env vars are reliably passed through. Use `settings.local.json`
only for non-env settings like permissions.

## Available settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RLMGW_UPSTREAM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL (OpenAI-compatible endpoint) |
| `RLMGW_REPO_ROOT` | `.` (current project) | Target repository to analyze |
| `RLMGW_MAX_CONTEXT_PACK_CHARS` | `12000` | Max characters in context pack output |
| `RLMGW_MAX_INTERNAL_CALLS` | `3` | Max RLM recursive iterations for context selection |
| `RLMGW_USE_RLM_CONTEXT_SELECTION` | `true` | Use intelligent RLM selection (`true`) or simple keywords (`false`) |

## Instructions

$ARGUMENTS

Based on the user's request:

### If they want to change the vLLM URL/port:
1. Read `.claude/settings.json`
2. Update `RLMGW_UPSTREAM_BASE_URL` in the `env` field to the new value
3. Write the updated file
4. Test the connection: `curl -s <new_url>/models | python3 -m json.tool`
5. Remind the user to restart Claude Code for changes to take effect

### If they want to see current config:
1. Read `.claude/settings.json`
2. Show the current env settings
3. Test vLLM connectivity: `curl -s <url>/models`

### If they want to change context selection settings:
1. Read `.claude/settings.json`
2. Update the relevant `env` field
3. Write the file back
4. Remind the user to restart Claude Code for changes to take effect

### If they want to point at a different repository:
1. Set `RLMGW_REPO_ROOT` in `.claude/settings.json` env field
2. Remind the user to restart Claude Code for changes to take effect

### Common configurations:

**Local vLLM (default):**
```json
"RLMGW_UPSTREAM_BASE_URL": "http://localhost:8000/v1"
```

**Remote vLLM on LAN:**
```json
"RLMGW_UPSTREAM_BASE_URL": "http://192.168.1.100:8000/v1"
```

**Faster but less accurate (reduce iterations):**
```json
"RLMGW_MAX_INTERNAL_CALLS": "1"
```

**More context in results:**
```json
"RLMGW_MAX_CONTEXT_PACK_CHARS": "24000"
```

**Disable RLM (keyword-only, no LLM needed):**
```json
"RLMGW_USE_RLM_CONTEXT_SELECTION": "false"
```

After changing any setting, remind the user to restart Claude Code (`/exit` and relaunch) for the MCP server to pick up the new environment variables.
