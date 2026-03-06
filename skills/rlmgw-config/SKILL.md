---
description: Configure RLMgw vLLM connection settings — set the upstream vLLM host, port, and context selection parameters.
disable-model-invocation: true
allowed-tools: Read, Edit, Bash(curl -s http://localhost:*)
---

# RLMgw Configuration

Help the user configure their vLLM and RLMgw settings.

## Configuration files

Settings are stored in two places:

1. **`.claude/settings.json`** (shared, committed) — team defaults
2. **`.claude/settings.local.json`** (personal, gitignored) — local overrides

The local file takes priority over the shared one.

## Available settings

All settings are environment variables in the `env` field:

| Variable | Default | Description |
|----------|---------|-------------|
| `RLMGW_UPSTREAM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL (OpenAI-compatible endpoint) |
| `RLMGW_REPO_ROOT` | `.` | Target repository to analyze |
| `RLMGW_MAX_CONTEXT_PACK_CHARS` | `12000` | Max characters in context pack output |
| `RLMGW_MAX_INTERNAL_CALLS` | `3` | Max RLM recursive iterations for context selection |
| `RLMGW_USE_RLM_CONTEXT_SELECTION` | `true` | Use intelligent RLM selection (`true`) or simple keywords (`false`) |

## Instructions

$ARGUMENTS

Based on the user's request:

### If they want to change the vLLM URL/port:
1. Read `.claude/settings.local.json`
2. Update `RLMGW_UPSTREAM_BASE_URL` to the new value (e.g., `http://192.168.1.100:8000/v1`)
3. Write the updated file
4. Test the connection: `curl -s <new_url>/models | python3 -m json.tool`

### If they want to see current config:
1. Read `.claude/settings.json` (shared defaults)
2. Read `.claude/settings.local.json` (local overrides)
3. Show the effective configuration (local overrides shared)
4. Test vLLM connectivity: `curl -s $RLMGW_UPSTREAM_BASE_URL/models`

### If they want to change context selection settings:
1. Read `.claude/settings.json`
2. Update the relevant `env` field
3. Write the file back

### If they want to point at a different repository:
1. Update `RLMGW_REPO_ROOT` in `.claude/settings.local.json`
2. The MCP server will pick up the new path on next tool call

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

After changing settings, remind the user to restart Claude Code for the MCP server to pick up the new environment variables.
