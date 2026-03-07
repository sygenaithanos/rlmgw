---
description: Use RLM Gateway to intelligently explore and select relevant code context from large repositories. Powered by a local vLLM instance with recursive language model exploration.
allowed-tools: mcp__rlmgw__repo_select_context, mcp__rlmgw__repo_tree, mcp__rlmgw__repo_fingerprint, mcp__rlmgw__vllm_status, Read, Grep, Glob
---

# RLMgw — Intelligent Context Selection

You have access to **RLM Gateway** tools via MCP. These tools use a local LLM (served by vLLM) to recursively explore a codebase and select the most relevant files for a given query.

## When to use this skill

- When working with a **large codebase** where manual grep/read would be slow
- When you need to understand **how a feature works end-to-end** across many files
- When the user asks a **complex architectural question** about the codebase
- When you need a **curated context pack** rather than raw file contents

## How to use the tools

### Step 1: Check vLLM availability
```
Call vllm_status to verify the upstream LLM is running.
```
If vLLM is unavailable, the tools will fall back to simple keyword-based selection (still useful, just less intelligent).

### Step 2: Select context for a query
```
Call repo_select_context with the user's query.
```
This returns a JSON object with:
- `relevant_files` — list of file paths selected as most relevant
- `file_contents` — dict mapping file paths to their contents (truncated to fit)
- `repo_fingerprint` — git hash for cache invalidation
- `selection_mode` — "rlm" (intelligent) or "simple" (keyword fallback)
- `upstream_model` — the model used (dynamically discovered from vLLM)

### Step 3: Use the context
Read the returned `file_contents` to answer the user's question. If you need more detail on specific files, use the built-in `Read` tool to get the full contents.

### Optional: Get repo overview
```
Call repo_tree for a directory structure overview.
Call repo_fingerprint to check if the repo has changed since last query.
```

## Arguments

$ARGUMENTS

If the user provides a query after `/rlmgw`, use it as the `query` parameter for `repo_select_context`. For example:

- `/rlmgw how does authentication work` → call `repo_select_context("how does authentication work")`
- `/rlmgw find the database migration logic` → call `repo_select_context("find the database migration logic")`

If no arguments provided, call `vllm_status` first to show the current setup, then ask the user what they'd like to explore.

## Tips

- The RLM mode works best with **natural language queries** describing what you're looking for
- For simple lookups (e.g., "find function X"), prefer built-in `Grep` — it's faster
- Use `max_chars` parameter to control context size (default: 12000 chars)
- The model name is **auto-discovered** from your vLLM server — no need to configure it
