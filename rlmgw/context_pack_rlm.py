"""RLM-based context pack builder for RLMgw."""

import json
import logging
import re

from .config import RLMgwConfig
from .models import ContextPack
from .repo_context import RepoContextCollector

logger = logging.getLogger(__name__)


# System prompt for context selection RLM
CONTEXT_SELECTION_PROMPT = """You are a context selection assistant. Your task is to find the MOST RELEVANT source code files for a user's query.

You have access to repo tools in your REPL. You MUST use print() to see results — bare expressions are truncated.

Available tools:
- repo.grep(pattern, extensions=None) -> Dict[str, List[str]]  # Search file contents
- repo.list_files(extensions=None) -> List[str]  # List files by extension
- repo.read_file(path) -> str  # Read a file
- repo.get_tree() -> Dict  # Directory structure

IMPORTANT RULES:
1. Always wrap tool calls in print() so you can see the output
2. Search broadly — the codebase may use ANY language (.py, .rs, .ts, .go, .java, .cpp, etc.)
3. If a grep returns empty for one extension, try without extensions or with different ones
4. PRIORITIZE source code over docs, configs, tests, or generated output
5. Do NOT reference or use the `context` variable — use repo.* tools instead
6. When done, call FINAL() with your answer — not just bare JSON

Example workflow:
```repl
# First, find what languages are in the repo
files = repo.list_files()
print(f"Total files: {len(files)}, first 20: {files[:20]}")
```
```repl
# Search for the key term
results = repo.grep("my_function")
print(results)
```
```repl
# Read a promising file
content = repo.read_file("src/module.rs")
print(content[:2000])
```

When you have identified the relevant files, provide your answer:
FINAL({"relevant_files": ["path/to/file1.rs", "path/to/file2.py"], "reasoning": "Brief explanation"})

Do NOT use FINAL() until you have explored the codebase. Keep selections COMPACT and HIGH-SIGNAL.
"""


class RLMContextPackBuilder:
    """Builds context packs using RLM-based intelligent selection."""

    def __init__(self, repo_collector: RepoContextCollector, config: RLMgwConfig):
        self.repo_collector = repo_collector
        self.config = config
        self.max_chars = config.max_context_pack_chars
        self.max_internal_calls = config.max_internal_calls

        # We'll initialize RLM when needed to avoid import issues
        self.rlm = None

    def _initialize_rlm(self):
        """Initialize RLM with context selection configuration."""
        if self.rlm is not None:
            return

        try:
            from rlm import RLM

            # Build setup code that creates auto-printing repo tool wrappers.
            # The REPL's format_execution_result() only shows variable NAMES, not values.
            # Without print(), the model cannot see any tool output. Auto-printing
            # ensures results are always visible regardless of whether the model uses print().
            #
            # CONSTRAINTS from LocalREPL sandbox:
            # 1. Variables starting with '_' are filtered out between exec() calls
            # 2. __build_class__ is not in _SAFE_BUILTINS, so 'class' statements fail
            # 3. Must use a SimpleNamespace or dict wrapper instead of a class
            repo_root_escaped = str(self.repo_collector.repo_root).replace("'", "\\'")
            setup_code = f"""
import sys
from types import SimpleNamespace
sys.path.insert(0, '{repo_root_escaped}')

from rlmgw.repo_env import RepoContextTools
base_repo = RepoContextTools('{repo_root_escaped}')

def repo_list_files(extensions=None):
    result = base_repo.list_files(extensions)
    print(f"Found {{len(result)}} files. First 30: {{result[:30]}}")
    return result

def repo_grep(pattern, extensions=None):
    result = base_repo.grep(pattern, extensions)
    print(f"grep '{{pattern}}' matched {{len(result)}} files: {{dict(list(result.items())[:15])}}")
    return result

def repo_read_file(path):
    result = base_repo.read_file(path)
    if result:
        print(f"=== {{path}} ({{len(result)}} chars) ===")
        print(result[:3000])
    else:
        print(f"File not found: {{path}}")
    return result

def repo_get_tree():
    result = base_repo.get_tree()
    print(f"Repo tree (top-level keys): {{list(result.keys())}}")
    return result

repo = SimpleNamespace(
    list_files=repo_list_files,
    grep=repo_grep,
    read_file=repo_read_file,
    get_tree=repo_get_tree,
)
"""

            # Create RLM instance for context selection
            # Use the same upstream as configured
            self.rlm = RLM(
                backend="openai",  # Will use vLLM through OpenAI-compatible API
                backend_kwargs={
                    "model_name": self.config.upstream_model,
                    "base_url": self.config.upstream_base_url,
                    "api_key": "dummy",  # vLLM doesn't require API key
                },
                environment="local",  # Use local REPL for speed
                environment_kwargs={
                    "setup_code": setup_code,
                },
                max_depth=1,
                max_iterations=self.max_internal_calls,
                custom_system_prompt=CONTEXT_SELECTION_PROMPT,
                verbose=False,
            )

            logger.info("Initialized RLM for context selection")
        except Exception as e:
            logger.warning(f"Failed to initialize RLM: {e}. Falling back to simple selection.")
            self.rlm = None

    def build_from_query(self, query: str) -> ContextPack:
        """Build context pack using RLM-based intelligent selection."""

        # Try RLM-based selection first
        try:
            return self._build_with_rlm(query)
        except Exception as e:
            logger.warning(f"RLM-based selection failed: {e}. Falling back to simple selection.")
            return self._build_simple(query)

    def _build_with_rlm(self, query: str) -> ContextPack:
        """Use RLM to intelligently select context."""
        self._initialize_rlm()

        if self.rlm is None:
            raise Exception("RLM not available")

        # Create context selection query — this string becomes the `context` variable
        # in the REPL. The RLM's default user prompt tells the model to "look at the
        # context variable", so we put redirect instructions here.
        selection_query = f"""=== CONTEXT SELECTION TASK ===
User Query: {query}

INSTRUCTIONS (this IS the context — there is no other data to inspect):
1. Use repo.grep() to search for key terms. Always use print() to see results.
2. The codebase may be in ANY language (.rs, .py, .ts, .go, .java, etc.) — search broadly.
3. If grep returns empty, try different search terms or drop the extension filter.
4. Use repo.read_file() to verify promising files.
5. When done, call FINAL() with JSON: FINAL({{"relevant_files": [...], "reasoning": "..."}})
6. Do NOT ask clarifying questions — just find the files.

Example:
```repl
results = repo.grep("generate_rag_response")
print(results)
```
"""

        # Let RLM explore and select context
        logger.info(f"Running RLM context selection for query: {query[:100]}")
        result = self.rlm.completion(selection_query, root_prompt=query)

        # find_final_answer() returns tuple (type, content) when FINAL() is used,
        # or _default_answer() returns a raw string when iterations exhaust
        response_text = result.response
        if isinstance(response_text, tuple):
            response_text = response_text[1]  # Extract content from ("FINAL", content)

        # Parse RLM's selection
        try:
            selection_data = json.loads(response_text)
            relevant_files = selection_data.get("relevant_files", [])
            reasoning = selection_data.get("reasoning", "")

            logger.info(f"RLM selected {len(relevant_files)} files: {reasoning}")

            return self._build_context_pack(query, relevant_files)
        except (json.JSONDecodeError, TypeError):
            # Try to extract JSON from surrounding prose
            json_match = re.search(
                r'\{[^{}]*"relevant_files"\s*:\s*\[.*?\][^{}]*\}',
                str(response_text),
                re.DOTALL,
            )
            if json_match:
                try:
                    selection_data = json.loads(json_match.group())
                    relevant_files = selection_data.get("relevant_files", [])
                    reasoning = selection_data.get("reasoning", "")
                    logger.info(
                        f"RLM selected {len(relevant_files)} files (extracted from text): {reasoning}"
                    )
                    return self._build_context_pack(query, relevant_files)
                except (json.JSONDecodeError, TypeError):
                    pass
            logger.warning("RLM didn't return valid JSON. Falling back to simple selection.")
            return self._build_simple(query)

    def _build_simple(self, query: str) -> ContextPack:
        """Fallback to simple keyword-based selection."""
        logger.info("Using simple keyword-based context selection")

        # Simple keyword extraction
        keywords = self._extract_keywords(query)
        relevant_files = self._find_relevant_files(keywords)

        return self._build_context_pack(query, relevant_files)

    def _build_context_pack(self, query: str, relevant_files: list[str]) -> ContextPack:
        """Build context pack from selected files."""
        repo_fingerprint = self.repo_collector.get_repo_fingerprint()

        # Read relevant files with size limits
        file_contents = {}
        total_chars = 0

        for file_path in relevant_files:
            content = self.repo_collector.read_file_safe(file_path)
            if content:
                # Truncate to fit within max_chars limit
                remaining_chars = self.max_chars - total_chars
                if remaining_chars > 0:
                    truncated_content = self._truncate_content(content, remaining_chars)
                    file_contents[file_path] = truncated_content
                    total_chars += len(truncated_content)
                else:
                    break

        # Build context pack
        context_pack = ContextPack(
            repo_fingerprint=repo_fingerprint,
            relevant_files=relevant_files,
            file_contents=file_contents,
            symbols=[],
            constraints=[],
            risks=[],
            suggested_actions=[],
        )

        logger.info(f"Built context pack with {len(file_contents)} files, {total_chars} chars")
        return context_pack

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to max length."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "... (truncated)"

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query (simple fallback)."""
        words = query.lower().split()
        common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "and", "or"}
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        return keywords

    def _find_relevant_files(self, keywords: list[str]) -> list[str]:
        """Find files relevant to keywords (simple fallback)."""
        if not keywords:
            return []

        relevant_files = set()

        for keyword in keywords[:5]:  # Limit to first 5 keywords
            grep_results = self.repo_collector.grep_repo(keyword)
            relevant_files.update(grep_results.keys())
            if len(relevant_files) >= 20:  # Limit total files
                break

        # Also include common project files
        common_files = ["README.md", "pyproject.toml", "setup.py", "requirements.txt"]
        for file in common_files:
            if (self.repo_collector.repo_root / file).exists():
                relevant_files.add(file)

        return list(relevant_files)[:20]  # Limit to 20 files
