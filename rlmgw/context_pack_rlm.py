"""RLM-based context pack builder for RLMgw."""

import json
import logging
import re

from .config import RLMgwConfig
from .models import ContextPack
from .repo_context import RepoContextCollector

logger = logging.getLogger(__name__)


# System prompt for context selection RLM
CONTEXT_SELECTION_PROMPT = """You are a code search agent. You MUST use REPL tools to explore a codebase.

CRITICAL: The `context` variable is just a task description string — it is NOT the codebase.
The codebase is accessed ONLY through the `repo` object in the REPL. You MUST write ```repl``` code blocks to use it.

Available repo tools (REPL only):
- repo.grep(pattern, extensions=None) -> Dict[str, List[str]]
- repo.list_files(extensions=None) -> List[str]
- repo.read_file(path) -> str
- repo.get_tree() -> Dict

MANDATORY WORKFLOW — you MUST follow these steps:
1. FIRST action: write a ```repl``` block to grep or list files. Example:
```repl
results = repo.grep("search_term")
```
2. Read promising source files with repo.read_file()
3. When done, call FINAL() with your selection:
FINAL({"relevant_files": ["path/to/file.rs"], "reasoning": "brief explanation"})

RULES:
- ALWAYS write ```repl``` code blocks — never just text responses
- All repo.* calls auto-print results, no need for print()
- Prioritize source code (.rs, .py, .ts, .go, .java) over docs/configs/tests
- Do NOT call FINAL() until you have explored with repo tools
- Do NOT inspect or analyze the `context` variable — use repo.* instead
- Do NOT ask clarifying questions — just search the codebase
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
    print(f"Found {{len(result)}} files (ext={{extensions}}). First 30: {{result[:30]}}")
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

        # This string becomes the `context` variable in the REPL.
        # The default user prompt (prompts.py) tells the model to "use the context
        # variable". The model will see this when it inspects `context`. We redirect
        # it to use repo.* tools instead.
        selection_query = f"""IGNORE this variable. Use repo.* tools in the REPL instead.
Find source files for: {query}"""

        # Let RLM explore and select context
        logger.info(f"Running RLM context selection for query: {query[:100]}")
        result = self.rlm.completion(selection_query, root_prompt=query)

        # find_final_answer() returns tuple (type, content) when FINAL() is used,
        # or _default_answer() returns a raw string when iterations exhaust
        response_text = result.response
        logger.debug(f"RLM raw response type={type(response_text).__name__}: {repr(str(response_text))[:500]}")
        if isinstance(response_text, tuple):
            response_text = response_text[1]  # Extract content from ("FINAL", content)

        # Parse RLM's selection
        try:
            selection_data = json.loads(response_text)
            relevant_files = selection_data.get("relevant_files", [])
            reasoning = selection_data.get("reasoning", "")

            logger.info(f"RLM selected {len(relevant_files)} files: {reasoning}")

            return self._build_context_pack(relevant_files)
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
                    return self._build_context_pack(relevant_files)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Last resort: extract file paths from REPL output/response text.
            # The RLM may have explored files but never produced FINAL() JSON.
            files_from_text = self._extract_file_paths_from_text(str(response_text))
            if files_from_text:
                logger.info(
                    f"RLM selected {len(files_from_text)} files (extracted paths from text)"
                )
                return self._build_context_pack(relevant_files=files_from_text)

            logger.warning("RLM didn't return valid JSON. Falling back to simple selection.")
            return self._build_simple(query)

    def _build_simple(self, query: str) -> ContextPack:
        """Fallback to simple keyword-based selection."""
        logger.info("Using simple keyword-based context selection")

        # Simple keyword extraction
        keywords = self._extract_keywords(query)
        relevant_files = self._find_relevant_files(keywords)

        return self._build_context_pack(relevant_files)

    def _extract_file_paths_from_text(self, text: str) -> list[str]:
        """Extract file paths from RLM response text as a last resort."""
        path_pattern = re.findall(r'[\w./-]+\.(?:rs|py|ts|tsx|js|go|java|cpp|c|h)', text)
        # Verify they exist in the repo
        valid_files = []
        seen = set()
        for path in path_pattern:
            if path not in seen and self.repo_collector.read_file_safe(path) is not None:
                valid_files.append(path)
                seen.add(path)
                if len(valid_files) >= 10:
                    break
        return valid_files

    def _build_context_pack(self, relevant_files: list[str]) -> ContextPack:
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
