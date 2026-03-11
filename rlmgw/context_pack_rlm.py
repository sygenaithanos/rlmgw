"""RLM-based context pack builder for RLMgw."""

import json
import logging
import re

from .config import RLMgwConfig
from .models import ContextPack
from .repo_context import RepoContextCollector

logger = logging.getLogger(__name__)


# System prompt for context selection RLM — includes few-shot examples
# to teach the model the exact workflow expected.
CONTEXT_SELECTION_PROMPT = """You are a code search agent. You explore a codebase using REPL tools and select the most relevant source files.

The `context` variable contains the task description. Read it, then use `repo.*` tools to find files.

Available tools (use in ```repl``` code blocks):
- repo.grep(pattern) -> Dict[str, List[str]]
- repo.list_files() -> List[str]
- repo.read_file(path) -> str
- repo.get_tree() -> Dict

RULES:
- ALWAYS respond with ```repl``` code blocks
- Prioritize source code (.rs, .py, .ts, .go) over docs/configs — do NOT return .md or .json files
- Use file paths from repo.grep() and repo.list_files() output — do not invent paths
- After grep finds files, read them to verify, then call FINAL()
- When done, call FINAL() with JSON

=== EXAMPLE 1: Specific function search ===

User query: "Find the authentication middleware"

Turn 1:
```repl
results = repo.grep("auth")
```
Output: grep 'auth' matched 4 files: ['src/middleware/auth.py', 'src/models/user.py', 'tests/test_auth.py', 'config/auth.yaml']
  src/middleware/auth.py: ['def authenticate(request):']
  src/models/user.py: ['class AuthUser:']
Next step: use repo.read_file() on these files, then call FINAL().

Turn 2:
```repl
content = repo.read_file("src/middleware/auth.py")
```
Output: === src/middleware/auth.py (800 chars) === def authenticate(request): ...

Turn 3:
FINAL({"relevant_files": ["src/middleware/auth.py", "src/models/user.py"], "reasoning": "Auth middleware and user model"})

=== EXAMPLE 2: Architecture question (workspace repo) ===

User query: "How is the project structured?"

Turn 1:
```repl
tree = repo.get_tree()
```
Output: Repo tree (5 top-level entries):
  Cargo.toml
  crates/: ['app-auth', 'app-core', 'app-db']
  scripts/: ['deploy.sh']
  tests/: ['integration']

Turn 2:
```repl
files = repo.list_files()
```
Output: Found 200 files. First 50: ['crates/app-core/src/lib.rs', 'crates/app-core/src/config.rs', 'crates/app-auth/src/lib.rs', 'crates/app-auth/src/middleware.rs', 'crates/app-db/src/lib.rs', ...]

Use EXACT paths from list_files() output. Do NOT guess paths like 'src/lib.rs'.

Turn 3:
```repl
content = repo.read_file("crates/app-core/src/lib.rs")
```
Output: === crates/app-core/src/lib.rs (500 chars) === pub mod config; pub mod db; ...

Turn 4:
```repl
content2 = repo.read_file("crates/app-auth/src/lib.rs")
```
Output: === crates/app-auth/src/lib.rs (300 chars) === pub mod middleware; pub mod jwt; ...

Turn 5:
FINAL({"relevant_files": ["crates/app-core/src/lib.rs", "crates/app-auth/src/lib.rs", "crates/app-db/src/lib.rs", "Cargo.toml"], "reasoning": "Entry point lib.rs from each crate shows module structure; Cargo.toml defines workspace"})
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
    print(f"Found {{len(result)}} files. First 50: {{result[:50]}}")
    return result

def repo_grep(pattern, extensions=None):
    result = base_repo.grep(pattern, extensions)
    file_list = list(result.keys())[:15]
    print(f"grep '{{pattern}}' matched {{len(result)}} files: {{file_list}}")
    for f, lines in list(result.items())[:5]:
        print(f"  {{f}}: {{lines[:2]}}")
    if result:
        print("Next step: use repo.read_file() on these files, then call FINAL().")
    return result

def repo_read_file(path):
    result = base_repo.read_file(path)
    if result:
        print(f"=== {{path}} ({{len(result)}} chars) ===")
        print(result[:3000])
    else:
        print(f"ERROR: File not found: {{path}}")
        # Show actual files that partially match to help the model recover
        partial = path.rsplit("/", 1)[-1]
        matches = [f for f in base_repo.list_files() if partial in f][:5]
        if matches:
            print(f"Did you mean one of: {{matches}}")
        else:
            print("Use repo.list_files() or repo.get_tree() to see actual paths.")
    return result

def repo_get_tree():
    result = base_repo.get_tree()
    summary = []
    for key in sorted(result.keys()):
        val = result[key]
        if isinstance(val, dict) and val:
            children = sorted(val.keys())[:15]
            summary.append(f"  {{key}}/: {{children}}")
        else:
            summary.append(f"  {{key}}")
    tree_str = "\\n".join(summary)
    print(f"Repo tree ({{len(result)}} top-level entries):\\n{{tree_str}}")
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
            # Sampling params tuned for reliable instruction following:
            # - temperature=0.02: near-deterministic for format compliance
            # - top_p=0.95, top_k=40: Qwen3-Coder-Next recommended settings
            # - repetition_penalty=1.05: prevents degenerate loops
            # - max_tokens=4096: enough room for reasoning + code blocks
            self.rlm = RLM(
                backend="openai",  # Will use vLLM through OpenAI-compatible API
                backend_kwargs={
                    "model_name": self.config.upstream_model,
                    "base_url": self.config.upstream_base_url,
                    "api_key": "dummy",  # vLLM doesn't require API key
                    "temperature": 0.02,
                    "top_p": 0.95,
                    "top_k": 40,
                    "repetition_penalty": 1.05,
                    "max_tokens": 4096,
                },
                environment="local",  # Use local REPL for speed
                environment_kwargs={
                    "setup_code": setup_code,
                },
                max_depth=1,
                max_iterations=self.max_internal_calls,
                custom_system_prompt=CONTEXT_SELECTION_PROMPT,
                # Override the default user prompt which says "use the context variable"
                # — that conflicts with our system prompt. Keep the structural cues
                # (step-by-step, don't FINAL yet) but reinforce repo.* tools.
                custom_user_prompt=(
                    'Find source files for: "{root_prompt}"\n'
                    "Iteration {iteration}. "
                    "Step 1: grep for key terms. "
                    "Step 2: read found files with repo.read_file(). "
                    "Step 3: call FINAL() with the files. "
                    "Do NOT repeat a grep you already ran. "
                    "Do NOT inspect the `context` variable. "
                    "Your next ```repl``` code block:"
                ),
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
        # The default user prompt tells the model to "use the context variable" — and
        # it WILL inspect it. So we pre-populate it with useful data: the query + repo
        # tree summary. This way, when the model does print(context), it gets useful
        # data that guides its next grep/read_file calls.
        try:
            tree = self.repo_collector.get_repo_tree()
            tree_summary = ", ".join(sorted(tree.keys())[:30])
        except Exception:
            tree_summary = "(tree unavailable)"

        selection_query = f"""=== TASK: Find source files relevant to this query ===
QUERY: {query}

REPO STRUCTURE (top-level): {tree_summary}

INSTRUCTIONS:
1. Use repo.grep("pattern") to search for key terms from the query
2. Use repo.read_file("path") to verify promising files
3. When done, call FINAL({{"relevant_files": ["path/to/file"], "reasoning": "..."}})

Available tools: repo.grep(pattern), repo.list_files(), repo.read_file(path), repo.get_tree()
Prioritize source code (.rs, .py, .ts, .go) over docs/configs/tests.
"""

        # Let RLM explore and select context
        logger.info(f"Running RLM context selection for query: {query[:100]}")
        result = self.rlm.completion(selection_query, root_prompt=query)

        # find_final_answer() returns tuple (type, content) when FINAL() is used,
        # or _default_answer() returns a raw string when iterations exhaust
        response_text = result.response
        logger.debug(
            f"RLM raw response type={type(response_text).__name__}: {repr(str(response_text))[:500]}"
        )
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
            files_from_text = self._extract_file_paths_from_text(str(response_text), query)
            if files_from_text:
                # If text extraction found very few files, supplement with keyword search
                # to ensure we don't miss the primary target file
                if len(files_from_text) < 3:
                    simple_files = self._find_relevant_files(self._extract_keywords(query))
                    seen = set(files_from_text)
                    for f in simple_files:
                        if f not in seen:
                            files_from_text.append(f)
                            seen.add(f)
                            if len(files_from_text) >= 10:
                                break
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

    def _extract_file_paths_from_text(self, text: str, query: str = "") -> list[str]:
        """Extract file paths from RLM response text as a last resort.

        Prioritizes files whose names match query terms for better relevance.
        """
        path_pattern = re.findall(r"[\w./-]+\.(?:rs|py|ts|tsx|js|go|java|cpp|c|h)", text)
        # Verify they exist in the repo
        valid_files = []
        seen = set()
        for path in path_pattern:
            if path not in seen and self.repo_collector.read_file_safe(path) is not None:
                valid_files.append(path)
                seen.add(path)
                if len(valid_files) >= 10:
                    break
        # Sort: files whose name matches query terms come first
        if query and valid_files:
            query_terms = {w.lower() for w in query.split() if len(w) > 2}
            valid_files.sort(key=lambda f: -sum(1 for t in query_terms if t in f.lower()))
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
        """Extract keywords from query (simple fallback).

        Generates compound terms (underscore-joined adjacent words) to match
        identifiers like 'output_parser' from queries like 'output parser'.
        Sorts by length descending so the most specific terms are searched first.
        """
        words = query.lower().split()
        common_words = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "and",
            "or",
            "how",
            "does",
            "what",
            "is",
            "are",
            "this",
            "that",
            "find",
            "show",
            "where",
            "which",
            "why",
            "can",
            "overall",
        }
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        # Generate underscore-joined adjacent pairs to match compound identifiers
        compounds = []
        for i in range(len(keywords) - 1):
            compounds.append(f"{keywords[i]}_{keywords[i + 1]}")
        # Merge and sort by length descending (most specific first)
        all_terms = compounds + keywords
        all_terms.sort(key=len, reverse=True)
        # Deduplicate while preserving order
        seen = set()
        result = []
        for t in all_terms:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def _find_relevant_files(self, keywords: list[str]) -> list[str]:
        """Find files relevant to keywords (simple fallback).

        Searches both file contents (grep) and file names for keyword matches.
        Preserves keyword specificity order: results from the most specific
        keyword come first. Prioritizes source code over docs/tests.
        """
        if not keywords:
            return []

        relevant_files = []
        seen = set()

        for keyword in keywords[:3]:  # Top 3 most specific keywords
            # Search file names first — catches files like output_parser.rs
            # where the filename matches but the content doesn't contain the term
            file_list = self.repo_collector.get_file_list()
            for path in file_list:
                if path not in seen:
                    basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                    if keyword.lower() in basename.lower():
                        seen.add(path)
                        relevant_files.insert(0, path)  # Name matches first

            # Then search file contents
            grep_results = self.repo_collector.grep_repo(keyword)
            for path in grep_results:
                if path not in seen:
                    seen.add(path)
                    relevant_files.append(path)
            if len(relevant_files) >= 20:
                break

        # Sort: source code first, test-reports/docs last
        source_exts = (".rs", ".py", ".ts", ".tsx", ".go", ".java", ".c", ".cpp", ".h")
        relevant_files.sort(
            key=lambda f: (
                0 if any(f.endswith(e) for e in source_exts) else 1,
                1 if f.startswith("test") or "/tests/" in f else 0,
            )
        )

        return relevant_files[:20]
