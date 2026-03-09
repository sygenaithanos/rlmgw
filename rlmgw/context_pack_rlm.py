"""RLM-based context pack builder for RLMgw."""

import json
import logging
import re

from .config import RLMgwConfig
from .models import ContextPack
from .repo_context import RepoContextCollector

logger = logging.getLogger(__name__)


# System prompt for context selection RLM
CONTEXT_SELECTION_PROMPT = """You are a context selection assistant for a coding agent.

Your task is to analyze a user's query and select the MOST RELEVANT source code files from a codebase.

You have access to these tools in your REPL environment:
- repo.list_files(extensions=None) -> List[str]  # List files, optionally filtered by extension
- repo.grep(pattern: str, extensions=None) -> Dict[str, List[str]]  # Search for pattern
- repo.read_file(path: str) -> str  # Read file contents
- repo.get_tree() -> Dict  # Get directory structure

Your goal:
1. Understand what the user is asking about
2. Use the tools to explore and find relevant source code
3. PRIORITIZE source code files (.py, .rs, .ts, .go, .java, etc.) over docs, configs, and generated files
4. Select the MINIMAL set of files that provide high-signal context
5. Return your selection as a JSON object wrapped in FINAL()

Strategy:
- Start by grepping for key terms from the query
- Read promising source files to verify relevance
- Prefer implementation files over tests, docs, configs, or generated output
- Skip markdown plans, migration docs, JSON output files, and settings files unless directly relevant

When you want to execute Python code, wrap it in triple backticks with 'repl' language identifier:
```repl
results = repo.grep("function_name")
print(results)
```

IMPORTANT: When you have completed your selection, you MUST provide your final answer using:
FINAL({"relevant_files": ["path/to/file1.py", "path/to/file2.rs"], "reasoning": "Brief explanation"})

Do NOT just output JSON — wrap it in FINAL(). Do NOT use FINAL() until you have explored the codebase and are confident in your selection.

Keep context COMPACT but HIGH-SIGNAL. Quality over quantity. Source code over documentation.
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

            # Build setup code that creates the repo tools in the environment
            setup_code = f"""
import sys
sys.path.insert(0, '{str(self.repo_collector.repo_root)}')

# Import and setup repo context tools
from rlmgw.repo_env import RepoContextTools
repo = RepoContextTools('{str(self.repo_collector.repo_root)}')
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

        # Create context selection query
        selection_query = f"""
User Query: {query}

Analyze this query and select the most relevant source code files from the codebase.
Use the repo tools to explore:
1. Search for keywords related to the query using repo.grep()
2. Read promising source code files to verify relevance
3. Select the MINIMAL set of files that provide the context needed
4. Wrap your final answer in FINAL()
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
