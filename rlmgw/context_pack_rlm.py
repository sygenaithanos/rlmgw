"""RLM-based context pack builder for RLMgw."""

import json
import logging
import re

from .config import RLMgwConfig
from .models import ContextPack
from .repo_context import ENTRY_POINT_NAMES, RepoContextCollector

logger = logging.getLogger(__name__)


# System prompt for context selection RLM — includes LSP-enhanced tools
# and few-shot examples teaching the exact workflow expected.
CONTEXT_SELECTION_PROMPT = """You are a code search agent with semantic code intelligence tools.

The `context` variable contains the task and INITIAL LEADS (pre-computed candidates). Review the leads, then use tools to verify and expand.

Available tools (use in ```repl``` code blocks):
- repo.find_symbols(query) -> List[dict]  # BEST: semantic symbol search via LSP
- repo.entry_points() -> List[str]        # Find architectural entry points (lib.rs, main.rs, etc.)
- repo.doc_symbols(path) -> List[dict]    # List all symbols in a file
- repo.grep(pattern) -> Dict[str, List[str]]  # Text search
- repo.read_file(path) -> str             # Read file contents
- repo.list_files() -> List[str]          # List all files

RULES:
- ALWAYS respond with ```repl``` code blocks
- Start by reviewing INITIAL LEADS in `context`, then verify with repo.read_file()
- Use repo.find_symbols() for semantic search — it's more precise than grep
- Prioritize source code (.rs, .py, .ts, .go) over docs/configs
- Use EXACT paths from tool output — do not invent paths
- When done, call FINAL() with JSON

=== EXAMPLE 1: Starting from initial leads ===

User query: "Find the authentication middleware"

Turn 1 (review leads and verify):
```repl
print(context)  # See initial leads
content = repo.read_file("src/middleware/auth.py")  # Verify top lead
```

Turn 2 (expand with semantic search):
```repl
symbols = repo.find_symbols("Auth")
```
Output: find_symbols 'Auth' found 3 symbols:
  Struct AuthMiddleware in src/middleware/auth.py:15
  Function authenticate in src/middleware/auth.py:42
  Struct AuthUser in src/models/user.py:8

Turn 3:
FINAL({"relevant_files": ["src/middleware/auth.py", "src/models/user.py"], "reasoning": "Auth middleware and user model found via LSP"})

=== EXAMPLE 2: Architecture question ===

User query: "How is the project structured?"

Turn 1:
```repl
entries = repo.entry_points()
print(entries)
```
Output: Entry points: ['crates/app-core/src/lib.rs', 'crates/app-auth/src/lib.rs', 'crates/app-db/src/lib.rs', 'Cargo.toml']

Turn 2:
```repl
syms = repo.doc_symbols("crates/app-core/src/lib.rs")
```
Output: doc_symbols found 8 symbols in crates/app-core/src/lib.rs:
  Module config, Module db, Module handlers, Struct AppState, Function init

Turn 3:
FINAL({"relevant_files": ["crates/app-core/src/lib.rs", "crates/app-auth/src/lib.rs", "crates/app-db/src/lib.rs", "Cargo.toml"], "reasoning": "Entry point lib.rs from each crate; Cargo.toml defines workspace"})
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

        # LSP manager for semantic code intelligence (lazy init)
        self._lsp = None

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
from rlmgw.repo_context import RepoContextCollector
base_repo = RepoContextTools('{repo_root_escaped}')
base_collector = RepoContextCollector('{repo_root_escaped}')

# Reuse the already-initialized LSP manager (same process, shared via registry)
lsp_mgr = None
try:
    from rlmgw.lsp_client import get_shared_lsp, LSPManager
    lsp_mgr = get_shared_lsp('{repo_root_escaped}')
    if lsp_mgr is None:
        lsp_mgr = LSPManager('{repo_root_escaped}')
        lsp_mgr.initialize(timeout_per_server=90)
    if not lsp_mgr.available:
        lsp_mgr = None
except Exception:
    lsp_mgr = None

def repo_find_symbols(query):
    if lsp_mgr is None:
        print("LSP not available. Use repo.grep() instead.")
        return []
    symbols = lsp_mgr.workspace_symbol(query)
    print(f"find_symbols '{{query}}' found {{len(symbols)}} symbols:")
    for s in symbols[:15]:
        print(f"  {{s.kind_name}} {{s.name}} in {{s.file_path}}:{{s.line}}")
    return symbols

def repo_entry_points():
    result = base_collector.find_entry_points()
    print(f"Entry points: {{result[:30]}}")
    return result

def repo_doc_symbols(path):
    if lsp_mgr is None:
        print("LSP not available.")
        return []
    symbols = lsp_mgr.document_symbols(path)
    print(f"doc_symbols found {{len(symbols)}} symbols in {{path}}:")
    for s in symbols[:20]:
        print(f"  {{s.kind_name}} {{s.name}}")
    return symbols

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
    find_symbols=repo_find_symbols,
    entry_points=repo_entry_points,
    doc_symbols=repo_doc_symbols,
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
                    "Step 1: review INITIAL LEADS in context, verify with repo.read_file(). "
                    "Step 2: use repo.find_symbols() for semantic search. "
                    "Step 3: call FINAL() with verified files. "
                    "Do NOT repeat a search you already ran. "
                    "Your next ```repl``` code block:"
                ),
                verbose=False,
            )

            logger.info("Initialized RLM for context selection")
        except Exception as e:
            logger.warning(f"Failed to initialize RLM: {e}. Falling back to simple selection.")
            self.rlm = None

    def build_from_query(self, query: str) -> ContextPack:
        """Build context pack using combined symbolic + RLM pipeline.

        1. Pre-compute symbolic candidates (LSP, entry points, grep) and classify query
        2. Inject seed candidates into RLM REPL context
        3. RLM explores with LSP-powered tools, starting from strong initial leads
        4. If RLM fails, fall back to the pre-computed symbolic results (zero extra cost)
        """
        # Step 1: Symbolic pre-computation (fast, deterministic)
        query_type = "specific"
        seed_candidates: list[tuple[str, str]] = []
        try:
            query_type = self._classify_query(query)
            seed_candidates = self._symbolic_retrieve(query, query_type)
            logger.info(
                f"Pre-computed {len(seed_candidates)} seed candidates (query_type={query_type})"
            )
        except Exception as e:
            logger.warning(f"Symbolic pre-computation failed: {e}")

        # Step 2: RLM exploration with enhanced tools + seed data
        try:
            return self._build_with_rlm(query, seed_candidates, query_type)
        except Exception as e:
            logger.warning(f"RLM exploration failed: {e}")

        # Step 3: Fall back to neural reranking of pre-computed seeds
        if seed_candidates:
            try:
                ranked = self._neural_rerank(query, seed_candidates, query_type)
                if ranked:
                    logger.info(f"Falling back to seed reranking: {len(ranked)} files")
                    return self._build_context_pack(ranked)
            except Exception as e:
                logger.warning(f"Neural reranking failed: {e}")

        # Step 4: Last resort — simple keyword search
        return self._build_simple(query)

    def _build_with_rlm(
        self,
        query: str,
        seed_candidates: list[tuple[str, str]] | None = None,
        query_type: str = "specific",
    ) -> ContextPack:
        """Use RLM to explore with LSP-enhanced tools, starting from seed candidates."""
        self._initialize_rlm()

        if self.rlm is None:
            raise Exception("RLM not available")

        # Build seed data summary for the context variable
        try:
            tree = self.repo_collector.get_repo_tree()
            tree_summary = ", ".join(sorted(tree.keys())[:30])
        except Exception:
            tree_summary = "(tree unavailable)"

        # Format seed candidates as initial leads
        leads_text = ""
        if seed_candidates:
            lead_paths = [c[0] for c in seed_candidates[:15]]
            leads_text = (
                "\n\nINITIAL LEADS (pre-computed, verify with repo.read_file()):\n"
                + "\n".join(f"  - {p}" for p in lead_paths)
            )

        selection_query = f"""=== TASK: Find source files relevant to this query ===
QUERY: {query}
TYPE: {query_type}

REPO STRUCTURE (top-level): {tree_summary}{leads_text}

INSTRUCTIONS:
1. Review the INITIAL LEADS above — verify the best ones with repo.read_file()
2. Use repo.find_symbols("term") for semantic code search (better than grep)
3. Select 3-10 files maximum. Quality over quantity.
4. When confident, call FINAL({{"relevant_files": ["path/to/file"], "reasoning": "..."}})

Available tools: repo.find_symbols(query), repo.entry_points(), repo.doc_symbols(path), repo.grep(pattern), repo.read_file(path), repo.list_files()
Prioritize source code (.rs, .py, .ts, .go) over docs/configs/tests.
"""

        # Let RLM explore and select context
        logger.info(f"Running RLM context selection for query: {query[:100]}")
        result = self.rlm.completion(selection_query, root_prompt=query)

        # Parse the RLM response into a file list
        relevant_files = self._parse_rlm_response(result.response, query)

        if not relevant_files:
            raise ValueError("RLM returned no files")

        # Quality gate: catch noisy "dump everything" results
        relevant_files = self._quality_gate(relevant_files, seed_candidates, query_type)
        logger.info(f"RLM final selection: {len(relevant_files)} files after quality gate")

        return self._build_context_pack(relevant_files)

    def _parse_rlm_response(self, response, query: str) -> list[str]:
        """Extract file list from RLM response (JSON, regex, or text extraction)."""
        response_text = response
        logger.debug(
            f"RLM raw response type={type(response_text).__name__}: "
            f"{repr(str(response_text))[:500]}"
        )
        if isinstance(response_text, tuple):
            response_text = response_text[1]  # Extract content from ("FINAL", content)

        # Try 1: direct JSON parse
        try:
            selection_data = json.loads(response_text)
            relevant_files = selection_data.get("relevant_files", [])
            reasoning = selection_data.get("reasoning", "")
            logger.info(f"RLM selected {len(relevant_files)} files: {reasoning}")
            return relevant_files
        except (json.JSONDecodeError, TypeError):
            pass

        # Try 2: extract JSON from surrounding prose
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
                return relevant_files
            except (json.JSONDecodeError, TypeError):
                pass

        # Try 3: extract file paths from REPL output/response text
        files_from_text = self._extract_file_paths_from_text(str(response_text), query)
        if files_from_text:
            logger.info(f"RLM selected {len(files_from_text)} files (extracted paths from text)")
            return files_from_text

        return []

    def _quality_gate(
        self,
        rlm_files: list[str],
        seed_candidates: list[tuple[str, str]] | None,
        query_type: str,
    ) -> list[str]:
        """Filter noisy RLM results using seed candidates as quality signal.

        When the RLM dumps >10 files (a common failure mode for small models),
        boost files that overlap with pre-computed seeds and cap the count.
        If no overlap exists at all, raise to trigger neural reranking fallback.
        """
        max_files = 15 if query_type == "architecture" else 10

        if len(rlm_files) <= max_files:
            return rlm_files  # Selective enough, trust the RLM

        logger.info(f"Quality gate: RLM returned {len(rlm_files)} files (>{max_files}), filtering")

        if not seed_candidates:
            return rlm_files[:max_files]

        seed_paths = {c[0] for c in seed_candidates}
        overlap = [f for f in rlm_files if f in seed_paths]
        rlm_only = [f for f in rlm_files if f not in seed_paths]

        if not overlap:
            # RLM went completely off track — no overlap with pre-computed seeds.
            # Raise to fall through to neural reranking of seeds.
            raise ValueError(
                f"RLM returned {len(rlm_files)} files with zero seed overlap — "
                "falling back to seed reranking"
            )

        # Combine: seed-validated files first, then RLM discoveries
        result = overlap + rlm_only
        logger.info(
            f"Quality gate: {len(overlap)} seed-validated + "
            f"{min(len(rlm_only), max_files - len(overlap))} RLM discoveries"
        )
        return result[:max_files]

    # ── Neuro-symbolic pipeline ──────────────────────────────────────────

    def _build_with_symbolic_pipeline(self, query: str) -> ContextPack:
        """Neuro-symbolic pipeline: symbolic retrieval → neural reranking → assembly.

        1. Classify query type symbolically (architecture vs. specific)
        2. Retrieve candidates deterministically (entry points, hub files, grep, filenames)
        3. Rerank candidates with a simple LLM call (classification, not REPL)
        4. Assemble the final context pack
        """
        query_type = self._classify_query(query)
        logger.info(f"Symbolic pipeline: query_type={query_type}, query={query[:80]}")

        # Step 1: Symbolic candidate retrieval
        candidates = self._symbolic_retrieve(query, query_type)
        if not candidates:
            logger.info("Symbolic pipeline: no candidates found")
            raise ValueError("No candidates from symbolic retrieval")

        logger.info(f"Symbolic pipeline: {len(candidates)} candidates retrieved")

        # Step 2: Neural reranking
        ranked_files = self._neural_rerank(query, candidates, query_type)
        logger.info(f"Symbolic pipeline: {len(ranked_files)} files after reranking")

        # Step 3: Symbolic assembly
        return self._build_context_pack(ranked_files)

    def _classify_query(self, query: str) -> str:
        """Classify query type using LLM.

        Uses a single LLM call for robust classification. Falls back to
        "specific" if the LLM call fails.
        """
        prompt = (
            "Classify this codebase query into exactly one category.\n\n"
            "Categories:\n"
            '- "architecture": asking about overall project structure, module layout, '
            "how the system is organized, entry points, workspace structure\n"
            '- "specific": asking about a specific function, file, feature, component, '
            "or implementation detail\n\n"
            f'Query: "{query}"\n\n'
            "Respond with ONLY the category name, nothing else."
        )
        try:
            from rlm.clients import get_client

            client = get_client(
                "openai",
                {
                    "model_name": self.config.upstream_model,
                    "base_url": self.config.upstream_base_url,
                    "api_key": "dummy",
                    "temperature": 0.01,
                    "max_tokens": 32,
                },
            )
            response = client.completion(prompt).strip().lower().strip("\"'")
            if "architecture" in response:
                return "architecture"
            return "specific"
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, defaulting to specific")
            return "specific"

    def _get_lsp(self):
        """Lazily initialize LSP manager and register it for REPL reuse."""
        if self._lsp is None:
            from .lsp_client import LSPManager, _lsp_registry

            self._lsp = LSPManager(str(self.repo_collector.repo_root))
            # Register so REPL setup code reuses this instance (same process)
            _lsp_registry[self._lsp.repo_root] = self._lsp
        return self._lsp

    def _symbolic_retrieve(self, query: str, query_type: str) -> list[tuple[str, str]]:
        """Candidate retrieval using LSP (primary) + grep/filenames (fallback).

        LSP provides semantic symbol search (workspace/symbol) which is
        categorically better than text grep for finding code definitions.
        Falls back to grep/filename matching when LSP is unavailable.
        """
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add_candidate(path: str) -> None:
            if path in seen:
                return
            # Normalize to relative path
            repo_root = str(self.repo_collector.repo_root)
            if path.startswith(repo_root):
                path = path[len(repo_root) :].lstrip("/")
            seen.add(path)
            summary = self.repo_collector.get_file_summary(path, max_lines=8)
            candidates.append((path, summary))

        keywords = self._extract_keywords(query)

        # 1. LSP semantic symbol search (highest quality)
        lsp_found = 0
        try:
            lsp = self._get_lsp()
            lsp.initialize(timeout_per_server=90.0)
            if lsp.available:
                for keyword in keywords[:5]:
                    symbols = lsp.workspace_symbol(keyword)
                    for sym in symbols:
                        if sym.file_path:
                            add_candidate(sym.file_path)
                            lsp_found += 1
                if lsp_found:
                    logger.info(f"LSP workspace/symbol found {lsp_found} symbols")
        except Exception as e:
            logger.debug(f"LSP retrieval failed: {e}")

        # 2. Filename matching (catches files like output_parser.rs)
        file_list = self.repo_collector.get_file_list()
        for path in file_list:
            basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if any(kw.lower() in basename for kw in keywords):
                add_candidate(path)

        # 3. Content grep (fallback, or supplement when LSP found few results)
        if lsp_found < 5:
            for keyword in keywords[:4]:
                if len(candidates) >= 30:
                    break
                grep_results = self.repo_collector.grep_repo(keyword)
                for path in grep_results:
                    if len(candidates) >= 30:
                        break
                    add_candidate(path)

        # 4. For architecture queries, supplement with entry points and hubs
        if query_type == "architecture":
            for f in self.repo_collector.find_entry_points():
                add_candidate(f)
            for f in self.repo_collector.find_hub_files(max_files=15):
                add_candidate(f)

        return candidates[:40]

    def _neural_rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        query_type: str,
    ) -> list[str]:
        """Use LLM to rerank candidates via a simple classification prompt.

        This is NOT free-form REPL exploration — it's a single structured LLM call
        asking "which of these files are relevant?" Much easier for small models.
        Falls back to heuristic ordering if the LLM call fails.
        """
        # Build candidate descriptions
        candidate_lines = []
        for i, (path, summary) in enumerate(candidates):
            # Show first line of summary as a hint
            first_line = summary.split("\n")[0].strip() if summary else "(empty)"
            candidate_lines.append(f"{i + 1}. {path} — {first_line}")

        candidate_text = "\n".join(candidate_lines)

        if query_type == "architecture":
            instruction = (
                "Select the files that best represent the system's architecture: "
                "entry points (lib.rs, main.rs, __init__.py), module declarations, "
                "and build configs. Prefer source code over docs. Select 5-15 files."
            )
        else:
            instruction = (
                "Select the files most relevant to the query. "
                "Prefer implementation files over tests and docs. Select 3-10 files."
            )

        prompt = (
            f'Query: "{query}"\n\n'
            f"{instruction}\n\n"
            f"Candidates:\n{candidate_text}\n\n"
            "Respond with ONLY a JSON array of file paths, e.g.:\n"
            '["path/to/file1.rs", "path/to/file2.py"]'
        )

        try:
            from rlm.clients import get_client

            client = get_client(
                "openai",
                {
                    "model_name": self.config.upstream_model,
                    "base_url": self.config.upstream_base_url,
                    "api_key": "dummy",
                    "temperature": 0.01,
                    "max_tokens": 1024,
                },
            )
            response = client.completion(prompt)
            logger.debug(f"Neural rerank response: {response[:500]}")

            # Extract JSON array from response
            json_match = re.search(r"\[.*?\]", response, re.DOTALL)
            if json_match:
                selected = json.loads(json_match.group())
                # Validate: only keep paths that were actual candidates
                candidate_paths = {c[0] for c in candidates}
                valid = [f for f in selected if f in candidate_paths]
                if valid:
                    logger.info(
                        f"Neural reranking selected {len(valid)} files from {len(candidates)} candidates"
                    )
                    return valid

        except Exception as e:
            logger.warning(f"Neural reranking failed: {e}")

        # Fallback: heuristic ordering (source code first, entry points prioritized)
        logger.info("Neural reranking failed, using heuristic ordering")
        return self._heuristic_rank(candidates, query_type)

    def _heuristic_rank(self, candidates: list[tuple[str, str]], query_type: str) -> list[str]:
        """Deterministic fallback ranking when neural reranking fails."""
        source_exts = {".rs", ".py", ".ts", ".tsx", ".go", ".java", ".c", ".cpp", ".h"}

        def score(path: str) -> tuple[int, int, int]:
            ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
            basename = path.rsplit("/", 1)[-1]
            is_source = 0 if ext in source_exts else 1
            is_entry = 0 if basename in ENTRY_POINT_NAMES else 1
            is_test = 1 if "/test" in path or path.startswith("test") else 0
            return (is_source, is_test, is_entry)

        paths = [c[0] for c in candidates]
        paths.sort(key=score)
        limit = 15 if query_type == "architecture" else 10
        return paths[:limit]

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
