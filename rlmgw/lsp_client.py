"""Lightweight LSP client for code intelligence in context selection.

Spawns language servers (rust-analyzer, pyright, typescript-language-server)
as subprocesses and communicates over stdio using JSON-RPC 2.0 with
Content-Length framing. Zero external dependencies — uses only stdlib.

Architecture:
    LSPManager (per-repo, reusable)
      └── LSPConnection (per-language server)
              ├── subprocess (the language server)
              └── reader thread (routes JSON-RPC responses)
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _noop_event() -> threading.Event:
    """Create an unset Event for progress tracking."""
    return threading.Event()


# LSP SymbolKind constants (useful subset)
SYMBOL_KIND = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    22: "Struct",
    23: "Event",
    24: "Operator",
    25: "TypeParameter",
}


@dataclass
class SymbolInfo:
    """Parsed LSP symbol information."""

    name: str
    kind: int
    kind_name: str
    file_path: str
    line: int
    character: int
    container_name: str = ""


@dataclass
class LSPConnection:
    """Single LSP server connection over stdio with reader thread."""

    cmd: list[str]
    root_uri: str
    process: subprocess.Popen | None = field(default=None, repr=False)
    _request_id: int = field(default=0, repr=False)
    _pending: dict = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _reader_thread: threading.Thread | None = field(default=None, repr=False)
    _shutdown: bool = field(default=False, repr=False)
    _opened_files: set = field(default_factory=set, repr=False)
    ready: bool = False
    _indexed: threading.Event = field(default_factory=_noop_event, repr=False)
    _progress_started: threading.Event = field(default_factory=_noop_event, repr=False)
    _active_progress: set = field(default_factory=set, repr=False)
    server_capabilities: dict = field(default_factory=dict, repr=False)

    def start(self, timeout: float = 90.0) -> bool:
        """Start the LSP server and complete initialization handshake."""
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Own process group for clean tree kill
            )
        except FileNotFoundError:
            logger.warning(f"LSP server not found: {self.cmd[0]}")
            return False

        # Start reader thread
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        # Initialize handshake
        try:
            result = self._request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": self.root_uri,
                    "capabilities": {
                        "textDocument": {
                            "documentSymbol": {
                                "hierarchicalDocumentSymbolSupport": True,
                            },
                            "definition": {"linkSupport": True},
                            "references": {},
                        },
                        "workspace": {
                            "symbol": {
                                "symbolKind": {
                                    "valueSet": list(range(1, 27)),
                                },
                            },
                        },
                    },
                },
                timeout=timeout,
            )

            self.server_capabilities = result.get("capabilities", {})
            self._notify("initialized", {})
            self.ready = True
            logger.info(f"LSP server ready: {self.cmd[0]}")
            return True

        except Exception as e:
            logger.warning(f"LSP initialization failed for {self.cmd[0]}: {e}")
            self.stop()
            return False

    def workspace_symbol(self, query: str, timeout: float = 30.0) -> list[SymbolInfo]:
        """Search for symbols matching query across the workspace."""
        if not self.ready:
            return []
        try:
            result = self._request("workspace/symbol", {"query": query}, timeout=timeout)
            return self._parse_workspace_symbols(result or [])
        except Exception as e:
            logger.debug(f"workspace/symbol failed: {e}")
            return []

    def document_symbols(self, file_path: str, timeout: float = 15.0) -> list[SymbolInfo]:
        """Get all symbols in a file."""
        if not self.ready:
            return []
        file_uri = Path(file_path).as_uri()
        self._ensure_open(file_path, file_uri)
        try:
            result = self._request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": file_uri}},
                timeout=timeout,
            )
            return self._parse_document_symbols(result or [], file_path)
        except Exception as e:
            logger.debug(f"documentSymbol failed for {file_path}: {e}")
            return []

    def _ensure_open(self, file_path: str, file_uri: str):
        """Send didOpen if we haven't opened this file yet."""
        if file_uri in self._opened_files:
            return
        try:
            content = Path(file_path).read_text(errors="ignore")
        except Exception:
            return

        lang_map = {
            ".rs": "rust",
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact",
        }
        ext = Path(file_path).suffix
        lang_id = lang_map.get(ext, "plaintext")

        self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": file_uri,
                    "languageId": lang_id,
                    "version": 1,
                    "text": content,
                },
            },
        )
        self._opened_files.add(file_uri)

    def _parse_workspace_symbols(self, symbols: list) -> list[SymbolInfo]:
        """Parse workspace/symbol response (flat SymbolInformation[])."""
        results = []
        for sym in symbols:
            loc = sym.get("location", {})
            uri = loc.get("uri", "")
            pos = loc.get("range", {}).get("start", {})
            kind = sym.get("kind", 0)
            results.append(
                SymbolInfo(
                    name=sym.get("name", ""),
                    kind=kind,
                    kind_name=SYMBOL_KIND.get(kind, "Unknown"),
                    file_path=_uri_to_path(uri),
                    line=pos.get("line", 0),
                    character=pos.get("character", 0),
                    container_name=sym.get("containerName", ""),
                )
            )
        return results

    def _parse_document_symbols(self, symbols: list, file_path: str) -> list[SymbolInfo]:
        """Parse textDocument/documentSymbol response (may be hierarchical)."""
        results: list[SymbolInfo] = []

        def flatten(syms: list, container: str = ""):
            for sym in syms:
                sel = sym.get("selectionRange", sym.get("range", {}))
                pos = sel.get("start", {})
                kind = sym.get("kind", 0)
                results.append(
                    SymbolInfo(
                        name=sym.get("name", ""),
                        kind=kind,
                        kind_name=SYMBOL_KIND.get(kind, "Unknown"),
                        file_path=file_path,
                        line=pos.get("line", 0),
                        character=pos.get("character", 0),
                        container_name=container,
                    )
                )
                if "children" in sym:
                    flatten(sym["children"], sym.get("name", ""))

        flatten(symbols)
        return results

    # ── JSON-RPC transport ───────────────────────────────────────────

    def _request(self, method: str, params, timeout: float = 30.0):
        """Send a JSON-RPC request and wait for the matching response."""
        with self._lock:
            self._request_id += 1
            req_id = self._request_id

        event = threading.Event()
        result_holder: list = [None, None]  # [result, error]
        self._pending[req_id] = (event, result_holder)

        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        self._send(msg)

        if not event.wait(timeout=timeout):
            self._pending.pop(req_id, None)
            raise TimeoutError(f"LSP request {method} timed out after {timeout}s")

        self._pending.pop(req_id, None)
        if result_holder[1] is not None:
            raise Exception(f"LSP error: {result_holder[1]}")
        return result_holder[0]

    def _notify(self, method: str, params):
        """Send a JSON-RPC notification (no response expected)."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._send(msg)

    def _send(self, msg: dict):
        """Send a Content-Length framed JSON-RPC message."""
        if not self.process or self.process.stdin is None:
            return
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        try:
            self.process.stdin.write(header + body)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            logger.debug(f"LSP write failed: {e}")

    def _reader_loop(self):
        """Background thread: read JSON-RPC messages and route responses."""
        if not self.process or not self.process.stdout:
            return
        stdout = self.process.stdout
        while not self._shutdown and self.process.poll() is None:
            try:
                msg = self._read_one_message(stdout)
                if msg is None:
                    break

                # Route response to waiting request
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    event, result_holder = self._pending[msg_id]
                    if "error" in msg:
                        result_holder[1] = msg["error"].get("message", str(msg["error"]))
                    else:
                        result_holder[0] = msg.get("result")
                    event.set()

                # Track $/progress notifications for indexing readiness
                method = msg.get("method", "")
                if method == "$/progress":
                    self._handle_progress(msg.get("params", {}))

            except Exception as e:
                if not self._shutdown:
                    logger.debug(f"LSP reader error: {e}")
                break

    def _read_one_message(self, stream) -> dict | None:
        """Read one Content-Length framed message from stream."""
        # Read headers
        content_length = 0
        while True:
            line = stream.readline()
            if not line:
                return None  # EOF
            line = line.decode("ascii", errors="ignore").strip()
            if not line:
                break  # Empty line = end of headers
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":")[1].strip())

        if content_length == 0:
            return None

        body = stream.read(content_length)
        if not body:
            return None
        return json.loads(body)

    def _handle_progress(self, params: dict):
        """Track work done progress notifications from the server.

        rust-analyzer sends $/progress with tokens like "rustAnalyzer/Indexing".
        pyright sends "pyright.analysis". When a "begin" comes, we track it;
        when "end" comes, we remove it. When all active progress tokens finish,
        we signal _indexed.
        """
        token = params.get("token", "")
        value = params.get("value", {})
        kind = value.get("kind", "")

        if kind == "begin":
            self._active_progress.add(token)
            self._progress_started.set()
            self._indexed.clear()
            logger.debug(f"LSP progress begin: {token}")
        elif kind == "end":
            self._active_progress.discard(token)
            logger.debug(f"LSP progress end: {token}")
            if not self._active_progress:
                self._indexed.set()
                logger.info(f"LSP server fully indexed: {self.cmd[0]}")

    def wait_until_indexed(self, timeout: float = 120.0) -> bool:
        """Wait for the server to finish indexing. Returns True if indexed.

        Two-phase wait:
        1. Wait up to 5s for any progress "begin" notification
        2. If received, wait for all progress to "end" (up to timeout)
        3. If no progress starts within 5s, assume server is immediately ready
        """
        # Phase 1: wait briefly for progress to start
        if not self._progress_started.wait(timeout=2.0):
            # No progress notifications — server is ready immediately
            logger.info(f"LSP server ready (no indexing needed): {self.cmd[0]}")
            return True

        # Phase 2: progress started — wait for it to finish
        if self._indexed.is_set():
            return True
        logger.info(f"Waiting for {self.cmd[0]} to finish indexing...")
        return self._indexed.wait(timeout=timeout)

    def stop(self):
        """Shutdown the LSP server and reap all child processes."""
        self._shutdown = True
        if self.process and self.process.returncode is None:
            try:
                # Graceful shutdown via LSP protocol
                self._request("shutdown", None, timeout=5)
                self._notify("exit", None)
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Kill entire process group (handles rust-analyzer child procs)
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                # Always wait to reap — prevents zombie
                try:
                    self.process.wait(timeout=3)
                except Exception:
                    pass
            except Exception:
                pass
        self.ready = False


def _uri_to_path(uri: str) -> str:
    """Convert file:// URI to filesystem path."""
    if uri.startswith("file://"):
        return uri[7:]
    return uri


# ── LSP Server configurations ───────────────────────────────────────

SERVER_CONFIGS: dict[str, list[str]] = {
    "rust": ["rust-analyzer"],
    "python": ["pyright-langserver", "--stdio"],
    "typescript": ["typescript-language-server", "--stdio"],
}

LANG_EXTENSIONS: dict[str, str] = {
    ".rs": "rust",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
}


# Module-level registry for sharing LSP managers across contexts.
# Keyed by absolute repo_root path. Used to avoid spawning duplicate
# LSP servers when the RLM REPL setup code runs in the same process.
_lsp_registry: dict[str, "LSPManager"] = {}


def get_shared_lsp(repo_root: str) -> "LSPManager | None":
    """Retrieve an already-initialized LSPManager for this repo, if one exists."""
    return _lsp_registry.get(str(Path(repo_root).absolute()))


class LSPManager:
    """Manages LSP server connections for code intelligence.

    Lazily initializes servers on first use. Detects which languages
    are present in the repo and only starts relevant servers.
    Keeps servers alive across queries for reuse.
    """

    def __init__(self, repo_root: str):
        self.repo_root = str(Path(repo_root).absolute())
        self.root_uri = Path(repo_root).absolute().as_uri()
        self.servers: dict[str, LSPConnection] = {}
        self._initialized = False
        self._init_lock = threading.Lock()

    def initialize(self, timeout_per_server: float = 90.0):
        """Detect languages, start LSP servers, and wait for indexing.

        Thread-safe and idempotent. Blocks until servers are ready for queries.
        """
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            languages = self._detect_languages()
            logger.info(f"LSP: detected languages in repo: {languages}")

            for lang in languages:
                if lang not in SERVER_CONFIGS:
                    continue
                cmd = SERVER_CONFIGS[lang]
                conn = LSPConnection(cmd=cmd, root_uri=self.root_uri)
                if conn.start(timeout=timeout_per_server):
                    self.servers[lang] = conn
                else:
                    logger.warning(f"LSP: failed to start {lang} server")

            # Wait for all servers to finish indexing in parallel
            wait_threads = []
            for conn in self.servers.values():
                t = threading.Thread(
                    target=conn.wait_until_indexed,
                    args=(timeout_per_server,),
                    daemon=True,
                )
                t.start()
                wait_threads.append(t)
            for t in wait_threads:
                t.join(timeout=timeout_per_server)

            self._initialized = True
            logger.info(f"LSP: {len(self.servers)} servers ready: {list(self.servers.keys())}")

    def _detect_languages(self) -> set[str]:
        """Detect which languages are present using git ls-files."""
        languages: set[str] = set()
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for filepath in result.stdout.strip().split("\n"):
                    ext = Path(filepath).suffix
                    if ext in LANG_EXTENSIONS:
                        languages.add(LANG_EXTENSIONS[ext])
                    if len(languages) >= len(SERVER_CONFIGS):
                        break
        except Exception:
            # Fallback: walk first 100 files
            count = 0
            for _, dirs, files in os.walk(self.repo_root):
                dirs[:] = [
                    d
                    for d in dirs
                    if d not in {".git", "node_modules", "target", "__pycache__", ".venv"}
                ]
                for f in files:
                    ext = Path(f).suffix
                    if ext in LANG_EXTENSIONS:
                        languages.add(LANG_EXTENSIONS[ext])
                    count += 1
                    if count > 100:
                        break
                if count > 100:
                    break
        return languages

    def workspace_symbol(self, query: str) -> list[SymbolInfo]:
        """Search for symbols matching query across all running servers."""
        self.initialize()
        all_symbols: list[SymbolInfo] = []
        for lang, server in self.servers.items():
            try:
                symbols = server.workspace_symbol(query)
                all_symbols.extend(symbols)
            except Exception as e:
                logger.debug(f"workspace/symbol failed for {lang}: {e}")
        return all_symbols

    def document_symbols(self, file_path: str) -> list[SymbolInfo]:
        """Get all symbols in a specific file."""
        self.initialize()
        abs_path = str(Path(self.repo_root, file_path).absolute())
        ext = Path(file_path).suffix
        lang = LANG_EXTENSIONS.get(ext)
        if lang and lang in self.servers:
            try:
                return self.servers[lang].document_symbols(abs_path)
            except Exception as e:
                logger.debug(f"documentSymbol failed for {file_path}: {e}")
        return []

    @property
    def available(self) -> bool:
        """Whether any LSP servers are running."""
        return bool(self.servers)

    def shutdown(self):
        """Shutdown all LSP servers and deregister from shared registry."""
        for server in self.servers.values():
            server.stop()
        self.servers.clear()
        self._initialized = False
        # Remove from registry to prevent stale references
        _lsp_registry.pop(self.repo_root, None)

    def __del__(self):
        self.shutdown()


def _atexit_cleanup():
    """Kill all registered LSP servers on interpreter shutdown."""
    for manager in list(_lsp_registry.values()):
        try:
            manager.shutdown()
        except Exception:
            pass
    _lsp_registry.clear()


atexit.register(_atexit_cleanup)
