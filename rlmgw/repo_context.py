"""Repository context collectors for RLMgw."""

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Source code extensions (high priority for context selection)
SOURCE_EXTENSIONS = [
    ".py",
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".swift",
    ".kt",
    ".scala",
    ".ex",
    ".exs",
    ".zig",
    ".lua",
    ".sh",
]

# Config/doc extensions (lower priority)
CONFIG_EXTENSIONS = [".toml", ".cfg", ".ini"]
DOC_EXTENSIONS = [".md", ".txt", ".rst"]
DATA_EXTENSIONS = [".json", ".yaml", ".yml", ".xml"]

ALL_EXTENSIONS = SOURCE_EXTENSIONS + CONFIG_EXTENSIONS + DOC_EXTENSIONS + DATA_EXTENSIONS


def _file_priority(filepath: str) -> int:
    """Lower number = higher priority. Source code first."""
    ext = "." + filepath.rsplit(".", 1)[-1] if "." in filepath else ""
    if ext in SOURCE_EXTENSIONS:
        return 0
    if ext in CONFIG_EXTENSIONS:
        return 1
    if ext in DOC_EXTENSIONS:
        return 2
    return 3


class RepoContextCollector:
    """Collects context from repository in a read-only, safe manner."""

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).absolute()
        self.excluded_dirs = {
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            "build",
            "dist",
            "target",
        }
        self.max_file_size = 1024 * 1024  # 1MB
        self.max_file_read = 1024 * 100  # 100KB per file
        self.max_grep_results = 50

    def _is_excluded(self, path: Path) -> bool:
        """Check if path should be excluded."""
        for part in path.parts:
            if part in self.excluded_dirs:
                return True
        return False

    def _safe_path(self, path: Path) -> Path | None:
        """Ensure path is safe and within repo root."""
        try:
            # Resolve and check if within repo root
            abs_path = path.absolute()
            if not str(abs_path).startswith(str(self.repo_root)):
                logger.warning(f"Path traversal attempt: {path}")
                return None
            if self._is_excluded(abs_path):
                return None
            return abs_path
        except Exception as e:
            logger.warning(f"Invalid path {path}: {e}")
            return None

    def get_repo_fingerprint(self) -> str:
        """Get repository fingerprint using git HEAD or file hashes."""
        try:
            # Try git first
            git_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if git_head.returncode == 0:
                return git_head.stdout.strip()
        except Exception:
            pass

        # Fallback: hash directory structure
        hasher = hashlib.sha256()
        for root, dirs, files in os.walk(self.repo_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            for file in files:
                file_path = Path(root) / file
                if not self._is_excluded(file_path):
                    hasher.update(file.encode())
                    try:
                        with open(file_path, "rb") as f:
                            hasher.update(f.read(self.max_file_size))
                    except Exception:
                        pass

        return hasher.hexdigest()

    def get_repo_tree(self) -> dict[str, Any]:
        """Get repository tree structure using git ls-files (respects .gitignore)."""
        tree = {}

        # Try git ls-files first for .gitignore-aware listing
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
                    if not filepath:
                        continue
                    parts = filepath.split("/")
                    current_level = tree
                    for part in parts[:-1]:
                        if part not in current_level:
                            current_level[part] = {}
                        current_level = current_level[part]
                    current_level[parts[-1]] = "file"
                return tree
        except Exception:
            pass

        # Fallback to os.walk
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            rel_path = Path(root).relative_to(self.repo_root)
            current_level = tree

            for part in rel_path.parts:
                if not part:
                    continue
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

            for file in files:
                file_path = Path(root) / file
                if not self._is_excluded(file_path):
                    current_level[file] = "file"

        return tree

    def read_file_safe(self, file_path: str) -> str | None:
        """Read file with size limits and path safety checks."""
        # Resolve relative paths against repo_root
        path = self.repo_root / file_path
        path = self._safe_path(path)
        if not path or not path.is_file():
            return None

        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read(self.max_file_read)
                return content
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return None

    def grep_repo(
        self, pattern: str, file_extensions: list[str] | None = None
    ) -> dict[str, list[str]]:
        """Search for pattern in repository files using git grep (respects .gitignore)."""
        if file_extensions is None:
            file_extensions = ALL_EXTENSIONS

        # Try git grep first for speed and .gitignore support
        try:
            cmd = ["git", "grep", "-n", "--no-color", "-I", pattern]
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode in (0, 1):  # 1 = no matches
                results = {}
                for line in result.stdout.strip().split("\n"):
                    if not line or ":" not in line:
                        continue
                    # git grep -n format: "file:lineno:content"
                    file_path, _, rest = line.partition(":")
                    _, _, match_line = rest.partition(":")
                    if any(file_path.endswith(ext) for ext in file_extensions):
                        if file_path not in results:
                            results[file_path] = []
                        if len(results[file_path]) < self.max_grep_results:
                            results[file_path].append(match_line.strip())
                # Sort: source files first, then limit total files
                sorted_results = dict(sorted(results.items(), key=lambda x: _file_priority(x[0])))
                return dict(list(sorted_results.items())[: self.max_grep_results])
        except Exception as e:
            logger.debug(f"git grep failed, falling back to os.walk: {e}")

        # Fallback to pure-Python grep
        results = {}
        try:
            for root, dirs, files in os.walk(self.repo_root):
                dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

                for file in files:
                    if any(file.endswith(ext) for ext in file_extensions):
                        file_path = Path(root) / file
                        if not self._is_excluded(file_path):
                            content = self.read_file_safe(str(file_path))
                            if content:
                                lines = content.split("\n")
                                matches = [line for line in lines if pattern in line]
                                if matches:
                                    rel_path = str(file_path.relative_to(self.repo_root))
                                    results[rel_path] = matches[: self.max_grep_results]
                                    if len(results) >= self.max_grep_results:
                                        break
        except Exception as e:
            logger.warning(f"Grep failed: {e}")

        return results

    def get_file_list(self, extensions: list[str] | None = None, max_files: int = 500) -> list[str]:
        """Get list of files in repository using git ls-files (respects .gitignore)."""
        if extensions is None:
            extensions = ALL_EXTENSIONS

        # Try git ls-files first for .gitignore support
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                all_files = result.stdout.strip().split("\n")
                filtered = [f for f in all_files if any(f.endswith(ext) for ext in extensions)]
                # Sort: source code first, then configs, then docs
                filtered.sort(key=_file_priority)
                return filtered[:max_files]
        except Exception as e:
            logger.debug(f"git ls-files failed, falling back to os.walk: {e}")

        # Fallback to os.walk
        file_list = []
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = Path(root) / file
                    if not self._is_excluded(file_path):
                        rel_path = str(file_path.relative_to(self.repo_root))
                        file_list.append(rel_path)

        file_list.sort(key=_file_priority)
        return file_list[:max_files]
