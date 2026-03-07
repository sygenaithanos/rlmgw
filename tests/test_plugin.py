"""Tests for Claude Code plugin structure, skills, hooks, and MCP integration."""

import json
import os
import stat
import subprocess

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_DIR = os.path.join(ROOT, "plugins", "rlmgw")
PLUGIN_MANIFEST_DIR = os.path.join(PLUGIN_DIR, ".claude-plugin")
MARKETPLACE_DIR = os.path.join(ROOT, ".claude-plugin")


# ---------------------------------------------------------------------------
# Plugin manifest
# ---------------------------------------------------------------------------


class TestPluginManifest:
    """Validate plugins/rlmgw/.claude-plugin/plugin.json structure."""

    @pytest.fixture(autouse=True)
    def load_manifest(self):
        with open(os.path.join(PLUGIN_MANIFEST_DIR, "plugin.json")) as f:
            self.manifest = json.load(f)

    def test_required_fields(self):
        assert "name" in self.manifest
        assert "version" in self.manifest
        assert "description" in self.manifest

    def test_name_is_kebab_case(self):
        name = self.manifest["name"]
        assert name == name.lower()
        assert " " not in name
        assert name.replace("-", "").isalnum()

    def test_version_semver(self):
        parts = self.manifest["version"].split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# Marketplace catalog
# ---------------------------------------------------------------------------


class TestMarketplace:
    """Validate .claude-plugin/marketplace.json."""

    @pytest.fixture(autouse=True)
    def load_marketplace(self):
        with open(os.path.join(MARKETPLACE_DIR, "marketplace.json")) as f:
            self.marketplace = json.load(f)

    def test_has_plugins_list(self):
        assert "plugins" in self.marketplace
        assert isinstance(self.marketplace["plugins"], list)
        assert len(self.marketplace["plugins"]) >= 1

    def test_plugin_entries_have_required_fields(self):
        for plugin in self.marketplace["plugins"]:
            assert "name" in plugin
            assert "source" in plugin
            assert "description" in plugin

    def test_plugin_source_resolves_to_plugin_json(self):
        for plugin in self.marketplace["plugins"]:
            source = plugin["source"]
            manifest = os.path.join(ROOT, source, ".claude-plugin", "plugin.json")
            assert os.path.isfile(manifest), f"No plugin.json at {manifest}"

    def test_marketplace_names_match_manifests(self):
        for plugin in self.marketplace["plugins"]:
            source = plugin["source"]
            manifest_path = os.path.join(ROOT, source, ".claude-plugin", "plugin.json")
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert plugin["name"] == manifest["name"]


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


class TestSkills:
    """Validate skill SKILL.md files have correct frontmatter."""

    @pytest.fixture(autouse=True)
    def load_skills(self):
        skills_dir = os.path.join(PLUGIN_DIR, "skills")
        self.skills = {}
        for name in os.listdir(skills_dir):
            skill_file = os.path.join(skills_dir, name, "SKILL.md")
            if os.path.isfile(skill_file):
                with open(skill_file) as f:
                    self.skills[name] = f.read()

    def test_at_least_one_skill(self):
        assert len(self.skills) >= 1

    def test_skills_have_yaml_frontmatter(self):
        for name, content in self.skills.items():
            assert content.startswith("---"), f"Skill {name} missing YAML frontmatter"
            parts = content.split("---", 2)
            assert len(parts) >= 3, f"Skill {name} has malformed frontmatter"
            frontmatter = yaml.safe_load(parts[1])
            assert isinstance(frontmatter, dict), f"Skill {name} frontmatter not a dict"

    def test_skills_have_description(self):
        for name, content in self.skills.items():
            frontmatter = yaml.safe_load(content.split("---", 2)[1])
            assert "description" in frontmatter, f"Skill {name} missing description"
            assert len(frontmatter["description"]) > 10

    def test_skills_have_allowed_tools(self):
        for name, content in self.skills.items():
            frontmatter = yaml.safe_load(content.split("---", 2)[1])
            if not frontmatter.get("disable-model-invocation"):
                assert "allowed-tools" in frontmatter, f"Skill {name} missing allowed-tools"

    def test_skills_have_arguments_placeholder(self):
        for name, content in self.skills.items():
            assert "$ARGUMENTS" in content, f"Skill {name} missing $ARGUMENTS placeholder"

    def test_rlmgw_skill_references_mcp_tools(self):
        """The main rlmgw skill must reference the MCP tools."""
        content = self.skills["rlmgw"]
        frontmatter = yaml.safe_load(content.split("---", 2)[1])
        tools = frontmatter["allowed-tools"]
        assert "mcp__rlmgw__repo_select_context" in tools
        assert "mcp__rlmgw__repo_tree" in tools
        assert "mcp__rlmgw__vllm_status" in tools


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class TestHooks:
    """Validate hooks configuration and scripts."""

    @pytest.fixture(autouse=True)
    def load_hooks(self):
        with open(os.path.join(PLUGIN_DIR, "hooks", "hooks.json")) as f:
            self.hooks_config = json.load(f)

    def test_hooks_json_has_hooks_key(self):
        assert "hooks" in self.hooks_config

    def test_session_start_hook_defined(self):
        assert "SessionStart" in self.hooks_config["hooks"]
        matchers = self.hooks_config["hooks"]["SessionStart"]
        assert len(matchers) >= 1

    def test_hook_scripts_exist(self):
        for _event, matchers in self.hooks_config["hooks"].items():
            for matcher in matchers:
                for hook in matcher.get("hooks", []):
                    cmd = hook.get("command", "")
                    resolved = cmd.replace("${CLAUDE_PLUGIN_ROOT}", PLUGIN_DIR)
                    parts = resolved.split()
                    script = None
                    for part in parts:
                        if "/" in part and not part.startswith("-"):
                            script = part
                    if script:
                        assert os.path.isfile(script), f"Hook script not found: {script}"

    def test_session_start_script_executable(self):
        script = os.path.join(PLUGIN_DIR, "hooks", "session-start.sh")
        assert os.path.isfile(script)
        mode = os.stat(script).st_mode
        assert mode & stat.S_IXUSR, "session-start.sh not executable"

    def test_session_start_script_runs(self):
        """Hook script should run without errors (vLLM unavailable is fine)."""
        result = subprocess.run(
            ["bash", os.path.join(PLUGIN_DIR, "hooks", "session-start.sh")],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "RLMGW_UPSTREAM_BASE_URL": "http://localhost:99999/v1"},
        )
        assert result.returncode == 0
        assert "RLMgw:" in result.stdout


# ---------------------------------------------------------------------------
# MCP server config
# ---------------------------------------------------------------------------


class TestMCPConfig:
    """Validate .mcp.json configuration."""

    @pytest.fixture(autouse=True)
    def load_mcp(self):
        with open(os.path.join(PLUGIN_DIR, ".mcp.json")) as f:
            self.mcp = json.load(f)

    def test_has_mcp_servers_key(self):
        assert "mcpServers" in self.mcp

    def test_rlmgw_server_defined(self):
        assert "rlmgw" in self.mcp["mcpServers"]

    def test_server_uses_plugin_root_variable(self):
        """Paths must use ${CLAUDE_PLUGIN_ROOT} for remote install compatibility."""
        server = self.mcp["mcpServers"]["rlmgw"]
        args_str = " ".join(server["args"])
        assert "${CLAUDE_PLUGIN_ROOT}" in args_str

    def test_server_has_required_env_vars(self):
        server = self.mcp["mcpServers"]["rlmgw"]
        env = server.get("env", {})
        required = ["RLMGW_REPO_ROOT", "RLMGW_UPSTREAM_BASE_URL"]
        for var in required:
            assert var in env, f"Missing env var {var} in MCP config"


# ---------------------------------------------------------------------------
# MCP server stdio protocol
# ---------------------------------------------------------------------------


class TestMCPProtocol:
    """Test MCP server responds correctly over stdio JSON-RPC."""

    SERVER_CMD = [
        "uv",
        "--directory",
        PLUGIN_DIR,
        "run",
        "python3",
        "-m",
        "rlmgw.mcp_server",
    ]
    SERVER_ENV = {
        **os.environ,
        "RLMGW_REPO_ROOT": ROOT,
        "RLMGW_UPSTREAM_BASE_URL": "http://localhost:99999/v1",
    }

    @staticmethod
    def _send(proc, method, params=None, req_id=None):
        """Send a JSON-RPC message (request if req_id, notification otherwise)."""
        msg = {"jsonrpc": "2.0", "method": method}
        if req_id is not None:
            msg["id"] = req_id
        if params is not None:
            msg["params"] = params
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    @staticmethod
    def _read_response(proc, req_id, timeout=30):
        """Read lines until we get a JSON-RPC response matching req_id."""
        import select

        deadline = __import__("time").time() + timeout
        while __import__("time").time() < deadline:
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                continue
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" not in msg:
                continue
            if msg["id"] == req_id:
                return msg
        return None

    def _start_and_initialize(self):
        """Start the MCP server and complete the initialize handshake."""
        proc = subprocess.Popen(
            self.SERVER_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.SERVER_ENV,
            cwd=ROOT,
        )
        self._send(
            proc,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
            req_id=1,
        )
        resp = self._read_response(proc, req_id=1)
        assert resp is not None, "MCP server did not respond to initialize"
        assert "result" in resp
        self._send(proc, "notifications/initialized")
        return proc, resp

    def test_initialize_and_list_tools(self):
        """MCP server should respond to initialize and list tools."""
        proc, init_resp = self._start_and_initialize()
        try:
            assert init_resp["result"]["protocolVersion"] == "2024-11-05"

            self._send(proc, "tools/list", req_id=2)
            resp = self._read_response(proc, req_id=2)
            assert resp is not None
            assert "result" in resp
            tools = resp["result"]["tools"]
            tool_names = {t["name"] for t in tools}
            assert "repo_select_context" in tool_names
            assert "repo_tree" in tool_names
            assert "repo_fingerprint" in tool_names
            assert "vllm_status" in tool_names

            for tool in tools:
                assert "description" in tool
                assert "inputSchema" in tool
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_call_repo_tree_tool(self):
        """MCP server should execute repo_tree via tools/call."""
        proc, _ = self._start_and_initialize()
        try:
            self._send(
                proc,
                "tools/call",
                {
                    "name": "repo_tree",
                    "arguments": {},
                },
                req_id=2,
            )
            resp = self._read_response(proc, req_id=2)
            assert resp is not None
            assert "result" in resp
            content = resp["result"]["content"]
            assert len(content) >= 1
            tree = json.loads(content[0]["text"])
            assert "rlmgw" in tree
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_call_vllm_status_tool(self):
        """MCP server should return unavailable status when vLLM is down."""
        proc, _ = self._start_and_initialize()
        try:
            self._send(
                proc,
                "tools/call",
                {
                    "name": "vllm_status",
                    "arguments": {},
                },
                req_id=2,
            )
            resp = self._read_response(proc, req_id=2)
            assert resp is not None
            status = json.loads(resp["result"]["content"][0]["text"])
            assert status["status"] == "unavailable"
        finally:
            proc.terminate()
            proc.wait(timeout=5)
