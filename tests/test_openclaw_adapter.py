"""Tests for clampai.adapters.openclaw_adapter.

Covers every public class and method in the super-wrapper:

    OpenClawResponse         — dataclass fields
    GatewayHealth            — dataclass fields
    OpenClawGateway          — health, status, call, list_models,
                               list_sessions, search_memory, version,
                               async variants, repr
    OpenClawAdapter          — init, with_new_session, properties, complete,
                               complete_rich, acomplete, retry, session_id
                               propagation, agent_id, local, check_gateway
    AsyncOpenClawAdapter     — init, with_new_session, properties, acomplete
                               (blocking + streaming), retry, NotImplementedError
    openclaw_session         — context manager
    async_openclaw_session   — context manager
    helpers                  — _strip_ansi, _build_full_prompt, THINKING_LEVELS
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clampai.adapters.openclaw_adapter import (
    THINKING_LEVELS,
    AsyncOpenClawAdapter,
    GatewayHealth,
    OpenClawAdapter,
    OpenClawGateway,
    OpenClawResponse,
    _build_full_prompt,
    _strip_ansi,
    async_openclaw_session,
    openclaw_session,
)

# ── factories ─────────────────────────────────────────────────────────────────


def _cp(
    stdout: str = "Hello from OpenClaw",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    return cp


def _async_proc(
    stdout_bytes: bytes = b"Hello from OpenClaw",
    stderr_bytes: bytes = b"",
    returncode: int = 0,
) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    proc.stdout = None
    return proc


def _async_proc_streaming(
    lines: List[bytes],
    returncode: int = 0,
) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    mock_stdout = MagicMock()
    mock_stdout.readline = AsyncMock(side_effect=[*lines, b""])
    proc.stdout = mock_stdout
    return proc


# ── helpers ───────────────────────────────────────────────────────────────────


class TestStripAnsi:
    def test_strips_color_codes(self):
        assert _strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_strips_cursor_movement(self):
        assert _strip_ansi("\x1b[2Jhello") == "hello"

    def test_no_ansi_unchanged(self):
        assert _strip_ansi("plain text") == "plain text"

    def test_empty_string(self):
        assert _strip_ansi("") == ""

    def test_mixed_ansi_and_text(self):
        assert _strip_ansi("\x1b[1mBold\x1b[0m and normal") == "Bold and normal"


class TestBuildFullPrompt:
    def test_with_system_prompt(self):
        assert _build_full_prompt("user msg", "sys") == "sys\n\nuser msg"

    def test_without_system_prompt(self):
        assert _build_full_prompt("user msg", "") == "user msg"

    def test_empty_system_prompt_returns_prompt_only(self):
        assert _build_full_prompt("only", "") == "only"


class TestThinkingLevels:
    def test_all_six_levels_present(self):
        for level in ("off", "minimal", "low", "medium", "high", "xhigh"):
            assert level in THINKING_LEVELS

    def test_exactly_six_levels(self):
        assert len(THINKING_LEVELS) == 6


# ── OpenClawResponse ──────────────────────────────────────────────────────────


class TestOpenClawResponse:
    def test_required_text_field(self):
        r = OpenClawResponse(text="hello")
        assert r.text == "hello"

    def test_optional_fields_default_none(self):
        r = OpenClawResponse(text="x")
        assert r.session_id is None
        assert r.thinking_level is None
        assert r.raw is None

    def test_all_fields_set(self):
        r = OpenClawResponse(
            text="answer",
            session_id="s-123",
            thinking_level="high",
            raw="answer",
        )
        assert r.session_id == "s-123"
        assert r.thinking_level == "high"
        assert r.raw == "answer"


# ── GatewayHealth ─────────────────────────────────────────────────────────────


class TestGatewayHealth:
    def test_running_true(self):
        h = GatewayHealth(running=True)
        assert h.running is True

    def test_running_false(self):
        h = GatewayHealth(running=False)
        assert h.running is False

    def test_default_url(self):
        h = GatewayHealth(running=True)
        assert "18789" in h.url

    def test_latency_and_raw(self):
        h = GatewayHealth(running=True, latency_ms=5.2, raw={"status": "ok"})
        assert h.latency_ms == pytest.approx(5.2)
        assert h.raw == {"status": "ok"}


# ── OpenClawGateway ───────────────────────────────────────────────────────────


class TestOpenClawGatewayInit:
    def test_defaults(self):
        gw = OpenClawGateway()
        assert gw._executable == "openclaw"
        assert gw._timeout == pytest.approx(15.0)

    def test_custom_values(self):
        gw = OpenClawGateway(executable="/opt/oc", timeout=5.0)
        assert gw._executable == "/opt/oc"
        assert gw._timeout == pytest.approx(5.0)

    def test_repr(self):
        assert "openclaw" in repr(OpenClawGateway())


class TestOpenClawGatewayHealth:
    def test_running_when_exit_zero(self):
        cp = _cp(stdout='{"status":"ok"}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            h = OpenClawGateway().health()
        assert h.running is True

    def test_not_running_when_exit_nonzero(self):
        cp = _cp(stdout="", returncode=1)
        with patch("subprocess.run", return_value=cp):
            h = OpenClawGateway().health()
        assert h.running is False

    def test_not_running_on_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("no oc")):
            h = OpenClawGateway().health()
        assert h.running is False

    def test_not_running_on_timeout(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            h = OpenClawGateway().health()
        assert h.running is False

    def test_not_running_on_oserror(self):
        with patch("subprocess.run", side_effect=OSError("socket error")):
            h = OpenClawGateway().health()
        assert h.running is False

    def test_latency_set_on_success(self):
        cp = _cp(stdout='{"running":true}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            h = OpenClawGateway().health()
        assert h.latency_ms is not None
        assert h.latency_ms >= 0

    def test_raw_parsed_from_json(self):
        cp = _cp(stdout='{"gateway":"up","port":18789}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            h = OpenClawGateway().health()
        assert h.raw == {"gateway": "up", "port": 18789}

    def test_raw_none_on_non_json_output(self):
        cp = _cp(stdout="Gateway is up", returncode=0)
        with patch("subprocess.run", return_value=cp):
            h = OpenClawGateway().health()
        assert h.raw is None

    def test_uses_json_flag_in_cmd(self):
        cp = _cp(stdout="{}", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().health()
        cmd = mock_run.call_args.args[0]
        assert "--json" in cmd
        assert "gateway" in cmd
        assert "health" in cmd

    def test_ansi_stripped_from_json(self):
        raw = "\x1b[32m" + '{"ok":true}' + "\x1b[0m"
        cp = _cp(stdout=raw, returncode=0)
        with patch("subprocess.run", return_value=cp):
            h = OpenClawGateway().health()
        assert h.raw == {"ok": True}


class TestOpenClawGatewayStatus:
    def test_returns_dict_on_success(self):
        cp = _cp(stdout='{"sessions":3}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            s = OpenClawGateway().status()
        assert s == {"sessions": 3}

    def test_returns_empty_dict_on_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            s = OpenClawGateway().status()
        assert s == {}

    def test_returns_empty_dict_on_timeout(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            s = OpenClawGateway().status()
        assert s == {}

    def test_returns_empty_dict_on_non_json(self):
        cp = _cp(stdout="not json", returncode=0)
        with patch("subprocess.run", return_value=cp):
            s = OpenClawGateway().status()
        assert s == {}

    def test_deep_flag_added(self):
        cp = _cp(stdout="{}", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().status(deep=True)
        cmd = mock_run.call_args.args[0]
        assert "--deep" in cmd

    def test_returns_empty_dict_on_nonzero_exit(self):
        cp = _cp(stdout="", returncode=1)
        with patch("subprocess.run", return_value=cp):
            s = OpenClawGateway().status()
        assert s == {}


class TestOpenClawGatewayCall:
    def test_returns_parsed_json(self):
        cp = _cp(stdout='{"result":"ok"}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            result = OpenClawGateway().call("health")
        assert result == {"result": "ok"}

    def test_returns_raw_text_when_not_json(self):
        cp = _cp(stdout="plain text response", returncode=0)
        with patch("subprocess.run", return_value=cp):
            result = OpenClawGateway().call("health")
        assert result == "plain text response"

    def test_params_serialized_to_json(self):
        cp = _cp(stdout='{"ok":true}', returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().call("status.get", params={"deep": True})
        cmd = mock_run.call_args.args[0]
        assert "--params" in cmd
        params_idx = cmd.index("--params") + 1
        assert json.loads(cmd[params_idx]) == {"deep": True}

    def test_raises_runtime_error_on_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="not found"):
                OpenClawGateway().call("health")

    def test_raises_runtime_error_on_nonzero_exit(self):
        cp = _cp(stdout="", stderr="error", returncode=1)
        with patch("subprocess.run", return_value=cp):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                OpenClawGateway().call("bad.method")

    def test_raises_timeout_error_on_timeout(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            with pytest.raises(TimeoutError, match="timed out"):
                OpenClawGateway().call("slow")

    def test_method_in_cmd(self):
        cp = _cp(stdout="{}", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().call("my.rpc")
        cmd = mock_run.call_args.args[0]
        assert "my.rpc" in cmd

    def test_no_params_flag_when_params_is_none(self):
        cp = _cp(stdout="{}", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().call("health")
        cmd = mock_run.call_args.args[0]
        assert "--params" not in cmd


class TestOpenClawGatewayListModels:
    def test_returns_list_on_success(self):
        cp = _cp(stdout='[{"id":"anthropic/claude-opus-4-6"}]', returncode=0)
        with patch("subprocess.run", return_value=cp):
            models = OpenClawGateway().list_models()
        assert models == [{"id": "anthropic/claude-opus-4-6"}]

    def test_returns_empty_list_on_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            assert OpenClawGateway().list_models() == []

    def test_returns_empty_list_on_timeout(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            assert OpenClawGateway().list_models() == []

    def test_returns_empty_list_on_non_json(self):
        cp = _cp(stdout="not json", returncode=0)
        with patch("subprocess.run", return_value=cp):
            assert OpenClawGateway().list_models() == []

    def test_wraps_non_list_response(self):
        cp = _cp(stdout='{"model":"x"}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            result = OpenClawGateway().list_models()
        assert result == [{"model": "x"}]

    def test_uses_json_flag(self):
        cp = _cp(stdout="[]", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().list_models()
        cmd = mock_run.call_args.args[0]
        assert "--json" in cmd
        assert "models" in cmd


class TestOpenClawGatewayListSessions:
    def test_returns_list_on_success(self):
        cp = _cp(stdout='[{"id":"abc","active":true}]', returncode=0)
        with patch("subprocess.run", return_value=cp):
            sessions = OpenClawGateway().list_sessions()
        assert sessions == [{"id": "abc", "active": True}]

    def test_returns_empty_list_on_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            assert OpenClawGateway().list_sessions() == []

    def test_agent_id_in_cmd(self):
        cp = _cp(stdout="[]", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().list_sessions(agent_id="myagent")
        cmd = mock_run.call_args.args[0]
        assert "--agent" in cmd
        assert "myagent" in cmd

    def test_all_agents_flag(self):
        cp = _cp(stdout="[]", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().list_sessions(all_agents=True)
        cmd = mock_run.call_args.args[0]
        assert "--all-agents" in cmd

    def test_wraps_non_list_dict(self):
        cp = _cp(stdout='{"session":"x"}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            result = OpenClawGateway().list_sessions()
        assert result == [{"session": "x"}]


class TestOpenClawGatewaySearchMemory:
    def test_returns_text_on_success(self):
        cp = _cp(stdout="Found: invoice #42", returncode=0)
        with patch("subprocess.run", return_value=cp):
            result = OpenClawGateway().search_memory("unpaid invoices")
        assert result == "Found: invoice #42"

    def test_raises_runtime_error_when_cli_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="not found"):
                OpenClawGateway().search_memory("query")

    def test_returns_empty_on_timeout(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            assert OpenClawGateway().search_memory("q") == ""

    def test_query_in_cmd(self):
        cp = _cp(stdout="result", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().search_memory("find invoices")
        cmd = mock_run.call_args.args[0]
        assert "find invoices" in cmd
        assert "memory" in cmd

    def test_agent_id_in_cmd(self):
        cp = _cp(stdout="result", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().search_memory("query", agent_id="assistant")
        cmd = mock_run.call_args.args[0]
        assert "--agent" in cmd
        assert "assistant" in cmd

    def test_ansi_stripped(self):
        cp = _cp(stdout="\x1b[32mresult\x1b[0m", returncode=0)
        with patch("subprocess.run", return_value=cp):
            result = OpenClawGateway().search_memory("q")
        assert "\x1b" not in result
        assert "result" in result


class TestOpenClawGatewayVersion:
    def test_returns_version_string(self):
        cp = _cp(stdout="2026.2.26", returncode=0)
        with patch("subprocess.run", return_value=cp):
            v = OpenClawGateway().version()
        assert v == "2026.2.26"

    def test_returns_empty_on_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            assert OpenClawGateway().version() == ""

    def test_returns_empty_on_timeout(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            assert OpenClawGateway().version() == ""

    def test_version_flag_in_cmd(self):
        cp = _cp(stdout="2026.2.26", returncode=0)
        with patch("subprocess.run", return_value=cp) as mock_run:
            OpenClawGateway().version()
        cmd = mock_run.call_args.args[0]
        assert "--version" in cmd


class TestOpenClawGatewayAsyncVariants:
    def test_health_async_returns_gateway_health(self):
        cp = _cp(stdout='{"ok":true}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            h = asyncio.run(OpenClawGateway().health_async())
        assert isinstance(h, GatewayHealth)
        assert h.running is True

    def test_status_async_returns_dict(self):
        cp = _cp(stdout='{"sessions":1}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            s = asyncio.run(OpenClawGateway().status_async())
        assert s == {"sessions": 1}

    def test_call_async_returns_json(self):
        cp = _cp(stdout='{"result":"ok"}', returncode=0)
        with patch("subprocess.run", return_value=cp):
            r = asyncio.run(OpenClawGateway().call_async("health"))
        assert r == {"result": "ok"}

    def test_list_models_async_returns_list(self):
        cp = _cp(stdout='[{"id":"m1"}]', returncode=0)
        with patch("subprocess.run", return_value=cp):
            models = asyncio.run(OpenClawGateway().list_models_async())
        assert models == [{"id": "m1"}]

    def test_list_sessions_async_returns_list(self):
        cp = _cp(stdout='[{"id":"s1"}]', returncode=0)
        with patch("subprocess.run", return_value=cp):
            sessions = asyncio.run(OpenClawGateway().list_sessions_async())
        assert sessions == [{"id": "s1"}]

    def test_search_memory_async_returns_text(self):
        cp = _cp(stdout="hit: file.txt", returncode=0)
        with patch("subprocess.run", return_value=cp):
            hits = asyncio.run(OpenClawGateway().search_memory_async("query"))
        assert hits == "hit: file.txt"

    def test_version_async_returns_string(self):
        cp = _cp(stdout="2026.2.26", returncode=0)
        with patch("subprocess.run", return_value=cp):
            v = asyncio.run(OpenClawGateway().version_async())
        assert v == "2026.2.26"


# ── OpenClawAdapter — init ────────────────────────────────────────────────────


class TestOpenClawAdapterInit:
    def test_defaults(self):
        a = OpenClawAdapter()
        assert a._executable == "openclaw"
        assert a._thinking == "low"
        assert a._timeout == pytest.approx(60.0)
        assert a._default_system_prompt == ""
        assert a._extra_args == []
        assert a._session_id is None
        assert a._agent_id is None
        assert a._local is False
        assert a._max_retries == 1

    def test_custom_values(self):
        a = OpenClawAdapter(
            executable="/usr/local/bin/openclaw",
            thinking="high",
            timeout=30.0,
            default_system_prompt="Be concise.",
            extra_args=["--verbose"],
            session_id="s-abc",
            agent_id="assistant",
            local=True,
            max_retries=3,
        )
        assert a._executable == "/usr/local/bin/openclaw"
        assert a._thinking == "high"
        assert a._timeout == pytest.approx(30.0)
        assert a._default_system_prompt == "Be concise."
        assert a._extra_args == ["--verbose"]
        assert a._session_id == "s-abc"
        assert a._agent_id == "assistant"
        assert a._local is True
        assert a._max_retries == 3

    def test_invalid_thinking_raises(self):
        with pytest.raises(ValueError, match="thinking must be one of"):
            OpenClawAdapter(thinking="ultra")

    def test_negative_max_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            OpenClawAdapter(max_retries=-1)

    def test_all_valid_thinking_levels(self):
        for level in THINKING_LEVELS:
            a = OpenClawAdapter(thinking=level)
            assert a._thinking == level

    def test_extra_args_copied(self):
        original = ["--verbose"]
        a = OpenClawAdapter(extra_args=original)
        original.append("--tamper")
        assert "--tamper" not in a._extra_args

    def test_check_gateway_raises_when_unhealthy(self):
        h = GatewayHealth(running=False)
        with patch.object(OpenClawGateway, "health", return_value=h):
            with pytest.raises(RuntimeError, match="Gateway is not running"):
                OpenClawAdapter(check_gateway=True)

    def test_check_gateway_passes_when_healthy(self):
        h = GatewayHealth(running=True)
        with patch.object(OpenClawGateway, "health", return_value=h):
            a = OpenClawAdapter(check_gateway=True)
        assert a is not None

    def test_repr_minimal(self):
        r = repr(OpenClawAdapter(executable="oc", thinking="medium"))
        assert "oc" in r
        assert "medium" in r

    def test_repr_with_session_and_agent(self):
        r = repr(
            OpenClawAdapter(
                executable="oc",
                thinking="low",
                session_id="s-123",
                agent_id="bot",
                local=True,
            )
        )
        assert "s-123" in r
        assert "bot" in r
        assert "local=True" in r


class TestOpenClawAdapterWithNewSession:
    def test_generates_session_id(self):
        a = OpenClawAdapter.with_new_session()
        assert a._session_id is not None
        assert a._session_id.startswith("clampai-")

    def test_custom_prefix(self):
        a = OpenClawAdapter.with_new_session(prefix="myapp")
        assert a._session_id is not None
        assert a._session_id.startswith("myapp-")

    def test_unique_ids(self):
        ids = {OpenClawAdapter.with_new_session()._session_id for _ in range(10)}
        assert len(ids) == 10

    def test_kwargs_forwarded(self):
        a = OpenClawAdapter.with_new_session(thinking="high", max_retries=2)
        assert a._thinking == "high"
        assert a._max_retries == 2


class TestOpenClawAdapterProperties:
    def test_session_id_property(self):
        a = OpenClawAdapter(session_id="abc-123")
        assert a.session_id == "abc-123"

    def test_session_id_none_when_not_set(self):
        assert OpenClawAdapter().session_id is None

    def test_agent_id_property(self):
        a = OpenClawAdapter(agent_id="my-agent")
        assert a.agent_id == "my-agent"

    def test_agent_id_none_when_not_set(self):
        assert OpenClawAdapter().agent_id is None

    def test_gateway_property_returns_gateway(self):
        a = OpenClawAdapter(executable="/opt/oc")
        gw = a.gateway
        assert isinstance(gw, OpenClawGateway)
        assert gw._executable == "/opt/oc"


# ── OpenClawAdapter — complete() ──────────────────────────────────────────────


class TestOpenClawAdapterComplete:
    def test_success_returns_text(self):
        with patch("subprocess.run", return_value=_cp(stdout="The answer is 42.")):
            assert OpenClawAdapter().complete("What is 42?") == "The answer is 42."

    def test_prompt_in_cmd(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter().complete("my prompt")
        cmd = m.call_args.args[0]
        assert "--message" in cmd
        assert "my prompt" in cmd[cmd.index("--message") + 1]

    def test_thinking_in_cmd(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter(thinking="xhigh").complete("prompt")
        cmd = m.call_args.args[0]
        assert cmd[cmd.index("--thinking") + 1] == "xhigh"

    def test_session_id_in_cmd(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter(session_id="s-999").complete("prompt")
        cmd = m.call_args.args[0]
        assert "--session-id" in cmd
        assert "s-999" in cmd

    def test_no_session_id_when_not_set(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter().complete("prompt")
        cmd = m.call_args.args[0]
        assert "--session-id" not in cmd

    def test_agent_id_in_cmd(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter(agent_id="analyst").complete("prompt")
        cmd = m.call_args.args[0]
        assert "--agent" in cmd
        assert "analyst" in cmd

    def test_no_agent_id_when_not_set(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter().complete("prompt")
        cmd = m.call_args.args[0]
        assert "--agent" not in cmd

    def test_local_flag_in_cmd(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter(local=True).complete("prompt")
        cmd = m.call_args.args[0]
        assert "--local" in cmd

    def test_no_local_flag_by_default(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter().complete("prompt")
        cmd = m.call_args.args[0]
        assert "--local" not in cmd

    def test_system_prompt_prepended(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter().complete("do X", system_prompt="Be helpful.")
        cmd = m.call_args.args[0]
        msg = cmd[cmd.index("--message") + 1]
        assert "Be helpful." in msg
        assert "do X" in msg

    def test_default_system_prompt_used_when_empty(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter(default_system_prompt="Default ctx.").complete("q")
        cmd = m.call_args.args[0]
        assert "Default ctx." in cmd[cmd.index("--message") + 1]

    def test_explicit_system_prompt_overrides_default(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter(default_system_prompt="Default.").complete(
                "q", system_prompt="Override."
            )
        cmd = m.call_args.args[0]
        msg = cmd[cmd.index("--message") + 1]
        assert "Override." in msg
        assert "Default." not in msg

    def test_extra_args_appended(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            OpenClawAdapter(extra_args=["--verbose"]).complete("p")
        cmd = m.call_args.args[0]
        assert "--verbose" in cmd

    def test_ansi_stripped(self):
        with patch("subprocess.run", return_value=_cp(stdout="\x1b[32mGreen\x1b[0m")):
            assert OpenClawAdapter().complete("p") == "Green"

    def test_nonzero_exit_raises(self):
        with patch(
            "subprocess.run",
            return_value=_cp(stdout="", stderr="fatal", returncode=1),
        ):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                OpenClawAdapter().complete("p")

    def test_empty_response_raises(self):
        with patch("subprocess.run", return_value=_cp(stdout="", returncode=0)):
            with pytest.raises(RuntimeError, match="empty response"):
                OpenClawAdapter().complete("p")

    def test_whitespace_only_raises(self):
        with patch("subprocess.run", return_value=_cp(stdout="  \n  ", returncode=0)):
            with pytest.raises(RuntimeError, match="empty response"):
                OpenClawAdapter().complete("p")

    def test_file_not_found_raises(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="not found"):
                OpenClawAdapter(executable="missing").complete("p")

    def test_timeout_raises(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            with pytest.raises(TimeoutError, match="timed out"):
                OpenClawAdapter(timeout=5.0).complete("p")

    def test_stderr_in_error_message(self):
        with patch(
            "subprocess.run",
            return_value=_cp(stdout="", stderr="\x1b[31mCritical failure\x1b[0m", returncode=2),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                OpenClawAdapter().complete("p")
        assert "Critical failure" in str(exc_info.value)

    def test_whitespace_trimmed_from_response(self):
        with patch("subprocess.run", return_value=_cp(stdout="  result  ")):
            assert OpenClawAdapter().complete("p") == "result"

    def test_stream_tokens_called_per_line(self):
        with patch(
            "subprocess.run",
            return_value=_cp(stdout="line one\nline two\nline three"),
        ):
            collected: List[str] = []
            result = OpenClawAdapter().complete("p", stream_tokens=collected.append)
        assert len(collected) == 3
        assert result == "line one\nline two\nline three"

    def test_stream_tokens_skips_blank_lines(self):
        with patch(
            "subprocess.run",
            return_value=_cp(stdout="line1\n\n\nline2\n"),
        ):
            collected: List[str] = []
            OpenClawAdapter().complete("p", stream_tokens=collected.append)
        assert all(tok.strip() for tok in collected)

    def test_temperature_accepted(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")):
            OpenClawAdapter().complete("p", temperature=0.9)

    def test_max_tokens_accepted(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")):
            OpenClawAdapter().complete("p", max_tokens=500)


class TestOpenClawAdapterRetry:
    def test_retries_on_runtime_error(self):
        side_effects = [
            RuntimeError("transient error"),
            _cp(stdout="ok on second try"),
        ]
        call_count = 0

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            effect = side_effects[call_count - 1]
            if isinstance(effect, Exception):
                raise effect
            return effect

        with patch("subprocess.run", side_effect=fake_run):
            with patch("time.sleep"):
                result = OpenClawAdapter(max_retries=1).complete("p")
        assert result == "ok on second try"
        assert call_count == 2

    def test_raises_after_exhausting_retries(self):
        with patch(
            "subprocess.run",
            side_effect=RuntimeError("always fails"),
        ):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="always fails"):
                    OpenClawAdapter(max_retries=2).complete("p")

    def test_timeout_not_retried(self):
        call_count = 0

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise subprocess.TimeoutExpired("cmd", 5)

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(TimeoutError):
                OpenClawAdapter(max_retries=3).complete("p")
        assert call_count == 1

    def test_zero_retries_raises_immediately(self):
        call_count = 0

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError):
                OpenClawAdapter(max_retries=0).complete("p")
        assert call_count == 1

    def test_exponential_backoff_applied(self):
        side_effects = [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            _cp(stdout="ok"),
        ]
        call_count = 0
        sleep_calls: List[float] = []

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            effect = side_effects[call_count - 1]
            if isinstance(effect, Exception):
                raise effect
            return effect

        def fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with patch("subprocess.run", side_effect=fake_run):
            with patch("time.sleep", side_effect=fake_sleep):
                OpenClawAdapter(max_retries=2).complete("p")
        assert len(sleep_calls) == 2
        assert sleep_calls[1] > sleep_calls[0]


class TestOpenClawAdapterCompleteRich:
    def test_returns_open_claw_response(self):
        with patch("subprocess.run", return_value=_cp(stdout="rich response")):
            r = OpenClawAdapter(
                thinking="medium", session_id="s-rich"
            ).complete_rich("prompt")
        assert isinstance(r, OpenClawResponse)
        assert r.text == "rich response"
        assert r.session_id == "s-rich"
        assert r.thinking_level == "medium"
        assert r.raw == "rich response"

    def test_session_id_none_when_not_set(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")):
            r = OpenClawAdapter().complete_rich("p")
        assert r.session_id is None

    def test_raises_on_empty_response(self):
        with patch("subprocess.run", return_value=_cp(stdout="", returncode=0)):
            with pytest.raises(RuntimeError):
                OpenClawAdapter().complete_rich("p")


class TestOpenClawAdapterAcomplete:
    def test_acomplete_delegates_to_complete(self):
        with patch("subprocess.run", return_value=_cp(stdout="async result")):
            result = asyncio.run(OpenClawAdapter().acomplete("prompt"))
        assert result == "async result"

    def test_acomplete_passes_system_prompt(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")) as m:
            asyncio.run(OpenClawAdapter().acomplete("q", system_prompt="Sys."))
        cmd = m.call_args.args[0]
        assert "Sys." in cmd[cmd.index("--message") + 1]


# ── AsyncOpenClawAdapter — init ───────────────────────────────────────────────


class TestAsyncOpenClawAdapterInit:
    def test_defaults(self):
        a = AsyncOpenClawAdapter()
        assert a._executable == "openclaw"
        assert a._thinking == "low"
        assert a._timeout == pytest.approx(60.0)
        assert a._default_system_prompt == ""
        assert a._extra_args == []
        assert a._session_id is None
        assert a._agent_id is None
        assert a._local is False
        assert a._max_retries == 1

    def test_invalid_thinking_raises(self):
        with pytest.raises(ValueError, match="thinking must be one of"):
            AsyncOpenClawAdapter(thinking="turbo")

    def test_negative_max_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            AsyncOpenClawAdapter(max_retries=-1)

    def test_complete_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="async-only"):
            AsyncOpenClawAdapter().complete("hello")

    def test_repr_minimal(self):
        r = repr(AsyncOpenClawAdapter(executable="oc", thinking="xhigh"))
        assert "oc" in r
        assert "xhigh" in r

    def test_repr_with_session_agent_local(self):
        r = repr(
            AsyncOpenClawAdapter(
                executable="oc",
                thinking="low",
                session_id="s-x",
                agent_id="bot",
                local=True,
            )
        )
        assert "s-x" in r
        assert "bot" in r
        assert "local=True" in r

    def test_extra_args_copied(self):
        original = ["--session", "s1"]
        a = AsyncOpenClawAdapter(extra_args=original)
        original.append("tamper")
        assert "tamper" not in a._extra_args

    def test_check_gateway_raises_when_unhealthy(self):
        h = GatewayHealth(running=False)
        with patch.object(OpenClawGateway, "health", return_value=h):
            with pytest.raises(RuntimeError, match="Gateway is not running"):
                AsyncOpenClawAdapter(check_gateway=True)

    def test_check_gateway_passes_when_healthy(self):
        h = GatewayHealth(running=True)
        with patch.object(OpenClawGateway, "health", return_value=h):
            a = AsyncOpenClawAdapter(check_gateway=True)
        assert a is not None


class TestAsyncOpenClawAdapterWithNewSession:
    def test_generates_session_id(self):
        a = AsyncOpenClawAdapter.with_new_session()
        assert a._session_id is not None
        assert a._session_id.startswith("clampai-")

    def test_custom_prefix(self):
        a = AsyncOpenClawAdapter.with_new_session(prefix="taskbot")
        assert a._session_id is not None
        assert a._session_id.startswith("taskbot-")

    def test_unique_ids(self):
        ids = {AsyncOpenClawAdapter.with_new_session()._session_id for _ in range(10)}
        assert len(ids) == 10


class TestAsyncOpenClawAdapterProperties:
    def test_session_id_property(self):
        a = AsyncOpenClawAdapter(session_id="s-async")
        assert a.session_id == "s-async"

    def test_agent_id_property(self):
        a = AsyncOpenClawAdapter(agent_id="async-bot")
        assert a.agent_id == "async-bot"

    def test_gateway_property(self):
        a = AsyncOpenClawAdapter(executable="/opt/oc")
        gw = a.gateway
        assert isinstance(gw, OpenClawGateway)
        assert gw._executable == "/opt/oc"


# ── AsyncOpenClawAdapter — acomplete() blocking ───────────────────────────────


class TestAsyncAdapterAcompleteBlocking:
    def test_success_returns_text(self):
        proc = _async_proc(stdout_bytes=b"Async response")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = asyncio.run(AsyncOpenClawAdapter().acomplete("prompt"))
        assert result == "Async response"

    def test_session_id_in_cmd(self):
        proc = _async_proc(stdout_bytes=b"ok")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as m:
            asyncio.run(AsyncOpenClawAdapter(session_id="s-777").acomplete("p"))
        cmd_args = m.call_args.args
        assert "--session-id" in cmd_args
        idx = list(cmd_args).index("--session-id")
        assert cmd_args[idx + 1] == "s-777"

    def test_no_session_id_when_not_set(self):
        proc = _async_proc(stdout_bytes=b"ok")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as m:
            asyncio.run(AsyncOpenClawAdapter().acomplete("p"))
        cmd_args = m.call_args.args
        assert "--session-id" not in cmd_args

    def test_agent_id_in_cmd(self):
        proc = _async_proc(stdout_bytes=b"ok")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as m:
            asyncio.run(AsyncOpenClawAdapter(agent_id="mybot").acomplete("p"))
        cmd_args = m.call_args.args
        assert "--agent" in cmd_args
        idx = list(cmd_args).index("--agent")
        assert cmd_args[idx + 1] == "mybot"

    def test_local_flag_in_cmd(self):
        proc = _async_proc(stdout_bytes=b"ok")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as m:
            asyncio.run(AsyncOpenClawAdapter(local=True).acomplete("p"))
        cmd_args = m.call_args.args
        assert "--local" in cmd_args

    def test_thinking_in_cmd(self):
        proc = _async_proc(stdout_bytes=b"ok")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as m:
            asyncio.run(AsyncOpenClawAdapter(thinking="high").acomplete("p"))
        cmd_args = m.call_args.args
        idx = list(cmd_args).index("--thinking")
        assert cmd_args[idx + 1] == "high"

    def test_system_prompt_prepended(self):
        proc = _async_proc(stdout_bytes=b"ok")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as m:
            asyncio.run(AsyncOpenClawAdapter().acomplete("task", system_prompt="Ctx."))
        cmd_args = m.call_args.args
        idx = list(cmd_args).index("--message")
        assert "Ctx." in cmd_args[idx + 1]

    def test_ansi_stripped(self):
        proc = _async_proc(stdout_bytes=b"\x1b[32mGreen\x1b[0m")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = asyncio.run(AsyncOpenClawAdapter().acomplete("p"))
        assert result == "Green"

    def test_nonzero_exit_raises(self):
        proc = _async_proc(stdout_bytes=b"", stderr_bytes=b"err", returncode=1)
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                asyncio.run(AsyncOpenClawAdapter().acomplete("p"))

    def test_empty_response_raises(self):
        proc = _async_proc(stdout_bytes=b"", returncode=0)
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with pytest.raises(RuntimeError, match="empty response"):
                asyncio.run(AsyncOpenClawAdapter().acomplete("p"))

    def test_file_not_found_raises(self):
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=FileNotFoundError()),
        ):
            with pytest.raises(RuntimeError, match="not found"):
                asyncio.run(AsyncOpenClawAdapter(executable="missing").acomplete("p"))

    def test_timeout_raises_and_kills(self):
        proc = _async_proc()
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with patch("asyncio.wait_for", AsyncMock(side_effect=asyncio.TimeoutError())):
                with pytest.raises(TimeoutError, match="timed out"):
                    asyncio.run(AsyncOpenClawAdapter(timeout=1.0).acomplete("p"))
        proc.kill.assert_called_once()

    def test_temperature_accepted(self):
        proc = _async_proc(stdout_bytes=b"ok")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            asyncio.run(AsyncOpenClawAdapter().acomplete("p", temperature=1.0))

    def test_max_tokens_accepted(self):
        proc = _async_proc(stdout_bytes=b"ok")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            asyncio.run(AsyncOpenClawAdapter().acomplete("p", max_tokens=9999))


# ── AsyncOpenClawAdapter — acomplete() streaming ─────────────────────────────


class TestAsyncAdapterStreaming:
    def test_stream_tokens_called_per_line(self):
        proc = _async_proc_streaming(
            lines=[b"line one\n", b"line two\n", b"line three\n"]
        )
        collected: List[str] = []
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = asyncio.run(
                AsyncOpenClawAdapter().acomplete("p", stream_tokens=collected.append)
            )
        assert len(collected) == 3
        assert "line one" in result
        assert "line two" in result

    def test_streaming_returns_full_text(self):
        proc = _async_proc_streaming(lines=[b"part1\n", b"part2\n"])
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = asyncio.run(
                AsyncOpenClawAdapter().acomplete("p", stream_tokens=lambda t: None)
            )
        assert "part1" in result
        assert "part2" in result

    def test_streaming_strips_ansi(self):
        proc = _async_proc_streaming(lines=[b"\x1b[32mGreen line\x1b[0m\n"])
        collected: List[str] = []
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            asyncio.run(
                AsyncOpenClawAdapter().acomplete("p", stream_tokens=collected.append)
            )
        assert "\x1b" not in collected[0]
        assert "Green line" in collected[0]

    def test_streaming_blank_lines_not_emitted(self):
        proc = _async_proc_streaming(
            lines=[b"real\n", b"\n", b"   \n", b"another\n"]
        )
        collected: List[str] = []
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            asyncio.run(
                AsyncOpenClawAdapter().acomplete("p", stream_tokens=collected.append)
            )
        assert all(tok.strip() for tok in collected)

    def test_streaming_empty_raises(self):
        # max_retries=0 prevents a second attempt from exhausting the mock's
        # side_effect list after the first attempt raises RuntimeError.
        proc = _async_proc_streaming(lines=[b"\n", b"   \n"])
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with pytest.raises(RuntimeError, match="empty response"):
                asyncio.run(
                    AsyncOpenClawAdapter(max_retries=0).acomplete(
                        "p", stream_tokens=lambda t: None
                    )
                )

    def test_streaming_nonzero_exit_raises(self):
        # max_retries=0 prevents retry exhausting the mock's side_effect list.
        proc = _async_proc_streaming(lines=[b"partial\n"], returncode=1)
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                asyncio.run(
                    AsyncOpenClawAdapter(max_retries=0).acomplete(
                        "p", stream_tokens=lambda t: None
                    )
                )

    def test_streaming_file_not_found_raises(self):
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=FileNotFoundError()),
        ):
            with pytest.raises(RuntimeError, match="not found"):
                asyncio.run(
                    AsyncOpenClawAdapter().acomplete(
                        "p", stream_tokens=lambda t: None
                    )
                )

    def test_streaming_timeout_raises_and_kills(self):
        proc = _async_proc_streaming(lines=[b"part\n"])
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with patch("asyncio.wait_for", AsyncMock(side_effect=asyncio.TimeoutError())):
                with pytest.raises(TimeoutError, match="timed out"):
                    asyncio.run(
                        AsyncOpenClawAdapter(timeout=0.01).acomplete(
                            "p", stream_tokens=lambda t: None
                        )
                    )
        proc.kill.assert_called_once()


class TestAsyncAdapterRetry:
    def test_retries_on_runtime_error_blocking(self):
        call_count = 0
        procs = [
            _async_proc(stdout_bytes=b"", stderr_bytes=b"err", returncode=1),
            _async_proc(stdout_bytes=b"ok on retry"),
        ]

        async def fake_exec(*args, **kwargs):
            nonlocal call_count
            proc = procs[call_count]
            call_count += 1
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with patch("asyncio.sleep", AsyncMock()):
                result = asyncio.run(
                    AsyncOpenClawAdapter(max_retries=1).acomplete("p")
                )
        assert result == "ok on retry"
        assert call_count == 2

    def test_timeout_not_retried_blocking(self):
        call_count = 0

        async def fake_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _async_proc()

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with patch(
                "asyncio.wait_for", AsyncMock(side_effect=asyncio.TimeoutError())
            ):
                with pytest.raises(TimeoutError):
                    asyncio.run(
                        AsyncOpenClawAdapter(max_retries=3).acomplete("p")
                    )
        assert call_count == 1


# ── context managers ──────────────────────────────────────────────────────────


class TestOpenClawSessionContextManager:
    def test_yields_adapter_with_session_id(self):
        with openclaw_session() as adapter:
            assert isinstance(adapter, OpenClawAdapter)
            assert adapter.session_id is not None

    def test_session_id_starts_with_prefix(self):
        with openclaw_session(prefix="test") as adapter:
            assert adapter.session_id is not None
            assert adapter.session_id.startswith("test-")

    def test_default_prefix_is_clampai(self):
        with openclaw_session() as adapter:
            assert adapter.session_id is not None
            assert adapter.session_id.startswith("clampai-")

    def test_thinking_propagated(self):
        with openclaw_session(thinking="high") as adapter:
            assert adapter._thinking == "high"

    def test_executable_propagated(self):
        with openclaw_session(executable="/opt/oc") as adapter:
            assert adapter._executable == "/opt/oc"

    def test_unique_session_ids_across_calls(self):
        ids = []
        for _ in range(5):
            with openclaw_session() as a:
                ids.append(a.session_id)
        assert len(set(ids)) == 5

    def test_kwargs_forwarded(self):
        with openclaw_session(max_retries=3) as adapter:
            assert adapter._max_retries == 3


class TestAsyncOpenClawSessionContextManager:
    def test_yields_async_adapter_with_session_id(self):
        async def _run():
            async with async_openclaw_session() as adapter:
                assert isinstance(adapter, AsyncOpenClawAdapter)
                assert adapter.session_id is not None

        asyncio.run(_run())

    def test_session_id_starts_with_prefix(self):
        async def _run():
            async with async_openclaw_session(prefix="async-test") as adapter:
                assert adapter.session_id is not None
                assert adapter.session_id.startswith("async-test-")

        asyncio.run(_run())

    def test_default_prefix_is_clampai(self):
        async def _run():
            async with async_openclaw_session() as adapter:
                assert adapter.session_id is not None
                assert adapter.session_id.startswith("clampai-")

        asyncio.run(_run())

    def test_thinking_propagated(self):
        async def _run():
            async with async_openclaw_session(thinking="medium") as adapter:
                assert adapter._thinking == "medium"

        asyncio.run(_run())

    def test_unique_session_ids(self):
        async def _run():
            ids = []
            for _ in range(5):
                async with async_openclaw_session() as a:
                    ids.append(a.session_id)
            assert len(set(ids)) == 5

        asyncio.run(_run())

    def test_kwargs_forwarded(self):
        async def _run():
            async with async_openclaw_session(max_retries=0) as adapter:
                assert adapter._max_retries == 0

        asyncio.run(_run())


# ── cross-adapter parity ──────────────────────────────────────────────────────


class TestAdapterParity:
    def test_both_have_complete(self):
        assert callable(OpenClawAdapter().complete)
        assert callable(AsyncOpenClawAdapter().complete)

    def test_both_have_acomplete(self):
        assert callable(OpenClawAdapter().acomplete)
        assert callable(AsyncOpenClawAdapter().acomplete)

    def test_async_complete_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AsyncOpenClawAdapter().complete("hi")

    def test_sync_complete_does_not_raise_not_implemented(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")):
            OpenClawAdapter().complete("hi")

    def test_both_accept_same_complete_signature(self):
        with patch("subprocess.run", return_value=_cp(stdout="ok")):
            OpenClawAdapter().complete(
                "prompt",
                system_prompt="sys",
                temperature=0.5,
                max_tokens=100,
                stream_tokens=None,
            )
        proc = _async_proc(stdout_bytes=b"ok")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            asyncio.run(
                AsyncOpenClawAdapter().acomplete(
                    "prompt",
                    system_prompt="sys",
                    temperature=0.5,
                    max_tokens=100,
                    stream_tokens=None,
                )
            )

    def test_both_have_session_id_property(self):
        assert OpenClawAdapter(session_id="s").session_id == "s"
        assert AsyncOpenClawAdapter(session_id="s").session_id == "s"

    def test_both_have_agent_id_property(self):
        assert OpenClawAdapter(agent_id="a").agent_id == "a"
        assert AsyncOpenClawAdapter(agent_id="a").agent_id == "a"

    def test_both_have_gateway_property(self):
        assert isinstance(OpenClawAdapter().gateway, OpenClawGateway)
        assert isinstance(AsyncOpenClawAdapter().gateway, OpenClawGateway)

    def test_both_have_with_new_session(self):
        s = OpenClawAdapter.with_new_session()
        a = AsyncOpenClawAdapter.with_new_session()
        assert s.session_id is not None
        assert a.session_id is not None
