"""
clampai.adapters.openclaw_adapter — OpenClaw personal AI assistant adapters.

OpenClaw (https://github.com/openclaw/openclaw) is a local-first personal AI
assistant runtime that connects to messaging platforms (WhatsApp, Telegram,
Discord, Slack, iMessage, and more) and exposes a local Gateway control plane
at ``ws://127.0.0.1:18789``.

This module provides three tiers of integration:

**1. Direct adapters** — wrap ``openclaw agent`` for use with the ClampAI
    Orchestrator.  No extra Python package required; only the OpenClaw CLI.

    Sync::

        from clampai.adapters import OpenClawAdapter
        adapter = OpenClawAdapter(thinking="medium")
        engine = Orchestrator(task, llm=adapter)

    Async::

        from clampai.adapters import AsyncOpenClawAdapter
        adapter = AsyncOpenClawAdapter(thinking="high")
        response = await adapter.acomplete("Choose the next action.")

**2. Session context managers** — auto-generate a unique session ID so the
    Gateway maintains conversation history across calls in the same block::

        from clampai.adapters import openclaw_session, async_openclaw_session

        with openclaw_session(thinking="medium") as adapter:
            r1 = adapter.complete("Step 1.")
            r2 = adapter.complete("Step 2, building on the above.")

        async with async_openclaw_session(thinking="high") as adapter:
            r = await adapter.acomplete("Summarise my inbox.")

**3. Gateway client** — programmatic access to the Gateway control plane::

        from clampai.adapters import OpenClawGateway

        gw = OpenClawGateway()
        health = gw.health()                  # liveness probe
        models = gw.list_models()             # enumerate available models
        sessions = gw.list_sessions()         # active session list
        result = gw.call("health")            # raw RPC call
        hits = gw.search_memory("invoices")   # semantic memory search

Thinking levels (in increasing depth):
    ``"off"``, ``"minimal"``, ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``

Prerequisites::

    npm install -g openclaw@latest
    openclaw onboard --install-daemon   # first-time setup
    openclaw gateway                    # start Gateway (keep running)
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple

THINKING_LEVELS: Tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHFJABCDfnsu]")

_DEFAULT_GATEWAY_URL = "ws://127.0.0.1:18789"


# ── helpers ────────────────────────────────────────────────────────────────────


def _strip_ansi(text: str) -> str:
    """Remove ANSI terminal escape sequences from a string."""
    return _ANSI_RE.sub("", text)


def _build_full_prompt(prompt: str, system_prompt: str) -> str:
    """Merge system prompt and user prompt into a single string."""
    return f"{system_prompt}\n\n{prompt}" if system_prompt else prompt


def _build_agent_cmd(
    executable: str,
    thinking: str,
    full_prompt: str,
    extra_args: List[str],
    *,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    local: bool = False,
) -> List[str]:
    """Assemble the ``openclaw agent`` command list."""
    cmd: List[str] = [executable, "agent", "--message", full_prompt,
                      "--thinking", thinking]
    if session_id:
        cmd += ["--session-id", session_id]
    if agent_id:
        cmd += ["--agent", agent_id]
    if local:
        cmd.append("--local")
    cmd += extra_args
    return cmd


def _run_cli(
    cmd: List[str],
    timeout: float,
) -> str:
    """
    Run a CLI command and return stripped stdout text.

    Raises:
        RuntimeError: If the CLI binary is not found or exits non-zero.
        TimeoutError: If the process exceeds *timeout* seconds.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"OpenClaw CLI timed out after {timeout}s: {cmd[1]!r}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"OpenClaw CLI not found: {cmd[0]!r}. "
            "Install with: npm install -g openclaw@latest"
        ) from exc

    if result.returncode != 0:
        stderr = _strip_ansi(result.stderr or "").strip()
        raise RuntimeError(
            f"openclaw {cmd[1]!r} exited with code {result.returncode}. "
            f"stderr: {stderr[:500]}"
        )

    return _strip_ansi(result.stdout or "").strip()


# ── data classes ───────────────────────────────────────────────────────────────


@dataclass
class OpenClawResponse:
    """
    Structured response from the OpenClaw agent.

    Attributes:
        text: The assistant response text (ANSI codes stripped).
        session_id: The session ID used for this call, if any.
        thinking_level: The thinking depth used for this call.
        raw: The raw stdout text before any post-processing.
    """

    text: str
    session_id: Optional[str] = None
    thinking_level: Optional[str] = None
    raw: Optional[str] = None


@dataclass
class GatewayHealth:
    """
    Health status of the OpenClaw Gateway.

    Attributes:
        running: True if the Gateway responded to a health probe.
        url: The Gateway WebSocket URL.
        latency_ms: Round-trip latency in milliseconds, or None if unreachable.
        raw: The parsed JSON body from ``openclaw gateway health --json``,
            or None if the probe failed or returned non-JSON output.
    """

    running: bool
    url: str = _DEFAULT_GATEWAY_URL
    latency_ms: Optional[float] = None
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


# ── Gateway client ─────────────────────────────────────────────────────────────


class OpenClawGateway:
    """
    Programmatic interface to the OpenClaw Gateway control plane.

    Wraps the following OpenClaw CLI subcommands:

    - ``openclaw gateway health`` — liveness probe
    - ``openclaw gateway status`` — detailed session and provider status
    - ``openclaw gateway call <method>`` — JSON-RPC invocation
    - ``openclaw models list`` — enumerate available models
    - ``openclaw sessions`` — list active sessions
    - ``openclaw memory search`` — semantic memory query

    Every method has a ``*_async`` counterpart that runs the blocking
    subprocess in ``asyncio.to_thread``.

    Args:
        executable: Path or name of the OpenClaw CLI binary.  Default: ``"openclaw"``.
        timeout: Subprocess wall-clock timeout in seconds.  Default: 15.0.

    Usage::

        gw = OpenClawGateway()

        health = gw.health()
        if not health.running:
            raise RuntimeError("Start the Gateway: openclaw gateway")

        models = gw.list_models()
        sessions = gw.list_sessions(all_agents=True)
        result = gw.call("health")
        hits = gw.search_memory("unpaid invoices")

    Guarantee: CONDITIONAL (correct when the OpenClaw CLI is installed).
    """

    def __init__(
        self,
        *,
        executable: str = "openclaw",
        timeout: float = 15.0,
    ) -> None:
        self._executable = executable
        self._timeout = timeout

    # ── sync methods ────────────────────────────────────────────────────────

    def health(self) -> GatewayHealth:
        """
        Probe the Gateway and return its health.

        Calls ``openclaw gateway health --json``.  Never raises; returns a
        ``GatewayHealth`` with ``running=False`` if the Gateway is unreachable.

        Returns:
            GatewayHealth with ``running=True`` if the Gateway responds.

        Guarantee: CONDITIONAL.
        """
        start = time.monotonic()
        try:
            result = subprocess.run(
                [self._executable, "gateway", "health", "--json"],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return GatewayHealth(running=False)

        latency_ms = round((time.monotonic() - start) * 1000.0, 1)

        if result.returncode != 0:
            return GatewayHealth(running=False, latency_ms=latency_ms)

        raw_data: Optional[Dict[str, Any]] = None
        try:
            raw_data = json.loads(_strip_ansi(result.stdout or ""))
        except (json.JSONDecodeError, ValueError):
            pass

        return GatewayHealth(
            running=True,
            latency_ms=latency_ms,
            raw=raw_data,
        )

    def status(self, *, deep: bool = False) -> Dict[str, Any]:
        """
        Retrieve detailed Gateway status via ``openclaw gateway status --json``.

        Args:
            deep: Pass ``--deep`` for extended probe information.

        Returns:
            Parsed JSON dict from the Gateway, or an empty dict on error.

        Guarantee: CONDITIONAL.
        """
        cmd = [self._executable, "gateway", "status", "--json"]
        if deep:
            cmd.append("--deep")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return {}

        text = _strip_ansi(result.stdout or "").strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Invoke a Gateway RPC method via ``openclaw gateway call <method>``.

        Args:
            method: RPC method name (e.g. ``"health"``, ``"status.get"``).
            params: Optional JSON-serializable parameter dict.

        Returns:
            Parsed JSON response from the Gateway, or the raw text if the
            response is not valid JSON.

        Raises:
            RuntimeError: If the CLI binary is not found or the call fails.
            TimeoutError: If the subprocess exceeds the timeout.

        Guarantee: CONDITIONAL.
        """
        cmd = [self._executable, "gateway", "call", method]
        if params:
            cmd += ["--params", json.dumps(params)]
        text = _run_cli(cmd, self._timeout)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Return available models via ``openclaw models list --json``.

        Returns:
            List of model descriptor dicts, or an empty list on error.

        Guarantee: CONDITIONAL.
        """
        try:
            result = subprocess.run(
                [self._executable, "models", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return []

        text = _strip_ansi(result.stdout or "").strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            return [data] if data else []
        except (json.JSONDecodeError, ValueError):
            return []

    def list_sessions(
        self,
        *,
        agent_id: Optional[str] = None,
        all_agents: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return active sessions via ``openclaw sessions --json``.

        Args:
            agent_id: Restrict to sessions for a specific agent.
            all_agents: Aggregate sessions across all agents.

        Returns:
            List of session descriptor dicts, or an empty list on error.

        Guarantee: CONDITIONAL.
        """
        cmd = [self._executable, "sessions", "--json"]
        if agent_id:
            cmd += ["--agent", agent_id]
        if all_agents:
            cmd.append("--all-agents")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return []

        text = _strip_ansi(result.stdout or "").strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            return [data] if data else []
        except (json.JSONDecodeError, ValueError):
            return []

    def search_memory(
        self,
        query: str,
        *,
        agent_id: Optional[str] = None,
    ) -> str:
        """
        Run a semantic memory search via ``openclaw memory search <query>``.

        Args:
            query: Natural language search query.
            agent_id: Restrict search to a specific agent's memory store.

        Returns:
            Raw text output from the memory search command.

        Raises:
            RuntimeError: If the CLI binary is not found.

        Guarantee: CONDITIONAL.
        """
        cmd = [self._executable, "memory", "search", query]
        if agent_id:
            cmd += ["--agent", agent_id]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"OpenClaw CLI not found: {self._executable!r}. "
                "Install with: npm install -g openclaw@latest"
            ) from exc
        except subprocess.TimeoutExpired:
            return ""

        return _strip_ansi(result.stdout or "").strip()

    def version(self) -> str:
        """
        Return the installed OpenClaw version string.

        Calls ``openclaw --version``.

        Returns:
            Version string (e.g. ``"2026.2.26"``), or an empty string on error.

        Guarantee: CONDITIONAL.
        """
        try:
            result = subprocess.run(
                [self._executable, "--version"],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return ""

        return _strip_ansi(result.stdout or "").strip()

    # ── async variants ───────────────────────────────────────────────────────

    async def health_async(self) -> GatewayHealth:
        """Async variant of :meth:`health`."""
        return await asyncio.to_thread(self.health)

    async def status_async(self, *, deep: bool = False) -> Dict[str, Any]:
        """Async variant of :meth:`status`."""
        return await asyncio.to_thread(self.status, deep=deep)

    async def call_async(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Async variant of :meth:`call`."""
        return await asyncio.to_thread(self.call, method, params)

    async def list_models_async(self) -> List[Dict[str, Any]]:
        """Async variant of :meth:`list_models`."""
        return await asyncio.to_thread(self.list_models)

    async def list_sessions_async(
        self,
        *,
        agent_id: Optional[str] = None,
        all_agents: bool = False,
    ) -> List[Dict[str, Any]]:
        """Async variant of :meth:`list_sessions`."""
        return await asyncio.to_thread(
            self.list_sessions, agent_id=agent_id, all_agents=all_agents
        )

    async def search_memory_async(
        self,
        query: str,
        *,
        agent_id: Optional[str] = None,
    ) -> str:
        """Async variant of :meth:`search_memory`."""
        return await asyncio.to_thread(
            self.search_memory, query, agent_id=agent_id
        )

    async def version_async(self) -> str:
        """Async variant of :meth:`version`."""
        return await asyncio.to_thread(self.version)

    def __repr__(self) -> str:
        return f"OpenClawGateway(executable={self._executable!r})"


# ── sync adapter ───────────────────────────────────────────────────────────────


class OpenClawAdapter:
    """
    ClampAI LLM adapter for OpenClaw (sync, subprocess-based).

    Calls ``openclaw agent --message <prompt> --thinking <level>`` and returns
    the stripped stdout as the assistant response.

    Session continuity
        Pass ``session_id`` (or use :meth:`with_new_session`) to send all calls
        within the same Gateway conversation.  The Gateway maintains the full
        message history for the session, so the agent can build on prior turns.

    Agent targeting
        Pass ``agent_id`` to route requests to a specific configured agent.
        If omitted, the Gateway uses the default agent.

    Local mode
        Pass ``local=True`` to add ``--local`` and bypass the Gateway.  Useful
        for offline environments or when the Gateway is not running.

    Retry
        Transient ``RuntimeError`` failures (non-zero exit, empty response) are
        retried up to ``max_retries`` times with exponential back-off (0.5 s,
        1 s, 2 s, …).  ``TimeoutError`` is never retried.

    Args:
        executable: Path or name of the OpenClaw CLI binary.
            Default: ``"openclaw"`` (must be on ``PATH``).
        thinking: Thinking depth passed to the agent.
            One of ``THINKING_LEVELS``.  Default: ``"low"``.
        timeout: Subprocess wall-clock timeout in seconds.  Default: 60.0.
        default_system_prompt: Prepended to prompts when the caller does not
            supply a ``system_prompt``.
        extra_args: Additional CLI arguments appended to every invocation,
            e.g. ``["--verbose"]``.
        session_id: Optional session identifier forwarded as ``--session-id``.
            Enables conversation history across multiple calls.
        agent_id: Optional agent identifier forwarded as ``--agent``.
            Targets a specific registered agent.
        local: If True, pass ``--local`` to skip the Gateway.
        max_retries: Number of retry attempts on transient failures.  Default: 1.
        check_gateway: If True, verify the Gateway is reachable at construction
            time and raise ``RuntimeError`` if it is not.  Default: False.

    Raises:
        ValueError: If ``thinking`` is not a valid level, or ``max_retries < 0``.
        RuntimeError: At construction if ``check_gateway=True`` and the Gateway
            does not respond.
    """

    def __init__(
        self,
        *,
        executable: str = "openclaw",
        thinking: str = "low",
        timeout: float = 60.0,
        default_system_prompt: str = "",
        extra_args: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        local: bool = False,
        max_retries: int = 1,
        check_gateway: bool = False,
    ) -> None:
        if thinking not in THINKING_LEVELS:
            raise ValueError(
                f"thinking must be one of {THINKING_LEVELS}, got {thinking!r}"
            )
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries!r}")
        self._executable = executable
        self._thinking = thinking
        self._timeout = timeout
        self._default_system_prompt = default_system_prompt
        self._extra_args: List[str] = list(extra_args or [])
        self._session_id = session_id
        self._agent_id = agent_id
        self._local = local
        self._max_retries = max_retries
        if check_gateway:
            h = OpenClawGateway(executable=executable, timeout=10.0).health()
            if not h.running:
                raise RuntimeError(
                    "OpenClaw Gateway is not running. "
                    "Start it with: openclaw gateway"
                )

    # ── factory ─────────────────────────────────────────────────────────────

    @classmethod
    def with_new_session(
        cls,
        *,
        prefix: str = "clampai",
        **kwargs: Any,
    ) -> "OpenClawAdapter":
        """
        Create an adapter with an auto-generated unique session ID.

        Each call to the returned adapter shares the same Gateway session,
        enabling conversation history.

        Args:
            prefix: Human-readable prefix for the session ID.
            **kwargs: Forwarded to ``OpenClawAdapter.__init__``.

        Returns:
            A new ``OpenClawAdapter`` with ``session_id`` set.
        """
        return cls(session_id=f"{prefix}-{uuid.uuid4().hex[:8]}", **kwargs)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def session_id(self) -> Optional[str]:
        """The active session identifier, or None if stateless."""
        return self._session_id

    @property
    def agent_id(self) -> Optional[str]:
        """The targeted agent identifier, or None if using the default agent."""
        return self._agent_id

    @property
    def gateway(self) -> OpenClawGateway:
        """A :class:`OpenClawGateway` bound to the same CLI executable."""
        return OpenClawGateway(executable=self._executable)

    # ── public API ──────────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream_tokens: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Send a prompt to the OpenClaw agent and return the response.

        Args:
            prompt: User-turn content.
            system_prompt: System context prepended to the prompt.
                Falls back to ``default_system_prompt`` if empty.
            temperature: Accepted for interface compatibility; not forwarded.
            max_tokens: Accepted for interface compatibility; not forwarded.
            stream_tokens: If provided, called once per non-empty output line
                as the response is processed.  The complete text is returned
                regardless.

        Returns:
            The complete assistant response text (ANSI codes stripped,
            leading/trailing whitespace removed).

        Raises:
            RuntimeError: If the CLI is not found, exits non-zero after all
                retries, or returns an empty response.
            TimeoutError: If the subprocess exceeds ``timeout`` seconds.

        Guarantee: CONDITIONAL (requires OpenClaw CLI and, unless
        ``local=True``, a running Gateway).
        """
        sys_p = system_prompt or self._default_system_prompt
        full_prompt = _build_full_prompt(prompt, sys_p)
        cmd = _build_agent_cmd(
            self._executable,
            self._thinking,
            full_prompt,
            self._extra_args,
            session_id=self._session_id,
            agent_id=self._agent_id,
            local=self._local,
        )
        stdout_text = self._run_with_retry(cmd)

        if stream_tokens is not None:
            for line in stdout_text.splitlines(keepends=True):
                if line.strip():
                    stream_tokens(line)

        return stdout_text

    def complete_rich(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> OpenClawResponse:
        """
        Send a prompt and return an :class:`OpenClawResponse` with metadata.

        The ``text`` field always contains the assistant response.
        ``session_id`` reflects the active session (if any), enabling callers
        to correlate responses to a specific conversation.

        Args:
            prompt: User-turn content.
            system_prompt: System context prepended to the prompt.

        Returns:
            :class:`OpenClawResponse` with ``text``, ``session_id``, and
            ``thinking_level`` populated.

        Guarantee: CONDITIONAL.
        """
        text = self.complete(prompt, system_prompt)
        return OpenClawResponse(
            text=text,
            session_id=self._session_id,
            thinking_level=self._thinking,
            raw=text,
        )

    async def acomplete(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream_tokens: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Async variant using ``asyncio.to_thread``.

        Runs :meth:`complete` in a thread pool so the event loop is not
        blocked.  For fully non-blocking async I/O use
        :class:`AsyncOpenClawAdapter` instead.

        Guarantee: HEURISTIC (thread pool; not native async).
        """
        return await asyncio.to_thread(
            self.complete,
            prompt,
            system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_tokens=stream_tokens,
        )

    # ── internal ────────────────────────────────────────────────────────────

    def _run_subprocess(self, cmd: List[str]) -> str:
        """Execute *cmd* and return stripped stdout text."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"OpenClaw agent timed out after {self._timeout}s. "
                "Increase timeout or use a lower thinking level."
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"OpenClaw CLI not found: {self._executable!r}. "
                "Install with: npm install -g openclaw@latest"
            ) from exc

        if result.returncode != 0:
            stderr = _strip_ansi(result.stderr or "").strip()
            raise RuntimeError(
                f"OpenClaw agent exited with code {result.returncode}. "
                f"stderr: {stderr[:500]}"
            )

        text = _strip_ansi(result.stdout or "").strip()
        if not text:
            raise RuntimeError(
                "OpenClaw agent returned an empty response. "
                "Verify the Gateway is running: openclaw gateway"
            )
        return text

    def _run_with_retry(self, cmd: List[str]) -> str:
        """Run *cmd*, retrying up to ``_max_retries`` times on RuntimeError."""
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                time.sleep(0.5 * (2 ** (attempt - 1)))
            try:
                return self._run_subprocess(cmd)
            except TimeoutError:
                raise
            except RuntimeError:
                if attempt >= self._max_retries:
                    raise
        raise RuntimeError("Unreachable")  # pragma: no cover

    def __repr__(self) -> str:
        parts = [f"executable={self._executable!r}", f"thinking={self._thinking!r}"]
        if self._session_id:
            parts.append(f"session_id={self._session_id!r}")
        if self._agent_id:
            parts.append(f"agent_id={self._agent_id!r}")
        if self._local:
            parts.append("local=True")
        return f"OpenClawAdapter({', '.join(parts)})"


# ── async adapter ──────────────────────────────────────────────────────────────


class AsyncOpenClawAdapter:
    """
    Native-async OpenClaw adapter using ``asyncio.create_subprocess_exec``.

    The subprocess runs concurrently with other coroutines without occupying
    an OS thread.  This is the recommended adapter for use with
    :class:`~clampai.AsyncSafetyKernel` or any coroutine-based orchestration.

    Supports the same session, agent, local, retry, and gateway-check features
    as :class:`OpenClawAdapter`.

    Args:
        executable: Path or name of the OpenClaw CLI binary.
        thinking: Thinking depth — one of ``THINKING_LEVELS``.
        timeout: Subprocess wall-clock timeout in seconds.
        default_system_prompt: Prepended to prompts when empty.
        extra_args: Additional CLI arguments appended to every invocation.
        session_id: Optional session identifier forwarded as ``--session-id``.
        agent_id: Optional agent identifier forwarded as ``--agent``.
        local: If True, pass ``--local`` to skip the Gateway.
        max_retries: Number of retry attempts on transient failures.  Default: 1.
        check_gateway: If True, verify the Gateway is reachable at construction
            time (synchronous probe).  Default: False.

    Raises:
        ValueError: If ``thinking`` is not a valid level, or ``max_retries < 0``.
        RuntimeError: At construction if ``check_gateway=True`` and the Gateway
            does not respond.

    Usage::

        from clampai.adapters import AsyncOpenClawAdapter
        from clampai import AsyncSafetyKernel

        kernel = AsyncSafetyKernel(budget=100.0, invariants=[...])
        adapter = AsyncOpenClawAdapter(thinking="high")
        response = await adapter.acomplete("Choose the next action.")

    Guarantee: CONDITIONAL (requires OpenClaw CLI and, unless ``local=True``,
    a running Gateway; I/O errors propagate as exceptions).
    """

    def __init__(
        self,
        *,
        executable: str = "openclaw",
        thinking: str = "low",
        timeout: float = 60.0,
        default_system_prompt: str = "",
        extra_args: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        local: bool = False,
        max_retries: int = 1,
        check_gateway: bool = False,
    ) -> None:
        if thinking not in THINKING_LEVELS:
            raise ValueError(
                f"thinking must be one of {THINKING_LEVELS}, got {thinking!r}"
            )
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries!r}")
        self._executable = executable
        self._thinking = thinking
        self._timeout = timeout
        self._default_system_prompt = default_system_prompt
        self._extra_args: List[str] = list(extra_args or [])
        self._session_id = session_id
        self._agent_id = agent_id
        self._local = local
        self._max_retries = max_retries
        if check_gateway:
            h = OpenClawGateway(executable=executable, timeout=10.0).health()
            if not h.running:
                raise RuntimeError(
                    "OpenClaw Gateway is not running. "
                    "Start it with: openclaw gateway"
                )

    # ── factory ─────────────────────────────────────────────────────────────

    @classmethod
    def with_new_session(
        cls,
        *,
        prefix: str = "clampai",
        **kwargs: Any,
    ) -> "AsyncOpenClawAdapter":
        """
        Create an async adapter with an auto-generated unique session ID.

        Args:
            prefix: Human-readable prefix for the session ID.
            **kwargs: Forwarded to ``AsyncOpenClawAdapter.__init__``.

        Returns:
            A new ``AsyncOpenClawAdapter`` with ``session_id`` set.
        """
        return cls(session_id=f"{prefix}-{uuid.uuid4().hex[:8]}", **kwargs)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def session_id(self) -> Optional[str]:
        """The active session identifier, or None if stateless."""
        return self._session_id

    @property
    def agent_id(self) -> Optional[str]:
        """The targeted agent identifier, or None if using the default agent."""
        return self._agent_id

    @property
    def gateway(self) -> OpenClawGateway:
        """A :class:`OpenClawGateway` bound to the same CLI executable."""
        return OpenClawGateway(executable=self._executable)

    # ── public API ──────────────────────────────────────────────────────────

    async def acomplete(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream_tokens: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Send a prompt to the OpenClaw agent asynchronously.

        Uses ``asyncio.create_subprocess_exec`` — no OS thread is occupied
        while waiting for subprocess output.

        Args:
            prompt: User-turn content.
            system_prompt: System context prepended to the prompt.
            temperature: Accepted for interface compatibility; not forwarded.
            max_tokens: Accepted for interface compatibility; not forwarded.
            stream_tokens: If provided, called once per non-empty output line.

        Returns:
            The complete assistant response (ANSI codes stripped).

        Raises:
            RuntimeError: If the CLI is not found, exits non-zero after all
                retries, or returns an empty response.
            TimeoutError: If the subprocess exceeds ``timeout`` seconds.

        Guarantee: CONDITIONAL.
        """
        sys_p = system_prompt or self._default_system_prompt
        full_prompt = _build_full_prompt(prompt, sys_p)
        cmd = _build_agent_cmd(
            self._executable,
            self._thinking,
            full_prompt,
            self._extra_args,
            session_id=self._session_id,
            agent_id=self._agent_id,
            local=self._local,
        )

        if stream_tokens is not None:
            return await self._acomplete_with_retry_streaming(cmd, stream_tokens)
        return await self._acomplete_with_retry_blocking(cmd)

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream_tokens: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Not implemented on the async adapter.

        Use :meth:`acomplete` from an async context, or use
        :class:`OpenClawAdapter` for synchronous use.
        """
        raise NotImplementedError(
            "AsyncOpenClawAdapter is async-only. "
            "Use acomplete() from an async context, or "
            "OpenClawAdapter for synchronous use."
        )

    # ── internal ────────────────────────────────────────────────────────────

    async def _acomplete_blocking(self, cmd: List[str]) -> str:
        """Run the subprocess and collect all output via communicate()."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"OpenClaw CLI not found: {cmd[0]!r}. "
                "Install with: npm install -g openclaw@latest"
            ) from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"OpenClaw agent timed out after {self._timeout}s. "
                "Increase timeout or use a lower thinking level."
            )

        if proc.returncode != 0:
            stderr = _strip_ansi(
                (stderr_bytes or b"").decode("utf-8", errors="replace")
            ).strip()
            raise RuntimeError(
                f"OpenClaw agent exited with code {proc.returncode}. "
                f"stderr: {stderr[:500]}"
            )

        text = _strip_ansi(
            (stdout_bytes or b"").decode("utf-8", errors="replace")
        ).strip()
        if not text:
            raise RuntimeError(
                "OpenClaw agent returned an empty response. "
                "Verify the Gateway is running: openclaw gateway"
            )
        return text

    async def _acomplete_streaming(
        self,
        cmd: List[str],
        stream_tokens: Callable[[str], None],
    ) -> str:
        """Run the subprocess and call stream_tokens per line of stdout."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"OpenClaw CLI not found: {cmd[0]!r}. "
                "Install with: npm install -g openclaw@latest"
            ) from exc

        lines: List[str] = []

        async def _read_lines() -> None:
            assert proc.stdout is not None
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = _strip_ansi(line_bytes.decode("utf-8", errors="replace"))
                if line.strip():
                    stream_tokens(line)
                lines.append(line)

        try:
            await asyncio.wait_for(_read_lines(), timeout=self._timeout)
            await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"OpenClaw agent stream timed out after {self._timeout}s."
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenClaw agent exited with code {proc.returncode}."
            )

        full = "".join(lines).strip()
        if not full:
            raise RuntimeError(
                "OpenClaw agent returned an empty response. "
                "Verify the Gateway is running: openclaw gateway"
            )
        return full

    async def _acomplete_with_retry_blocking(self, cmd: List[str]) -> str:
        """Blocking path with exponential-backoff retry."""
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
            try:
                return await self._acomplete_blocking(cmd)
            except TimeoutError:
                raise
            except RuntimeError:
                if attempt >= self._max_retries:
                    raise
        raise RuntimeError("Unreachable")  # pragma: no cover

    async def _acomplete_with_retry_streaming(
        self,
        cmd: List[str],
        stream_tokens: Callable[[str], None],
    ) -> str:
        """Streaming path with exponential-backoff retry."""
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
            try:
                return await self._acomplete_streaming(cmd, stream_tokens)
            except TimeoutError:
                raise
            except RuntimeError:
                if attempt >= self._max_retries:
                    raise
        raise RuntimeError("Unreachable")  # pragma: no cover

    def __repr__(self) -> str:
        parts = [f"executable={self._executable!r}", f"thinking={self._thinking!r}"]
        if self._session_id:
            parts.append(f"session_id={self._session_id!r}")
        if self._agent_id:
            parts.append(f"agent_id={self._agent_id!r}")
        if self._local:
            parts.append("local=True")
        return f"AsyncOpenClawAdapter({', '.join(parts)})"


# ── session context managers ───────────────────────────────────────────────────


@contextmanager
def openclaw_session(
    *,
    prefix: str = "clampai",
    thinking: str = "low",
    executable: str = "openclaw",
    timeout: float = 60.0,
    **kwargs: Any,
) -> Iterator[OpenClawAdapter]:
    """
    Context manager that provides a session-scoped :class:`OpenClawAdapter`.

    Generates a unique session ID on entry and passes it to the adapter so
    the Gateway maintains conversation history for the duration of the block.

    Args:
        prefix: Human-readable prefix for the auto-generated session ID.
        thinking: Thinking depth for the adapter.
        executable: Path or name of the OpenClaw CLI binary.
        timeout: Subprocess timeout in seconds.
        **kwargs: Additional keyword arguments forwarded to
            :class:`OpenClawAdapter`.

    Yields:
        An :class:`OpenClawAdapter` with a persistent session ID.

    Usage::

        with openclaw_session(thinking="medium") as adapter:
            r1 = adapter.complete("Analyse step 1.")
            r2 = adapter.complete("Now build on that for step 2.")
    """
    session_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    yield OpenClawAdapter(
        executable=executable,
        thinking=thinking,
        timeout=timeout,
        session_id=session_id,
        **kwargs,
    )


@asynccontextmanager
async def async_openclaw_session(
    *,
    prefix: str = "clampai",
    thinking: str = "low",
    executable: str = "openclaw",
    timeout: float = 60.0,
    **kwargs: Any,
) -> AsyncIterator[AsyncOpenClawAdapter]:
    """
    Async context manager that provides a session-scoped
    :class:`AsyncOpenClawAdapter`.

    Generates a unique session ID on entry so the Gateway maintains
    conversation history for all calls within the block.

    Args:
        prefix: Human-readable prefix for the auto-generated session ID.
        thinking: Thinking depth for the adapter.
        executable: Path or name of the OpenClaw CLI binary.
        timeout: Subprocess timeout in seconds.
        **kwargs: Additional keyword arguments forwarded to
            :class:`AsyncOpenClawAdapter`.

    Yields:
        An :class:`AsyncOpenClawAdapter` with a persistent session ID.

    Usage::

        async with async_openclaw_session(thinking="high") as adapter:
            r = await adapter.acomplete("Solve part 1.")
            followup = await adapter.acomplete("Now solve part 2.")
    """
    session_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    yield AsyncOpenClawAdapter(
        executable=executable,
        thinking=thinking,
        timeout=timeout,
        session_id=session_id,
        **kwargs,
    )
