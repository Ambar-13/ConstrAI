# Changelog

All notable changes to ClampAI are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Versioning policy (0.x phase)

ClampAI is in the **0.x** phase while the API stabilises.

- **Patch releases** (`0.y.Z`) never break backwards compatibility.
  You can always upgrade a patch version safely.
- **Minor releases** (`0.Y.0`) may include breaking changes.
  Read the migration notes in the relevant section before upgrading.
- **1.0.0** is the API-stability milestone.  Once published, the public API
  (everything in `clampai/__init__.py`) will follow full SemVer guarantees:
  no breaking changes without a major-version bump.

---

## [1.0.1] — 2026-02-28

### Changed

- PyPI metadata: expanded keywords to multi-word search phrases for better
  Elasticsearch ranking (`"llm safety"`, `"budget enforcement"`, `"agent guardrails"`,
  `"langchain safety"`, `"openai guardrails"`, `"ai automation"`, and 35 more).
- README: added keyword-rich opening paragraph for Google indexing of the PyPI page.
- `pyproject.toml`: bumped Development Status to `5 - Production/Stable`,
  added `Topic :: Security` classifier, `Operating System :: OS Independent`.

---

## [1.0.0] — 2026-02-28

API-stability milestone. The public API (`clampai/__init__.py`) is now frozen
under full SemVer: no breaking changes without a major-version bump.

### Added

**LangGraph adapter** (`clampai/adapters/langgraph_adapter.py`)

- `SafetyNode` — a LangGraph-compatible callable that wraps any node function
  with ClampAI budget enforcement and invariant checking.  Budget and step
  count persist across calls; `reset()` restores the initial state.
- `@clampai_node(budget, cost_per_step, invariants)` — decorator equivalent
  of `SafetyNode`, preserving `__name__` and `__doc__` via `functools.update_wrapper`.
- `budget_guard(budget, cost_per_step)` — factory that returns a pass-through
  LangGraph node enforcing a budget cap.  Raises `ClampAIBudgetError` (HTTP
  analogue: 429) when exhausted.
- `invariant_guard(invariants)` — factory that returns a pass-through LangGraph
  node checking invariants with no budget charge.  Raises
  `ClampAIInvariantError` on blocking-mode violations.
- Three typed exception classes: `ClampAISafetyError` (base),
  `ClampAIBudgetError`, `ClampAIInvariantError`.
- Install: `pip install clampai[langgraph]`

**FastAPI / Starlette middleware** (`clampai/adapters/fastapi_middleware.py`)

- `ClampAIMiddleware(app, budget, cost_per_request, invariants, state_fn)` —
  Starlette `BaseHTTPMiddleware` subclass.  Every incoming request is evaluated
  by a `SafetyKernel` before the endpoint handler is called.
- Budget exhaustion returns **429 Too Many Requests** (JSON).
  Invariant violation returns **422 Unprocessable Entity** (JSON).
- `state_fn` hook lets callers inject extra fields (user ID, tenant, etc.)
  into the state dict checked by invariants.
- `budget_remaining` and `requests_processed` properties for observability.
- `reset()` method for re-use across test suites and app restarts.
- Install: `pip install clampai[fastapi]`

**Improved invariant suggestions** (`clampai/invariants.py`)

- All 25 pre-built invariant factories now return actionable, specific
  `suggestion` strings that tell the agent exactly what to do when violated.
  Previously some suggestions were generic placeholders.
- Fixed a bug in `monotone_increasing_invariant` and
  `monotone_decreasing_invariant` where the suggestion string captured the
  `initial_value` (always 0.0 / ∞) instead of describing the violated condition.

**New optional extras** (`pyproject.toml`)

- `clampai[langgraph]` — `langgraph>=0.2`
- `clampai[fastapi]` — `fastapi>=0.100.0`, `starlette>=0.27.0`

**Tests** (`tests/`)

- `tests/test_langgraph_adapter.py` — 55 tests across 9 classes covering
  `SafetyNode`, `@clampai_node`, `budget_guard`, `invariant_guard`, and the
  three exception types.
- `tests/test_fastapi_middleware.py` — 50 tests across 8 classes covering
  `ClampAIMiddleware` budget enforcement, invariant violation, state_fn,
  error response format, and reset behaviour.

**LangChain callback handler** (`clampai/adapters/langchain_callback.py`)

- `ClampAICallbackHandler(budget, cost_per_action, invariants, state_fn, raise_on_block)` —
  LangChain `BaseCallbackHandler` subclass. Enforces ClampAI budget and
  invariants in `on_agent_action()` before any tool executes. Requires zero
  changes to the agent — attach via `config={"callbacks": [handler]}`.
- `raise_on_block=False` for monitoring-only mode.
- `budget_remaining`, `actions_blocked`, `tool_calls_made`, `step_count` properties.
- `reset()` for reuse across agent runs.
- `ClampAICallbackError(RuntimeError)` — raised (by default) when blocked.
- Install: `pip install clampai[langchain]`

**CrewAI adapter** (`clampai/adapters/crewai_adapter.py`)

- `ClampAISafeCrewTool(func, name, description, budget, cost, invariants)` —
  callable CrewAI tool wrapper with full ClampAI enforcement.
- `ClampAICrewCallback(budget, cost_per_step, invariants)` — step and task
  callbacks for `Crew(step_callback=..., task_callback=...)`.
- `@safe_crew_tool(budget, cost, name, description, invariants)` — decorator.
- `ClampAICrewBudgetError`, `ClampAICrewInvariantError` exception types.

**AutoGen adapter** (`clampai/adapters/autogen_adapter.py`)

- `ClampAISafeAutoGenAgent(fn, budget, cost_per_reply, invariants, agent_name)` —
  AutoGen reply function wrapper. `check(message, sender, extra_state)` as a
  standalone gate; `__call__` for `agent.register_reply(...)`.
- `@autogen_reply_fn(budget, cost_per_reply, agent_name, invariants)` — decorator.
- Returns `(True, result)` on success, `(False, None)` when `fn=None`.
- `ClampAIAutoGenBudgetError`, `ClampAIAutoGenInvariantError` exception types.

**HTTP sidecar server** (`clampai/server.py`)

- `ClampAIServer(budget, invariants, host, port, min_action_cost)` — HTTP
  server wrapping a shared `SafetyKernel`. Zero external dependencies (stdlib
  `HTTPServer` only; FastAPI optional for full ASGI).
- Endpoints: `POST /evaluate`, `POST /execute`, `POST /reset`,
  `GET /status`, `GET /health`.
- `start_background()` for non-blocking use; `stop()` for teardown.
- CLI: `python -m clampai.server --budget 1000.0 --port 8765`.

**`reconcile_fn` hook in `SafetyKernel`** (`clampai/formal.py`)

- `SafetyKernel(budget, invariants, ..., reconcile_fn=None)` — new optional
  parameter. Called post-commit with `(model_state, action, trace_entry)`.
  If it returns a non-None `State`, that state is used instead of the model
  state, bridging the spec-reality gap. Failures are silently swallowed.
- Applies to both `execute()` and `evaluate_and_execute_atomic()`.

**`OTelTraceExporter`** (`clampai/adapters/metrics.py`)

- `OTelTraceExporter(tracer)` — converts `TraceEntry` objects to OpenTelemetry
  spans. `export_entry(entry)` for manual export; `make_reconcile_fn()` returns
  a closure compatible with `SafetyKernel(reconcile_fn=...)` for automatic
  per-commit export.

**Examples**

- `examples/06_langgraph_agent.py` — four LangGraph integration patterns:
  `@clampai_node`, `budget_guard`, `invariant_guard`, combined usage.
- `examples/07_fastapi_middleware.py` — FastAPI middleware demo with standalone
  runner (no uvicorn required).

**Tests** (`tests/`)

- `tests/test_langchain_callback.py` — 45 tests for `ClampAICallbackHandler`.
- `tests/test_crewai_adapter.py` — 53 tests for `ClampAISafeCrewTool`,
  `ClampAICrewCallback`, and `@safe_crew_tool`.
- `tests/test_autogen_adapter.py` — 43 tests for `ClampAISafeAutoGenAgent`
  and `@autogen_reply_fn`.
- `tests/test_server.py` — 44 tests for `ClampAIServer` endpoints using real
  stdlib HTTP connections.
- `tests/test_reconcile_fn.py` — 30 tests for `reconcile_fn` in `SafetyKernel`.

Total test suite: **1296 tests passing**, ruff clean, mypy clean.
Chaos fuzzer: **45/45 attacks blocked**. Coverage: **91%**.

### Changed

- `clampai/__init__.py`: `__version__` bumped to `"1.0.0"`.
- `clampai/adapters/__init__.py`: exports `SafetyNode`, `clampai_node`,
  `budget_guard`, `invariant_guard`, `ClampAISafetyError`,
  `ClampAIBudgetError`, `ClampAIInvariantError`, `ClampAIMiddleware`,
  `ClampAICallbackHandler`, `ClampAICallbackError`,
  `ClampAISafeCrewTool`, `ClampAICrewCallback`, `safe_crew_tool`,
  `ClampAICrewBudgetError`, `ClampAICrewInvariantError`,
  `ClampAISafeAutoGenAgent`, `autogen_reply_fn`,
  `ClampAIAutoGenBudgetError`, `ClampAIAutoGenInvariantError`,
  `OTelTraceExporter`.

---

## [0.4.1] — 2026-02-28

### Fixed

- `README.md`: changed three relative file links (`MATHEMATICAL_COMPLIANCE.md`,
  `VULNERABILITIES.md`, `BENCHMARKS.md`) to absolute GitHub URLs so they
  resolve correctly on PyPI.
- `CITATION.cff`: bumped `version` to `0.4.1` and `date-released` to
  `2026-02-28`.

---

## [0.4.0] — 2026-02-27

### Added

**`AsyncOrchestrator`** (`clampai/orchestrator.py`)

- Native-async drop-in replacement for `Orchestrator`. The execution loop,
  LLM call, and kernel commit are all coroutines. Uses `AsyncSafetyKernel`
  internally so competing coroutines yield to the event loop rather than
  blocking OS threads.
- All T1–T8 guarantees are preserved: the `asyncio.Lock` in
  `AsyncSafetyKernel.execute_atomic()` serialises budget charge, step
  increment, and trace append exactly as the `threading.Lock` does in the
  synchronous kernel.
- Accepts an optional `kernel=` keyword argument, enabling multiple
  `AsyncOrchestrator` instances to share a single `AsyncSafetyKernel` for
  joint budget enforcement across concurrent agents.
- TOCTOU safety: the fast pre-check (`await kernel.evaluate()`) is followed
  by a locked atomic commit (`await kernel.execute_atomic()`). If the budget
  changes between the two calls, `execute_atomic` catches it and the action
  is rejected — never silently over-spent.
- Exported from `clampai` as `AsyncOrchestrator`.

**`AsyncSafetyKernel.record_rejection()`** (`clampai/formal.py`)

- Delegation method added to `AsyncSafetyKernel`; routes to the wrapped
  synchronous kernel's `record_rejection()` so the async orchestrator can
  append rejection trace entries without acquiring the async lock (the
  operation is read-only with respect to budget and step count).

**Test coverage** (`tests/test_async_orchestrator.py`)

- 40 tests covering `AsyncOrchestrator`:
  - All seven `TerminationReason` paths (GOAL_ACHIEVED, BUDGET_EXHAUSTED,
    STEP_LIMIT, LLM_STOP, MAX_FAILURES, ERROR, STUCK-adjacent)
  - `AsyncSafetyKernel.record_rejection` delegation
  - T1 budget guarantee verified at scale
  - Blocking vs monitoring invariant enforcement
  - Dominant-strategy LLM skip (no adapter call with 1 action)
  - LLM exception → fallback selection
  - Invalid JSON response → fallback
  - Shared-kernel T1 enforcement across two concurrent agents
  - Ten concurrent independent runs with `asyncio.gather`
  - `ExecutionResult` shape, `to_dict()`, `summary()`
  - Custom `goal_progress_fn`
  - Proof-path artifact written on async run
  - Bayesian priors initialised correctly

**Test utilities module** (`clampai/testing.py`)

- `make_state(**vars)` — construct a `State` from keyword arguments; one-liner
  for test fixtures.
- `make_action(name, cost, *, reversible, tags, **effects)` — construct an
  `ActionSpec` with each keyword argument becoming an `Effect("set", ...)`;
  auto-generates a UUID-based action id.
- `SafetyHarness(budget, invariants, *, min_action_cost, emergency_actions)` —
  context-manager test harness wrapping a `SafetyKernel` with assertion helpers:
  - `.assert_allowed(state, action)` — raises `AssertionError` if action is
    blocked.
  - `.assert_blocked(state, action, *, reason_contains)` — raises if action is
    approved; optional substring check on the rejection reason.
  - `.assert_budget_remaining(expected, tol)` — checks `kernel.budget.remaining`.
  - `.assert_step_count(expected)` — checks `kernel.step_count`.
  - `.execute(state, action)` — calls `evaluate_and_execute_atomic`; returns new
    `State` or raises `RuntimeError` if blocked.
  - `.reset()` — recreates the kernel with original budget and invariants.
- All three exported from `clampai.__init__`.

**New invariant factory functions** (`clampai/invariants.py`)

- `string_length_invariant(key, max_length, *, enforcement, suggestion)` —
  blocks if `len(str(state[key]))` exceeds `max_length`.  Absent and `None`
  values pass.  Non-string values are coerced via `str()`.
- `pii_guard_invariant(*keys, enforcement, suggestion)` — blocks if any of the
  given state keys contain patterns for SSN (`\d{3}-\d{2}-\d{4}`), 16-digit
  credit-card numbers, email addresses, or North American phone numbers.
  Patterns are compiled once at factory call time.  Absent / `None` values pass.
- `time_window_rate_invariant(key, max_count, window_seconds, *, enforcement,
  suggestion)` — blocks if `state[key]` contains `>= max_count` float Unix
  timestamps within the last `window_seconds`.  Uses `time.time()` for "now"
  (patchable in tests via `clampai.invariants._time.time`).  Absent, `None`,
  and non-list values pass.
- `json_schema_invariant(key, schema, *, enforcement, suggestion)` — blocks if
  `state[key]` is not a `dict`, or if any field present in both `state[key]`
  and `schema` has the wrong Python type.  `schema` is `Dict[str, type]`.
  Missing fields in the dict are not checked (use `required_fields_invariant`
  for that).  Absent and `None` values pass.
- All four exported from `clampai.invariants.__all__` and `clampai.__init__`.

**Test coverage — new invariants** (`tests/test_new_invariants.py`)

- 57 tests across 4 classes covering all edge cases: absent keys, `None`
  values, non-string coercion, enforcement modes, name and suggestion
  propagation, boundary conditions, and time-window boundary semantics.

**Test coverage — orchestrator** (`tests/test_orchestrator_coverage.py`)

- 50 tests across 11 classes covering: `Outcome` properties (`succeeded`,
  `state_matches_expected`), `ProgressMonitor` (record, current_progress,
  progress_rate, estimated_steps_to_goal, is_stuck, to_llm_text),
  `ExecutionResult` serialization (`to_dict`, `summary`), and seven
  `Orchestrator` integration scenarios (initial state violation → ERROR,
  goal satisfied immediately, goal achieved via action, step limit, max
  consecutive failures, budget exhausted, fallback selection).

**Test coverage — reference monitor** (`tests/test_reference_monitor_coverage.py`)

- 40 tests across 8 classes covering: `DataLabel` lattice algebra (`__le__`,
  `__ge__`, `__eq__`, `__hash__`, `join`), `SafeHoverState.to_action()`,
  `ControlBarrierFunction.evaluate()`, `CaptureBasin.evaluate_reachability()`,
  `ContractSpecification.is_satisfied_by()`, `OperadicComposition.compose()`,
  and `ReferenceMonitor` registration methods.

**Test coverage — testing utilities** (`tests/test_testing_module.py`)

- 60 tests across 9 classes covering `make_state`, `make_action`, and all
  `SafetyHarness` assertion methods plus context-manager lifecycle and
  integration scenarios.

**README additions**

- LLM Adapters section with a full table of all supported backends including
  OpenClaw.
- OpenClaw integration code example with prerequisites.
- "Why ClampAI vs. alternatives" comparison table against custom validation,
  LangChain callbacks, and METR-style eval harnesses.

**OpenClaw super-wrapper** (`clampai/adapters/openclaw_adapter.py`)

Based on a full audit of the [OpenClaw](https://github.com/openclaw/openclaw)
CLI (`openclaw agent`, `openclaw gateway`, `openclaw models`, `openclaw sessions`,
`openclaw memory`).  The adapter now wraps all relevant Gateway capabilities,
not just the `agent` subcommand.

New classes:

- `OpenClawGateway(*, executable, timeout)` — programmatic interface to the OpenClaw
  Gateway control plane.  Methods: `health()` → `GatewayHealth` (liveness probe
  via `openclaw gateway health --json`); `status(*, deep)` → `dict` (detailed
  Gateway status); `call(method, params)` (raw JSON-RPC via
  `openclaw gateway call`); `list_models()` → list (available model descriptors
  via `openclaw models list --json`); `list_sessions(*, agent_id, all_agents)` →
  list (active sessions via `openclaw sessions --json`);
  `search_memory(query, *, agent_id)` → str (semantic memory via
  `openclaw memory search`); `version()` → str (CLI version string).  Every
  method has a `*_async` counterpart running in `asyncio.to_thread`.
- `GatewayHealth(running, url, latency_ms, raw)` — structured health probe
  result dataclass; `running=False` when the Gateway is unreachable (never
  raises).
- `OpenClawResponse(text, session_id, thinking_level, raw)` — structured
  response dataclass returned by `complete_rich()`.
- `openclaw_session(*, prefix, thinking, executable, timeout, **kwargs)` — sync
  context manager that auto-generates a unique `--session-id`; the Gateway
  maintains conversation history for all calls within the block.
- `async_openclaw_session(*, prefix, thinking, executable, timeout, **kwargs)` —
  async context manager equivalent.

New keyword-only parameters added to `OpenClawAdapter` and `AsyncOpenClawAdapter`
(placed after `*`; existing callers are unaffected):

- `session_id=` — forwarded as `--session-id`; enables conversation continuity
  across multiple calls (the Gateway maintains the full message history).
- `agent_id=` — forwarded as `--agent`; targets a specific configured agent.
- `local=True` — adds `--local`; bypasses the Gateway for offline use.
- `max_retries=1` — retries transient `RuntimeError` failures with exponential
  back-off (0.5 s, 1 s, 2 s, …); `TimeoutError` is never retried.
- `check_gateway=True` — probes `openclaw gateway health` at construction time;
  raises `RuntimeError` immediately if the Gateway is not running.
- `.with_new_session(prefix, **kwargs)` — classmethod factory that generates a
  unique session ID automatically.
- `.session_id` / `.agent_id` — read-only properties.
- `.gateway` — returns an `OpenClawGateway` bound to the same executable.
- `.complete_rich(prompt, system_prompt)` — returns `OpenClawResponse` instead
  of a plain string.

New exports from `clampai.adapters`: `OpenClawGateway`, `GatewayHealth`,
`OpenClawResponse`, `openclaw_session`, `async_openclaw_session`.

**Test coverage — OpenClaw super-wrapper** (`tests/test_openclaw_adapter.py`)

- 187 tests across 24 test classes covering: `OpenClawResponse` and
  `GatewayHealth` dataclasses; `OpenClawGateway` all seven sync methods and all
  seven async variants; `OpenClawAdapter` init (new params, `check_gateway`,
  `repr`), `with_new_session`, properties, `complete` (session/agent/local flags
  in CLI, system prompt, extra args, ANSI stripping, all error paths), retry
  logic (exponential back-off, timeout not retried, zero retries), `complete_rich`,
  `acomplete`; `AsyncOpenClawAdapter` init (same new params), `with_new_session`,
  properties, blocking path (all flags, all error paths, timeout + process kill),
  streaming path (per-line tokens, ANSI, blank lines, empty/nonzero-exit, timeout),
  retry (blocking and streaming); `openclaw_session` and `async_openclaw_session`
  context managers; cross-adapter parity assertions.

**Repository hygiene**

- `.gitignore` — added `FIRST_30_DAYS.md` and `safety_evaluation/` to exclude
  AI-generated planning and evaluation files from the repository and PyPI sdist.

**Chaos fuzzer cleanup** (`tests/chaos_fuzzer.py`)

- Replaced emoji output markers (`🛡`, `✅`, `⚠`, `💥`) with plain ASCII
  equivalents (`[OK]`, `ALL ATTACKS BLOCKED`, `[FAIL]`, `[!!]`) for
  professional terminal output and clean CI logs.

**Distributed multi-agent budget** (`clampai/formal.py`)

- `ProcessSharedBudgetController(budget)` — cross-process-safe budget controller
  using `multiprocessing.Value` (int64 shared memory) and `multiprocessing.Lock`
  for atomicity.  Safe across OS processes spawned with fork, forkserver, or
  spawn.  Preserves T1 (spent_net ≤ B₀) and T4 (monotone gross spend) via the
  same `_BudgetLogic` arithmetic as `BudgetController`.  Process-local ledger;
  for a full cross-process audit log write entries to an external store.
  Exported from `clampai.__init__`.

**Native-async safety kernel** (`clampai/formal.py`)

- `AsyncSafetyKernel(budget, invariants, *, min_action_cost, emergency_actions,
  metrics)` — wraps `SafetyKernel` with a lazily-initialised `asyncio.Lock`.
  `evaluate()` acquires no lock (read-only).  `execute_atomic()` acquires the
  asyncio lock — concurrent coroutines yield to the event loop rather than
  blocking OS threads.  All T1–T8 guarantees are preserved.  Full proxy
  surface: `budget`, `invariants`, `step_count`, `max_steps`, `trace`,
  `add_precondition()`, `rollback()`, `status()`.  Exported from
  `clampai.__init__`.

**Native-async adapters** (`clampai/adapters/`)

- `AsyncAnthropicAdapter(client, model, default_system_prompt)` — native-async
  Anthropic adapter using `anthropic.AsyncAnthropic`.  No thread pool.
  `acomplete()` dispatches to blocking or streaming path.  `complete()` raises
  `NotImplementedError` (async-only).  Exported from `clampai.adapters`.
- `AsyncOpenAIAdapter(client, model, default_system_prompt)` — native-async
  OpenAI adapter using `openai.AsyncOpenAI` (or `AsyncAzureOpenAI`).
  `_acomplete_streaming()` uses `async for chunk in stream` without a thread
  pool.  `complete()` raises `NotImplementedError` (async-only).  Exported
  from `clampai.adapters`.

**Mock adapter async support** (`clampai/reasoning.py`)

- `MockLLMAdapter.acomplete()` — async variant of `complete()`.  No I/O
  occurs, so no thread pool is needed.  Suitable for use with
  `AsyncSafetyKernel` in tests and deterministic simulations.

**Test coverage**

- `tests/test_async_and_distributed.py` — 71 tests covering all four new
  classes plus `MockLLMAdapter.acomplete()`.  Includes concurrency test for
  `AsyncSafetyKernel.execute_atomic()` (6 coroutines via `asyncio.gather`).

### Changed

- `AsyncSafetyKernel.evaluate()` signature simplified: removed unused
  `dry_run` and `timeout_ms` kwargs (delegated `SafetyKernel.evaluate` does
  not accept them).
- `AnthropicAdapter.acomplete()` docstring updated to recommend
  `AsyncAnthropicAdapter` for production async use.
- `OpenAIAdapter.acomplete()` docstring note about "planned v0.6 adapter"
  removed; `AsyncOpenAIAdapter` is now available.
- **All OpenClaw public APIs are now fully keyword-only** (0.x breaking
  change, permitted by the versioning policy above).  The following previously
  accepted positional arguments and no longer do:
  - `OpenClawGateway.__init__`: `executable`, `timeout`
  - `OpenClawAdapter.__init__`: `executable`, `thinking`, `timeout`,
    `default_system_prompt`, `extra_args`
  - `AsyncOpenClawAdapter.__init__`: same five params
  - `openclaw_session()`: `prefix`, `thinking`, `executable`, `timeout`
  - `async_openclaw_session()`: same four params

  Migration: add the argument name explicitly.  `OpenClawAdapter("openclaw",
  "medium")` → `OpenClawAdapter(executable="openclaw", thinking="medium")`.
  All documented usage examples already use keyword form.

---

## [0.3.1] — 2026-02-27

### Added

**Zero-config decorator API** (`clampai/api.py`)

- `safe(budget, *, cost_per_call, invariants, state_fn, action_name)` — one-line
  `@safe` decorator that wraps any callable with a dedicated `SafetyKernel`.
  Charges `cost_per_call` on every invocation; evaluates all invariants on the
  *projected next state* (T3 semantics — invariants see the state *after* effects,
  not before).  Thread-safe via `threading.Lock`.
- `SafetyViolation(reason, verdict)` — raised when the kernel blocks a call;
  carries the structured `SafetyVerdict` for programmatic handling.
- `_SafeWrapper` — runtime object installed by `safe()`; exposes `.kernel`,
  `.audit_log`, and `.reset()` for introspection and test isolation.
- `clampai_safe` — backwards-compatibility alias for `safe`.
- Both `safe` and `SafetyViolation` exported from `clampai.__init__`.

**MCP adapter** (`clampai/adapters/mcp_server.py`)

- `SafeMCPServer(name, budget, cost_per_tool, invariants, min_action_cost)` — a
  ClampAI-guarded wrapper around `FastMCP` (from the `mcp` package, optional
  extra).  All tools registered via `@server.tool()` share a single
  `SafetyKernel`.  Per-tool `cost=` and `invariants=` override the server
  defaults.  Thread-safe for concurrent tool calls.
- Install with `pip install clampai[mcp]`.

**Progressive examples** (`examples/`)

- `01_hello_safety.py` — minimal: one kernel, one `evaluate`, one `execute`.
- `02_budget_enforcement.py` — budget tracking and the `@safe` decorator.
- `03_invariants.py` — custom and factory invariants, blocking vs monitoring.
- `04_orchestrator.py` — full `TaskDefinition` → `Orchestrator` pipeline.
- `05_safe_patterns.py` — four `@safe` patterns: basic, `state_fn`, chained
  pipeline, and per-test `reset()`.

**Packaging and infrastructure**

- `pyproject.toml` — replaces `setup.py` with PEP 621-compliant packaging.
  Extras: `anthropic`, `openai`, `langchain`, `mcp`, `prometheus`,
  `opentelemetry`, `dev`.
- `clampai/py.typed` — PEP 561 marker file for downstream type-checkers.
- `.github/workflows/test.yml` — CI matrix: Python 3.9, 3.11, 3.13 ×
  ubuntu-latest and macos-latest; 85 % coverage floor (adapter modules excluded
  from measurement; covered by `test-integrations.yml`).  `ruff check` and
  `mypy` steps run before pytest.
- `.github/workflows/publish.yml` — automated PyPI publishing via OIDC Trusted
  Publisher; gated on the test workflow passing.
- `.github/workflows/test-integrations.yml` — nightly integration matrix for
  `[anthropic]`, `[openai]`, `[langchain]`, and `[prometheus]` extras.

**Adapters package** (`clampai/adapters/`)

- `AnthropicAdapter` — production-ready Anthropic Claude adapter with streaming
  (`stream_tokens` callback) and async (`acomplete()`) support.
- `OpenAIAdapter` — production-ready OpenAI adapter with the same streaming and
  async interface as `AnthropicAdapter`.
- `langchain_tool.py` — `ClampAISafeTool`, a `BaseTool` subclass that wraps any
  ClampAI action as a kernel-gated LangChain tool.  Every tool call passes
  through `SafetyKernel.evaluate()` before execution.
- `metrics.py` — `PrometheusMetrics` and `OTelMetrics`, concrete implementations
  of the `MetricsBackend` protocol defined in `formal.py`.
  - `PrometheusMetrics`: fine-grained latency buckets (0.1 ms – 10 s), lazy
    registration, label-aware counters/histograms/gauges.
  - `OTelMetrics`: OpenTelemetry API adapter; lazy instrument creation; never
    raises (failures silently absorbed so the kernel is never disrupted).
  Both are exported from `clampai.adapters`.

**LLMAdapter API hardening**

- Keyword-only `*` separator added between `system_prompt` and `temperature` in
  every public `complete()` / `acomplete()` signature:
  - `clampai.reasoning.LLMAdapter` (Protocol)
  - `clampai.reasoning.MockLLMAdapter`
  - `AnthropicAdapter.complete()` / `acomplete()`
  - `OpenAIAdapter.complete()` / `acomplete()`
  This makes all callers pass `temperature=`, `max_tokens=`, and `stream_tokens=`
  as keywords, preventing silent positional-argument mis-wiring.
- `acomplete()` added to `LLMAdapter` Protocol (async variant; experimental —
  wraps `complete()` in `asyncio.to_thread()`).  Native async adapters planned
  for v0.6.

**Documentation**

- `docs/STREAMING.md` — design rationale for buffering LLM responses before
  safety evaluation: T5 atomicity requires a complete `ActionSpec`; the 0.061 ms
  safety check is unmeasurable against 1 000–5 000 ms LLM latency; the
  `stream_tokens` callback provides UX streaming without compromising the kernel.
- `docs/MULTI_AGENT_RFC.md` — RFC for shared-kernel multi-agent patterns:
  thread-safe budget sharing, per-agent invariant namespacing, and the
  single-process vs. multi-process distinction.
- `FIRST_30_DAYS.md` — 24-item actionable sprint plan for the v0.3.1 launch
  window: bug fixes, PyPI publication, DX test, email demo, multi-agent example.
- `safety_evaluation/LLM_RED_TEAM_PROTOCOL.md` — formal adversarial LLM
  red-team protocol.  Classifies 6 attack vectors with pre-assessed kernel
  responses; defines a 3-model test (Claude Sonnet 4.6, GPT-4o, open-source
  baseline); commits to verbatim publication of results; scheduled weeks 8–10
  post-launch.
- `MATHEMATICAL_COMPLIANCE.md` — added Lean 4 prerequisites assessment: skills
  gate, dependency map, concrete learning path, and estimated timeline.
- `CONTRIBUTING.md` — keyword-only parameter convention fully documented with
  rationale and examples.
- `SECURITY.md` — added slow-predicate DoS note (invariant predicates with
  `max_eval_ms` timeout), response SLAs, CVE criteria, and reporter credit
  programme.
- `README.md` — guarantee map table; async and LangChain integration notes;
  "Zero-config API: `@safe`" section; updated installation extras table and
  progressive-examples table.
- `OWASP_MAPPING.md` — maps all ten OWASP LLM Top 10 (2025) risks to the
  ClampAI theorems that address them.
- `CHANGELOG.md` — this file.

**Examples**

- `examples/email_safety.py --adversarial` flag — runs a scripted adversary that
  attempts prompt injection, budget exhaustion, and invariant-boundary walking;
  all three attacks are blocked by the kernel and logged.
- `examples/multi_agent_shared_kernel.py` — 8 concurrent agents sharing one
  `SafetyKernel`; demonstrates thread-safety of `BudgetController` under
  concurrent load.

**Safety evaluation**

- `safety_evaluation/llm_red_team_informal_2026.md` — structured capture template
  for informal red-team observations from running `email_safety.py --real-llm`;
  three session slots (baseline, explicit deletion, adversarial) with per-turn
  transcript tables and post-session assessment questions.

### Changed

- `setup.py` simplified to a single `setup()` call; all metadata now lives
  exclusively in `pyproject.toml`.  `setup.py` is retained for legacy
  editable-install compatibility and will be removed at 1.0.0.
- `pyproject.toml` `[langchain]` extra pinned to `langchain>=0.2.0` and
  `langchain-core>=0.3.0` (was `langchain>=0.1.0`); `pydantic>=2.0` added as
  explicit dependency (LangChain 0.2+ requires Pydantic v2).
- `pyproject.toml` `[dev]` extra gains `pytest-asyncio>=0.21` for testing
  `acomplete()` coroutines.

### Fixed

- `clampai/active_hjb_barrier.py` — `Dict` used in type annotations but not
  imported; added to `from typing import`.
- `clampai/formal.py` — `State.__slots__` attributes (`_vars`, `_json`, `_hash`)
  lacked class-level annotations; mypy could not resolve them.  Added
  `_vars: Mapping[str, Any]`, `_json: str`, `_hash: int`.  Also added missing
  `-> None` return types on `__init__`, `__setattr__`, `__delattr__`,
  `__post_init__`, `ExecutionTrace.__init__`, and `SafetyKernel.add_precondition`.
- `clampai/hardening.py` — three `implicit-optional` violations (`working_dir`,
  `command_allowlist`, `ids`) changed to `Optional[...]`.  Loop variable `p` in
  `_chi2p` shadowed the method parameter `p` (renamed to `pval`).
  `VALID_TRANSITIONS` annotated as `ClassVar`.
- `clampai/guards.py` — `Callable` used in a return type annotation but not
  imported (F821 undefined name); added to `from typing import`.  Ambiguous
  variable names `l`/`r` renamed to `lv`/`rv`.
- `clampai/invariants.py` — missing default-argument type annotations on inner
  closures in `no_sensitive_substring_invariant`; added `fl: List[str]`.
- `clampai/adapters/mcp_server.py` — removed redundant `# type: ignore[import]`
  on the `FastMCP` import; `ignore_missing_imports = true` in mypy already handles
  missing optional stubs.
- `SafeHoverState.to_action()` (`reference_monitor.py`): now constructs
  `ActionSpec` with only declared fields; previously passed an unrecognised
  `metadata` keyword argument that raised `TypeError` at runtime.
- `QPProjector.project_cost()` (`reference_monitor.py`): `budget_remaining`
  parameter now defaults to `float('inf')` instead of the hardcoded `10.0` that
  caused incorrect QP projections when the actual budget exceeded 10 resource
  units.
- `SafetyKernel.__init__()` (`formal.py`): removed duplicate
  `self.emergency_actions` assignment; the set is now initialised once from the
  `emergency_actions` constructor parameter.
- `pyproject.toml` build backend changed from `setuptools.backends.legacy:build`
  to `setuptools.build_meta`; the `backends` subpackage is absent from
  conda-distributed setuptools builds, causing `pip install` to fail with
  `BackendUnavailable`.
- `pyproject.toml` classifiers: removed `"License :: OSI Approved :: MIT License"`
  classifier; it conflicts with the `license = "MIT"` SPDX expression field under
  PEP 639 (enforced by setuptools ≥ 67).
- `BudgetController` (`formal.py`): all arithmetic now delegates to `_BudgetLogic`
  static methods (`to_millicents`, `from_millicents`, `can_afford_i`, `remaining_i`,
  `utilization`, new `can_refund_i`).  `_BudgetLogic` was already present but was
  dead code — `BudgetController` had duplicate inline arithmetic.  The extraction is
  now complete: a future `AsyncBudgetController` can wrap `_BudgetLogic` with
  `asyncio.Lock` without duplicating any arithmetic logic.

---

## [0.3.0] — 2026-02-08

### Added

- **Eight proven theorems** (T1–T8) enforced by the safety kernel:
  - T1 — Budget safety: `spent(t) <= budget` for all t.
  - T2 — Bounded termination: halts in at most `floor(budget / min_cost)` steps
    (conditional on `min_cost > 0`).
  - T3 — Invariant preservation: every declared blocking invariant holds on
    every reachable state.
  - T4 — Monotone resource consumption: gross spend is non-decreasing.
  - T5 — Atomicity: rejected actions leave state and budget unchanged.
  - T6 — Trace integrity: the execution log is append-only and SHA-256
    hash-chained; verified by `ExecutionTrace.verify_integrity()`.
  - T7 — Exact rollback: `rollback(execute(s, a)) == s` exactly, realised via
    `InverseAlgebra` in `inverse_algebra.py`.
  - T8 — Emergency escape: the escape action is always executable regardless of
    budget or invariant state (conditional on correct configuration).
- **Layer 0 — Safety Kernel** (`formal.py`): `State`, `Effect`, `ActionSpec`,
  `Invariant`, `SafetyKernel`, `BudgetController`, `ExecutionTrace`.
- **Layer 1 — Reasoning Engine** (`reasoning.py`): Bayesian beliefs (Beta
  posterior), causal graph, multi-dimensional action value computation,
  Integral Sensitivity Filter for prompt pruning, `LLMAdapter` interface,
  `MockLLMAdapter` for zero-dependency testing.
- **Layer 2 — Orchestrator** (`orchestrator.py`): main execution loop,
  LLM-failure fallback to highest-value READY action, `ProgressMonitor`.
- **Layer 3 — Hardening** (`hardening.py`): `SubprocessAttestor` (binary
  allowlist, no `shell=True`), `TemporalCausalGraph` (readiness probes with
  exponential backoff), `CostAwarePriorFactory`, `EnvironmentReconciler`,
  `MultiDimensionalAttestor`.
- **Reference monitor** (`reference_monitor.py`): information flow control
  (IFC), control barrier functions (CBF), QP action repair via `QPProjector`.
- **Operadic composition** (`operadic_composition.py`): compositional
  verification — `Verified(A) and Verified(B)` implies `Verified(A composed B)`
  for compatible interface signatures.
- **Active HJB barrier** (`active_hjb_barrier.py`): k-step lookahead to detect
  multi-step traps before they close.
- **Gradient tracker** (`gradient_tracker.py`): finite-difference Jacobian for
  boundary-proximity scoring.
- **Jacobian fusion** (`jacobian_fusion.py`): boundary sensitivity scoring for
  prompt saliency.
- **Safe hover** (`safe_hover.py`): hard enforcement gate / emergency stop.
- **Saliency engine** (`saliency.py`): Integral Sensitivity Filter for reducing
  token usage while preserving safety-relevant state visibility.
- **Verification log** (`verification_log.py`): proof record writer.
- Full test suite: `test_clampai.py` (T1–T8 unit tests), `test_monte_carlo.py`
  (1,000-run probabilistic validation), `chaos_fuzzer.py` (45 adversarial
  attack scenarios), `test_composition.py`, `test_integration.py`,
  `test_soft_gaps_fixed.py`, `test_boundary_enforcement.py`.
- Safety evaluation harness: 39 attack vectors across 9 threat categories;
  89.7 % recall, zero false positives at sub-millisecond latency.

---

## [0.0.1] — 2026-01-15

### Added

- PyPI name reservation for `clampai`.
- Placeholder `README.md` and `LICENSE` (MIT).
