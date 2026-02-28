# ClampAI

[![PyPI version](https://badge.fury.io/py/clampai.svg)](https://pypi.org/project/clampai/)
[![Python Versions](https://img.shields.io/pypi/pyversions/clampai.svg)](https://pypi.org/project/clampai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Ambar-13/ClampAI/actions/workflows/test.yml/badge.svg)](https://github.com/Ambar-13/ClampAI/actions/workflows/test.yml)

Formal safety for AI agents — enforced at execution, not in the prompt.

**ClampAI** is a Python safety framework for LLM agents and AI automation — budget enforcement, safety guardrails, invariant checking, and rate limiting at the execution layer. Works with LangChain, LangGraph, OpenAI, Anthropic, OpenClaw, AutoGen, CrewAI, and FastAPI. Zero runtime dependencies.

---

## Install

```bash
pip install clampai
```

Zero runtime dependencies. Python 3.9+.

---

## Quickstart

The simplest case: one line gives you provable budget enforcement.

```python
from clampai import safe

@safe(budget=10.0)
def call_api(endpoint: str) -> dict:
    ...

call_api("/search")   # $1.00 charged
call_api("/search")   # $1.00 charged
# ... after 10 calls:
# SafetyViolation: budget exhausted ($10.00 spent of $10.00)
```

Add invariants in the same decorator:

```python
from clampai.invariants import rate_limit_invariant, pii_guard_invariant

@safe(
    budget=50.0,
    cost_per_call=2.0,
    invariants=[
        rate_limit_invariant("emails_sent", max_count=100),
        pii_guard_invariant("output"),
    ],
    state_fn=lambda: {"emails_sent": db.get_count(), "output": ""},
)
def send_email(recipient: str, body: str) -> None:
    ...
```

For multi-step agents, see [Building an agent with Orchestrator](#building-an-agent-with-orchestrator).

---

## What you get

- **Budget enforcement (T1):** Total spend never exceeds what you set. Enforced by a locked counter — not a trust-based check.
- **Invariant preservation (T3):** Every rule you declare holds on every state the kernel allows to exist. The bad state never becomes real.
- **Atomicity (T5):** Rejected actions leave state and budget exactly as they were before the attempt.
- **Audit trail (T6):** SHA-256 chained log of every decision — tamper-evident by construction.
- **Exact rollback (T7):** For reversible actions, `rollback(execute(s, a)) == s` exactly.
- **Bounded termination (T2):** Given a finite budget and a minimum action cost, the loop cannot run forever.

**Battle-tested:** The chaos fuzzer passes 45/45 adversarial attack scenarios. Red-team evaluation across 39 attack vectors spanning 9 threat categories: **89.7% recall, zero false positives**, at sub-millisecond latency (~45,600 checks/second). The 4 vectors that evade classification do so before an `ActionSpec` is constructed — the kernel itself has no known bypass.

**NeMo Guardrails checks what the LLM says. ClampAI checks what the agent does.**

ClampAI is complementary to text-layer guardrails, not a replacement. A production agent needs both: text-layer safety (NeMo, LLM Guard) catches prompt injection and toxic outputs; ClampAI catches budget overruns, invariant violations, and state corruption at the execution layer.

---

## Table of contents

1. [The problem this solves](#the-problem-this-solves)
2. [How it works — the mental model](#how-it-works--the-mental-model)
3. [Installation](#installation)
4. [Core concepts](#core-concepts)
5. [The @safe decorator](#the-safe-decorator)
6. [Building an agent with Orchestrator](#building-an-agent-with-orchestrator)
7. [Pre-built invariants](#pre-built-invariants)
8. [Testing your safety rules](#testing-your-safety-rules)
9. [LLM adapters](#llm-adapters)
10. [Framework integrations](#framework-integrations) — LangGraph, FastAPI, LangChain, CrewAI, AutoGen, HTTP Sidecar
11. [Formal guarantees in plain English](#formal-guarantees-in-plain-english)
12. [Architecture](#architecture)
13. [Advanced features](#advanced-features)
14. [Known limitations](#known-limitations)
15. [Performance](#performance)
16. [FAQ](#faq)
17. [Examples](#examples)
18. [Project structure](#project-structure)
19. [Citation](#citation)
20. [License](#license)

---

## The problem this solves

When you give an LLM the ability to take real actions — send emails, write files, call APIs, provision cloud resources — a few things can go wrong:

1. **The LLM spends more than you budgeted.** A loop that calls an expensive API 1,000 times when you expected 10 is a real risk with autonomous agents.
2. **The LLM violates a rule you declared.** "Never delete records", "never send more than 20 emails per hour", "stop if the error rate exceeds 5%" — these rules need to be enforced, not just stated in a prompt.
3. **Something crashes mid-run and leaves state inconsistent.** An action was half-applied and you don't know what happened.

Prompt engineering ("please don't do bad things") is not enforcement, the model can hallucinate past it. Post-hoc filtering fires after the damage. ClampAI enforces at the **execution layer**: the LLM produces a decision; the kernel decides whether it executes. The two are separate, and neither can override the other.

---

## How it works — the mental model

Think of it as a **strict bank teller**.

You (or your LLM) want to make a transaction. You hand in a slip that says: "I want to execute action X, which will cost $2.00 and will change the state in the following ways." The teller checks:

- Do you have $2.00 left in your budget?
- Does the resulting state violate any of the rules you declared?
- Would this push any metric over a declared limit?

If everything is fine, the transaction goes through. If anything fails, the slip is returned with a reason. The existing balance is untouched.

The "slip" is an `ActionSpec`. The "bank balance" is a `BudgetController`. The "rules" are `Invariant` objects. The "teller" is a `SafetyKernel`.

A key detail: the kernel does not execute the action to check it. It **simulates** the action on a copy of state, checks the simulated result against your invariants, and only commits if everything passes. The bad state never becomes real.

---

## Installation

```bash
pip install clampai
```

The core package has zero external dependencies — it runs on the Python standard library alone.

Optional extras:

```bash
pip install clampai[anthropic]     # Anthropic Claude adapter
pip install clampai[openai]        # OpenAI and Azure OpenAI adapter
pip install clampai[langchain]     # LangChain tool adapter
pip install clampai[langgraph]     # LangGraph node and edge safety
pip install clampai[fastapi]       # FastAPI/Starlette request middleware
pip install clampai[mcp]           # Model Context Protocol server
pip install clampai[prometheus]    # Prometheus metrics export
pip install clampai[opentelemetry] # OpenTelemetry metrics export
pip install clampai[dev]           # pytest, mypy, ruff
```

---

## Core concepts

### State

`State` is an immutable snapshot of your agent's world — a frozen dictionary.

```python
from clampai import State

s = State({"counter": 0, "errors": 0, "status": "pending"})
s.get("counter")       # 0
s.get("missing", 42)   # 42 — safe default, key absent
```

`State` objects are **never mutated in place**. Every action produces a new `State`. This is what lets the kernel simulate an action on a copy before committing.

### Effect

An `Effect` describes one atomic change to state: `(key, operation, value)`.

```python
from clampai import Effect

Effect("counter", "increment", 1)      # counter += 1
Effect("status",  "set",       "done") # status = "done"
Effect("items",   "append",    "x")    # items.append("x")
Effect("items",   "remove",    "x")    # items.remove("x")
Effect("cost",    "decrement", 5)      # cost -= 5
Effect("price",   "multiply",  0.9)    # price *= 0.9
Effect("tmp_key", "delete",    None)   # removes the key from state
```

Available operations: `"set"`, `"increment"`, `"decrement"`, `"multiply"`, `"append"`, `"remove"`, `"delete"`.

Because effects are **data, not code**, the kernel can apply them to a copy of state, check the result, and decide whether to commit — all without touching the real state.

```python
# Wrong: action as code (cannot be formally checked before execution)
def deploy():
    subprocess.run(["kubectl", "apply", ...])

# Right: action as a declarative spec (simulated before execution)
deploy_action = ActionSpec(
    id="deploy",
    effects=(Effect("deployed", "set", True),),
    cost=10.0,
)
```

Data can be simulated, diffed, inspected, and inverted. Code cannot.

### ActionSpec

An `ActionSpec` bundles effects with metadata used for planning and reasoning.

```python
from clampai import ActionSpec, Effect

action = ActionSpec(
    id="process_batch",         # unique identifier
    name="Process Batch",       # human-readable name
    description="Process 5 records from the queue",
    effects=(
        Effect("processed", "increment", 5),
        Effect("queue_size", "decrement", 5),
    ),
    cost=2.0,         # how much budget this costs
    reversible=True,  # supports rollback (T7)
)
```

### Invariant

An `Invariant` is a rule that must hold on every state the kernel allows to exist. Write it as a function from `State` to `bool`.

```python
from clampai import Invariant

# Hard stop: if this returns False, the action is rejected.
no_over_send = Invariant(
    name="no_over_send",
    predicate=lambda s: s.get("emails_sent", 0) <= 100,
    description="Never send more than 100 emails",
    enforcement="blocking",   # "blocking" (default) or "monitoring"
    suggestion="Stop sending and report to the user.",
    max_eval_ms=5.0,          # treated as violation if predicate is too slow
)

# Soft warning: violation is logged but execution continues.
low_budget_warning = Invariant(
    name="low_budget_warning",
    predicate=lambda s: s.get("spend_usd", 0) < 40,
    description="Warn when approaching the $40 spend limit",
    enforcement="monitoring",
)
```

**Rules for invariant predicates:**
- Pure function of `State` — no I/O, no side effects, no randomness.
- Fast — sub-millisecond target. Use `max_eval_ms` for a hard timeout: a predicate that exceeds it is treated as a violation (fail-safe).
- Decidable — returns a clear True/False; no exceptions, no external calls.

### SafetyKernel

The kernel is the core enforcement object. Everything flows through it.

```python
from clampai import SafetyKernel, Invariant

kernel = SafetyKernel(
    budget=50.0,
    invariants=[
        Invariant("no_delete", lambda s: not s.get("deleted", False),
                  "Deletion not allowed", enforcement="blocking"),
    ],
    min_action_cost=0.01,   # T2: prevents infinite loops (must be > 0)
)
```

Key methods:

| Method | What it does |
|--------|-------------|
| `evaluate(state, action)` | Returns a `SafetyVerdict` — approved or rejected with reasons |
| `execute(state, action)` | Returns `(new_state, trace_entry)`; must be called after `evaluate` approves |
| `evaluate_and_execute_atomic(state, action)` | Returns `(new_state, trace_entry)`; evaluates and commits in one locked step |
| `rollback(state_before, state_after, action)` | Refunds budget and returns `state_before` exactly (T7) |

---

## The @safe decorator

For wrapping a single function — rate-limiting an API call, enforcing a spend ceiling — the `@safe` decorator is the simplest path.

```python
from clampai import safe, SafetyViolation

# Charge 1.0 against a budget of 10 on each call.
@safe(budget=10.0)
def call_external_api(endpoint: str) -> dict:
    ...

# After 10 calls, this raises SafetyViolation.
```

With invariants and dynamic state (reading live counters from your database or state store):

```python
from clampai.invariants import rate_limit_invariant, value_range_invariant

@safe(
    budget=50.0,
    cost_per_call=2.0,
    invariants=[
        rate_limit_invariant("calls_today", max_count=20),
        value_range_invariant("cost_usd", 0, 40),
    ],
    state_fn=lambda: {
        "calls_today": db.get_call_count_today(),
        "cost_usd": db.get_total_spend(),
    },
)
def send_email(recipient: str, body: str) -> None:
    ...

try:
    send_email("alice@example.com", "Hello")
except SafetyViolation as exc:
    print(f"Blocked: {exc}")

# Inspect after the fact.
print(send_email.audit_log)               # [{step, action, cost, timestamp, approved}, ...]
print(send_email.kernel.budget.remaining)

# Reset for a new session or test run.
send_email.reset()
```

`@safe` is thread-safe. Each call acquires a lock before evaluating, so concurrent calls from different threads do not race on the budget or counters.

---

## Building an agent with Orchestrator

For multi-step tasks where an LLM chooses what to do at each step, use `TaskDefinition` + `Orchestrator` (synchronous) or `AsyncOrchestrator` (async).

### Minimal example

```python
from clampai import (
    TaskDefinition, State, ActionSpec, Effect, Invariant, Orchestrator
)

task = TaskDefinition(
    goal="Process 10 records",
    initial_state=State({"processed": 0, "errors": 0}),
    available_actions=[
        ActionSpec(
            id="process_batch",
            name="Process Batch",
            description="Process the next 5 records",
            effects=(Effect("processed", "increment", 5),),
            cost=2.0,
        ),
    ],
    invariants=[
        Invariant(
            "max_errors",
            lambda s: s.get("errors", 0) <= 3,
            "Stop if more than 3 errors",
            enforcement="blocking",
        ),
    ],
    budget=20.0,
    goal_predicate=lambda s: s.get("processed", 0) >= 10,
)

engine = Orchestrator(task)
result = engine.run()
print(result.summary())
```

No API key needed — the built-in `MockLLMAdapter` drives execution for development and testing.

### Connecting a real LLM

```python
import anthropic
from clampai.adapters import AnthropicAdapter

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
engine = Orchestrator(task, llm=AnthropicAdapter(client))
result = engine.run()
```

### Reading the result

```python
result.goal_achieved          # True / False
result.termination_reason     # GOAL_ACHIEVED / BUDGET_EXHAUSTED / STEP_LIMIT / ...
result.total_steps            # number of actions executed
result.total_cost             # total budget spent
result.final_state            # the State at termination
result.to_dict()              # JSON-serialisable summary
result.summary()              # human-readable one-pager with ✅ / ❌
```

Termination reasons:

| Reason | Meaning |
|--------|---------|
| `GOAL_ACHIEVED` | `goal_predicate` returned True |
| `BUDGET_EXHAUSTED` | No affordable action remains |
| `STEP_LIMIT` | `int(budget / min_action_cost)` steps reached |
| `MAX_FAILURES` | Too many consecutive rejected actions |
| `STUCK` | Progress monitor detects no improvement |
| `LLM_STOP` | LLM explicitly chose to stop |
| `ERROR` | Initial invariant violation or unrecoverable error |

### How the execution loop works

```
while not done:
    1. Check invariants on current state
    2. Check if goal is already achieved
    3. Check step limit and remaining budget
    4. Get actions whose cost fits in the remaining budget
    5. Score each action (expected progress, information gain,
       cost ratio, risk, opportunity cost)
    6. Ask the LLM to choose — or skip the call if one action
       clearly dominates (saves latency and API cost)
    7. Kernel validates the chosen action
    8. If approved: apply it, update beliefs and progress
    9. If rejected: record failure, update failure beliefs
   10. Check termination conditions
```

If the LLM call fails for any reason (timeout, parse error, returns a non-existent action ID), the orchestrator falls back to the highest-scored affordable action. Execution never stalls on an LLM failure.

---

## Pre-built invariants

`clampai/invariants.py` ships 25 ready-to-use factory functions. Import from `clampai.invariants` or directly from `clampai`.

### Complete reference

| Function | What it blocks | Signature (positional args) |
|----------|---------------|----------------------------|
| `rate_limit_invariant` | `state[key] > max_count` | `(key, max_count)` |
| `resource_ceiling_invariant` | `state[key] > ceiling` | `(key, ceiling)` |
| `value_range_invariant` | `state[key]` outside `[min, max]` | `(key, min_val, max_val)` |
| `max_retries_invariant` | Retry counter exceeded | `(key, limit)` |
| `api_call_limit_invariant` | API call count exceeded | `(key, max_calls)` |
| `file_operation_limit_invariant` | File op count exceeded | `(key, max_ops)` |
| `no_delete_invariant` | Key becomes falsy/None | `(key)` |
| `read_only_keys_invariant` | Any key changes from initial | `(keys, initial_state)` |
| `required_fields_invariant` | Any field is absent or None | `(fields)` — a list |
| `no_sensitive_substring_invariant` | Key contains forbidden strings | `(key, forbidden)` |
| `no_regex_match_invariant` | Key matches a regex pattern | `(key, pattern)` |
| `email_safety_invariant` | `emails_deleted > 0` | `()` — no args |
| `pii_guard_invariant` | Any key contains SSN/CC/email/phone | `(*keys)` — variadic |
| `string_length_invariant` | `len(str(state[key])) > max_length` | `(key, max_length)` |
| `human_approval_gate_invariant` | Approval flag not set | `(approval_key)` |
| `no_action_after_flag_invariant` | Flag is set | `(flag_key)` |
| `allowed_values_invariant` | Value not in allowed set | `(key, allowed)` |
| `monotone_increasing_invariant` | Key decreases | `(key)` |
| `monotone_decreasing_invariant` | Key increases | `(key)` |
| `non_empty_invariant` | Key is empty/None | `(key)` |
| `list_length_invariant` | List/string longer than max | `(key, max_length)` |
| `no_duplicate_ids_invariant` | List contains duplicates | `(key)` |
| `time_window_rate_invariant` | Too many timestamps in window | `(key, max_count, window_seconds)` |
| `json_schema_invariant` | Dict field has wrong type | `(key, schema)` — schema maps `field → type` |
| `custom_invariant` | Whatever your validator says | `(name, validator, key)` |

### Examples

```python
from clampai.invariants import (
    rate_limit_invariant,
    resource_ceiling_invariant,
    value_range_invariant,
    required_fields_invariant,
    allowed_values_invariant,
    no_sensitive_substring_invariant,
    pii_guard_invariant,
    string_length_invariant,
    human_approval_gate_invariant,
    no_action_after_flag_invariant,
    monotone_increasing_invariant,
    time_window_rate_invariant,
    json_schema_invariant,
)

# Rate / resource limits
rate_limit_invariant("api_calls", max_count=100)
resource_ceiling_invariant("active_threads", ceiling=20)
value_range_invariant("confidence", 0.0, 1.0)

# Structural
required_fields_invariant(["user_id", "session_token"])   # takes a list
allowed_values_invariant("environment", {"staging", "prod"})

# Security
no_sensitive_substring_invariant("output", ["sk-", "Bearer ", "password"])
pii_guard_invariant("message", "summary")   # variadic: any number of keys

# Compliance
string_length_invariant("llm_output", max_length=4096)
human_approval_gate_invariant("human_approved")    # blocks all until flag set
no_action_after_flag_invariant("task_complete")   # blocks all once flag set

# Progress
monotone_increasing_invariant("tasks_completed")  # counter must never go down

# Time-windowed rate limit (state[key] must be a list of float timestamps)
time_window_rate_invariant("request_timestamps", max_count=10, window_seconds=60.0)

# Type checking on a dict field
json_schema_invariant("payload", {"amount": float, "currency": str})
```

### Writing your own

```python
from clampai import Invariant

my_rule = Invariant(
    name="no_large_transfers",
    predicate=lambda s: s.get("transferred_usd", 0) <= 1000,
    description="No single transfer may exceed $1,000",
    enforcement="blocking",
    suggestion="Split into smaller transfers or request approval first.",
    max_eval_ms=5.0,
)
```

---

## Testing your safety rules

`clampai.testing` provides utilities for writing unit tests against your safety configuration without spinning up a full Orchestrator.

### `make_state` and `make_action`

```python
from clampai import make_state, make_action

state  = make_state(counter=0, status="pending")
action = make_action("increment", cost=1.0, counter=5)  # sets counter=5 in state
```

### `SafetyHarness`

A context manager that creates a `SafetyKernel` from `budget` and `invariants` and exposes assertion helpers for testing.

```python
from clampai import SafetyHarness, make_state, make_action
from clampai.invariants import value_range_invariant

def test_spend_ceiling():
    inv = value_range_invariant("spend_usd", 0, 100)
    with SafetyHarness(budget=50.0, invariants=[inv]) as h:
        # Should be allowed:
        h.assert_allowed(make_state(spend_usd=10), make_action("buy", cost=5.0))

        # Should be blocked (invariant violation):
        h.assert_blocked(
            make_state(spend_usd=99),
            make_action("buy_big", cost=5.0, spend_usd=200),
            reason_contains="spend_usd",
        )

def test_budget_exhaustion():
    with SafetyHarness(budget=10.0) as h:
        h.execute(make_state(), make_action("step1", cost=6.0))
        h.assert_budget_remaining(4.0)
        h.assert_blocked(make_state(), make_action("step2", cost=6.0))
```

`SafetyHarness` methods:

| Method | What it does |
|--------|-------------|
| `assert_allowed(state, action)` | Fails the test if the action is blocked |
| `assert_blocked(state, action, *, reason_contains=None)` | Fails if the action is approved |
| `assert_budget_remaining(expected, tol=0.01)` | Fails if remaining budget differs |
| `assert_step_count(expected)` | Fails if step count differs |
| `execute(state, action)` | Runs the action; raises `RuntimeError` if blocked |
| `reset()` | Recreates the kernel with original budget and invariants |

---

## LLM adapters

All adapters implement the same interface: `complete(prompt, ...) -> str` and `acomplete(...)` for async. Swapping providers requires changing only the adapter — no safety logic changes.

| Adapter class | Backend | Install |
|---------------|---------|---------|
| `AnthropicAdapter` / `AsyncAnthropicAdapter` | Anthropic Claude | `pip install clampai[anthropic]` |
| `OpenAIAdapter` / `AsyncOpenAIAdapter` | OpenAI / Azure OpenAI | `pip install clampai[openai]` |
| `OpenClawAdapter` / `AsyncOpenClawAdapter` | OpenClaw local CLI | `npm install -g openclaw@latest` |
| `ClampAISafeTool` | LangChain `BaseTool` wrapper | `pip install clampai[langchain]` |
| `SafeMCPServer` | MCP tool server with shared budget | `pip install clampai[mcp]` |
| `MockLLMAdapter` | Deterministic mock (built-in) | — |

### Anthropic Claude

```python
import anthropic
from clampai.adapters import AnthropicAdapter

client = anthropic.Anthropic()
engine = Orchestrator(task, llm=AnthropicAdapter(client, model="claude-opus-4-6"))
```

### OpenAI

```python
import openai
from clampai.adapters import OpenAIAdapter

client = openai.OpenAI()
engine = Orchestrator(task, llm=OpenAIAdapter(client, model="gpt-4o"))
```

### OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is a local-first AI assistant runtime. ClampAI wraps it with the same budget and invariant enforcement applied to cloud LLMs.

Prerequisites:
```bash
npm install -g openclaw@latest
openclaw gateway   # keeps the WebSocket control plane running
```

```python
from clampai.adapters import AsyncOpenClawAdapter
from clampai.invariants import pii_guard_invariant

adapter = AsyncOpenClawAdapter(thinking="medium")
task = TaskDefinition(
    goal="Summarise my inbox",
    initial_state=State({"emails_read": 0}),
    available_actions=[...],
    invariants=[pii_guard_invariant("summary")],
    budget=20.0,
    goal_predicate=lambda s: s.get("emails_read", 0) >= 10,
)
engine = Orchestrator(task, llm=adapter)
```

### LangChain

**`ClampAICallbackHandler`** — wrap any existing LangChain agent with budget and invariant enforcement in two lines. No changes to the agent required.

```python
from clampai.adapters import ClampAICallbackHandler
from clampai.invariants import pii_guard_invariant, rate_limit_invariant

handler = ClampAICallbackHandler(
    budget=50.0,
    invariants=[
        rate_limit_invariant("tool_calls", max_count=20),
        pii_guard_invariant("tool_input"),
    ],
)

# Works with any LangChain agent unchanged:
result = agent.invoke({"input": "..."}, config={"callbacks": [handler]})

# Inspect after:
print(f"Budget remaining: {handler.budget_remaining:.1f}")
print(f"Actions blocked:  {handler.actions_blocked}")
```

**`ClampAISafeTool`** wraps a ClampAI `Orchestrator` as a standard LangChain `BaseTool`. Every `_run()` call runs the orchestrator's full goal-directed loop with its safety kernel enforced throughout.

```python
from clampai.adapters import ClampAISafeTool
from clampai import Orchestrator

tool = ClampAISafeTool(
    orchestrator=Orchestrator(task),
    name="email_agent",
    description="Manages an email inbox with safety guarantees.",
)
# Drop into any LangChain agent or LangGraph graph unchanged.
```

**Version note:** LangChain has breaking API changes across minor versions. Pin in `requirements.txt`:
```
langchain==0.3.19
langchain-core==0.3.40
```

### MCP server

For multi-model pipelines using the Model Context Protocol:

```python
from clampai.adapters import SafeMCPServer

server = SafeMCPServer("my-agent", budget=100.0)

@server.tool(cost=5.0)
def search(query: str) -> str: ...

@server.tool(cost=10.0)
def write_file(path: str, content: str) -> str: ...

server.run()   # stdio MCP server; all tools share one SafetyKernel
```

---

## Framework integrations

### LangGraph

`pip install clampai[langgraph]`

Wrap any LangGraph node with budget and invariant enforcement using the `@clampai_node` decorator or `SafetyNode` class. Blocked nodes raise `ClampAIBudgetError` or `ClampAIInvariantError`, which your graph can catch and route to an error handler.

```python
from clampai.adapters import clampai_node, budget_guard, invariant_guard
from clampai.invariants import rate_limit_invariant, value_range_invariant
from langgraph.graph import StateGraph, END

# Wrap a node — safety is enforced before the function body runs
@clampai_node(budget=50.0, cost_per_step=2.0,
               invariants=[rate_limit_invariant("api_calls", max_count=20)])
def call_api_node(state: dict) -> dict:
    # Only runs if budget remains and all invariants hold
    result = my_api.call(state["query"])
    return {"result": result, "api_calls": state.get("api_calls", 0) + 1}

# Standalone budget gate node (no wrapped function — pure enforcement)
graph = StateGraph(dict)
graph.add_node("call_api", call_api_node)
graph.add_node("budget_gate", budget_guard(budget=50.0, cost_per_step=1.0))
graph.add_node("error_handler", lambda s: {"error": "safety limit reached"})
```

Use `invariant_guard` to add a safety checkpoint that checks invariants without charging budget:

```python
graph.add_node("data_guard", invariant_guard([
    rate_limit_invariant("api_calls", max_count=20),
    value_range_invariant("confidence", 0.0, 1.0),
]))
```

See [`examples/06_langgraph_agent.py`](examples/06_langgraph_agent.py) for a complete runnable example demonstrating all four integration patterns.

### FastAPI / Starlette

`pip install clampai[fastapi]`

`ClampAIMiddleware` enforces a per-application budget across all HTTP requests. Requests that exhaust the budget receive HTTP 429.

```python
from fastapi import FastAPI
from clampai.adapters import ClampAIMiddleware
from clampai.invariants import rate_limit_invariant

app = FastAPI()

app.add_middleware(
    ClampAIMiddleware,
    budget=10_000.0,
    cost_per_request=1.0,
    invariants=[
        rate_limit_invariant("requests_served", max_count=5_000),
    ],
)

@app.get("/search")
def search(q: str) -> dict:
    return {"results": [...]}
# Each request costs 1.0. After 10,000 budget units: 429.
```

Inspect budget and reset by traversing the Starlette middleware stack:

```python
def _get_clampai_mw(app):
    stack = app.middleware_stack
    while stack is not None:
        if isinstance(stack, ClampAIMiddleware):
            return stack
        stack = getattr(stack, "app", None)

mw = _get_clampai_mw(app)
print(mw.budget_remaining)
mw.reset()
```

See [`examples/07_fastapi_middleware.py`](examples/07_fastapi_middleware.py) for a complete standalone demo (no uvicorn required).

### CrewAI

`pip install crewai`

**`ClampAISafeCrewTool`** wraps any callable as a CrewAI tool with budget and invariant enforcement. **`ClampAICrewCallback`** enforces safety at each agent step and task completion.

```python
from clampai.adapters import ClampAISafeCrewTool, ClampAICrewCallback, safe_crew_tool
from clampai.invariants import rate_limit_invariant, pii_guard_invariant

# Decorator form:
@safe_crew_tool(budget=50.0, cost=2.0, name="web_search",
                description="Search the web",
                invariants=[pii_guard_invariant("query")])
def web_search(query: str) -> str:
    return search_api(query)

# Callback form (enforces per-step and per-task budget):
callback = ClampAICrewCallback(
    budget=100.0,
    cost_per_step=1.0,
    invariants=[rate_limit_invariant("steps", max_count=50)],
)

crew = Crew(
    agents=[...],
    tasks=[...],
    step_callback=callback.step_callback,
    task_callback=callback.task_callback,
)
```

### AutoGen

`pip install pyautogen`

**`ClampAISafeAutoGenAgent`** wraps any AutoGen reply function with budget and invariant enforcement. **`autogen_reply_fn`** is the decorator form.

```python
from clampai.adapters import autogen_reply_fn, ClampAISafeAutoGenAgent
from clampai.invariants import rate_limit_invariant

# Decorator form:
@autogen_reply_fn(budget=50.0, cost_per_reply=2.0, agent_name="researcher",
                  invariants=[rate_limit_invariant("messages", max_count=20)])
def my_reply_fn(recipient, messages, sender, config):
    return True, generate_reply(messages)

# Register with AutoGen:
agent = ConversableAgent("researcher")
agent.register_reply(
    [ConversableAgent, None],
    my_reply_fn,
)

# Standalone gate (check before sending a message):
safe_fn = ClampAISafeAutoGenAgent(
    my_reply_fn, budget=50.0, cost_per_reply=1.0
)
safe_fn.check(message="run the deployment script", sender=agent)
```

### HTTP Sidecar Server

Run ClampAI as a standalone HTTP service. Any language or runtime can enforce the same budget and invariant guarantees via simple JSON requests — no Python required in the agent process.

```bash
# Start the server (zero external dependencies beyond ClampAI):
python -m clampai.server --budget 1000.0 --port 8765
# ClampAI sidecar server listening on http://127.0.0.1:8765
```

```python
from clampai.server import ClampAIServer
from clampai.invariants import rate_limit_invariant, pii_guard_invariant

server = ClampAIServer(
    budget=1000.0,
    invariants=[
        rate_limit_invariant("api_calls", 200),
        pii_guard_invariant("output"),
    ],
    port=8765,
)
server.start_background()   # non-blocking; server.stop() to shut down

# Any HTTP client can now evaluate/execute actions:
# POST /evaluate  → {"approved": true, "budget_remaining": 999.0}
# POST /execute   → {"approved": true, "new_state": {...}, "step_count": 1}
# GET  /status    → {"budget_remaining": 999.0, "step_count": 1}
# POST /reset     → {"status": "ok", "budget_remaining": 1000.0}
```

```bash
# From any language:
curl -s -X POST http://127.0.0.1:8765/execute \
  -H 'Content-Type: application/json' \
  -d '{"state": {"api_calls": 5}, "action": {"id": "call_api", "name": "Call API",
       "description": "API call", "cost": 1.0, "effects": [], "reversible": false}}'
# → {"approved": true, "new_state": {"api_calls": 5}, "step_count": 1, ...}
```

---

## Formal guarantees in plain English

The safety kernel makes eight claims. Here is what each means, without notation.

| Label | Plain English |
|-------|--------------|
| **T1** | Total budget spend never exceeds the budget you set. Enforced by a locked counter — not a trust-based check. |
| **T2** | The loop terminates. Given a finite budget and a minimum action cost greater than zero, there is a hard ceiling on the number of steps. |
| **T3** | Every invariant you declared holds on every state the kernel allows to exist. The kernel checks invariants on a simulated next state before committing — the bad state never becomes real. |
| **T4** | Gross budget spend never goes down. Once spent, budget does not return (unless you explicitly reset). |
| **T5** | Actions are atomic. If rejected, the state and budget are exactly as they were before the attempt — no partial writes. |
| **T6** | The execution trace is append-only. Each entry is SHA-256 chained to the previous, so tampering is detectable. |
| **T7** | For reversible actions, `rollback(execute(s, a)) == s` exactly. The rolled-back state is mathematically identical to the pre-execution state. |
| **T8** | Any action you designate as an emergency action can always execute, regardless of budget. This prevents the system from getting permanently stuck. |

**What these guarantees cover:** the formal model — state, budget, declared invariants, and the trace. They hold as long as you describe actions using `Effect` objects.

**What they do not cover:**
- The gap between your `Effect` declarations and what actually happens in the world. If you declare `Effect("files_deleted", "increment", 1)` but the code behind the action deletes 100 files, the kernel tracks 1.
- LLM decision quality — the kernel approves or rejects; it does not make the LLM smarter.
- Multi-process or cross-machine state. A `threading.Lock` does not cross process boundaries.

### Guarantee taxonomy

Every claim in ClampAI carries one of four epistemic labels:

| Label | Meaning |
|-------|---------|
| `PROVEN` | Holds unconditionally by construction — induction over code |
| `CONDITIONAL` | Proven under stated assumptions (e.g., T2 requires `min_action_cost > 0`) |
| `EMPIRICAL` | Measured on test suites; confidence intervals available |
| `HEURISTIC` | Best-effort; no formal guarantee (gradient tracker, HJB barrier) |

T1, T3, T4, T5, T6, T7 are `PROVEN`. T2 and T8 are `CONDITIONAL`. See [MATHEMATICAL_COMPLIANCE.md](https://github.com/Ambar-13/ClampAI/blob/main/MATHEMATICAL_COMPLIANCE.md) for the full proofs.

### What is and is not guaranteed for real-world actions

T1–T8 operate over *declared* `Effect` objects. The table below shows what is and is not enforced for each category of real-world action.

| Action type | Example | T1 Budget | T3 Invariants | T7 Rollback |
|-------------|---------|-----------|---------------|-------------|
| Abstract state only | `pages_written += 1` | PROVEN | PROVEN | PROVEN |
| Real-world with proxy tracking | `emails_sent += 1` as proxy for `send_email()` | PROVEN | PROVEN on proxy | PROVEN on proxy |
| Real-world, no proxy declared | `http_request()` with no declared effects | PROVEN | NOT APPLICABLE | NOT APPLICABLE |
| Opaque shell/code execution | `execute_shell(...)` with no declared effects | PROVEN | NOT APPLICABLE | NOT APPLICABLE |

NOT APPLICABLE is not a bug — the kernel can only simulate and invert effects it knows about. Undeclared side effects leave the formal model intact but create a spec-reality gap. The `EnvironmentReconciler` in `hardening.py` is designed to detect this at the next reconciliation cycle.

---

## Architecture

ClampAI has four layers. Each layer can only clampain the layers above it — not bypass them.

```
┌─────────────────────────────────────────────────────┐
│  Layer 3 — Hardening                                │
│  Environment reconciliation, temporal dependencies, │
│  subprocess sandboxing, multi-dimensional attestors │
├─────────────────────────────────────────────────────┤
│  Layer 2 — Orchestrator                             │
│  Main execution loop, LLM interface, fallback logic │
├─────────────────────────────────────────────────────┤
│  Layer 1 — Reasoning Engine                         │
│  Bayesian beliefs, causal graph, action valuation   │
├─────────────────────────────────────────────────────┤
│  Layer 0 — Safety Kernel          [T1–T8]           │
│  Immutable state, declarative effects, budget,      │
│  invariant checks, hash-chained trace               │
└─────────────────────────────────────────────────────┘
```

**Layer 0 — Safety Kernel:** The innermost gate. For each proposed action: (1) check minimum cost, (2) check budget, (3) check step limit, (4) simulate action on a copy of state, (5) check all blocking invariants on the simulated result, (6) if all pass, commit atomically; if anything fails, reject with state and budget unchanged.

**Layer 1 — Reasoning Engine:** Structured intelligence that informs the LLM rather than blindly deferring to it. Bayesian Beta(α, β) beliefs per action, updated after every outcome. Causal graph of action dependencies — blocked actions are never offered to the LLM. Multi-dimensional value scores (progress, information gain, cost ratio, risk). Integral Sensitivity Filter: prunes state variables from the prompt that the available actions do not affect, reducing token usage without hiding safety-relevant data.

**Layer 2 — Orchestrator:** The main execution loop. Picks actions, calls or skips the LLM, runs the kernel, updates beliefs, checks termination conditions. Falls back to highest-scored READY action if the LLM call fails.

**Layer 3 — Hardening:** Practical fixes for adversarial and real-world deployment conditions.

| Mechanism | What it prevents |
|-----------|-----------------|
| `SubprocessAttestor` | Command injection; enforces a binary allowlist |
| `TemporalCausalGraph` | Race conditions between provisioned and ready states |
| `CostAwarePriorFactory` | First-strike budget waste; expensive actions start pessimistic |
| `EnvironmentReconciler` | Model drift; halts if live environment diverges from model state |
| `MultiDimensionalAttestor` | Reward hacking; all quality dimensions must pass simultaneously |

---

## Advanced features

### Information flow control

Tag state fields with security levels and block actions that would write high-security data to a lower-security field.

```python
from clampai import DataLabel, SecurityLevel, ReferenceMonitor

monitor = ReferenceMonitor()
monitor.set_ifc_label("user_email", DataLabel(SecurityLevel.PII))
# Actions that write PII to a lower-security field are blocked.
```

Security levels form a lattice: `PUBLIC ⊑ INTERNAL ⊑ PII ⊑ SECRET`. Information can only flow from lower to higher security levels, never the reverse.

### Control barrier functions

Add a mathematical barrier that limits how quickly the state can move toward unsafe regions.

```python
from clampai import ReferenceMonitor

monitor = ReferenceMonitor()
# h(s) > 0 means "safe". alpha controls how fast h may decrease per step.
monitor.add_cbf(h=lambda s: 1.0 - s.get("risk", 0.0), alpha=0.1)
```

### Compositional task verification

If you build a pipeline from separately verified tasks, you can compose their contracts:

```python
from clampai import OperadicComposition, ContractSpecification

spec_a = ContractSpecification(name="fetch", assume=..., guarantee=...)
spec_b = ContractSpecification(name="process", assume=..., guarantee=...)
composed = OperadicComposition.compose(spec_a, spec_b)
# composed.name == "(fetch;process)"
# composed.assume comes from spec_a, composed.guarantee from spec_b
```

### Rollback

For reversible actions (declared with `reversible=True`), undo is exact:

```python
new_state, _entry = kernel.evaluate_and_execute_atomic(state, reversible_action)
restored = kernel.rollback(state, new_state, reversible_action)
# restored == state  (exactly — T7 guarantee)
```

---

## Known limitations

| Issue | Status |
|-------|--------|
| Spec-reality gap: proofs hold on the formal model, not on what code does in the world | Partially mitigated by `EnvironmentReconciler` |
| No text-layer safety: ClampAI does not do prompt injection detection, toxicity filtering, or jailbreak prevention | By design — use NeMo Guardrails or LLM Guard alongside ClampAI |
| Multi-process budget sharing | `ProcessSharedBudgetController` uses `multiprocessing.Value` (shared memory, int64) for cross-process budget enforcement. T1 and T4 guarantees are preserved across OS processes. Cross-machine multi-agent is not supported. |
| Subjective goals ("write good code", "be helpful") | No formal predicate is possible. Use `MultiDimensionalAttestor`. |
| Deep Python memory manipulation (`ctypes`, `gc`, `sys`) | Partially mitigated. Not memory-safe against intentional bypasses. |
| 4 adversarial evasion vectors (base64 payloads, `getattr` dynamic dispatch) | These bypass the classification layer before an `ActionSpec` is constructed. The kernel itself has no known bypass. |

See [docs/VULNERABILITIES.md](https://github.com/Ambar-13/ClampAI/blob/main/docs/VULNERABILITIES.md) for the full breakdown.

---

## Performance

Measured on 10,000 sequential safety checks in a single process (see [BENCHMARKS.md](https://github.com/Ambar-13/ClampAI/blob/main/BENCHMARKS.md) for methodology):

| Metric | Value |
|--------|-------|
| Average latency per check | 0.061 ms |
| P99 latency | 0.191 ms |
| Throughput | ~45,600 checks/second |
| Adversarial recall (39 attack vectors) | 89.7% |
| False positives | Zero |

The 89.7% recall is honest: 4 of the 39 attack vectors evade classification before an `ActionSpec` is constructed. Once an `ActionSpec` reaches the kernel, there are no known bypasses.

---

## FAQ

**Does ClampAI prevent the LLM from making bad decisions?**

No. ClampAI prevents bad decisions from executing. The LLM can propose anything — the kernel only approves actions that fit within the declared budget and invariants. If the LLM's decisions are consistently wrong, that is a prompting or model quality problem that ClampAI cannot fix.

---

**What is the difference between `@safe` and `Orchestrator`?**

`@safe` wraps a single Python function. One call equals one action — good for API wrappers, tools, and functions with a cost ceiling. `Orchestrator` runs a multi-step loop where an LLM picks actions from a task definition. Good for autonomous agents with a goal and a set of available tools.

---

**Does it support multiple agents?**

Yes, within one Python process. Multiple `Orchestrator` instances can share a single `SafetyKernel`. The budget controller is lock-protected; `evaluate_and_execute_atomic()` is atomic under that lock. Each orchestrator keeps its own belief state; only the budget and invariants are shared. See `examples/multi_agent_shared_kernel.py` for a working 8-agent demo.

For multi-process agent pools (e.g., concurrent workers each calling an LLM), use `ProcessSharedBudgetController` in place of the default `BudgetController`. It uses `multiprocessing.Value` with a process-safe lock, preserving the T1 budget guarantee across OS processes. Cross-machine multi-agent is not supported. See `docs/MULTI_AGENT_RFC.md`.

---

**Does it support async?**

Yes, fully. Both the kernel and the orchestrator have native async implementations.

**`AsyncSafetyKernel`** — drop-in async kernel. `evaluate()` and `execute_atomic()` are coroutines; the internal lock is `asyncio.Lock` so concurrent coroutines yield to the event loop rather than blocking OS threads. All T1–T8 guarantees are preserved.

```python
from clampai import AsyncSafetyKernel

kernel = AsyncSafetyKernel(budget=10.0, invariants=[])
verdict = await kernel.evaluate(state, action)
new_state, entry = await kernel.execute_atomic(state, action)
```

**`AsyncOrchestrator`** — drop-in async orchestrator. Same interface as `Orchestrator`; the execution loop, LLM call, and kernel commit are all async. Uses `AsyncSafetyKernel` internally. Pass a shared kernel to enforce a single budget across concurrent agents.

```python
import asyncio
import anthropic
from clampai import AsyncOrchestrator, AsyncSafetyKernel
from clampai.adapters import AsyncAnthropicAdapter

# Single agent.
engine = AsyncOrchestrator(
    task,
    llm=AsyncAnthropicAdapter(anthropic.AsyncAnthropic()),
)
result = await engine.run()

# Multi-agent: two orchestrators sharing one budget.
shared_kernel = AsyncSafetyKernel(budget=50.0, invariants=[])
results = await asyncio.gather(
    AsyncOrchestrator(task_a, kernel=shared_kernel).run(),
    AsyncOrchestrator(task_b, kernel=shared_kernel).run(),
)
```

`AsyncAnthropicAdapter` and `AsyncOpenAIAdapter` use the native async clients of their respective SDKs. `LLMAdapter.acomplete()` on the base class wraps the synchronous path with `asyncio.to_thread()` — sufficient for custom adapters that do not need native async.

---

**What happens if an invariant predicate takes too long?**

Set `max_eval_ms=N` on the invariant. If the predicate exceeds the timeout, the kernel treats it as a violation — fail-safe: when in doubt, block.

---

**Can the LLM reason its way around the kernel?**

The kernel reads only the `ActionSpec` and the `State` — not any text in the reasoning summary. Injected instructions in text fields are stored in the audit log but never evaluated. The realistic risk is the spec-reality gap: an `ActionSpec` with missing effects cannot be clampained by invariants that do not know about those effects.

---

**How do I debug a rejected action?**

```python
verdict = kernel.evaluate(state, action)
print(verdict.approved)           # False
print(verdict.rejection_reasons)  # ("Cannot afford $3.00: ...", "Invariant 'rate_limit' VIOLATED")
```

---

## Examples

| File | What it demonstrates |
|------|---------------------|
| `01_hello_safety.py` | One kernel, one check — the absolute minimum |
| `02_budget_enforcement.py` | Budget tracking with `@safe` decorator |
| `03_invariants.py` | Custom and pre-built invariants, blocking vs monitoring |
| `04_orchestrator.py` | Full `TaskDefinition` → `Orchestrator` pipeline |
| `05_safe_patterns.py` | `@safe` patterns: basic, `state_fn`, pipeline, reset |
| `06_langgraph_agent.py` | LangGraph: `@clampai_node`, `budget_guard`, `invariant_guard` |
| `07_fastapi_middleware.py` | FastAPI: `ClampAIMiddleware` — budget + invariant enforcement on HTTP requests |
| `email_safety.py` | Adversarial email-deletion demo; run with `--kernel-only` |
| `multi_agent_shared_kernel.py` | 8 concurrent agents sharing one safety kernel |

---

## Running tests

```bash
pip install clampai[dev]
pytest tests/ -v
```

Key test files:

```bash
pytest tests/test_clampai.py              # T1–T8 theorem unit tests
pytest tests/test_monte_carlo.py           # 1,000 random tasks
python tests/chaos_fuzzer.py               # 45 adversarial attack scenarios (standalone script)
pytest tests/test_api.py                   # @safe decorator
pytest tests/test_invariants_coverage.py   # All 25 invariant factories
pytest tests/test_testing_module.py        # SafetyHarness and test utilities
pytest tests/test_orchestrator_coverage.py # Orchestrator termination scenarios
pytest tests/test_reference_monitor_coverage.py
```

---

## Project structure

```
clampai/
├── formal.py               # Layer 0: safety kernel (T1–T8)
├── reasoning.py            # Layer 1: Bayesian beliefs, action valuation, LLM interface
├── orchestrator.py         # Layer 2: main execution loop
├── hardening.py            # Layer 3: environment reconciliation, sandboxing
├── reference_monitor.py    # IFC, control barrier functions, QP action repair
├── inverse_algebra.py      # T7: algebraic inverse effects for exact rollback
├── testing.py              # Test utilities: make_state, make_action, SafetyHarness
├── invariants.py           # 25 pre-built invariant factory functions
├── active_hjb_barrier.py   # Heuristic: k-step lookahead basin avoidance
├── gradient_tracker.py     # Heuristic: finite-difference boundary proximity
├── jacobian_fusion.py      # Heuristic: boundary sensitivity scoring for prompts
├── safe_hover.py           # Hard enforcement gate / emergency stop
├── operadic_composition.py # Compositional verification
├── saliency.py             # Prompt saliency engine
├── verification_log.py     # Proof record writer
├── api.py                  # @safe decorator
├── server.py               # HTTP sidecar server (zero-dep; python -m clampai.server)
├── adapters/
│   ├── anthropic_adapter.py
│   ├── openai_adapter.py
│   ├── openclaw_adapter.py
│   ├── langchain_tool.py
│   ├── langchain_callback.py  # ClampAICallbackHandler (any LangChain agent)
│   ├── langgraph_adapter.py   # LangGraph node and edge safety
│   ├── fastapi_middleware.py  # FastAPI/Starlette request middleware
│   ├── crewai_adapter.py      # CrewAI tool and callback adapters
│   ├── autogen_adapter.py     # AutoGen reply function wrapper
│   ├── mcp_server.py
│   └── metrics.py             # Prometheus + OpenTelemetry backends (+ OTelTraceExporter)
└── __init__.py              # Public API
docs/
├── ARCHITECTURE.md
├── THEOREMS.md
├── VULNERABILITIES.md
└── API.md
tests/
├── test_clampai.py
├── test_monte_carlo.py
├── chaos_fuzzer.py
├── test_api.py
├── test_invariants_coverage.py
├── test_testing_module.py
├── test_new_invariants.py
├── test_orchestrator_coverage.py
├── test_reference_monitor_coverage.py
└── ...
BENCHMARKS.md
CHANGELOG.md
MATHEMATICAL_COMPLIANCE.md
CONTRIBUTING.md
SECURITY.md
```

---

## Citation

```bibtex
@misc{ambar2026clampai,
    title  = {ClampAI: Formal safety framework for AI agents},
    author = {Ambar},
    year   = {2026},
    url    = {https://github.com/Ambar-13/ClampAI},
    note   = {Version 1.0.1}
}
```

**Contact:** Ambar · ambar13@u.nus.edu · National University of Singapore

---

## License

MIT. See `LICENSE`.
