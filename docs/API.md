# API Reference

## Primary entry point

The typical usage pattern:

```python
from constrai import (
    State, ActionSpec, Effect, Invariant,
    TaskDefinition, Orchestrator,
)

task = TaskDefinition(...)
engine = Orchestrator(task, llm=my_llm)
result = engine.run()
```

---

## `State` — Immutable world state

```python
State(variables: Dict[str, Any])
```

Immutable, hashable, JSON-deterministic world state. All keys are sorted at construction time to ensure canonical equality.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get(key, default=None)` | `Any` | Returns a deep copy of the value (never a mutable reference) |
| `has(key)` | `bool` | True if key exists |
| `keys()` | `List[str]` | Sorted list of all keys |
| `to_dict()` | `Dict` | Deep copy of all variables |
| `with_updates(updates)` | `State` | New State with given keys updated |
| `without_keys(keys)` | `State` | New State with given keys removed |
| `diff(other)` | `Dict` | `{key: (old, new)}` for differing keys |
| `fingerprint` | `str` | 16-char SHA-256 prefix (property) |
| `describe(max_keys=20)` | `str` | Human-readable summary |

**Immutability:** `__setattr__` and `__delattr__` raise `AttributeError`. The internal dict is wrapped in `MappingProxyType`.

---

## `Effect` — Declarative state mutation

```python
Effect(variable: str, mode: str, value: Any = None)
```

Frozen dataclass. An `Effect` is pure data — it describes a mutation without performing it.

**Modes:**

| Mode | Semantics |
|------|-----------|
| `"set"` | `var = value` |
| `"increment"` | `var += value` |
| `"decrement"` | `var -= value` |
| `"multiply"` | `var *= value` |
| `"append"` | `var.append(value)` |
| `"remove"` | `var.remove(value)` (no-op if absent) |
| `"delete"` | `del var` |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `apply(current)` | `Any` | Apply effect to current value; returns new value |
| `inverse()` | `Effect` | Algebraic inverse (raises `ValueError` for `set`/`delete`) |

---

## `ActionSpec` — Declarative action specification

```python
ActionSpec(
    id: str,
    name: str,
    description: str,
    effects: Tuple[Effect, ...],
    cost: float,
    category: str = "general",
    risk_level: str = "low",       # "low" | "medium" | "high" | "critical"
    reversible: bool = True,
    preconditions_text: str = "",
    postconditions_text: str = "",
    estimated_duration_s: float = 0.0,
    tags: Tuple[str, ...] = (),
)
```

Frozen dataclass. Actions are data, not code.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `simulate(state)` | `State` | Apply effects to state; return new state. Original unchanged. |
| `compute_inverse_effects(state_before)` | `Tuple[Effect, ...]` | Effects that exactly undo this action from `state_before` |
| `affected_variables()` | `Set[str]` | State keys touched by this action's effects |
| `to_llm_text()` | `str` | Full description for LLM prompts |
| `to_compact_text()` | `str` | Single-line summary |

---

## `Invariant` — Safety predicate

```python
Invariant(
    name: str,
    predicate: Callable[[State], bool],
    description: str = "",
    enforcement: str = "blocking",   # "blocking" | "monitoring"
)
```

**`enforcement="blocking"`:** T3 applies. Violated invariants block the action (state unchanged).

**`enforcement="monitoring"`:** Violations are logged but do not block execution.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `check(state)` | `Tuple[bool, str]` | `(holds, reason)`. Exceptions count as violations. |
| `violation_count` | `int` | Total violations observed (property) |

---

## `SafetyKernel` — The formal gate

```python
SafetyKernel(
    budget: float,
    invariants: List[Invariant],
    min_action_cost: float = 0.001,
    emergency_actions: Optional[Set[str]] = None,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(state, action)` | `SafetyVerdict` | Check action safety without committing (pure) |
| `execute(state, action, reasoning="")` | `Tuple[State, TraceEntry]` | Evaluate then commit |
| `evaluate_and_execute_atomic(state, action, reasoning="")` | `Tuple[State, TraceEntry]` | Thread-safe atomic evaluate + commit |
| `record_rejection(state, action, reasons, reasoning="")` | `TraceEntry` | Append a rejection entry to trace |
| `rollback(state_before, state_after, action)` | `State` | Refund budget and return `state_before` |
| `register_emergency_action(action_id)` | `None` | Mark action as emergency (bypasses cost/step limits, T8) |
| `add_precondition(fn)` | `None` | Register additional precondition `fn(state, action) -> (bool, str)` |
| `status()` | `str` | Summary of current step/budget/trace state |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `budget` | `BudgetController` | Budget ledger |
| `trace` | `ExecutionTrace` | Hash-chained execution log |
| `step_count` | `int` | Actions executed so far |
| `max_steps` | `int` | `⌊budget / min_action_cost⌋` |
| `invariants` | `List[Invariant]` | All registered invariants |

---

## `BudgetController` — Resource ledger

| Property | Type | Description |
|----------|------|-------------|
| `budget` | `float` | Total allocated budget |
| `spent_gross` | `float` | Cumulative charges; never decreases (T4) |
| `spent_net` | `float` | Net spend after refunds (= gross - refunded) |
| `remaining` | `float` | `budget - spent_net` |
| `utilization()` | `float` | `spent_net / budget` |
| `can_afford(cost)` | `Tuple[bool, str]` | Thread-safe affordability check |
| `ledger` | `List[Tuple]` | Append-only transaction log `(id, amount, timestamp)` |

---

## `ExecutionTrace` — Audit log

| Method | Returns | Description |
|--------|---------|-------------|
| `append(entry)` | `str` | Append entry; returns its hash |
| `verify_integrity()` | `Tuple[bool, str]` | Walk full hash chain; O(n) |
| `last_n(n)` | `List[TraceEntry]` | Most recent n entries |
| `entries` | `List[TraceEntry]` | All entries (property) |
| `length` | `int` | Number of entries (property) |

---

## `TaskDefinition` — Task configuration

```python
TaskDefinition(
    goal: str,
    initial_state: State,
    available_actions: List[ActionSpec],
    invariants: List[Invariant],
    budget: float,
    goal_predicate: Callable[[State], bool],
    goal_progress_fn: Optional[Callable[[State], float]] = None,
    min_action_cost: float = 0.001,
    dependencies: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    max_retries_per_action: int = 3,
    max_consecutive_failures: int = 5,
    stuck_patience: int = 5,
    system_prompt: str = "",
    risk_aversion: float = 1.0,
    sensitivity_threshold: float = 0.05,
    max_prompt_state_keys: int = 20,
    proof_path: str = "",
    capture_basins: Optional[List[CaptureBasin]] = None,
)
```

**Key fields:**

- `goal_predicate`: `lambda state: bool` — defines when the task is complete.
- `goal_progress_fn`: `lambda state: float` — optional progress estimate in `[0, 1]`.
- `dependencies`: `{"action_b": [("action_a", "reason")], ...}` — DAG edges.
- `priors`: `{"action:my_action:succeeds": (alpha, beta)}` — initial Beta priors.
- `proof_path`: Path to write a `.constrai_proof` JSON artifact at the end.
- `capture_basins`: Forbidden state regions for HJB barrier.

---

## `Orchestrator` — Main engine

```python
Orchestrator(task: TaskDefinition, llm: Optional[LLMAdapter] = None)
```

If `llm` is None, uses `MockLLMAdapter` (deterministic, no API key needed).

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run()` | `ExecutionResult` | Execute until termination; return full result |

---

## `ExecutionResult` — Final outcome

| Field | Type | Description |
|-------|------|-------------|
| `goal_achieved` | `bool` | Did the goal predicate become True? |
| `termination_reason` | `TerminationReason` | Why execution stopped |
| `final_state` | `State` | Last committed state |
| `total_cost` | `float` | Net spend |
| `total_steps` | `int` | Actions committed |
| `goal_progress` | `float` | Last computed progress value |
| `execution_time_s` | `float` | Wall-clock time |
| `actions_attempted` | `int` | Total actions sent to the safety gate |
| `actions_succeeded` | `int` | Actions that passed all checks |
| `actions_rejected_safety` | `int` | Rejected by safety kernel or monitor |
| `rollbacks` | `int` | HJB-triggered rollbacks |
| `llm_calls` | `int` | Calls to the LLM adapter |
| `errors` | `List[str]` | Non-fatal errors logged during execution |
| `summary()` | `str` | Human-readable summary |
| `to_dict()` | `Dict` | JSON-serializable metrics |

**Termination reasons:**

| Value | Meaning |
|-------|---------|
| `GOAL_ACHIEVED` | `goal_predicate(state) = True` |
| `BUDGET_EXHAUSTED` | No affordable actions remain |
| `STEP_LIMIT` | `step_count ≥ max_steps` |
| `LLM_STOP` | LLM set `should_stop = True` |
| `STUCK` | No progress for `stuck_patience` steps |
| `MAX_FAILURES` | `consecutive_failures ≥ max_consecutive_failures` |
| `ERROR` | Unrecoverable error at startup |

---

## `LLMAdapter` — Plugin protocol

```python
class MyLLM:
    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        ...
```

Pass an instance as `Orchestrator(task, llm=MyLLM())`. The adapter does one thing: take a prompt string, return a response string. All structured parsing happens in ConstrAI.

---

## Hardening (`constrai.hardening`)

### `SubprocessAttestor`

```python
SubprocessAttestor(
    command: Tuple[str, ...],        # Frozen; cannot be modified at runtime
    allowed_binaries: Set[str],      # Allowlist of permitted executables
    timeout_s: float = 30.0,
    max_output_bytes: int = 8192,
)
```

### `TemporalCausalGraph`

```python
TemporalCausalGraph()
# .add_temporal_dependency(action_id, deps, readiness_probe)
# .wait_for_ready(action_id)
```

### `CostAwarePriorFactory`

```python
CostAwarePriorFactory(budget: float)
# .create_prior(action) -> Belief
```

### `EnvironmentReconciler`

```python
EnvironmentReconciler(tolerance: float = 0.05)
# .register_probe(key, probe_fn)
# .reconcile(model_state) -> bool  (raises EnvironmentDriftError on failure)
```

### `MultiDimensionalAttestor`

```python
MultiDimensionalAttestor(
    dimensions: List[Callable[[State], float]],
    min_dimension_score: float = 0.3,
)
# .attest(state) -> bool
```

---

## Reference Monitor (`constrai.reference_monitor`)

### `ReferenceMonitor`

```python
ReferenceMonitor(
    ifc_enabled: bool = True,
    cbf_enabled: bool = True,
    hjb_enabled: bool = False,
)
# .add_label(variable, label)       # IFC: tag variable with security level
# .add_cbf(h, alpha)                # CBF: add barrier function h with decay alpha
# .enforce(action, state, available) -> Tuple[bool, str, Optional[ActionSpec]]
```

### `ControlBarrierFunction`

```python
ControlBarrierFunction(h: Callable[[State], float], alpha: float = 0.1)
# .check(state, next_state) -> Tuple[bool, str]
```

### `CaptureBasin`

```python
CaptureBasin(name: str, is_bad: Callable[[State], bool], max_steps: int = 3)
```

---

## Operadic Composition (`constrai.operadic_composition`)

### `InterfaceSignature`

```python
InterfaceSignature(
    required_inputs: Tuple[str, ...],
    produced_outputs: Tuple[str, ...],
    forbidden_variables: Tuple[str, ...] = (),
    precondition_predicates: Tuple[Callable, ...] = (),
    postcondition_predicates: Tuple[Callable, ...] = (),
)
# .compatible_with(other) -> bool
```

### `TaskComposer`

```python
TaskComposer()
# .register_task(super_task)
# .compose_chain(task_ids, composition_type) -> SuperTask
# .is_verified(task_id) -> bool
```

`CompositionType`: `SEQUENTIAL`, `PARALLEL`, `CONDITIONAL`, `REPEAT`.

---

## Guarantee levels (`constrai.formal`)

```python
GuaranteeLevel.PROVEN       # Holds unconditionally by construction
GuaranteeLevel.CONDITIONAL  # Proven under stated assumptions
GuaranteeLevel.EMPIRICAL    # Measured with statistical confidence
GuaranteeLevel.HEURISTIC    # Best-effort; no formal guarantee
```

All 8 theorems are available as `FORMAL_CLAIMS` (a list of `Claim` objects).
