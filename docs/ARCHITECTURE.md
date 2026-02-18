# Architecture

## The problem

AI agents need to take actions in the real world. Those actions cost money, break things, and often can't be undone. Standard approaches fail in predictable ways:

- **Unconstrained agents** (AutoGPT, LangChain) — let the LLM do whatever; hope it makes good decisions.
- **Rule-based constraints** — hardcode allowed actions; brittle, can't handle novel situations.
- **RLHF alignment** — shape what the model wants, not what it can do; training can drift.

ConstrAI takes a different approach: the LLM reasons freely, but its decisions pass through a formal verification layer before anything executes. The LLM can't bypass the math. The math can't bypass the LLM. Both must agree.

**Safety overhead is zero tokens.** Constraints live in the execution loop, not in the context window. No system-prompt bloat.

---

## Layer 0: Safety Kernel (`formal.py`)

Non-negotiable foundation. Nothing executes without passing through here.

### State

State is an immutable dictionary. Every mutation creates a new State object. The internal dict uses Python's `MappingProxyType` (read-only at the C level). `get()` deep-copies values before returning them. `__init__` deep-copies inputs.

```python
s = State({"x": 1, "items": [1, 2, 3]})
s.get("items").append(4)    # Returns a deep copy. Original unchanged.
s._vars["x"] = 99           # TypeError: mappingproxy doesn't support assignment
```

Why this matters: immutable state makes T7 (rollback) trivial. If you never mutate, you can always go back to any prior state.

### Effects

Actions don't contain code. They contain declarative Effect specs:

```python
Effect("counter", "increment", 1)
Effect("status", "set", "ready")
Effect("items", "append", "new_item")
Effect("temp_file", "delete")
```

This is deliberate. Code can't be formally checked before it runs. Data can. The kernel simulates effects on a copy of state with no side effects. If the simulation violates an invariant, the action is blocked before any real operation occurs.

Supported modes: `set`, `increment`, `decrement`, `multiply`, `append`, `remove`, `delete`.

### Invariants

User-defined predicates that must hold on every reachable state:

```python
Invariant("budget_positive", lambda s: s.get("balance", 0) >= 0, enforcement="blocking")
Invariant("max_instances", lambda s: s.get("count", 0) <= 10, enforcement="blocking")
```

The kernel checks invariants on the simulated next-state. If any blocking-mode invariant is violated, the action is rejected and the current state is unchanged.

Two enforcement modes:
- `"blocking"` — T3 applies. Violation rejects the action.
- `"monitoring"` — Violation is logged but does not block. Useful for diagnostics and soft limits.

### Budget Controller

Tracks spending in integer millicents (cost × 100,000) to avoid floating-point accumulation across many steps. `can_afford(cost)` checks before `charge(id, cost)` commits. Negative costs are rejected. Separate gross/refunded counters allow rollback refunds without violating T4 (gross spend never decreases).

### Execution Trace

Every action — approved or rejected — is recorded as a `TraceEntry` with a SHA-256 hash chain. Each entry's hash includes the previous entry's hash, creating a tamper-evident log. Modification of any historical entry breaks the chain, detected by `verify_integrity()`.

### Safety Kernel

The gate. For every proposed action:

1. Check min cost (T2 prerequisite)
2. Check budget (`can_afford`, T1)
3. Simulate state transition (`action.simulate(state)`) — pure, no side effects
4. Check all blocking invariants on the simulated state (T3)
5. Run pluggable precondition functions (user-supplied additional gates)
6. If everything passes: commit atomically (charge, increment step, append trace, T5)
7. If anything fails: reject — state and budget unchanged (T5)

The simulate-then-commit pattern gives T3 (invariant preservation) and T5 (atomicity).

**Atomic execution:** `evaluate_and_execute_atomic()` holds the kernel's lock across both the evaluation and the commit, eliminating the TOCTOU race that exists between separate `evaluate()` and `execute()` calls. Use this in concurrent settings.

---

## Layer 1: Reasoning Engine (`reasoning.py`)

This is where intelligence lives. The safety kernel is a cage; the reasoning engine is the brain inside it.

### Bayesian Beliefs

For each action, ConstrAI tracks a Beta(α, β) distribution representing the probability that this action succeeds:

- `observe(True)`: α += 1
- `observe(False)`: β += 1
- Mean = α/(α+β), uncertainty decreases with more observations

Optional decay for non-stationary environments: `α_new = 1 + (α_old - 1) × decay`. With decay=0.95, recent observations count more than old ones. This prevents stale optimistic beliefs from persisting after conditions change.

### Causal Graph

A DAG tracking which actions depend on which. If "deploy" depends on "test" and "test" hasn't completed, "deploy" stays BLOCKED regardless of what the LLM wants. The graph is populated from `TaskDefinition.dependencies` and can be extended at runtime.

### Action Value Computation

Scores each action across five dimensions:

- **Expected progress:** P(action contributes to goal) × (1 / steps_remaining)
- **Information gain:** Current belief variance (high uncertainty → high gain)
- **Cost ratio:** Fraction of remaining budget consumed
- **Risk:** Base risk from risk_level × (2 - confidence); lower data → higher risk
- **Opportunity cost:** Cost ratio × (1 - current_progress)

These scores are computed before the LLM is called. The LLM reasons over pre-computed analysis, not raw action lists.

### Integral Sensitivity Filter

Prunes state variables from the LLM prompt based on which variables the available actions actually affect:

```
S(key) = Σ_{a in available} 𝟙[key ∈ affected(a)] × |V(a)|
```

Variables with low integrated sensitivity are dropped from the prompt. This reduces token usage without affecting what the safety kernel sees — the kernel always operates on the full state.

### LLM Interface

The LLM receives a structured decision-support document, not a vague instruction. It contains: the goal, current state (filtered), all available actions ranked by computed value with READY/BLOCKED status, belief summaries, budget info, and recent history. The LLM returns JSON. If it returns garbage, the parser catches it and falls back to the highest-value READY action.

---

## Layer 2: Orchestrator (`orchestrator.py`)

The main loop:

```
while not done:
    1. Get available actions (affordable + dependency-ready)
    2. Compute action values
    3. If one action clearly dominates: skip LLM call (dominant-strategy shortcut)
    4. Ask LLM to select action (with structured prompt)
    5. Validate LLM response (catch hallucinations, parse errors)
    6. If invalid: fallback to highest-value READY action
    7. Run safety gauntlet:
       a. Gradient tracker (heuristic boundary proximity warning)
       b. HJB barrier (heuristic k-step lookahead for capture basins)
       c. Reference monitor (IFC, CBF, QP repair)
       d. Formal kernel (T1–T8)
    8. If approved: commit, update beliefs and causal graph, record rollback
    9. If rejected: record failure, update failure beliefs
    10. Check termination: goal? budget? stuck? max_failures?
```

Termination conditions: goal achieved, budget exhausted, step limit reached, LLM requests stop, stuck (no progress for N steps), too many consecutive failures.

---

## Layer 3: Hardening (`hardening.py`)

Fixes for specific vulnerabilities identified during adversarial testing.

### Sandboxed Attestors

External goal verification via subprocess. Commands are frozen tuples set at creation time. `shell=False` always. Binary allowlist prevents arbitrary execution. Output capped at 8KB. The agent cannot influence what command runs.

### Temporal Dependencies

Extends the causal graph with time awareness. A `ReadinessProbe` polls a resource with exponential backoff until it's truly ready (not just existing). Prevents "deploy the moment the database is provisioned, before it accepts connections" race conditions.

### Cost-Aware Priors

Bayesian priors proportional to risk. Cheap reversible actions get optimistic priors (Beta(3,1)); expensive irreversible actions get pessimistic priors (Beta(0.01, 100), effectively blocked until explicitly authorized). This reduces first-strike budget waste.

### Environment Reconciliation

After each action, compares the model's expected state to live environment probes. If the model says 5 files exist but `ls | wc -l` returns 3, execution halts with `EnvironmentDriftError`. Catches the spec-reality gap where `ActionSpec.effects` don't match what actually happens.

### Multi-Dimensional Attestation

Checks goals across multiple quality dimensions: existence, completeness, correctness, quality, safety, regression. Any dimension scoring zero fails the whole attestation. Harder to game than single-metric checks.

---

## Optional safety layers

These provide defence-in-depth between the reasoning layer and the formal kernel.

| Module | Guarantee | Purpose |
|--------|-----------|---------|
| `gradient_tracker.py` | HEURISTIC | Finite-difference Jacobian: which variables are near constraint boundaries? |
| `active_hjb_barrier.py` | HEURISTIC | k-step lookahead: can the agent get trapped in a capture basin within k steps? |
| `reference_monitor.py` | DETERMINISTIC | IFC lattice, CBF resource barriers, QP minimum-intervention repair |
| `inverse_algebra.py` | PROVEN | Algebraic inverse effects for exact rollback (T7 realization) |
| `operadic_composition.py` | CONDITIONAL | Compositional verification: verified subtasks compose without re-verification |
| `jacobian_fusion.py` | HEURISTIC | Sensitivity scoring for prompt saliency |

### Information Flow Control

Enforces a security lattice: `PUBLIC ⊑ INTERNAL ⊑ PII ⊑ SECRET`. Actions that would write higher-classification data to lower-classification sinks are blocked.

### Control Barrier Functions

For resource variables (budget, instance count, latency), CBFs enforce:
```
h(s_{t+1}) - h(s_t) ≥ -α × h(s_t)
```
where h(s) is the "distance to boundary" function. This bounds how fast the system can approach a resource limit.

### QP Minimum-Intervention Repair

When an action's numeric parameters violate a CBF, QP projection finds the minimum-norm modification that makes them safe. The action still executes, just with corrected parameters.

---

## What the LLM sees

A representative prompt:

```
## MISSION
Goal: Deploy a web application
Current Progress: 40.0%
Budget: Gross: $12.00 | Net: $12.00 | Remaining: $38.00
Steps: 4/50

## CURRENT STATE
  built = True
  tested = False
  deployed = False

## AVAILABLE ACTIONS (ranked by computed value)

### [READY] Run Tests (id=test)
  Execute test suite
  Cost: $2.00 | Risk: low | Reversible: True
  Value Score: 0.347
    Progress potential: 0.200
    Information gain: 0.150
    Risk: 0.050
    Analysis: high progress potential

### [BLOCKED] Deploy to Production (id=deploy) — needs: test
  Cost: $5.00 | Risk: high | Reversible: False
  Value Score: 0.122

## YOUR DECISION
{"chosen_action_id": "...", "reasoning": "...", ...}
```

The LLM isn't guessing. It's reasoning over computed quantities within formal constraints.
