# ConstrAI

**Formal safety framework for autonomous AI agents.**

ConstrAI sits between an LLM and the real world. Before any action executes, it runs a deterministic safety kernel that proves the action cannot exceed the budget, violate any declared invariant, or corrupt the audit log. The LLM reasons freely. The math decides what executes.

```
LLM proposes action
      ↓
  Safety Kernel  ←── proven: T1–T8
      ↓
Execute or Reject
```

---

## Why

Current approaches to agent safety are either too weak or too brittle:

- **Prompt-based guardrails** — "please don't do bad things" — can be forgotten, hallucinated past, or ignored.
- **Post-hoc filters** — reviewing output after decisions are made — fires after the damage.
- **RLHF alignment** — shapes what the model *wants*, not what it *can* do.

ConstrAI enforces safety at the **execution layer**. The LLM produces a decision; the kernel decides whether that decision executes. Neither can bypass the other.

---

## Core guarantees

Eight theorems, proven by construction and induction, hold for every execution:

| Theorem | Statement | Status |
|---------|-----------|--------|
| T1 | `spent(t) ≤ budget` for all t | Proven |
| T2 | Halts in ≤ ⌊budget/min\_cost⌋ steps | Conditional |
| T3 | Every declared invariant holds on every reachable state | Proven |
| T4 | Gross spend is monotonically non-decreasing | Proven |
| T5 | Actions are atomic: rejected actions leave state unchanged | Proven |
| T6 | The execution log is append-only and SHA-256 hash-chained | Proven |
| T7 | `rollback(execute(s, a)) == s` exactly | Proven |
| T8 | The emergency escape action is always executable | Conditional |

**What these proofs cover:** the formal model — state, budget, trace, and declared invariants.

**What they don't cover:** the gap between your ActionSpec and what actually happens in the world (addressed by environment reconciliation), LLM decision quality, and multi-agent coordination.

---

## Quick start

```python
from constrai import TaskDefinition, State, ActionSpec, Effect, Invariant, Orchestrator

task = TaskDefinition(
    goal="Process 10 records",
    initial_state=State({"processed": 0, "errors": 0}),
    available_actions=[
        ActionSpec(
            id="process_batch",
            name="Process Batch",
            description="Process the next 5 records",
            effects=(
                Effect("processed", "increment", 5),
            ),
            cost=2.0,
            reversible=True,
        ),
    ],
    invariants=[
        Invariant(
            "max_errors",
            lambda s: s.get("errors", 0) <= 3,
            description="Abort if error rate is too high",
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

No LLM API key needed — the built-in mock adapter drives the execution. Plug in a real LLM via the `llm=` parameter:

```python
class AnthropicAdapter:
    def __init__(self, client):
        self.client = client

    def complete(self, prompt, system_prompt="", temperature=0.3, max_tokens=2000):
        msg = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

engine = Orchestrator(task, llm=AnthropicAdapter(anthropic_client))
```

---

## Installation

```bash
pip install constrai
```

Optional extras:

```bash
pip install constrai[anthropic]   # Anthropic SDK
pip install constrai[openai]      # OpenAI SDK
pip install constrai[dev]         # pytest
```

---

## Architecture

ConstrAI has four layers. Each layer can only constrain the layers above it — not bypass them.

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
│  Layer 0 — Safety Kernel          [PROVEN T1–T8]   │
│  Immutable state, declarative effects, budget,      │
│  invariant checks, hash-chained trace               │
└─────────────────────────────────────────────────────┘
```

### Layer 0: Safety Kernel

The innermost, non-bypassable gate. For each proposed action:

1. Check minimum cost (T2 prerequisite)
2. Check budget (T1)
3. Check step limit (T2)
4. Simulate action on a copy of state — no side effects during checking
5. Check all blocking-mode invariants on the simulated result (T3)
6. If everything passes: commit atomically (charge budget, update state, append trace, T5)
7. If anything fails: reject — state and budget unchanged (T5)

**Key design choice — actions are data, not code:**

```python
# Wrong: action as a function (cannot be formally checked before execution)
def deploy():
    subprocess.run(["kubectl", "apply", ...])

# Right: action as a declarative spec (simulated before execution)
ActionSpec(
    id="deploy",
    effects=(Effect("deployed", "set", True),),
    cost=10.0,
)
```

Data can be simulated, diffed, inspected, and inverted. Code cannot.

### Layer 1: Reasoning Engine

Structured intelligence that informs the LLM rather than blindly deferring to it:

- **Bayesian beliefs:** Beta(α, β) posterior for each action's success probability, updated after every outcome. Supports decay for non-stationary environments.
- **Causal graph:** DAG of action dependencies. Blocked actions are never offered to the LLM.
- **Action value computation:** Multi-dimensional score per action — expected progress, information gain, cost ratio, risk, opportunity cost. The LLM reasons over computed analysis, not raw action lists.
- **Integral Sensitivity Filter:** Prunes state variables from the LLM prompt by computing which variables the available actions actually affect, reducing token usage without hiding safety-relevant data from the kernel.

### Layer 2: Orchestrator

The main execution loop:

```
while not done:
    available = affordable actions with satisfied dependencies
    values    = multi-dimensional value for each action
    response  = LLM decision (or dominant-strategy skip if margin is large)
    verdict   = kernel.evaluate(chosen_action)
    if approved:
        state, entry = kernel.execute(chosen_action)
        update beliefs, causal graph, progress monitor
    else:
        record rejection, update failure beliefs
    check: goal achieved? budget exhausted? stuck? too many failures?
```

If the LLM fails (timeout, parse error, hallucinated action ID), the orchestrator falls back to the highest-value READY action. Execution never stalls on an LLM failure.

### Layer 3: Hardening

Practical fixes for adversarial and real-world deployment conditions:

| Mechanism | What it prevents |
|-----------|-----------------|
| `SubprocessAttestor` | Command injection via shell metacharacters; binary allowlist enforced |
| `TemporalCausalGraph` | "Provisioned but not ready" race conditions via readiness probes with exponential backoff |
| `CostAwarePriorFactory` | First-strike budget waste; expensive actions start with pessimistic priors until explicitly authorized |
| `EnvironmentReconciler` | Model drift; halts if live environment probes diverge from model state |
| `MultiDimensionalAttestor` | Reward hacking; requires all quality dimensions to pass simultaneously |

---

## Advanced: optional safety layers

These run before the formal kernel and add defence-in-depth:

| Module | What it does | Level |
|--------|-------------|-------|
| `gradient_tracker.py` | Estimates which variables are near invariant boundaries (finite-difference Jacobian) | Heuristic |
| `active_hjb_barrier.py` | k-step lookahead: detects multi-step traps before they close | Heuristic |
| `reference_monitor.py` | Information flow control (IFC), control barrier functions (CBF), QP action repair | Deterministic |
| `operadic_composition.py` | Compositional verification: Verified(A) ∧ Verified(B) ⟹ Verified(A∘B) | Conditional |
| `inverse_algebra.py` | Exact rollback via algebraic inverse effects (T7 realization) | Proven |

### Information Flow Control

```python
from constrai import DataLabel, SecurityLevel, ReferenceMonitor

label = DataLabel("user_pii", SecurityLevel.PII)
monitor = ReferenceMonitor(ifc_enabled=True)
monitor.add_label("email_field", label)
# Actions that would write PII to a lower-security sink are blocked.
```

### Compositional Task Verification

```python
from constrai import SuperTask, TaskComposer, CompositionType

composer = TaskComposer()
composer.register_task(verified_fetch_task)
composer.register_task(verified_process_task)

# Conditionally verified: no re-verification needed when interfaces match.
composed = composer.compose_chain(["fetch", "process"], CompositionType.SEQUENTIAL)
```

---

## Invariant design

T3 (Invariant Preservation) is only as useful as the invariants you write. The kernel enforces whatever you declare.

**Good invariants:**

```python
# Binary, decidable, no side effects
Invariant(
    "budget_positive",
    lambda s: s.get("account_balance", 0) >= 0,
    enforcement="blocking",
)

Invariant(
    "instance_limit",
    lambda s: len(s.get("active_instances", [])) <= 10,
    enforcement="blocking",
)
```

**Monitoring-mode invariants** log violations without blocking — useful for soft warnings:

```python
Invariant(
    "high_latency_warning",
    lambda s: s.get("avg_latency_ms", 0) < 500,
    enforcement="monitoring",   # Logged but never blocks
)
```

**Avoid** blocking I/O, non-determinism, or slow computations inside invariant predicates.

---

## Guarantee taxonomy

Every claim in ConstrAI is tagged with its epistemic status:

| Level | Meaning |
|-------|---------|
| `PROVEN` | Holds unconditionally by construction (induction + code inspection) |
| `CONDITIONAL` | Proven under stated assumptions |
| `EMPIRICAL` | Measured with confidence intervals on test suites |
| `HEURISTIC` | Best-effort; no formal guarantee |

This taxonomy is consistent across code comments, docs, and test output.

---

## Known limitations

| Issue | Status |
|-------|--------|
| Spec-reality gap: proofs hold on the model, not the world | Partially mitigated by `EnvironmentReconciler` |
| Multi-agent coordination across processes | Not implemented |
| Subjective goals ("write good code") | No formal predicate possible; use `MultiDimensionalAttestor` |
| LLM adversarial prompting | Kernel limits damage to budget waste; no safety violations possible |
| Deep Python memory manipulation (`ctypes`, `gc`) | Partially mitigated; not memory-safe |

See [docs/VULNERABILITIES.md](docs/VULNERABILITIES.md) for the full breakdown.

---

## Running tests

```bash
pip install constrai[dev]
pytest tests/ -v
```

Key test categories:

```bash
pytest tests/test_constrai.py         # T1–T8 unit tests
pytest tests/test_monte_carlo.py      # 1,000 random tasks, all theorems verified
pytest tests/chaos_fuzzer.py          # 45 adversarial attack scenarios
pytest tests/test_composition.py      # Operadic task composition
```

---

## Project structure

```
constrai/
├── formal.py               # Layer 0: proven safety kernel (T1–T8)
├── reasoning.py            # Layer 1: Bayesian beliefs, action valuation, LLM interface
├── orchestrator.py         # Layer 2: main execution loop
├── hardening.py            # Layer 3: environment reconciliation, sandboxing
├── reference_monitor.py    # Enforcement: IFC, CBF, QP action repair
├── inverse_algebra.py      # T7: algebraic inverse effects for exact rollback
├── active_hjb_barrier.py   # Heuristic: k-step lookahead basin avoidance
├── gradient_tracker.py     # Heuristic: finite-difference boundary proximity
├── jacobian_fusion.py      # Heuristic: boundary sensitivity scoring for prompts
├── safe_hover.py           # Hard enforcement gate / emergency stop
├── operadic_composition.py # Compositional verification (OC-1, OC-2)
├── saliency.py             # Prompt saliency engine
├── verification_log.py     # Proof record writer
└── __init__.py             # Public API
docs/
├── ARCHITECTURE.md         # Design rationale and layer descriptions
├── THEOREMS.md             # All 13 formal claims with proofs
├── VULNERABILITIES.md      # Known issues, fixes, residual risks
└── API.md                  # Complete API reference
tests/
├── test_constrai.py        # Theorem unit tests
├── test_monte_carlo.py     # 1,000-run probabilistic validation
├── chaos_fuzzer.py         # 45 adversarial attack scenarios
├── test_soft_gaps_fixed.py # Inverse algebra, QP repair, monitor integration
├── test_boundary_enforcement.py
├── test_composition.py
└── test_integration.py
safety_evaluation/          # Adversarial evaluation against 39 attack vectors
```

---

## Citation

To cite ConstrAI in your work:

**BibTeX:**

```bibtex
@misc{ambar2026constrai,
    title = {ConstrAI: Formal safety framework for AI agents},
    author = {Ambar},
    year = {2026},
    howpublished = {\url{[https://github.com/Ambar-13/ConstrAI](https://github.com/Ambar-13/ConstrAI)}},
    note = {Version 0.3.0}
}
```

**Contact:** Ambar (ambar13@u.nus.edu)
Affiliation: National University of Singapore (NUS)
---

## License

MIT. See `LICENSE`.
