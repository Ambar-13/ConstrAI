# ConstrAI

Formal safety framework for AI agents. Math constrains what the agent can do, the LLM decides what it should do, neither can bypass the other.

## Why this exists

Current AI agent frameworks enforce safety through prompts ("please don't do anything dangerous") or post-hoc filtering. Both fail under adversarial conditions. ConstrAI enforces safety through state-transition math that runs *before* any action executes. The LLM cannot talk its way past the kernel.

**The key difference**: safety overhead is zero tokens. No system prompt bloat, no guardrail prompts. The constraints are in the execution loop, not in the context window.

## How It Works

ConstrAI uses three layers to keep agents safe:

1. **State & Actions** — Everything is immutable data, not code
   - States are snapshots you can't modify directly
   - Actions declare what they'll change, not how they'll do it
   - Effects are composable and reversible

2. **Safety Checks** — Multiple verification layers
   - Budget tracking: can't exceed spending limit
   - Invariant checking: safety properties always held
   - Rollback capability: undo any action that went wrong
   - Reference monitor: enforces information flow and constraints

3. **Execution** — Check before committing
   - Simulate the action first (on a copy)
   - Verify it's safe
   - Only then apply to real state
   - Record what was done so it can be undone

## Threat Model & Assumptions

These guarantees hold under the following assumptions:

- **The LLM is untrusted.** It can hallucinate, lie, or return garbage. ConstrAI treats it as potentially unreliable.
- **The ConstrAI kernel is trusted.** If an attacker has write access to the kernel code itself, all bets are off. This is the same assumption operating systems make.
- **ActionSpecs are human-authored and correct.** If a spec says "create file" but actually deletes things, the kernel protects the model, not the system.
- **Single-agent, single-process.** Concurrent access is protected by locks, but multi-agent coordination across processes is not implemented.
- **Budget checks work.** All budget checks happen before execution (with invariant checks on a simulated state).

If any of these assumptions are violated, specific guarantees may not hold. See [VULNERABILITIES.md](docs/VULNERABILITIES.md) for the full breakdown.

## Core Guarantees

ConstrAI provides 7 formal theorems. Detailed proofs in [THEOREMS.md](docs/THEOREMS.md).

| # | Theorem | What It Means |
|---|---------|--------------|
| T1 | Budget Safety | You cannot spend more than your budget, ever |
| T2 | Termination | Execution will eventually stop (won't loop forever) |
| T3 | Invariant Preservation | Safety properties are maintained at every step |
| T4 | Monotone Resources | Spending only increases, never decreases |
| T5 | Atomicity | Rejected actions don't change anything |
| T6 | Trace Integrity | Execution log cannot be tampered with |
| T7 | Rollback Exactness | Undo always restores the exact previous state |

**Conditional guarantees** (hold if checks are correct): temporal dependencies, environment reconciliation, goal verification.

**Measured properties** (tested but not formally proven): attestation is harder to game with multiple checks; dependency discovery reduces failures.

## What's New (v0.3.0)

Three new safety systems:

1. **Boundary Detection** — Knows when variables are getting close to constraint violations
2. **Enforcement Barriers** — Actively prevents dangerous actions before they execute
3. **Task Composition** — Combines verified subtasks safely without re-checking everything

See [SOFT_GAPS_FIXED.md](SOFT_GAPS_FIXED.md) for technical details.

## Architecture

```
┌─────────────────────────────────────────────┐
│                Orchestrator                 │
│  ┌───────────────────────────────────────┐  │
│  │ LLM (pluggable: Claude, GPT, local)   │  │
│  │  Receives: decision prompt             │  │
│  │  Returns: JSON action                  │  │
│  ├───────────────────────────────────────┤  │
│  │ Reasoning Engine                      │  │
│  │  • Track beliefs about state           │  │
│  │  • Compute action value               │  │
│  │  • Handle dependencies                │  │
│  ├───────────────────────────────────────┤  │
│  │ Safety Kernel (formal, non-bypassable)│  │
│  │  Budget, Invariants, Rollback, etc    │  │
│  ├───────────────────────────────────────┤  │
│  │ Reference Monitor                     │  │
│  │  • Information flow control           │  │
│  │  • Resource barrier functions         │  │
│  │  • Action repair via QP               │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Key Design**: Every action is checked before execution. The LLM proposes, the kernel verifies, only then we execute.

## API Overview

### Create a State

```python
from constrai import State

# Create initial state
state = State({"balance": 100, "deployed": False})

# Access values
balance = state.get("balance")  # Returns 100

# Create modified state (original unchanged)
new_state = state.with_updates({"balance": 95})
```

### Define Actions

```python
from constrai import ActionSpec, Effect

action = ActionSpec(
    id="spend",
    name="Spend money",
    description="Reduce balance by amount",
    effects=(
        Effect("balance", "decrement", 50),  # balance -= 50
    ),
    cost=1.0,
    risk_level="medium"
)

# Simulate what would happen
next_state = action.simulate(state)
print(next_state.get("balance"))  # 50
```

### Define Safety Rules

```python
from constrai import Invariant, CaptureBasin

# Invariant: safety property that must hold
invariant = Invariant(
    name="budget_safe",
    predicate=lambda s: s.get("balance", 0) >= 0,
    description="Balance must never be negative"
)

# Capture basin: region to avoid
danger_zone = CaptureBasin(
    name="bankruptcy",
    is_bad=lambda s: s.get("balance", 0) < 0,
    max_steps=5
)
```

### Create a Task

```python
from constrai import TaskDefinition

task = TaskDefinition(
    goal="Spend exactly 75 units",
    initial_state=State({"balance": 100}),
    available_actions=[action1, action2, action3],
    invariants=[invariant],
    budget=20.0,  # Max cost allowed
    goal_predicate=lambda s: s.get("balance") == 25,
    capture_basins=[danger_zone]
)
```

### Execute with an LLM

```python
from constrai import Orchestrator

# Create orchestrator with your LLM client
orch = Orchestrator(task, llm_client=your_llm)

# Run it
result = orch.execute_task(max_steps=30)

print(f"Success: {result.goal_achieved}")
print(f"Steps taken: {result.step_count}")
print(f"Budget used: {result.spent}")
```

## How Safety Checks Work

When you execute an action, ConstrAI does this:

```
1. Check Boundaries
   └─ Which variables are close to constraint violations?

2. Enforce Barriers
   └─ Would this action enter a forbidden region?

3. Reference Monitor
   └─ Information flow: can data flow to this variable?
   └─ Resource limits: would this exceed budget?
   └─ Repair if needed: nudge parameters to be safe

4. Simulate Action
   └─ Apply effects to a copy of state
   └─ Check invariants on the result
   └─ Is the next state valid?

5. Execute (if all checks pass)
   └─ Apply effects to real state
   └─ Record what was done for potential undo

6. Observe
   └─ Update beliefs about what happened
   └─ Log for audit trail
```

If any check fails, the action is rejected and state is unchanged.

## Testing & Guarantees

All 7 theorems are tested:

```bash
# Run all tests
python -m pytest tests/ -v

# Check specific system
python -m pytest tests/test_soft_gaps_fixed.py -v  # Latest features
python -m pytest tests/test_constrai.py -v          # Core system
```

**Current status**: 44 tests passing, 0 failures

**No breaking changes**: All existing code continues to work

## Documentation

- **[THEOREMS.md](docs/THEOREMS.md)** — Detailed proofs of all 7 theorems
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — System design and components
- **[VULNERABILITIES.md](docs/VULNERABILITIES.md)** — Security assumptions and limitations
- **[API.md](docs/API.md)** — Complete API reference
- **[REFERENCE_MONITOR.md](docs/REFERENCE_MONITOR.md)** — Safety enforcement details
- **[SOFT_GAPS_FIXED.md](SOFT_GAPS_FIXED.md)** — New boundary detection and enforcement features

## Example: Deploying a Service Safely

```python
from constrai import (
    State, Effect, ActionSpec, Invariant,
    TaskDefinition, Orchestrator, CaptureBasin
)

# Define what the system can do
actions = [
    ActionSpec(
        id="build",
        name="Build service",
        description="Compile source code",
        effects=(Effect("built", "set", True),),
        cost=10.0
    ),
    ActionSpec(
        id="test",
        name="Run tests",
        description="Execute test suite",
        effects=(Effect("tested", "set", True),),
        cost=5.0
    ),
    ActionSpec(
        id="deploy",
        name="Deploy",
        description="Ship to production",
        effects=(Effect("deployed", "set", True),),
        cost=15.0,
        risk_level="high"
    ),
]

# Define safety rules
invariants = [
    Invariant(
        "test_before_deploy",
        lambda s: not s.get("deployed") or s.get("tested"),
        "Cannot deploy without testing"
    ),
]

# Define forbidden states
danger_zones = [
    CaptureBasin(
        "untested_deploy",
        is_bad=lambda s: s.get("deployed") and not s.get("tested"),
        max_steps=1
    ),
]

# Create task
task = TaskDefinition(
    goal="Successfully deploy service",
    initial_state=State({
        "built": False,
        "tested": False,
        "deployed": False,
        "cost": 0.0
    }),
    available_actions=actions,
    invariants=invariants,
    capture_basins=danger_zones,
    budget=50.0,
    goal_predicate=lambda s: s.get("deployed", False),
    dependencies={
        "test": [("build", "Need to build first")],
        "deploy": [("test", "Need to test first")],
    }
)

# Run it
orch = Orchestrator(task, llm_client)
result = orch.execute_task(max_steps=50)

if result.goal_achieved:
    print(f"✓ Service deployed in {result.step_count} steps")
else:
    print(f"✗ Deployment failed: {result.failure_reason}")
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

## License

ConstrAI is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
        "deploy": [("test", "Must test before deploying")],
    },
)

engine = Orchestrator(task)  # Uses MockLLM by default
result = engine.run()
print(result.summary())
```

To use a real LLM, implement one method:

```python
class MyLLM:
    def complete(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
        # Call your LLM, return the text response
        ...

engine = Orchestrator(task, llm=MyLLM())
```

See `examples/claude_integration.py` for a working Anthropic Claude adapter.

## How the kernel works (30 seconds)

1. LLM selects an action (from structured prompt with pre-computed values)
2. Kernel **simulates** the action on an immutable copy of state
3. Checks: budget sufficient? All invariants hold on simulated state? Dependencies met?
4. If **any** check fails → reject. State and budget unchanged. (T5)
5. If **all** pass → commit. Charge budget, apply state, append to hash-chained trace.

The LLM never touches the state directly. It proposes; the kernel disposes.

## Test results

```
Core theorems (T1–T7) + integration + adversarial:   69/69
Chaos fuzzer (45 attack vectors):                     45/45
v2 hardening (5 vulnerability fixes):                 41/41
Monte Carlo (1000 random tasks):                      1000/1000 budget safe
                                                      1000/1000 invariant safe
                                                      1000/1000 terminated
                                                      1000/1000 trace integrity
```

The chaos fuzzer tests: hallucinated actions, budget overflow, invariant evasion, state poisoning, trace tampering, success cheating, resource exhaustion, 6 malicious LLM personalities, dependency bypass, type confusion, resource lifecycle violations, discovery poisoning, and 1000 randomized attack scenarios.

##NOTE
Current implementation works as good as the invariant. The well define dthe invariants are, the better the safety. Next version will focus on defining common invariants, helping users make well-defined invariants, and testing the invariants.

## Safety Evaluation

ConstrAI was evaluated against an internal adversarial evaluation suite covering 63 actions across 6 agent types and 39 attack vectors:

| Metric | Result |
|--------|:---:|
| **Safety Rate** | **89.7%** (35/39 attacks blocked) |
| **False Positives** | **0** (no legitimate actions blocked) |
| **Task Completion** | **100%** |
| **Precision** | **100%** |
| **Recall** | **89.7%** |
| **F1 Score** | **94.6%** |
| **Avg Latency** | **0.061ms** per check |
| **Throughput** | **45,613 checks/sec** |

**Agents tested:**
- FileSystemAgent (file ops, path traversal, credential theft)
- WebRequestAgent (exfiltration, SSRF, DNS covert channels)
- CodeExecutionAgent (privilege escalation, reverse shells)
- DataAccessAgent (SQL injection, PII extraction)
- MultiStepAgent (chained multi-step attacks)
- EvasionAgent (obfuscation, base64 encoding, semantic evasion)

**Threat coverage:** Data exfiltration, sandbox escape, privilege escalation, command injection, PII leaks, denial of service, path traversal, resource abuse, prompt injection.

**Key finding:** The 4 missed attacks (50% safety on EvasionAgent) are adversarial evasion techniques (string concatenation, base64 payloads, `getattr` dispatch) that bypass static pattern matching. The kernel's invariant checking has **zero known bypasses** — all failures are in the classification layer, which is modular and upgradeable.

See `safety_evaluation/constrai_safety_evaluation.py`; raw outputs are included in `safety_evaluation/evaluation_results_raw.json`.
See `safety_evaluation/Evaluation_report.md` for detailed breakdown, per-agent metrics, and architectural analysis.

## Known limitations

- **Spec-reality gap**: If your ActionSpec effects are wrong, the kernel protects the model, not the system. Environment reconciliation helps for probed variables. Unprobed variables can drift.
- **LLM quality**: ConstrAI limits damage from bad decisions, it doesn't make the LLM smarter. Wrong action sequences waste budget without safety violations.
- **Single-agent**: No multi-agent coordination. Concurrent access within one process is lock-protected.
- **State scale**: Dictionary-based. Works for hundreds of variables. Millions would need sharding.
- **Post-facto reconciliation**: Environment drift is detected *after* one bad action executes. Inherent to any system that interacts with the real world.

## Project structure

```
ConstrAI/
├── constrai/
│   ├── __init__.py          Public API
│   ├── formal.py            Safety kernel, state, effects, theorems T1–T7
│   ├── reasoning.py         Bayesian beliefs, causal graph, LLM interface
│   ├── orchestrator.py      Main execution loop
│   └── hardening.py         Attestors, temporal deps, reconciliation, priors
├── tests/
│   ├── test_constrai.py      Core theorem + integration tests
│   ├── test_v2_hardening.py Vulnerability fix tests
│   ├── test_monte_carlo.py  Statistical validation
│   └── chaos_fuzzer.py      Adversarial attack suite
├── examples/
│   └── claude_integration.py  Anthropic Claude adapter
├── docs/
│   ├── ARCHITECTURE.md      How the system works
│   ├── THEOREMS.md          Formal proofs
│   ├── VULNERABILITIES.md   Known flaws and mitigations
│   └── API.md               Class and method reference
├── CONTRIBUTING.md
├── setup.py
└── README.md
```

## Install

```bash
pip install -e .                     # Core (zero dependencies)
pip install -e ".[anthropic]"        # With Claude support
```

## Docs

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — How the layers work together
- [THEOREMS.md](docs/THEOREMS.md) — Formal proofs for T1–T7
- [VULNERABILITIES.md](docs/VULNERABILITIES.md) — Known flaws, mitigations, honest limitations
- [API.md](docs/API.md) — Class reference
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute

## License

MIT

## Citation

To cite ConstrAI in your work:

**BibTeX:**

```bibtex
@misc{ambar2026constrai,
    title = {ConstrAI: Formal safety framework for AI agents},
    author = {Ambar},
    year = {2026},
    howpublished = {\url{[https://github.com/Ambar-13/ConstrAI](https://github.com/Ambar-13/ConstrAI)}},
    note = {Version 0.2.0}
}
```

**Contact:** Ambar (ambar13@u.nus.edu)
Affiliation: National University of Singapore (NUS)
