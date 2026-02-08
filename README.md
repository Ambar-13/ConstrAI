# ConstrAI

Formal safety framework for AI agents. Math constrains what the agent can do, the LLM decides what it should do, neither can bypass the other.

## Why this exists

Current AI agent frameworks enforce safety through prompts ("please don't do anything dangerous") or post-hoc filtering. Both fail under adversarial conditions. ConstrAI enforces safety through state-transition math that runs *before* any action executes. The LLM cannot talk its way past the kernel.

**The key difference**: safety overhead is zero tokens. No system prompt bloat, no guardrail prompts. The constraints are in the execution loop, not in the context window.

## Threat Model & Assumptions

These guarantees hold under the following assumptions:

- **The LLM is untrusted.** It can hallucinate, lie, try to game metrics, or return garbage. ConstrAI treats it as a stochastic adversary.
- **The ConstrAI kernel is trusted.** If an attacker has write access to the kernel code itself, all bets are off. This is the same assumption operating systems make about the kernel.
- **ActionSpecs are human-authored and correct.** If the spec says "create file" but the real command deletes things, the kernel protects the model, not the system. Environment reconciliation catches this for probed variables only.
- **Single-agent, single-process.** Concurrent access is protected by locks, but multi-agent coordination across processes is not implemented.
- **Budget check.** All budget checks are done before execution (with invariant checks happening on a simulated state).

If any of these assumptions are violated, specific guarantees degrade. See [VULNERABILITIES.md](docs/VULNERABILITIES.md) for the full breakdown.

## Core Guarantees

7 theorems, enforced by construction. Each has a proof in [THEOREMS.md](docs/THEOREMS.md).

| # | Theorem | Statement | Proof technique |
|---|---------|-----------|-----------------|
| T1 | Budget Safety | `spent(t) ≤ B₀` for all t | Induction on check-then-charge |
| T2 | Termination | Halts in ≤ `⌊B₀/ε⌋` steps | Well-founded ordering on budget; ε > 0 enforced |
| T3 | Invariant Preservation | `I(s₀) ⟹ I(sₜ)` for all reachable sₜ | Simulate-then-commit; reject on violation |
| T4 | Monotone Resources | `spent(t) ≤ spent(t+1)` | Non-negative cost assertion |
| T5 | Atomicity | Rejected actions change nothing | Immutable state + simulation on copy |
| T6 | Trace Integrity | Append-only SHA-256 hash chain | Hash of entry includes previous hash |
| T7 | Rollback Exactness | `undo(execute(s, a)) == s` | Immutable state + stored inverse effects |

**Conditional guarantees** (hold if probes/attestors are correct): temporal dependencies, environment reconciliation, goal attestation.

**Empirical claims** (measured, not proven): multi-dimensional attestation is harder to game; dynamic dependency discovery reduces failures.

## Architecture

```
┌─────────────────────────────────────────────┐
│                Orchestrator                  │
│  ┌───────────────────────────────────────┐  │
│  │ LLM (pluggable: Claude, GPT, local)  │  │
│  │  Receives: structured decision prompt │  │
│  │  Returns: JSON action selection       │  │
│  ├───────────────────────────────────────┤  │
│  │ Reasoning Engine                      │  │
│  │  • Bayesian beliefs (Beta dist)       │  │
│  │  • Causal dependency graph            │  │
│  │  • Information-theoretic action vals  │  │
│  │  • Cost-aware pessimistic priors      │  │
│  ├───────────────────────────────────────┤  │
│  │ Safety Kernel (formal, non-bypassable)│  │
│  │  Theorems T1–T7 enforced here         │  │
│  ├───────────────────────────────────────┤  │
│  │ Hardening Layer                       │  │
│  │  • Sandboxed attestors (shell=False)  │  │
│  │  • Temporal deps + readiness probes   │  │
│  │  • Environment reconciliation         │  │
│  │  • Multi-dim quality attestation      │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

The LLM reasons and picks actions. Every action passes through the safety kernel before execution. The kernel can't be talked out of its constraints.

## Quick start

```python
from constrai import (
    State, Effect, ActionSpec, Invariant,
    TaskDefinition, Orchestrator
)

task = TaskDefinition(
    goal="Deploy a service",
    initial_state=State({"built": False, "tested": False, "deployed": False}),
    available_actions=[
        ActionSpec(id="build", name="Build", description="Compile",
                   effects=(Effect("built", "set", True),), cost=3.0),
        ActionSpec(id="test", name="Test", description="Run tests",
                   effects=(Effect("tested", "set", True),), cost=2.0),
        ActionSpec(id="deploy", name="Deploy", description="Ship it",
                   effects=(Effect("deployed", "set", True),), cost=5.0,
                   risk_level="high"),
    ],
    invariants=[
        Invariant("no_untested_deploy",
            lambda s: not s.get("deployed") or s.get("tested"),
            "Cannot deploy without passing tests"),
    ],
    budget=50.0,
    goal_predicate=lambda s: s.get("deployed", False),
    dependencies={
        "build": [],
        "test": [("build", "Need build first")],
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

## How the kernel works (30-second version)

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
