# Multi-Agent Architecture in ClampAI

**Status:** ClampAI v0.x — single-process multi-agent is supported; multi-process is not.  
**Date:** 2026-02-27

---

## Summary

| Deployment Pattern | Supported | Notes |
|---|---|---|
| Multiple Orchestrators, single process, shared `SafetyKernel` | **YES** | Correct by construction under `threading.Lock` |
| Multiple processes, each with its own `SafetyKernel` | **PARTIAL** | Each kernel enforces its own budget/invariants; no cross-process coordination |
| Distributed agents with a shared global budget/state | **NO** | Open research problem; see RFC below |

---

## Supported: Single-Process Shared SafetyKernel

### Architecture

```
  Process Boundary
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │   Orchestrator A        Orchestrator B        Orchestrator C │
  │   (Thread 1)            (Thread 2)            (Thread 3)     │
  │       │                     │                     │          │
  │       └─────────────────────┴─────────────────────┘          │
  │                             │                                │
  │                    ┌────────▼────────┐                       │
  │                    │  SafetyKernel   │  <── threading.Lock   │
  │                    │                 │                       │
  │                    │  BudgetCtrl     │  shared budget B0     │
  │                    │  Invariants     │  shared predicates    │
  │                    │  ExecutionTrace │  shared audit log     │
  │                    └─────────────────┘                       │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

### Why This Is Correct

`SafetyKernel.evaluate_and_execute_atomic()` acquires `self._lock` (a `threading.Lock`) before performing the check-charge-commit sequence and releases it only after all three operations complete. The lock is defined at `clampai/formal.py:746`.

This means:

1. **No TOCTOU within a process.** Thread A's "can_afford?" check and budget charge are one atomic unit. Thread B cannot interleave between the check and the charge because `_lock` is not released between them.
2. **Shared budget is globally enforced.** If the shared budget is 100 and Thread A charges 60 while Thread B charges 50, exactly one of them will be rejected - the total will never exceed 100.
3. **Shared invariants are globally enforced.** All three orchestrators check the same invariant predicates against the same simulated state.
4. **The trace is globally consistent.** All approvals, rejections, and rollbacks appear in a single hash-chained `ExecutionTrace`.

### Usage

```python
from clampai import SafetyKernel, Invariant, State, ActionSpec, Effect
import threading

# Create ONE kernel shared across all orchestrators
shared_kernel = SafetyKernel(
    budget=100.0,
    invariants=[
        Invariant(
            "global_resource_limit",
            lambda s: s.get("active_tasks", 0) <= 10,
            enforcement="blocking",
        )
    ],
    min_action_cost=1.0,
)

# Multiple orchestrators share it
def agent_worker(agent_id: str, shared_kernel: SafetyKernel, initial_state: State):
    action = ActionSpec(
        id=f"task_{agent_id}",
        name=f"Agent {agent_id} task",
        effects=(Effect("active_tasks", "increment", 1),),
        cost=10.0,
    )
    # evaluate_and_execute_atomic is thread-safe
    new_state, entry = shared_kernel.evaluate_and_execute_atomic(
        initial_state, action, reasoning_summary=f"Agent {agent_id} acting"
    )
    return new_state

state = State({"active_tasks": 0})
threads = [
    threading.Thread(target=agent_worker, args=(str(i), shared_kernel, state))
    for i in range(5)
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

See `examples/multi_agent_shared_kernel.py` for a complete working example.

### Correctness Scope

The shared-kernel pattern preserves all eight theorems (T1-T8) across threads **within the same process**. The proof is the same as the single-agent case: `evaluate_and_execute_atomic` is the single serialization point, and the lock ensures that no two threads can be in the critical section simultaneously.

The only cost is contention: if many threads call `evaluate_and_execute_atomic` at high frequency, they will serialize at the lock. For ClampAI's typical use case (agent decision loops at human-readable timescales), this is not a bottleneck - safety checks complete in ~0.061 ms per the benchmark results.

---

## Not Supported: Multi-Process Coordination

When multiple OS processes each hold their own `SafetyKernel` instance and need to coordinate over a shared logical budget or shared invariants, ClampAI provides no solution. This is not an implementation gap that can be closed by adding a flag - it is a class of distributed systems problems that each require deliberate architectural choices with significant trade-offs.

### Problem 1: Distributed TOCTOU

In a single process, `threading.Lock` prevents time-of-check-to-time-of-use races. Across processes, there is no shared lock primitive. Consider two processes, each with `budget=100` and a shared logical budget of 100:

```
Process A: can_afford(60)? -> YES (local remaining = 100)
Process B: can_afford(60)? -> YES (local remaining = 100)
Process A: charge(60)      -> local spent = 60
Process B: charge(60)      -> local spent = 60
Result:    total spent = 120 > 100  [T1 VIOLATED globally]
```

A distributed lock (e.g., Redis SETNX, ZooKeeper, etcd lease) can prevent this, but introduces latency, lock-holder failure modes, and a new consistency boundary.

### Problem 2: Budget Aggregation Under Partial Failure

If Process A charges its local budget, then crashes before reporting to a coordinator, the global budget accounting is corrupted. This is the classic problem that two-phase commit (2PC) addresses - but 2PC introduces blocking behavior: if the coordinator fails after Phase 1, all participants block waiting for a decision that may never come.

### Problem 3: Global State Consistency

T3 (Invariant Safety) requires that the kernel checks the invariant against the current state before committing. In a distributed system, "current state" is ambiguous: each process may have a different view of shared state variables. An invariant like `total_emails_sent <= 1000` cannot be safely enforced by any single process checking only its local count without reading from all other processes - which is an expensive distributed read.

CRDTs (Conflict-free Replicated Data Types) can provide eventual consistency for counters, but "eventual" is incompatible with blocking invariant enforcement: by the time a CRDT converges, the limit may already be exceeded.

### Known Solutions and Their Trade-offs

| Technique | What It Solves | Trade-offs |
|-----------|---------------|------------|
| Distributed lock (Redis SETNX, etcd) | TOCTOU across processes | Lock-holder failure → blocking; adds network round-trip to every safety check |
| Two-Phase Commit (2PC) | Atomic budget charge across coordinators | Blocking on coordinator failure; O(n) messages per commit |
| Optimistic Concurrency Control (OCC) | High-throughput check-then-commit with retry | Retry storms under contention; not suitable for hard budget limits |
| CRDTs (G-Counter for spend) | Eventual consistency of spend tracking | Eventual is not sufficient for hard limits; overshooting is possible |
| Token bucket / pre-allocated sub-budgets | Avoids coordination for most actions | Requires upfront budget partitioning; unused tokens are wasted |
| Saga pattern | Long-running distributed transactions with compensation | Compensating actions must be defined for every action; eventual consistency |

Each of these is a genuine engineering option for a future version, not a simple addition.

---

## Open Research Problem Statement

> **ClampAI v0.x does not solve distributed multi-agent safety. The RFC tracks this as an open research problem.**

The core difficulty is that ClampAI's safety guarantees (T1-T8) are proven over a single, coherent state machine with a single budget controller and a single execution trace. Distributing these across process or machine boundaries requires choosing a consistency model (strong, causal, eventual), a coordination protocol (locking, consensus, CRDT), and a failure model (fail-stop, Byzantine, partition-tolerant) - and each choice invalidates or weakens some subset of the current proofs.

Specific open questions:

1. **What is the right consistency model for a distributed invariant?** Is "no process ever observes a violation" (linearizability) required, or is "violations are detected within k steps" (bounded staleness) sufficient?
2. **How should budget be partitioned?** Static partitioning (each agent gets a fixed sub-budget) is simple but wasteful. Dynamic reallocation requires coordination.
3. **What does T7 (rollback) mean across processes?** If Agent A and Agent B each execute actions that together violate a global invariant, rolling back one agent's action does not undo the other's. Saga-style compensation is the standard approach, but requires pre-specifying compensating actions for every action in the system.
4. **What does T6 (trace integrity) mean across processes?** Each process has its own trace. A global, consistent, tamper-evident trace across processes requires a distributed ledger or a trusted log aggregator.

Contributions addressing these questions are welcome. See `CONTRIBUTING.md`.

---

## References

- `clampai/formal.py:703-970` - `SafetyKernel` and `evaluate_and_execute_atomic`
- `clampai/formal.py:746` - `threading.Lock` definition
- `examples/multi_agent_shared_kernel.py` - Working single-process shared kernel example
- `MATHEMATICAL_COMPLIANCE.md` - Full proof details for T1-T8
- `docs/VULNERABILITIES.md` - Known limitations including multi-agent gaps
