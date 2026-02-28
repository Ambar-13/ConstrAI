# RFC: Distributed Multi-Agent Safety for ClampAI

**Status:** Open Problem / Pre-RFC
**Author:** Ambar (ambar13@u.nus.edu)
**Date:** 2026-02-27
**Tracking:** https://github.com/Ambar-13/ClampAI/issues (label: `multi-agent`)

---

## Summary

ClampAI's safety kernel (T1–T8) is proven correct for a **single `SafetyKernel` instance in a single OS process**. This RFC documents the open problem of extending those guarantees to multiple agents running in separate processes, and surveys the distributed systems techniques that could be used to close that gap. No solution is proposed for v0.x. A concrete design is proposed for v1.0 consideration.

For the current single-process multi-agent pattern (multiple `Orchestrator` instances sharing one `SafetyKernel` via `threading.Lock`), see `docs/MULTI_AGENT_ARCHITECTURE.md`.

---

## Motivation

As ClampAI is used in production, deployments are increasingly multi-process:

- **Horizontal scaling:** Multiple worker processes handle independent tasks, each with a local `SafetyKernel`, but needing to respect a shared global budget.
- **Specialized agents:** A planner process and executor processes operate concurrently. The planner's budget estimates must be reflected in the executors' available spend.
- **Fault tolerance:** A crashed agent must not leave the global budget in an inconsistent state.

None of these are currently handled. This RFC characterises the problem and surveys options.

---

## Problem Statement

The three core problems are documented in detail in `docs/MULTI_AGENT_ARCHITECTURE.md §Not Supported`. A brief restatement:

### P1: Distributed TOCTOU

`SafetyKernel._lock` (a `threading.Lock`) is a **local primitive**. It cannot span process or machine boundaries.

```
Agent A: can_afford(60)?  → YES  (local remaining = 100)
Agent B: can_afford(60)?  → YES  (local remaining = 100)
Agent A: charge(60)       → local spent = 60
Agent B: charge(60)       → local spent = 60
T1 VIOLATED globally: total = 120 > 100
```

### P2: Budget Aggregation Under Partial Failure

If Agent A charges its local budget and then crashes, a coordinator that was tracking global spend now has stale data. 2PC addresses this but introduces coordinator-failure blocking.

### P3: Global State Consistency

T3 checks `inv.predicate(simulated_state)` before committing. In a multi-process system, `simulated_state` would need to include state changes from all other agents — requiring a global read that is either stale (eventual) or expensive (linearizable).

---

## Survey of Applicable Techniques

| Technique | Problem Addressed | ClampAI Fit | Trade-offs |
|-----------|------------------|--------------|------------|
| **Distributed lock** (Redis SETNX, etcd lease) | P1: TOCTOU | High — direct replacement for `threading.Lock` | ~1–5 ms per lock; lock-holder crash → blocked; must tune lease TTL |
| **Two-Phase Commit (2PC)** | P1, P2 | Medium — heavyweight for per-action coordination | Blocking on coordinator failure; O(n) messages per commit; latency scales with agent count |
| **Three-Phase Commit (3PC)** | P2 (non-blocking) | Low — impractical overhead for agent loops | Much more complex; still not partition-tolerant |
| **Optimistic Concurrency Control** | P1 (retry-based) | Low — unsuitable for hard limits | Retry storms under contention; T1 becomes probabilistic, not proven |
| **G-Counter CRDT** | P3 (eventual spend tracking) | Low | Eventual consistency ≠ hard limit; overshoot is possible before convergence |
| **Token bucket / sub-budget allocation** | P1 (avoids coordination) | **High** — simplest path to correctness | Wasted budget if partitions are uneven; requires upfront allocation decisions |
| **Saga pattern** | P2 (compensating transactions) | Medium | Compensating actions must be defined for every action; eventual consistency only |
| **Vector clocks / Lamport clocks** | Causal ordering of events | Medium — for trace reconstruction | Does not prevent budget violations; only enables post-hoc ordering |

---

## Proposed Design for v1.0 Consideration

This is a **sketch**, not a specification. It is intended to focus community discussion.

### Core Idea: Static Sub-Budget Allocation with Reconciliation

Instead of distributing the locking mechanism, partition the global budget statically at startup and let each agent enforce its own sub-budget locally (using the current proven kernel). A lightweight reconciliation step periodically reports actual spend to a coordinator that can reallocate unused budget.

```
┌─────────────────────────────────────────────────────────┐
│  BudgetCoordinator  (single process or external service) │
│                                                          │
│  global_budget = 1000                                    │
│  allocated = {A: 300, B: 300, C: 400}                    │
│  reported_spent = {A: 0, B: 0, C: 0}   (updated async)  │
└─────────────────────────────────────────────────────────┘
          ↑ heartbeat / reconcile (e.g., every 10s)
   ┌──────┼──────────┬──────────┐
   │      │          │          │
  Agent A  Agent B   Agent C
  kernel   kernel    kernel
  B=300    B=300     B=400
  (local,  (local,   (local,
   proven)  proven)   proven)
```

**Properties:**
- **T1 locally proven** for each agent's sub-budget. If Agent A has sub-budget 300 and spends 300, its kernel rejects any further actions.
- **T1 globally holds** if `sum(allocated) == global_budget` and the coordinator never over-allocates. This is a much simpler invariant than distributed locking.
- **No per-action coordination.** Each agent's safety check takes the same ~0.06 ms as the single-agent case.
- **Budget waste.** If Agent A finishes with 100 remaining, that 100 is unavailable to Agent B until the next reconciliation cycle. Tuning the reconciliation period trades waste against coordination overhead.

**What this does NOT solve:**
- T3 (Invariant Safety) for **shared** state variables. If all agents share a counter `total_emails_sent`, each agent only knows its own contributions. The global invariant `total_emails_sent <= 1000` cannot be checked locally.
- T7 (Rollback) across agents. Rolling back Agent A's action does not undo Agent B's actions that depended on A's state.

**Scope for shared invariants:** For v1.0, shared invariants over global state are out of scope for the sub-budget approach. Users who need global invariants must use the single-process shared-kernel pattern.

---

## Open Questions

These must be resolved before any distributed multi-agent implementation can claim the same epistemic status as the current single-process guarantees:

1. **Consistency model for T3.** Is "no agent ever observes a violation at the time of commit" (linearizability) required, or is "violations are detected and compensated within k steps" (bounded staleness) acceptable for some invariant categories?

2. **Saga vs. 2PC for T7.** If Agent A and Agent B each commit actions that together violate a global invariant, which agent rolls back? Who decides? How is the compensating action defined?

3. **Trace integrity across agents.** Each agent has its own `ExecutionTrace` (SHA-256 hash chain). A global tamper-evident trace requires either a shared chain (serialised writes) or a distributed ledger (complex, high latency). What is the minimum acceptable T6 guarantee in a multi-process setting?

4. **Budget coordinator failure.** If the coordinator crashes during a reconciliation cycle, what is the safe default? Freeze all agents (conservative, may stall tasks) or allow agents to continue on their last-known sub-budget (optimistic, may over-spend if reallocation was in flight)?

5. **API surface.** What is the minimal API change needed? A `DistributedSafetyKernel` that wraps `SafetyKernel` and adds a coordinator address is one option. A separate `BudgetCoordinator` service is another.

---

## Path Forward

| Phase | Goal | Status |
|-------|------|--------|
| v0.x | Single-process multi-agent via shared kernel | **Done** (`threading.Lock`) |
| v0.x | Document limitations honestly | **Done** (`docs/MULTI_AGENT_ARCHITECTURE.md`) |
| v0.x | RFC + community input | **This document** |
| v1.0 | Sub-budget allocation with coordinator (T1 only) | Proposal stage |
| v1.0+ | Shared invariant coordination (T3 extension) | Open research |
| v1.0+ | Cross-agent rollback / saga (T7 extension) | Open research |

Contributions are welcome. Open a GitHub Discussion tagged `multi-agent` to propose approaches or ask questions.

---

## References

- `docs/MULTI_AGENT_ARCHITECTURE.md` — Current limitations and single-process shared-kernel pattern
- `clampai/formal.py:703-970` — `SafetyKernel`, `evaluate_and_execute_atomic`, `threading.Lock`
- `MATHEMATICAL_COMPLIANCE.md §What These Proofs Do Not Cover` — Item 3: multi-process coordination
- Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." *CACM*.
- Gray, J. (1978). "Notes on Data Base Operating Systems." *Operating Systems: An Advanced Course*.
- Shapiro, M. et al. (2011). "Conflict-Free Replicated Data Types." *SSS 2011*.
- Garcia-Molina, H., Salem, K. (1987). "Sagas." *SIGMOD*.
