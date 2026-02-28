# ClampAI Mathematical Compliance Document

**Version:** 0.3.0  
**Date:** 2026-02-27  
**Purpose:** For each theorem T1-T8, state precisely what is proven, where the implementation lives, what the proof strategy is, what the guarantee level and assumptions are, and what the known limitations are.

This document is intentionally conservative. Claims are tagged `PROVEN`, `CONDITIONAL`, or `EMPIRICAL`. No claim exceeds what the code and proof strategy actually justify.

---

## Epistemic Status Tags

| Tag | Meaning |
|-----|---------|
| `PROVEN` | Holds unconditionally by construction and induction over the code as written. |
| `CONDITIONAL` | Proven under explicitly stated assumptions. The guarantee collapses if any assumption is violated. |
| `EMPIRICAL` | Measured on test suites; no formal proof. |

---

## T1 - Budget Safety

### Statement

> For all times t in an execution:
> `spent_net(t) <= B0`
> where `B0` is the budget provided at kernel construction and `spent_net(t)` is the net budget consumed at step t (gross charges minus refunds).

### Code Location

| Element | File | Lines |
|---------|------|-------|
| `BudgetController` class | `clampai/formal.py` | 446-574 |
| `BudgetController.__init__` - integer-scaled budget | `clampai/formal.py` | 472-480 |
| `BudgetController.can_afford` - pre-commit guard | `clampai/formal.py` | 511-522 |
| `BudgetController.charge` - commit + post-commit assertion | `clampai/formal.py` | 523-538 |
| `SafetyKernel.evaluate` - Check 1 (budget) | `clampai/formal.py` | 797-804 |
| `SafetyKernel.evaluate_and_execute_atomic` - atomic lock | `clampai/formal.py` | 918-970 |

### Proof Strategy

Induction over execution steps.

- **Base case:** At step 0, `spent_net(0) = 0`. Since `B0 >= 0` (enforced in `__init__`), `0 <= B0`. Holds.
- **Inductive step:** Assume `spent_net(t) <= B0`. At step t+1, before any charge, `can_afford(c)` is called. It evaluates `(spent_net(t) + c) <= B0`. If false, the action is rejected and `spent_net(t+1) = spent_net(t)`. If true, `charge(c)` sets `spent_net(t+1) = spent_net(t) + c <= B0` and asserts this post-commit. The kernel's `evaluate_and_execute_atomic` holds `_lock` across the check and charge, preventing interleaving.
- **Integer arithmetic:** Budgets are multiplied by `_SCALE = 1000` and stored as integers, eliminating floating-point accumulation error.

### Guarantee Level

`PROVEN`

### Assumptions

1. All mutations to `BudgetController` go through `charge()` (not direct attribute writes). This holds by construction: `_spent_gross_i` and `_refunded_i` are private integer attributes; Python's name mangling provides soft enforcement.
2. Single-process, single-kernel execution. In multi-process deployments, each process has an independent `BudgetController`; there is no cross-process budget aggregation. (See `docs/MULTI_AGENT_ARCHITECTURE.md`.)
3. `threading.Lock` is held for the full check-and-charge sequence in `evaluate_and_execute_atomic()`. If callers use `evaluate()` and `execute()` separately without their own locking, T1 holds per-call but TOCTOU between `evaluate` and `execute` is the caller's responsibility.

### Known Limitations

- Multi-process budget aggregation is not implemented. Two processes sharing a logical budget must coordinate externally; ClampAI does not provide this.
- The integer scaling factor (`_SCALE = 1000`) means costs below `0.001` are rounded to zero and bypass the budget check. Set `min_action_cost > 0` to prevent this (enforced by T2's precondition check).

---

## T2 - Termination

### Statement

> An execution halts in at most `floor(B0 / epsilon)` steps, where `epsilon` is `min_action_cost > 0`.

### Code Location

| Element | File | Lines |
|---------|------|-------|
| `SafetyKernel.__init__` - computes `max_steps` | `clampai/formal.py` | 786-796 |
| `SafetyKernel.evaluate` - Check 0 (min cost) | `clampai/formal.py` | 837-848 |
| `SafetyKernel.evaluate` - Check 2 (step limit) | `clampai/formal.py` | 807-819 |
| Precondition enforcement (`min_action_cost > 0`) | `clampai/formal.py` | 788-790 |

### Proof Strategy

Proof by contradiction / counting argument.

Assume the system runs for more than `N = floor(B0 / epsilon)` steps. Each step requires `cost >= epsilon` (Check 0 rejects cheaper actions unless emergency-flagged). After N steps, `spent_net >= N * epsilon > B0 - epsilon`. The next candidate action needs `cost >= epsilon`, but `remaining < epsilon`, so `can_afford(cost)` returns false and the action is rejected. The step counter also hard-caps at `max_steps`. No step N+1 can execute.

Additionally, the step counter check (Check 2) provides an independent hard stop at exactly `max_steps`.

### Guarantee Level

`CONDITIONAL`

### Assumptions

1. `min_action_cost > 0`. Enforced at `SafetyKernel.__init__` with a `ValueError`. If bypassed (e.g., via direct attribute write), the termination bound does not hold.
2. Emergency actions (`SAFE_HOVER`, any action in `emergency_actions`) are exempt from min-cost and step-limit checks. They can execute after the budget is exhausted. The caller is responsible for ensuring emergency actions themselves terminate.
3. The Orchestrator's `max_steps` parameter (a separate soft limit) is not part of T2; T2 is enforced solely by the kernel's budget/step checks.

### Known Limitations

- The bound `floor(B0 / epsilon)` can be extremely large for small `epsilon`. For example, `budget=1000.0, min_action_cost=0.001` gives a theoretical bound of 1,000,000 steps. Use a separate `max_steps` soft limit in the Orchestrator for practical runtime caps.
- Actions that raise Python exceptions during `simulate()` are rejected, not counted as steps. This is safe but means the actual step count can be lower than the budget bound suggests.

---

## T3 - Invariant Safety

### Statement

> For all invariants declared with `enforcement="blocking"`:
> If `I(s0) = True`, then `I(st) = True` for all reachable states `st`.

### Code Location

| Element | File | Lines |
|---------|------|-------|
| `Invariant` class | `clampai/formal.py` | 373-442 |
| `Invariant.__init__` - enforcement mode validation | `clampai/formal.py` | 400-418 |
| `Invariant.check` - predicate evaluation | `clampai/formal.py` | 419-430 |
| `SafetyKernel.evaluate` - Check 3 (invariant loop) | `clampai/formal.py` | 820-855 |
| `ActionSpec.simulate` - check-before-commit on copy | `clampai/formal.py` | 307-318 |

### Proof Strategy

Induction using check-before-commit discipline.

- **Base case:** `I(s0) = True` is a precondition supplied by the caller. The kernel does not verify the initial state; it is the caller's responsibility.
- **Inductive step:** Assume `I(st) = True`. Before any action `a` is committed, `evaluate()` calls `a.simulate(st)` to produce `st'` (a copy; `st` is unchanged). It then evaluates `I(st')`. If `I(st') = False` and the invariant mode is `blocking`, the action is rejected and `st+1 := st`, so `I(st+1) = True`. If `I(st') = True`, the action is committed and `st+1 := st'`. Either way, `I(st+1) = True`.

### Guarantee Level

`PROVEN` for `enforcement="blocking"` invariants.

`NOT A GUARANTEE` for `enforcement="monitoring"` invariants (violations are logged but do not block).

### Assumptions

1. `I(s0) = True`. The kernel does not check the initial state. If the initial state already violates an invariant, the guarantee does not apply.
2. Invariant predicates are pure (no side effects, no I/O, no randomness). Predicates that raise exceptions are treated as violations (conservative fallback), but non-deterministic predicates can produce inconsistent results.
3. Predicates do not modify the state they receive. `State` objects are immutable by construction (`__setattr__` raises `FrozenInstanceError`), making this structurally enforced.
4. The invariant covers all safety-relevant state variables. Undeclared variables are not checked.
5. Proofs are over the formal model. If a declared `Effect` does not match what the real world does (spec-reality gap), T3 holds on the model but not necessarily on the world.

### Known Limitations

- T3 is only as useful as the invariants declared. The kernel enforces whatever predicate is given; it cannot auto-generate invariants.
- Monitoring-mode invariants (`enforcement="monitoring"`) provide no blocking guarantee. They are useful for observability but must not be confused with T3 coverage.
- The initial state assumption is not checked at kernel construction. Call `Invariant.check(initial_state)` before constructing `SafetyKernel` if you need a hard guarantee from step 0.

---

## T4 - Monotone Gross Spend

### Statement

> `spent_gross(t) <= spent_gross(t+1)` for all t.
> Gross spend is monotonically non-decreasing.

### Code Location

| Element | File | Lines |
|---------|------|-------|
| `BudgetController` - `_spent_gross_i` field | `clampai/formal.py` | 476-480 |
| `BudgetController.charge` - increment + assertion | `clampai/formal.py` | 523-538 |
| `BudgetController.refund` - only `_refunded_i` incremented | `clampai/formal.py` | 540-552 |
| Post-commit assertion: `assert self._spent_gross_i >= old_gross` | `clampai/formal.py` | 534-535 |

### Proof Strategy

Construction: only one operation (`charge()`) modifies `_spent_gross_i`, and it only adds a non-negative value. `refund()` modifies only `_refunded_i`, not `_spent_gross_i`. The post-commit assertion in `charge()` verifies monotonicity after every charge.

T4 and T1 use separate tracking intentionally: `spent_gross` captures irreversible resource consumption (useful for auditing) while `spent_net = spent_gross - spent_refunded` is used for budget enforcement (T1). Rollback reduces `spent_net` without retroactively altering the audit trail.

### Guarantee Level

`PROVEN`

### Assumptions

1. `charge()` always receives `cost >= 0`. `ActionSpec.__post_init__` validates `cost >= 0`, providing defence in depth.
2. No direct writes to `_spent_gross_i` outside `charge()`.

### Known Limitations

None significant. This is the simplest of the eight theorems.

---

## T5 - Atomicity

### Statement

> Every action transition is all-or-nothing: either (budget is charged AND state is updated AND trace entry is appended) all succeed, or the action is rejected and none of those happen.

### Code Location

| Element | File | Lines |
|---------|------|-------|
| `SafetyKernel.evaluate` - simulate-only, no side effects | `clampai/formal.py` | 768-856 |
| `SafetyKernel.execute` - commit block | `clampai/formal.py` | 857-917 |
| `SafetyKernel.evaluate_and_execute_atomic` - under `_lock` | `clampai/formal.py` | 918-970 |
| `ActionSpec.simulate` - operates on copy, not original | `clampai/formal.py` | 307-318 |

### Proof Strategy

Construction.

- `evaluate()` calls `action.simulate(state)` which produces a new `State` object; the original `state` is never mutated (immutability enforced by `__setattr__` override at `formal.py:115-117`).
- No budget is charged and no trace entry is appended during `evaluate()`.
- `execute()` performs the commit sequence: re-evaluates (safety net), charges budget, constructs the new state, appends the trace entry, and returns. If any step raises an exception, the state returned to the caller is the pre-action state (Python exception unwinds the local assignment).
- `evaluate_and_execute_atomic()` wraps the full sequence under `threading.Lock`, preventing partial observations by other threads.

### Guarantee Level

`PROVEN` for the formal model (budget, state, trace).

`NOT COVERED` by T5: real-world side effects (HTTP calls, file writes, etc.) that happen outside the kernel after the kernel approves an action. T5 atomicity is at the model level, not the OS or network level.

### Assumptions

1. `State` immutability is not bypassed (no `object.__setattr__` calls on a `State` object from outside the class).
2. The kernel's `_lock` is used via `evaluate_and_execute_atomic()` in concurrent settings. Using `evaluate()` and `execute()` separately from multiple threads requires external synchronization.
3. Python's GIL provides a baseline of atomicity for pure Python operations; CPython does not guarantee atomicity across the budget charge + state update + trace append sequence without the explicit `threading.Lock`.

### Known Limitations

- T5 atomicity is a model-level property. If the approved action triggers an external side effect (e.g., an API call) and that call fails, the kernel's model is in the "committed" state while the real world is not. The `EnvironmentReconciler` in `hardening.py` (lines 413-469) addresses this by halting execution when model-reality drift is detected.

---

## T6 - Trace Integrity

### Statement

> The execution log is append-only and SHA-256 hash-chained. Any retrospective modification to any entry is detectable by `verify_integrity()`.

### Code Location

| Element | File | Lines |
|---------|------|-------|
| `TraceEntry` - frozen dataclass | `clampai/formal.py` | 576-611 |
| `TraceEntry.compute_hash` - SHA-256 over all fields | `clampai/formal.py` | 603-611 |
| `ExecutionTrace` class | `clampai/formal.py` | 613-671 |
| `ExecutionTrace.append` - links previous hash | `clampai/formal.py` | 620-636 |
| `ExecutionTrace.verify_integrity` - full chain check | `clampai/formal.py` | 638-647 |

### Proof Strategy

Construction.

- `TraceEntry` is a frozen dataclass (`@dataclass(frozen=True)`): Python raises `FrozenInstanceError` on any attempted mutation.
- `compute_hash()` computes SHA-256 over the serialization of all fields including `prev_hash`. This creates a hash chain: any modification to entry i changes its hash, which invalidates entry i+1's `prev_hash` field, cascading to all subsequent entries.
- `verify_integrity()` recomputes each entry's hash and checks it matches the stored hash of the next entry, detecting any chain break.
- `_entries` is a private Python list; entries are only added via `append()`, which calls `TraceEntry(prev_hash=last_hash, ...)`. There is no `remove()` or `pop()` method exposed.

### Guarantee Level

`PROVEN` within the Python process.

### Assumptions

1. The `_entries` list is not accessed directly via Python's name-mangling introspection. Python private attributes provide naming protection, not cryptographic protection.
2. SHA-256 is treated as collision-resistant for this application (standard cryptographic assumption, not a formally proven property within ClampAI).
3. Integrity only covers the in-memory trace. The trace is not persisted to disk by default. Persistence and verification across restarts require additional implementation.
4. An attacker with the ability to run arbitrary Python in the same process can forge trace entries. T6 deters accidental or casual tampering, not a determined attacker with code execution in the process.

### Known Limitations

- The trace is in-memory only. A process crash loses the full trace unless the caller persists `trace.entries()` externally.
- `verify_integrity()` detects tampering but cannot reconstruct the original entries.

---

## T7 - Rollback Exactness

### Statement

> `rollback(execute(s, a)) == s` exactly.
> Applying the inverse effects of action `a` to the post-execution state returns the original state `s`.

### Code Location

| Element | File | Lines |
|---------|------|-------|
| `ActionSpec.compute_inverse_effects` - T7 proof in docstring | `clampai/formal.py` | 319-394 |
| `Effect.inverse` - algebraic inverse per operation | `clampai/formal.py` | 237-265 |
| `SafetyKernel.rollback` | `clampai/formal.py` | 988-999 |
| `BudgetController.refund` - budget component of rollback | `clampai/formal.py` | 540-552 |
| `InverseAlgebra.compute_inverse_from_states` | `clampai/inverse_algebra.py` | 100-136 |
| `RollbackRecord.apply_rollback` - T7 runtime assertion | `clampai/inverse_algebra.py` | 57-87 |

### Proof Strategy

There are two independent mechanisms for T7; both are correct by construction.

**Mechanism 1 - Algebraic inverse effects** (`compute_inverse_effects`):
For each effect on variable `v`:
- `increment(x)` -> `decrement(x)` (or `set(old_value)` if old was not numeric)
- `decrement(x)` -> `increment(x)` (or `set(old_value)`)
- `set(v)` -> `set(old_value)` (old value captured from `state_before`)
- `append(x)` -> `set(old_list)` (entire list restored)
- `delete` -> `set(old_value)` if the variable existed, else no-op

By applying these in sequence to `s' = execute(s, a)`, each variable is restored to its value in `s`. Because `State` is immutable, `s` itself is unchanged throughout.

**Mechanism 2 - State diff inverse** (`InverseAlgebra.compute_inverse_from_states`):
Diff `state_before` and `state_after` directly. For each changed variable, emit a `set(old_value)` effect. For added keys, emit `delete`. For deleted keys, emit `set(old_value)`. This is guaranteed correct regardless of the effect type, because it operates on actual state values rather than declared effect semantics.

`RollbackRecord.apply_rollback` asserts `result == state_before` after applying the inverse, catching any edge-case failure at runtime.

**Budget component:** `BudgetController.refund(cost)` decrements `spent_net` by the action's cost, undoing the budget charge. T4 (`spent_gross` monotonicity) is preserved because only `_refunded_i` is incremented, not `_spent_gross_i` decremented.

### Guarantee Level

`PROVEN` for the formal state model (declared variables and declared effects).

### Assumptions

1. The state contains only JSON-serializable Python values (int, float, str, list, dict, None). Effects on non-serializable objects (e.g., file handles, thread objects) cannot be inverted by this mechanism.
2. Rollback must be applied to the state immediately following the action. If other actions have run since, rollback restores only the specific variables changed by `a`, leaving later changes intact - it is not a full-history multi-step rollback.
3. Budget rollback (`refund`) reduces `spent_net` but does not reverse the step counter. A rolled-back action still counted toward the step limit (T2's bound is not loosened by rollback).

### Known Limitations

- Rollback is a model operation. External side effects (emails sent, API calls made, files written) are not undone by kernel rollback. For real-world reversibility, the system must implement compensating actions (e.g., send a cancellation email, issue a DELETE request).
- `Effect.inverse()` on an `append` mode restores the entire list (`set(old_list)`). This is correct but coarse: if the list was modified by other means between execute and rollback, those additional changes are also undone.

---

## T8 - Emergency Escape

### Statement

> The emergency escape action (`SAFE_HOVER` or any action registered via `register_emergency_action()`) is always executable, regardless of budget remaining or step count.

### Code Location

| Element | File | Lines |
|---------|------|-------|
| `SafetyKernel.register_emergency_action` | `clampai/formal.py` | 764-766 |
| `SafetyKernel.evaluate` - Check 0 bypass for emergency | `clampai/formal.py` | 838-843 |
| `SafetyKernel.evaluate` - Check 2 bypass for emergency | `clampai/formal.py` | 808-813 |
| `SafetyKernel.__init__` - `emergency_actions` set | `clampai/formal.py` | 786-806 |
| Module docstring - T8 canonical statement | `clampai/formal.py` | 36 |

### Proof Strategy

Construction.

In `evaluate()`, Check 0 (min cost) and Check 2 (step limit) both begin with:

```python
if action.id in self.emergency_actions:
    # skip this check
```

The budget check (Check 1) is not a barrier for emergency actions with `cost=0.0`: `can_afford(0.0)` always passes because no budget is consumed. Invariant checks (Check 3) still run for emergency actions - the guarantee is executability past resource/step limits, not invariant bypass.

An emergency action with `cost=0.0` and `effects=()` (the canonical `SAFE_HOVER` pattern) trivially passes all remaining checks: budget is not consumed, state is unchanged, and no invariant can be violated by a no-op.

### Guarantee Level

`CONDITIONAL`

### Assumptions

1. The emergency action is registered before it is needed. `register_emergency_action()` must be called at kernel construction or before any step that might exhaust the budget.
2. The emergency action has `cost=0.0` for unconditional executability. A non-zero cost emergency action can still be blocked if the budget is exactly zero (integer arithmetic: `can_afford(0.0)` always passes; `can_afford(0.001)` may not if budget is exhausted).
3. The emergency action has `effects=()` or effects that do not violate any blocking invariant. If the emergency action itself would violate an invariant, it is still rejected by Check 3.

### Known Limitations

- T8 guarantees that the emergency action passes the kernel's checks. It does not guarantee that the action achieves a safe real-world state - that depends on what the action's effects do and what the real-world handler for that action implements.
- There is no automatic fallback: if no emergency action is registered and the budget is exhausted, the system halts. `SAFE_HOVER` must be explicitly configured by the user.
- The guarantee applies per-step. There is no built-in rate limit on emergency invocations; a loop calling only the emergency action will run indefinitely.

---

## Summary Table

| Theorem | Statement (short) | Level | Key Code Location | Primary Assumption |
|---------|------------------|-------|-------------------|--------------------|
| T1 | `spent_net(t) <= B0` always | `PROVEN` | `formal.py:511-538` | Single-process; lock held during check+charge |
| T2 | Halts in `<= floor(B0/epsilon)` steps | `CONDITIONAL` | `formal.py:786-848` | `min_action_cost > 0`; emergency actions excluded |
| T3 | Blocking invariants hold on all reachable states | `PROVEN` | `formal.py:820-855` | Initial state satisfies invariants; predicates are pure |
| T4 | `spent_gross` never decreases | `PROVEN` | `formal.py:523-538` | Only `charge()` modifies `_spent_gross_i` |
| T5 | Rejected actions leave state unchanged | `PROVEN` | `formal.py:768-970` | `State` immutability not bypassed; lock used in concurrent settings |
| T6 | Trace is tamper-detectable | `PROVEN` | `formal.py:576-671` | No direct `_entries` list access; SHA-256 collision resistance |
| T7 | `rollback(execute(s,a)) == s` | `PROVEN` | `formal.py:319-394`, `inverse_algebra.py:57-136` | State variables are JSON-serializable; effects match declared semantics |
| T8 | Emergency action always executable | `CONDITIONAL` | `formal.py:764-813` | Emergency action registered; `cost=0.0`; effects do not violate invariants |

---

## What These Proofs Do Not Cover

The following are explicit out-of-scope items. No theorem makes claims about them.

1. **Spec-reality gap.** Proofs hold over the formal model (declared `Effect` objects). If a declared effect does not match what actually happens in the world, the model is correct but the world is not. Partially mitigated by `EnvironmentReconciler` (`hardening.py:413-469`).

2. **LLM decision quality.** The theorems clampain what executes, not what the LLM proposes. A budget-compliant, invariant-satisfying action can still be the wrong action for the goal.

3. **Multi-process coordination.** All proofs assume a single `SafetyKernel` instance in a single process. Distributed multi-agent safety is an open research problem. See `docs/MULTI_AGENT_ARCHITECTURE.md`.

4. **Python memory-level attacks.** An adversary with the ability to call `ctypes` or manipulate the garbage collector can bypass Python's attribute protection. ClampAI does not provide memory-safety guarantees.

5. **Monitoring-mode invariants.** Only `enforcement="blocking"` invariants are covered by T3. Monitoring-mode invariants log violations but do not block execution.

6. **Classification layer accuracy.** The `SubprocessAttestor` and shell pattern classifier use heuristic pattern matching. Their coverage is `EMPIRICAL` (89.7% recall, 0 false positives across 39 attack vectors - see `BENCHMARKS.md`), not `PROVEN`.

---

## Machine-Checked Proof Roadmap (Lean 4)

The proofs in this document are pen-and-paper induction arguments, verified by code inspection. Converting them to machine-checked proofs in Lean 4 (or TLA+) is a long-term goal. This section documents the prerequisite skills assessment and a concrete First 30 Days sprint for anyone who wants to contribute to that effort.

### Prerequisites Assessment

Machine-checking these proofs requires specific background. The table below is an honest skills gate — do not start on the Lean 4 formalization before confirming all `Required` items.

| Skill | Level Required | Assessment |
|-------|---------------|------------|
| Lean 4 syntax and tactics (`rfl`, `simp`, `omega`, `linarith`) | Working familiarity | Can you prove `∀ n : ℕ, n + 0 = n` in Lean 4 without looking up the tactic? |
| Mathlib4 (Nat, Int, List, Finset) | Navigating the library | Can you find the `Nat.add_le_add_right` lemma in Mathlib4 and use it in a proof? |
| Induction over recursive datatypes | Comfortable | Can you prove properties of a simple list by structural induction in Lean 4? |
| Python semantics reasoning | Understand the gap | You must be comfortable reasoning about what the Python code *guarantees* vs. what Lean will verify (the formal model, not Python execution). |
| TLA+ (alternative path) | Basic | TLA+ is a lower barrier to entry for T1 and T4 (arithmetic invariants). It is less expressive than Lean 4 for T7 (algebraic rollback) but more accessible for first contributors. |

**Recommended self-test:** Before starting, attempt to prove the following in Lean 4:

```lean
-- Prove that a non-negative integer can never decrease by a positive increment
theorem spend_monotone (spent budget c : ℕ) (h : spent + c ≤ budget) :
    spent ≤ spent + c := Nat.le_add_right spent c
```

If this proof takes more than 10 minutes to find, invest time in the Lean 4 tutorial (`https://leanprover.github.io/lean4/doc/`) before starting.

### First 30 Days Sprint Plan

This sprint targets **T1 (Budget Safety)** as the first machine-checked theorem, because:
- It has the simplest statement (`spent ≤ budget` by induction).
- The Python proof is already well-structured (see §T1 above).
- Integer arithmetic (`omega` tactic in Lean 4) handles the core argument.

**Week 1: Setup and model**
- [ ] Install Lean 4 and Mathlib4 (`elan` + `lake`).
- [ ] Create `formal/ClampAI/BudgetModel.lean`.
- [ ] Define `BudgetState : Type` with `spent_i : Int` and `budget_i : Int` (scaled integers, matching `_SCALE = 1000` in `formal.py`).
- [ ] State T1 as a `theorem` without proving it yet: `∀ (s : BudgetState), s.spent_i ≤ s.budget_i`.

**Week 2: Base case and inductive step**
- [ ] Prove the base case: `BudgetState.initial.spent_i = 0` and `0 ≤ budget_i`.
- [ ] Model `can_afford (c : Int) (s : BudgetState) : Bool` as a pure function.
- [ ] Model `charge (c : Int) (s : BudgetState) : BudgetState` guarded by `can_afford`.
- [ ] Prove the inductive step: `can_afford c s = true → (charge c s).spent_i ≤ s.budget_i`.

**Week 3: Full induction**
- [ ] Define `ExecutionTrace` as a `List BudgetState`.
- [ ] Prove `∀ (trace : ExecutionTrace), ∀ (s ∈ trace), s.spent_i ≤ s.budget_i` by `List.induction`.
- [ ] Identify which parts of the Python `BudgetController` are not yet modelled and document them as `sorry` stubs with explanatory comments.

**Week 4: Integration and TLA+ alternative**
- [ ] If Lean 4 proof is complete: open a PR with the `.lean` file and update `MATHEMATICAL_COMPLIANCE.md §T1` to `MACHINE-CHECKED`.
- [ ] If blocked: as an alternative, write a TLA+ spec for T1 in `formal/tla/BudgetSafety.tla` and model-check it with TLC over a bounded state space (`budget ≤ 100`, `n_steps ≤ 20`).

### Tracked Work Items

Machine-checked proofs are tracked as GitHub issues with the `lean4-proof` label. Current status:

| Theorem | Status | Notes |
|---------|--------|-------|
| T1 Budget Safety | Open — seeking contributor | Simplest starting point; see sprint above |
| T4 Monotone Spend | Open — dependent on T1 model | Single `charge` step property; follows immediately from T1 model |
| T6 Trace Integrity | Open — SHA-256 assumptions need axiomatisation | Cryptographic assumption must be stated as an axiom in Lean 4 |
| T3 Invariant Safety | Open — requires State model | More complex; depends on `State` being modelled as a `Map String Value` |
| T7 Rollback Exactness | Open — requires InverseAlgebra model | Most complex; `Effect` inverse operations need full algebraic model |
| T2, T5, T8 | Not started | Lower priority pending T1 completion |
