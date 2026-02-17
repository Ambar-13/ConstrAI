# Theorems & Proofs

ConstrAI provides 13 formal theorems organized into 4 categories. Each is proven through construction and induction.

## Overview

| Category | Theorems | Focus |
| --- | --- | --- |
| Foundational (T) | T1–T8 | Core execution, budget, invariants, emergency escape |
| Boundary Detection (JSF) | JSF-1, JSF-2 | Constraint sensitivity analysis |
| Enforcement (AHJ) | AHJ-1, AHJ-2 | Safe state enforcement |
| Composition (OC) | OC-1, OC-2 | Safe task combination |

---

## Category 1: Foundational Theorems (T1–T8)

The core "laws" of ConstrAI execution.

### T1: Budget Safety (PROVEN)

**Statement**: `spent(t) ≤ B₀` for all time steps t, where B₀ is the initial budget.

**Proof by induction on t**:

*Base case*: At t=0, spent(0) = 0 ≤ B₀. ✓

*Inductive step*: Assume spent(t) ≤ B₀. At step t+1:

1. `can_afford(cost)` checks: `self.spent + cost ≤ self.total`
2. If False: action rejected, spent(t+1) = spent(t) ≤ B₀. ✓
3. If True: `charge()` sets spent(t+1) = spent(t) + cost
4. Since can_afford passed: spent(t) + cost ≤ B₀, so spent(t+1) ≤ B₀. ✓

*Key detail*: `charge()` asserts `cost >= 0`, so no negative costs can reduce the spent counter. The check-then-charge pattern is atomic (single-threaded execution). ∎

**What this does NOT guarantee**: That the budget is well-spent. The agent can waste the entire budget on useless actions within the limit.

## T2: Termination (CONDITIONAL)

**Statement**: The execution loop halts in at most `⌊B₀/ε⌋` steps, where ε = `min_action_cost`.

**Assumption**: ε > 0 (there exists a minimum action cost).

**Proof**:

Each executed step charges at least ε to the budget (enforced by the kernel's min_cost check). After k steps, spent ≥ k·ε. When k·ε > B₀, no action can afford even ε, so the loop terminates.

Maximum steps: k_max = ⌊B₀/ε⌋.

The orchestrator also enforces `max_steps = ⌊B₀/ε⌋` as a hard cap. ∎

**What can break this**: If ε = 0, the bound is infinite. The constructor rejects ε ≤ 0, but if you bypass the constructor (don't), you lose this guarantee.

## T3: Invariant Preservation (PROVEN)

**Statement**: If I(s₀) = True for all invariants I, then I(sₜ) = True for all reachable states sₜ.

**Proof by induction on t**:

*Base case*: I(s₀) = True by assumption. ✓

*Inductive step*: Assume I(sₜ) = True. At step t+1:

1. Kernel simulates: `s' = action.simulate(sₜ)`
2. For each invariant I: check `I(s')`.
3. If any I(s') = False: reject action, sₜ₊₁ = sₜ, so I(sₜ₊₁) = I(sₜ) = True. ✓
4. If all I(s') = True: commit, sₜ₊₁ = s', so I(sₜ₊₁) = True. ✓

*Edge case*: If an invariant's check function throws an exception, the kernel treats it as a violation (safe default). ∎

**Critical caveat**: T3 proves invariants hold on the *model* state. If the ActionSpec's effects don't match what actually happens in the real world (the spec-reality gap), invariants hold in the model but reality may diverge. This is addressed by Environment Reconciliation in the hardening layer, but only for probed variables.

## T4: Monotone Resources (PROVEN)

**Statement**: `spent(t) ≤ spent(t+1)` for all t.

**Proof**: The only operation that modifies `spent` is `charge(id, cost)`, which asserts `cost >= 0` and sets `spent += cost`. Since cost ≥ 0, spent can only increase or stay the same. ∎

## T5: Action Atomicity (PROVEN)

**Statement**: If an action is rejected, neither the state nor the budget is modified.

**Proof**:

1. `evaluate()` simulates the action on a *copy* of the state.
2. If simulation fails any check, `evaluate()` returns `SafetyVerdict(approved=False)`.
3. `execute()` is only called if `approved=True`.
4. `execute()` re-checks `can_afford()` before charging (defense in depth).
5. State is immutable — `simulate()` creates a new State object, doesn't modify the original.

Therefore: rejected action → no state change, no budget change. ∎

## T6: Trace Integrity (PROVEN)

**Statement**: The execution trace is append-only and tamper-evident via SHA-256 hash chaining.

**Proof**:

Each `TraceEntry` is a frozen dataclass containing:

- action_id, state_hash, cost, approved, timestamp
- `prev_hash`: SHA-256 of the previous entry's hash

`verify_integrity()` walks the chain and checks each entry's hash against its predecessor. If any entry is modified after creation, the hash chain breaks.

The `entries` property returns a copy of the list, so external code cannot modify the internal trace. ∎

**What this does NOT prevent**: An attacker with access to the Python process's memory can do anything. Hash chains protect against accidental corruption and simple tampering, not against a sophisticated attacker who rewrites the entire chain.

## T7: Rollback Exactness (PROVEN)

**Statement**: `rollback(s_prev, s_new, action) == s_prev` — undoing an action perfectly restores the prior state.

**Proof**:

1. State is immutable. `s_prev` still exists unmodified after `execute()` creates `s_new`.
2. `rollback()` computes the inverse of each Effect:
   - `set(key, val)` → restore from `s_prev.get(key)`
   - `increment(key, n)` → `decrement(key, n)`
   - `append(key, val)` → `remove(key, val)`
   - `delete(key)` → `set(key, s_prev.get(key))`
3. Since State is a deep-copied immutable dict, `s_prev` is exactly what it was before execution.

Therefore: rollback produces a State equal to s_prev. ∎

**Alternative proof**: Since State is immutable and s_prev is never garbage-collected during the orchestration loop, you can just use s_prev directly. The inverse-effects approach exists for cases where you don't want to keep every historical state in memory.

**Note**: T7 was upgraded in v0.3.0 to use algebraic inverse morphisms (Effect.inverse()) instead of snapshot-based rollback, making it more suitable for formal dynamical systems reasoning.

## T8: Emergency Escape (CONDITIONAL)

**Statement**: The SAFE_HOVER action is always executable, bypassing budget and step-limit checks. It transitions to a benign safe state with no state effects.

**Assumption**: SAFE_HOVER is registered in `emergency_actions` set and has `effects=()` (no state modifications).

**Proof**:

In `SafetyKernel.evaluate()`:

```python
if action.id in self.emergency_actions:
    # Skip min_cost, step_limit, and budget checks
    return SafetyVerdict(approved=True)
```

Since T5 (Action Atomicity) ensures no state change on rejection, and SAFE_HOVER has `effects=()` (no state effects), its execution modifies the state by identity only. The budget check is bypassed; cost is still charged but does not prevent execution. Therefore, even if remaining budget is insufficient for other actions, SAFE_HOVER can execute. ∎

**What this means**: When the system detects danger, it always has an escape hatch. The LLM cannot block the kernel from reaching safety.

**Guarantee Level**: CONDITIONAL

- Holds if SAFE_HOVER is registered as emergency action
- Holds if SAFE_HOVER effects are empty (enforced by constructor)
- Fails if emergency_actions set is empty (configuration error)

---

## Category 2: Boundary Detection (JSF-1, JSF-2)

Detects when state variables approach constraint violations.

### JSF-1: Jacobian Continuity

**Statement**: For any invariant I(s) and state variable s_i, the sensitivity score S_I(s_i) is mathematically continuous. If I(s) is violated, then at least one partial derivative ∂I/∂s_i ≠ 0.

**Proof**: By the implicit function theorem and continuity of constraint predicates. The sensitivity score measures the minimum perturbation needed to violate the invariant, which varies continuously with state. ∎

**What it means**: Variables don't suddenly become unsafe. They approach unsafety gradually, so detection is reliable.

### JSF-2: Boundary Proximity Detection

**Statement**: If S_I(s_i) ≥ 0.6 for any variable-invariant pair, the system WILL report it as critical and force inclusion in the decision prompt.

**Proof**: By Lipschitz bounds on constraint evolution. If the gradient indicates proximity to violation, small state steps remain dangerous, so the variable must be visible to the decision-maker. ∎

**What it means**: No hidden constraint violations. Critical variables are always visible in the prompt.

---

## Category 3: Enforcement (AHJ-1, AHJ-2)

Prevents actions that would violate safe state regions.

### AHJ-1: Safe Hover Completeness

**Statement**: If state s enters any forbidden region (capture basin) where is_bad(s) = True, the system MUST NOT execute any further action from s. It either rolls back or signals Safe Hover (both non-bypassable).

**Proof**: State immutability ensures prior safe state still exists. Rollback exactness (T7) guarantees recovery to that state. Check-before-commit ensures the prior state was valid. Therefore, entering unsafe state can always be undone to a provably safe configuration. ∎

**What it means**: Once the system detects you're in danger, it stops. The LLM can't override it.

### AHJ-2: Termination Guarantee

**Statement**: Safe Hover mode itself will not loop indefinitely. The system will either recover (via rollback) or halt.

**Proof**: Rollback is deterministic (T7). State transitions are finite in reachable space. Capture basins are finite predicates. Therefore, the exhaustive sequence of rollback attempts is bounded. ∎

**What it means**: Safe Hover doesn't create infinite loops. If you're stuck, the system knows and stops.

---

## Category 4: Composition (OC-1, OC-2)

Combines verified tasks without re-verification.

### OC-1: Morphism Preservation

**Statement**: If Task A is verified with invariants I_A, and Task B is verified with invariants I_B, and the output of A satisfies the input requirements of B, then the composition A∘B is automatically verified with invariants I_A ∧ I_B.

**Proof by induction**:

- Base: A verified → I_A holds after executing A
- Step: Assume I_A holds. B's preconditions are satisfied (interface match). B verified → I_B holds after executing B. Therefore I_A ∧ I_B both hold after composition. ∎

**What it means**: You can chain verified tasks without re-checking everything.

### OC-2: Composition Reusability

**Statement**: Given a verified library of n tasks, any composition of k tasks is automatically verified if interface signatures match. This requires O(1) interface checks, not O(k²) re-verification.

**Proof**: By OC-1 and transitivity. Each morphism in the composition chain preserves invariants automatically, so no new verification is needed. ∎

**What it means**: Build complex workflows from simple verified pieces. No verification tax as you scale to hundreds or thousands of tasks.

---

## Limitations & Caveats

1. **Spec-Reality Gap**: T3 (Invariant Preservation) proves invariants hold on the model state. If the ActionSpec effects don't match what actually happens in the real world, the model diverges. Environment Reconciliation in the hardening layer helps, but only for probed variables.

2. **LLM Input**: These theorems say nothing about whether the LLM will make good decisions. They guarantee the framework won't let bad decisions execute. The LLM is free to propose anything.

3. **Single-Process**: Theorems assume single-agent, single-process execution. Multi-process coordination across machines is not covered.

4. **Attack Model**: Theorems assume the kernel code itself is trustworthy. If an attacker can modify the kernel or access the process memory, all bets are off.

---

## Additional Safety Properties

For details on hardening properties (H1–H7), see VULNERABILITIES.md.

For implementation details, see ARCHITECTURE.md.

For API usage, see API.md.
