# Formal Claims

All claims are tagged with their epistemic status:

## Overview

| Category | Theorems | Focus |
| --- | --- | --- |
| Foundational (T) | T1–T8 | Core execution, budget, invariants, emergency escape |
| Boundary Detection (JSF) | JSF-1, JSF-2 | Constraint sensitivity analysis |
| Enforcement (AHJ) | AHJ-1, AHJ-2 | Safe state enforcement |
| Composition (OC) | OC-1, OC-2 | Safe task combination |

---

## Category 1: Foundational Theorems (T1–T8)

The core "laws" of ClampAI execution.

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

## T2 — Termination  `CONDITIONAL`

**Statement:** The system halts in at most `⌊B₀ / ε⌋` steps.

**Assumptions:** All actions cost ≥ ε > 0 (`min_action_cost > 0`). Budget B₀ is finite.

**Proof:**
After n steps, `spent_gross ≥ n × ε` (each step costs at least ε, T4).
When `n > ⌊B₀/ε⌋`: remaining budget `< ε`.
The budget check rejects any action costing ≥ ε. No actions can execute → halt. ∎

**Note:** Emergency actions (T8) bypass this check. They must have `cost = 0` and `effects = ()`.

---

## T3 — Invariant Preservation  `PROVEN`

**Statement:** For every blocking-mode invariant `I`:
`I(s₀) = True ⟹ I(sₜ) = True` for all t.

**Scope:** Only `enforcement="blocking"` invariants. Monitoring-mode invariants log violations but never block.

**Proof** (induction on t):
- Base: `I(s₀) = True` (checked at startup).
- Inductive step: Assume `I(sₜ) = True`.
  - `s' = action.simulate(sₜ)` (pure; no side effects).
  - If `I(s') = False` → action rejected, `sₜ₊₁ = sₜ`, `I(sₜ₊₁) = True`. ✓
  - If `I(s') = True` → committed, `sₜ₊₁ = s'`, `I(sₜ₊₁) = True`. ✓ ∎

---

## T4 — Monotone Gross Spend  `PROVEN`

**Statement:** `spent_gross(t) ≤ spent_gross(t+1)` for all t.

**Proof:** `charge()` asserts `cost ≥ 0` before adding to `spent_gross`. A non-negative addition cannot decrease the total. ∎

**Relevance to T7:** Rollback refunds `spent_net` via a separate `spent_refunded` counter, leaving `spent_gross` unchanged. T7 and T4 coexist without conflict.

---

## T5 — Atomicity  `PROVEN`

**Statement:** Actions are all-or-nothing. Rejected actions leave state, budget, and step count unchanged.

**Proof:**
`evaluate()` simulates on a copy of state — no shared state mutated.
`execute()` only commits after `evaluate()` returns `approved = True`.
If `approved = False`, no commit occurs. ∎

**Concurrency:** Use `evaluate_and_execute_atomic()` in concurrent settings to eliminate the TOCTOU race between `evaluate()` and `execute()`.

---

## T6 — Trace Integrity  `PROVEN`

**Statement:** The execution log is append-only and tamper-evident.

**Proof:**
`TraceEntry` is a frozen dataclass (immutable after construction).
Each entry's hash covers all fields including `prev_hash`, forming a chain.
`verify_integrity()` walks the chain in O(n): any modification to entry `i` breaks the hash at `i+1`. ∎

**Note:** This protects against software-layer tampering, not adversarial memory access (`gc`, `ctypes`).

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

## T7 — Rollback Exactness  `PROVEN`

**Statement:** `rollback(execute(s, a)) == s`.

**Proof:**
`State` is immutable (P1): once constructed, no State object is ever modified.
`state_before` stored at commit time is preserved unchanged by the immutability guarantee.
`apply_rollback()` returns `state_before` directly.
Budget refund uses `budget.refund()`, decrementing `spent_net` without touching `spent_gross` (T4 preserved). ∎

---

## T8 — Emergency Escape  `CONDITIONAL`

**Statement:** The SAFE_HOVER emergency action is always executable.

**Assumptions:**
1. The action is registered via `kernel.register_emergency_action(id)`.
2. The action has `cost = 0.0` and `effects = ()`.

**Proof:**
Emergency actions bypass Check 0 (min cost) and Check 2 (step limit).
With `cost = 0.0`, `can_afford()` always returns True.
With `effects = ()`, `simulate()` returns an identical state, passing all invariant checks trivially. ∎

---

## Reference Monitor Guarantees

### M1 — IFC Lattice  `DETERMINISTIC`

Data flows only to sinks at equal or higher security level (`PUBLIC ⊑ INTERNAL ⊑ PII ⊑ SECRET`). Violations blocked by `ReferenceMonitor.enforce()`.

### M2 — CBF Barrier  `DETERMINISTIC`

`h(s_{t+1}) ≥ (1 - α) × h(s_t)` enforced at each step, where h(s) is the distance-to-boundary function. Bounds the approach rate to resource limits.

### M3 — QP Minimality  `DETERMINISTIC`

When action parameters violate M2, the QP projector finds the minimum-norm modification that restores safety. The repaired action is the closest safe action to the original.

---

## Composition Theorems

### OC-1 — Compositional Safety  `CONDITIONAL`

**Statement:** `Verified(A) ∧ Verified(B) ∧ compatible(A, B) ⟹ Verified(A∘B)`.

Interface compatibility: A's output interface must cover B's required inputs. Postconditions of A must satisfy preconditions of B (checked semantically). The composed task inherits both verification certificates.

### OC-2 — Incremental Verification  `CONDITIONAL`

**Statement:** Verifying k-task compositions requires O(k) interface checks, not O(k²) re-verification.

Each task is verified once. Composition checks only interface compatibility. Re-running the full safety kernel is not required.

---

## Heuristic Claims

### JSF-1, JSF-2 — Boundary Sensitivity  `HEURISTIC`

`GradientTracker` estimates which variables are near invariant boundaries via finite-difference perturbation. Diagnostic signal only; not safety enforcement. May miss nonlinear or cross-variable clampaint interactions.

### AHJ-1, AHJ-2 — Active HJB Reachability  `HEURISTIC`

k-step lookahead to detect capture basins. Incomplete — bounded depth, finite action set. For complete reachability proofs, use TLA+ or SPIN.
