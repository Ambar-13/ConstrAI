# Known Limitations

Honest accounting of what ConstrAI can and cannot guarantee. No sugarcoating.

---

## V1 — Spec-Reality Gap  `Partially mitigated`

**The problem:** Formal proofs hold over the *model* (your `ActionSpec.effects`), not over reality. If your action spec says `Effect("files_uploaded", "set", True)` but the upload silently fails in production, the model stays clean while reality diverges.

**Why this is hard:** There's no way to formally verify arbitrary real-world effects from within Python.

**Mitigation:** `EnvironmentReconciler` (hardening.py) compares model state to live environment probes after each action. If they diverge beyond a threshold, execution halts with `EnvironmentDriftError`.

**Residual risk:** Only explicitly registered probes are checked. Unchecked variables can drift undetected.

---

## V2 — LLM Decision Quality  `Not mitigated`

**The problem:** ConstrAI limits the *damage* from bad LLM decisions but doesn't make the LLM smarter. An LLM can choose legal sequences of actions that waste the entire budget without achieving the goal.

**Why this is hard:** Solving arbitrary planning problems is PSPACE-hard. We can constrain the action space but not reason about the optimality of the chosen sequence.

**Mitigation:** Action value computation and Bayesian beliefs inform LLM decisions. Stuck detection triggers termination if progress stalls.

**Residual risk:** A sufficiently bad LLM will waste budget on legal-but-useless actions. No safety violation, but no goal achievement either.

---

## V3 — Multi-Agent Coordination  `Not implemented`

**The problem:** ConstrAI is single-agent. If multiple agents share state (e.g., multiple processes modifying the same database), ConstrAI's formal guarantees apply to one agent's model, not to the global system.

**Residual risk:** Cross-agent state conflicts, double-spends, and invariant violations not visible to any single agent's model.

---

## V4 — Subjective Goals  `Partially mitigated`

**The problem:** T3 (invariant preservation) requires binary predicates. Goals like "write good code" or "produce a high-quality report" cannot be precisely formalized.

**Mitigation:** `MultiDimensionalAttestor` scores quality across multiple dimensions (existence, completeness, correctness, quality, safety, regression). Any zero dimension fails the attestation. This is harder to game than a single score.

**Residual risk:** The attestation dimensions themselves require human judgment to define well.

---

## V5 — Evasion via Dynamic Dispatch  `Partially mitigated`

**The problem:** Invariant predicates that use static pattern matching (e.g., checking if a command string contains "rm -rf") can be bypassed via string concatenation, base64 encoding, or `getattr` dynamic dispatch.

**Evaluation result:** 4 of 39 adversarial attacks succeeded in the safety evaluation. All 4 exploited pattern matching in *user-written invariant predicates*, not the formal kernel. The kernel has no known bypasses when invariants are correctly written.

**Mitigation:** The kernel's formal invariant checking is immune when invariants are written against *state values* (e.g., `s.get("command")` after normalization) rather than raw string matching.

**Residual risk:** The quality of invariants determines the quality of safety. Writing invariants that are evasion-resistant is a user responsibility.

---

## V6 — Deep Python Memory Manipulation  `Partially mitigated`

**The problem:** Python is not memory-safe. An adversary with process access can use `ctypes`, `gc`, or `sys` internals to mutate objects that Python treats as immutable.

**Mitigation:** `State` uses `MappingProxyType` (read-only at the C level via Python's normal API). `TraceEntry` is a frozen dataclass. Both resist normal and moderate-effort mutation.

**Residual risk:** A sufficiently determined adversary with process-level access can bypass these protections. ConstrAI is not a sandboxing mechanism — it assumes the Python runtime is trusted.

---

## V7 — LLM Adversarial Prompting  `Accepted`

**The problem:** A crafted task goal or state description could cause the LLM to choose bad *sequences* of legal actions that drain the budget without achieving anything useful.

**Why this is accepted:** The formal layer prevents any individual action from violating invariants, exceeding budget, or corrupting the trace. The LLM can waste budget but cannot cause safety violations. Preventing all adversarial LLM behavior is an open research problem.

---

## V8 — T2 Termination Conditional  `Accepted`

**The problem:** T2 (termination) requires `min_action_cost > 0`. The `SafetyKernel` constructor enforces this. If the kernel is instantiated with `min_action_cost = 0`, the termination bound is undefined (infinite loop is possible).

**Mitigation:** The constructor raises `ValueError` if `min_action_cost ≤ 0`.

**Residual risk:** Code that bypasses the constructor (e.g., directly setting the attribute) removes the guarantee.

---

## V9 — No Global Termination for Monitoring Invariants  `Accepted by design`

**The problem:** Monitoring-mode invariants can be violated indefinitely. The system keeps running even if a monitoring invariant fires every step.

**Why this is accepted by design:** Monitoring-mode invariants are explicitly not safety-critical. They exist for diagnostics. If you want a hard stop, use `enforcement="blocking"`.

---

## Summary

| ID | Description | Status |
|----|-------------|--------|
| V1 | Spec-reality gap | Partially mitigated by EnvironmentReconciler |
| V2 | LLM decision quality | Not mitigated; formal layer limits damage |
| V3 | Multi-agent coordination | Not implemented |
| V4 | Subjective goals | Partially mitigated by MultiDimensionalAttestor |
| V5 | Evasion via dynamic dispatch | Mitigated when invariants check state values, not strings |
| V6 | Deep memory manipulation | Mitigated for normal API; not for ctypes/gc |
| V7 | LLM adversarial prompting | Accepted; formal layer prevents safety violations |
| V8 | T2 conditional on min_cost > 0 | Accepted; constructor enforces this |
| V9 | Monitoring invariants never halt | Accepted by design |
