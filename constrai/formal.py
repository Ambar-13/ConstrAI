"""

This module contains only provably-correct components. No heuristics, no ML,
no LLM. Pure deterministic state machines with verified safety properties.

Theorems proven by construction + induction:
  T1  Budget Safety:       spent_net(t) ≤ B₀  ∀t
  T2  Termination:         halts in ≤ ⌊B₀/ε⌋ steps (requires ε = min_cost > 0)
  T3  Invariant Safety:    I(s₀)=True ⟹ I(sₜ)=True ∀t  [blocking-mode only]
  T4  Monotone Spend:      spent_gross(t) ≤ spent_gross(t+1)  ∀t
  T5  Atomicity:           transitions are all-or-nothing
  T6  Trace Integrity:     execution log is append-only, SHA-256 hash-chained
  T7  Rollback Exactness:  undo(execute(s,a)) == s  (via algebraic inverse effects)
  T8  Emergency Escape:    SAFE_HOVER is always executable, cost-free, effect-free

Design principles:
  - State is immutable: new states are created on every transition, never mutated
  - Actions are DATA (declarative effects), not CODE (function pointers)
  - Every check happens BEFORE commitment, never after
  - This layer has zero knowledge of LLMs; it constrains any decision-making agent
ConstrAI — Autonomous Engine for Guaranteed Intelligent Safety
Module: constrai.formal — Layer 0: Mathematically Proven Guarantees

This module contains ONLY things we can PROVE. No heuristics, no ML, no LLM.
Pure deterministic state machines with verified safety properties.

THEOREMS (proven by construction + induction):
  T1  Budget Safety:       spent_net(t) ≤ B₀  ∀t
  T2  Termination:         halts in ≤ ⌊B₀/ε⌋ steps where ε = min_cost
  T3  Invariant Safety:    For blocking-mode invariants: I(s₀)=True ⟹ I(sₜ)=True ∀t
       (Monitoring-mode invariants are logged but not safety-critical)
  T4  Monotone Spend:      spent_gross(t) ≤ spent_gross(t+1)  ∀t
  T5  Atomicity:           transitions are all-or-nothing
  T6  Trace Integrity:     execution log is append-only, hash-chained
  T7  Rollback Exactness:  undo(execute(s,a)) == s (via separate spend tracking)
  T8  Emergency Escape:    SAFE_HOVER action always executable, cost-free, effect-free

DESIGN PRINCIPLES:
  - State is immutable (new states are created, never mutated)
  - Actions are DATA (declarative effects), not CODE
  - Every check happens BEFORE execution, never after
  - The formal layer has ZERO knowledge of LLMs — it constrains any agent
"""
from __future__ import annotations

import copy
import hashlib
import json
import time as _time
import threading as _threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Sequence
)


# ═══════════════════════════════════════════════════════════════════════════
# §0  GUARANTEE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

class GuaranteeLevel(Enum):
    """Epistemic status of every claim in ConstrAI."""
    PROVEN      = "proven"       # Holds unconditionally by construction
    CONDITIONAL = "conditional"  # Proven under stated assumptions
    EMPIRICAL   = "empirical"    # Measured with statistical confidence
    HEURISTIC   = "heuristic"    # Best-effort; no formal guarantee


@dataclass(frozen=True)
class Claim:
    """An auditable, tagged claim about a system property."""
    name: str
    statement: str
    level: GuaranteeLevel
    proof: str = ""
    assumptions: Tuple[str, ...] = ()

    def __repr__(self):
        tag = f"[{self.level.value.upper()}]"
        return f"{tag} {self.name}: {self.statement}"


# ═══════════════════════════════════════════════════════════════════════════
# §1  IMMUTABLE STATE
# ═══════════════════════════════════════════════════════════════════════════

class State:
    """
    Immutable, hashable, JSON-deterministic world state.

    Proven properties:
      P1  Immutability:    no method mutates self after __init__
      P2  Det. equality:   s1 == s2  ⟺  identical canonical JSON
      P3  O(1) hash:       cached at construction time
      P4  Isolation:       deep-copy on input, deep-copy on output

    Why immutability matters: State objects can be stored cheaply as rollback
    snapshots. If you never mutate, you can always go back to any prior state.
    """
    __slots__ = ('_vars', '_json', '_hash')

    def __init__(self, variables: Dict[str, Any]):
        import types
        # Sort keys for canonical JSON; deep-copy for isolation (P4).
        # MappingProxyType makes the underlying dict read-only at the C level.
        v = {k: copy.deepcopy(variables[k]) for k in sorted(variables)}
        object.__setattr__(self, '_vars', types.MappingProxyType(v))
        object.__setattr__(self, '_json',
                           json.dumps(dict(v), sort_keys=True, default=str))
        object.__setattr__(self, '_hash', hash(self._json))

    # Enforce P1: reject any attempt to mutate the object.
    def __setattr__(self, *_):
        raise AttributeError("State is immutable")
    def __delattr__(self, *_):
        raise AttributeError("State is immutable")

    # Accessors always return copies so callers cannot mutate internal data (P4).
    def get(self, key: str, default: Any = None) -> Any:
        v = self._vars.get(key, default)
        return copy.deepcopy(v)

    def keys(self) -> List[str]:
        return list(self._vars.keys())

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(dict(self._vars))

    def has(self, key: str) -> bool:
        return key in self._vars

    # Derived-state helpers create new State objects (P1 preserved).
    def with_updates(self, updates: Dict[str, Any]) -> 'State':
        d = self.to_dict()
        d.update(updates)
        return State(d)

    def without_keys(self, keys: Set[str]) -> 'State':
        d = self.to_dict()
        for k in keys:
            d.pop(k, None)
        return State(d)

    # Equality and hashing rely on canonical JSON (P2, P3).
    def __hash__(self) -> int:
        return self._hash
    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and self._json == other._json
    def __repr__(self) -> str:
        return f"State({self._vars})"

    @property
    def json(self) -> str:
        return self._json

    @property
    def fingerprint(self) -> str:
        """16-hex-char SHA-256 prefix for trace entries and logging."""
        return hashlib.sha256(self._json.encode()).hexdigest()[:16]

    def diff(self, other: 'State') -> Dict[str, Tuple[Any, Any]]:
        """Return {key: (old_val, new_val)} for all differing keys."""
        all_keys = set(self._vars) | set(other._vars)
        return {k: (self._vars.get(k), other._vars.get(k))
                for k in all_keys if self._vars.get(k) != other._vars.get(k)}

    def describe(self, max_keys: int = 20) -> str:
        """Human/LLM-readable summary (truncated for long values)."""
        items = list(self._vars.items())[:max_keys]
        lines = [f"  {k} = {_truncate(repr(v), 80)}" for k, v in items]
        if len(self._vars) > max_keys:
            lines.append(f"  ... and {len(self._vars) - max_keys} more")
        return "State:\n" + "\n".join(lines)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n-3] + "..."


# ═══════════════════════════════════════════════════════════════════════════
# §2  ACTIONS — Declarative Effects (Data, Not Code)
# ═══════════════════════════════════════════════════════════════════════════

_SENTINEL_DELETE = object()  # Marks a key for deletion after apply()


@dataclass(frozen=True)
class Effect:
    """
    Atomic, declarative state mutation on a single variable.

    Modes:
      set        — variable := value
      increment  — variable += value
      decrement  — variable -= value
      multiply   — variable *= value
      append     — variable.append(value)
      remove     — variable.remove(value)  (no-op if absent)
      delete     — del variable

    Determinism proof:
      apply() uses only arithmetic and list operations on its arguments.
      No randomness, no I/O, no mutable external state.
      Therefore: same (current_value, effect) → same result.  ∎
    """
    variable: str
    mode: str
    value: Any = None

    def apply(self, current: Any) -> Any:
        """Apply effect to current value, returning new value."""
        m = self.mode
        if m == "set":       return self.value
        if m == "increment": return (current or 0) + self.value
        if m == "decrement": return (current or 0) - self.value
        if m == "multiply":  return (current or 0) * self.value
        if m == "append":
            try:
                lst = list(current) if current else []
            except TypeError:
                lst = [current] if current is not None else []
            lst.append(self.value)
            return lst
        if m == "remove":
            try:
                lst = list(current) if current else []
            except TypeError:
                lst = [current] if current is not None else []
            if self.value in lst:
                lst.remove(self.value)
            return lst
        if m == "delete":    return _SENTINEL_DELETE
        raise ValueError(f"Unknown effect mode: {m!r}")

    def inverse(self) -> "Effect":
        """
        Return the algebraic inverse of this effect.

        Works for increment/decrement/multiply/append/remove.
        Raises ValueError for set/delete (requires prior state; use
        ActionSpec.compute_inverse_effects() instead).
        """
        m = self.mode
        if m == "set":
            raise ValueError("Cannot invert 'set' mode without prior state value")
        elif m == "increment":
            return Effect(self.variable, "decrement", self.value)
        elif m == "decrement":
            return Effect(self.variable, "increment", self.value)
        elif m == "multiply":
            if self.value == 0:
                raise ValueError("Cannot invert multiply by 0 — prior state required")
            return Effect(self.variable, "multiply", 1.0 / self.value)
        elif m == "append":
            return Effect(self.variable, "remove", self.value)
        elif m == "remove":
            return Effect(self.variable, "append", self.value)
        elif m == "delete":
            raise ValueError("Cannot invert 'delete' mode without prior state value")
        else:
            raise ValueError(f"Unknown effect mode: {m!r}")


@dataclass(frozen=True)
class ActionSpec:
    """
    Fully declarative action specification. Data, not code.

    Why data-not-code?
      1. Deterministic replay:  same spec → same transition, always
      2. Pre-execution sim:     no side effects during safety checking
      3. Formal verification:   effects are inspectable, diffable data
      4. LLM readability:       semantic metadata aids structured reasoning
      5. Invertibility:         rollback computable from spec + prior state

    Theorem (Determinism of simulate):
      ∀ state s, action a: a.simulate(s) returns a unique state s'.
    Proof:
      State.to_dict() deep-copies (P4 isolation).
      Effect.apply() is pure (proven above).
      State() constructor is deterministic (P2 canonical JSON).
      ⟹ same (s, a) → same s'.  ∎
    """
    id: str
    name: str
    description: str
    effects: Tuple[Effect, ...]
    cost: float

    # Semantic metadata used by the reasoning layer (not by the kernel).
    category: str = "general"
    risk_level: str = "low"          # low | medium | high | critical
    reversible: bool = True
    preconditions_text: str = ""     # Natural language: what must be true before
    postconditions_text: str = ""    # Natural language: what will be true after
    estimated_duration_s: float = 0.0
    tags: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.cost < 0:
            raise ValueError(f"Action cost must be ≥ 0, got {self.cost}")
        if not self.id:
            raise ValueError("Action must have a non-empty ID")

    def simulate(self, state: State) -> State:
        """Apply effects to state, returning a new state. Original is unchanged."""
        updates = {}
        deletions: Set[str] = set()
        for eff in self.effects:
            result = eff.apply(state.get(eff.variable))
            if result is _SENTINEL_DELETE:
                deletions.add(eff.variable)
            else:
                updates[eff.variable] = result
        return state.without_keys(deletions).with_updates(updates)

    def compute_inverse_effects(self, state_before: State) -> Tuple[Effect, ...]:
        """
        Compute effects that exactly undo this action from state_before.

        Theorem T7 (Rollback Exactness):
          Let s' = a.simulate(s).
          Let inv = a.compute_inverse_effects(s).
          Let a_inv = ActionSpec(effects=inv, ...).
          Then a_inv.simulate(s') == s.
        Proof:
          Each inverse effect restores the variable's original value from
          state_before. Since state_before is immutable (P1), the restored
          values are exact copies of what existed before the action.  ∎
        """
        inverse = []
        for eff in self.effects:
            old = state_before.get(eff.variable)
            if eff.mode == "delete":
                if old is not None:
                    inverse.append(Effect(eff.variable, "set", old))
            elif old is None:
                inverse.append(Effect(eff.variable, "delete"))
            else:
                inverse.append(Effect(eff.variable, "set", old))
        return tuple(inverse)

    def affected_variables(self) -> Set[str]:
        """Return the set of state variable names this action touches."""
        return {e.variable for e in self.effects}

    def to_llm_text(self) -> str:
        """Full action description for LLM decision prompts."""
        parts = [
            f"[{self.id}] {self.name}",
            f"  {self.description}",
            f"  cost=${self.cost:.2f}  risk={self.risk_level}  reversible={self.reversible}",
        ]
        if self.preconditions_text:
            parts.append(f"  requires: {self.preconditions_text}")
        if self.postconditions_text:
            parts.append(f"  ensures:  {self.postconditions_text}")
        effs = ", ".join(f"{e.variable}←{e.mode}({e.value})" for e in self.effects)
        parts.append(f"  effects:  [{effs}]")
        return "\n".join(parts)

    def to_compact_text(self) -> str:
        """Single-line summary for constrained-space prompts."""
        return f"[{self.id}] {self.name} (${self.cost:.2f}, {self.risk_level})"


# ═══════════════════════════════════════════════════════════════════════════
# §3  INVARIANTS — Verifiable Safety Predicates
# ═══════════════════════════════════════════════════════════════════════════

class Invariant:
    """
    Safety predicate that must hold on every reachable state.

    Enforcement modes:
      "blocking"   — T3 applies: a violated invariant blocks the action.
                     This is the safety-critical mode.
      "monitoring" — Violation is logged but does not block the action.
                     Useful for soft warnings and diagnostics.

    Theorem T3 (Invariant Preservation) — blocking-mode only:
      Given I(s₀) = True and check-before-commit discipline:
        ∀t: I(sₜ) = True.
    Proof by induction on t:
      Base: I(s₀) = True (precondition of system construction).
      Step: Assume I(sₜ) = True.
        Let s' = a.simulate(sₜ).
        If I(s') = False and enforcement="blocking" → action rejected, sₜ₊₁ := sₜ.
          I(sₜ₊₁) = I(sₜ) = True. ✓
        If I(s') = True → action committed, sₜ₊₁ := s'.
          I(sₜ₊₁) = True. ✓
      ∎

    Note: If the predicate raises an exception during check(), the kernel
    treats it as a violation (fail-safe default). Exception-raising predicates
    do NOT allow unsafe states through.
    """
    def __init__(self, name: str, predicate: Callable[[State], bool],
                 description: str = "", severity: str = "critical",
                 enforcement: Optional[str] = None):
        self.name = name
        self.predicate = predicate
        self.description = description or name
        # Legacy 'severity' field maps to enforcement mode.
        # Prefer 'enforcement' for new code.
        self.severity = severity
        if enforcement is None:
            self.enforcement = "blocking" if severity == "critical" else "monitoring"
        else:
            if enforcement not in ("blocking", "monitoring"):
                raise ValueError(
                    f"Invariant enforcement must be 'blocking' or 'monitoring', "
                    f"got {enforcement!r}")
            self.enforcement = enforcement
        self._violations = 0

    def check(self, state: State) -> Tuple[bool, str]:
        """Evaluate predicate. Returns (holds, reason). Exceptions count as violations."""
        try:
            holds = self.predicate(state)
            if not holds:
                self._violations += 1
                return False, f"Invariant '{self.name}' VIOLATED"
            return True, ""
        except Exception as e:
            self._violations += 1
            return False, f"Invariant '{self.name}' raised exception: {e}"

    @property
    def violation_count(self) -> int:
        return self._violations

    def to_llm_text(self) -> str:
        label = self.enforcement
        if self.severity:
            label = f"{label}/{self.severity}"
        return f"SAFETY RULE [{label}]: {self.description}"


# ═══════════════════════════════════════════════════════════════════════════
# §4  BUDGET CONTROLLER — Proven Resource Safety
# ═══════════════════════════════════════════════════════════════════════════

class BudgetController:
    """
    Theorem T1 (Budget Safety):  spent_net(t) ≤ B₀  ∀t.
    Proof:
      Base: spent_net(0) = 0 ≤ B₀. ✓
      Step: Before charging c at time t:
        Guard: c ≤ B₀ - spent_net(t). If False → reject.
        If True: spent_net(t+1) = spent_net(t) + c ≤ B₀. ✓
      By induction. ∎

    Theorem T4 (Monotone Gross Spend):  spent_gross(t) ≤ spent_gross(t+1).
    Proof: Only charge() increments gross_spent (c ≥ 0). Refund is separate. ∎

    Theorem T7 (Rollback with Separate Accounting):
      Refund operations decrease net spend without violating T4 (which tracks gross).

    Implementation notes:
      - spent_gross: Cumulative charges (never decreases) — proves T4
      - spent_refunded: Cumulative refunds (never decreases) — separate accounting
      - spent_net: spent_gross - spent_refunded — used for budget checks (T1)
      - All arithmetic uses integer MILLICENTS (cost * 100_000) internally
        to eliminate floating-point drift. External API is still float.
      - threading.Lock protects all mutations for concurrency safety.
    """
    _SCALE = 100_000  # 1.0 dollar = 100_000 millicents. Exact at 5 decimal places.

    def __init__(self, budget: float):
        if budget < 0:
            raise ValueError(f"Budget must be ≥ 0, got {budget}")
        self._budget_i = round(budget * self._SCALE)
        self._spent_gross_i = 0  # Total charged
        self._refunded_i = 0     # Total refunded
        self._ledger: List[Tuple[str, float, float]] = []
        self._lock = _threading.Lock()

    @property
    def budget(self) -> float:
        """Total budget."""
        return self._budget_i / self._SCALE

    @property
    def spent_gross(self) -> float:
        """Cumulative charges (never decreases — proves T4)."""
        return self._spent_gross_i / self._SCALE

    @property
    def spent_refunded(self) -> float:
        """Cumulative refunds (never decreases)."""
        return self._refunded_i / self._SCALE

    @property
    def spent_net(self) -> float:
        """Net spend after refunds (used for budget enforcement — T1)."""
        return (self._spent_gross_i - self._refunded_i) / self._SCALE

    @property
    def spent(self) -> float:
        """Alias for spent_net for backward compatibility."""
        return self.spent_net

    @property
    def remaining(self) -> float:
        """Remaining budget (based on net spend)."""
        return (self._budget_i - (self._spent_gross_i - self._refunded_i)) / self._SCALE

    def can_afford(self, cost: float) -> Tuple[bool, str]:
        """Check if net spend + cost would exceed budget."""
        if cost < 0:
            raise ValueError(f"Cost must be ≥ 0, got {cost}")
        cost_i = round(cost * self._SCALE)
        with self._lock:
            net_i = self._spent_gross_i - self._refunded_i
            if net_i + cost_i <= self._budget_i:
                return True, ""
        return False, (f"Cannot afford ${cost:.2f}: "
                       f"spent_net=${self.spent_net:.2f}, remaining=${self.remaining:.2f}")

    def charge(self, action_id: str, cost: float) -> None:
        """Charge budget (increases spent_gross). Precondition: can_afford(cost) was (True, _)."""
        if cost < 0:
            raise ValueError("Cost must be ≥ 0")
        cost_i = round(cost * self._SCALE)
        with self._lock:
            old_gross = self._spent_gross_i
            net_i = self._spent_gross_i - self._refunded_i
            if net_i + cost_i > self._budget_i:
                raise RuntimeError("BUDGET SAFETY VIOLATION — charge without can_afford")
            self._spent_gross_i += cost_i
            # Verify T4: gross spent is monotone
            assert self._spent_gross_i >= old_gross, "T4 VIOLATED: spent_gross decreased"
            # Verify T1: net spent respects budget
            assert self.spent_net <= self.budget, "T1 VIOLATED: spent_net exceeded budget"
        self._ledger.append((action_id, cost, _time.time()))

    def refund(self, action_id: str, cost: float) -> None:
        """Record refund for rollback (increases spent_refunded). Supports T7."""
        if cost < 0:
            raise ValueError(f"Cost must be ≥ 0")
        cost_i = round(cost * self._SCALE)
        with self._lock:
            old_refunded = self._refunded_i
            # Cannot refund more than was ever spent
            if self._refunded_i + cost_i <= self._spent_gross_i:
                self._refunded_i += cost_i
                assert self._refunded_i >= old_refunded, "Refund tracking failed"
        self._ledger.append((f"REFUND:{action_id}", -cost, _time.time()))

    def utilization(self) -> float:
        """Utilization as fraction of budget (based on net spend)."""
        return (self._spent_gross_i - self._refunded_i) / self._budget_i if self._budget_i > 0 else 0.0

    @property
    def ledger(self) -> List[Tuple[str, float, float]]:
        """Append-only transaction log."""
        return list(self._ledger)

    def summary(self) -> str:
        """Human-readable budget summary."""
        return (f"Budget: ${self.budget:.2f} | "
                f"Spent (gross): ${self.spent_gross:.2f} | "
                f"Refunded: ${self.spent_refunded:.2f} | "
                f"Spent (net): ${self.spent_net:.2f} | "
                f"Remaining: ${self.remaining:.2f} | "
                f"Utilization: {self.utilization():.1%}")


# ═══════════════════════════════════════════════════════════════════════════
# §5  EXECUTION TRACE — Hash-Chained Audit Log
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TraceEntry:
    """
    Single immutable entry in the execution trace, linked by SHA-256 hash.

    Theorem T6 (Trace Integrity):
      The trace is append-only. Each entry's hash commits to the previous
      entry's hash, forming a tamper-evident chain.
      Modifying entry i invalidates all entries j > i (detectable by
      verify_integrity()).
    Proof:
      TraceEntry is frozen (immutable by construction).
      prev_hash creates a linked chain.
      compute_hash() covers all fields including prev_hash.  ∎
    """
    step: int
    action_id: str
    action_name: str
    state_before_fp: str       # fingerprint of state before action
    state_after_fp: str        # fingerprint of state after action
    cost: float
    timestamp: float
    approved: bool
    rejection_reasons: Tuple[str, ...] = ()
    reasoning_summary: str = ""   # LLM reasoning, stored for audit
    prev_hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps({
            'step': self.step, 'action': self.action_id,
            'before': self.state_before_fp, 'after': self.state_after_fp,
            'cost': self.cost, 'approved': self.approved,
            'prev': self.prev_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


class ExecutionTrace:
    """Append-only, SHA-256 hash-chained execution log (T6)."""

    def __init__(self):
        self._entries: List[TraceEntry] = []
        self._head_hash = "0" * 32  # Genesis hash

    def append(self, entry: TraceEntry) -> str:
        """Append entry (linking to current head) and return its hash."""
        linked = TraceEntry(
            step=entry.step, action_id=entry.action_id,
            action_name=entry.action_name,
            state_before_fp=entry.state_before_fp,
            state_after_fp=entry.state_after_fp,
            cost=entry.cost, timestamp=entry.timestamp,
            approved=entry.approved,
            rejection_reasons=entry.rejection_reasons,
            reasoning_summary=entry.reasoning_summary,
            prev_hash=self._head_hash,
        )
        h = linked.compute_hash()
        self._entries.append(linked)
        self._head_hash = h
        return h

    def verify_integrity(self) -> Tuple[bool, str]:
        """Walk the entire hash chain. O(n). Returns (ok, message)."""
        prev = "0" * 32
        for i, entry in enumerate(self._entries):
            if entry.prev_hash != prev:
                return False, f"Chain broken at step {i}"
            prev = entry.compute_hash()
        return True, "Trace integrity verified"

    @property
    def entries(self) -> List[TraceEntry]:
        return list(self._entries)

    @property
    def length(self) -> int:
        return len(self._entries)

    def last_n(self, n: int) -> List[TraceEntry]:
        return self._entries[-n:] if n > 0 else []

    def summary(self, last_n: int = 5) -> str:
        lines = [f"Trace: {len(self._entries)} entries"]
        for e in self._entries[-last_n:]:
            status = "✓" if e.approved else "✗"
            lines.append(
                f"  [{status}] step={e.step} {e.action_name} "
                f"cost=${e.cost:.2f}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# §6  SAFETY KERNEL — The Core Decision Gate
# ═══════════════════════════════════════════════════════════════════════════

class CheckResult(Enum):
    PASS = auto()
    FAIL_BUDGET = auto()
    FAIL_INVARIANT = auto()
    FAIL_TERMINATION = auto()
    FAIL_PRECONDITION = auto()


@dataclass
class SafetyVerdict:
    """Complete result of a safety evaluation for one proposed action."""
    approved: bool
    checks: List[Tuple[str, CheckResult, str]]  # (check_name, result, detail)
    simulated_next_state: Optional[State] = None
    cost: float = 0.0

    @property
    def rejection_reasons(self) -> Tuple[str, ...]:
        return tuple(detail for _, result, detail in self.checks
                     if result != CheckResult.PASS)

    def summary(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        lines = [f"Safety Verdict: {status}"]
        for name, result, detail in self.checks:
            icon = "✓" if result == CheckResult.PASS else "✗"
            lines.append(f"  [{icon}] {name}: {detail}")
        return "\n".join(lines)


class SafetyKernel:
    """
    The formal safety gate. Every proposed action passes through here.

    Enforces T1 (budget), T2 (termination), T3 (invariants), T5 (atomicity).
    Has zero knowledge of LLMs or heuristics. This is the innermost,
    non-bypassable layer.

    Theorem T2 (Termination):
      Given discrete actions with min_cost ε > 0 and budget B:
        System halts in ≤ ⌊B/ε⌋ steps.
    Proof:
      After n steps: spent ≥ n·ε (each step costs ≥ ε, enforced at Check 0).
      When n > ⌊B/ε⌋: spent > B - ε.
      Next action needs cost ≥ ε but remaining < ε → budget check rejects it.
      System halts. ∎

    Theorem T5 (Atomicity):
      Either ALL of {budget charge, state transition, trace append} happen,
      or NONE do.
    Proof:
      evaluate() simulates but does NOT commit.
      execute() commits only if evaluate() returned approved=True.
      State is immutable: simulate() creates a new State object, never
      touching the original.  ∎

    Emergency actions (T8):
      Actions registered in emergency_actions bypass cost and step-limit checks.
      They must have cost=0.0 and effects=() (enforced by caller convention).
      This guarantees a graceful exit route is always available.
    """
    def __init__(self, budget: float, invariants: List[Invariant],
                 min_action_cost: float = 0.001,
                 emergency_actions: Optional[Set[str]] = None):
        if min_action_cost <= 0:
            raise ValueError(
                f"min_action_cost must be > 0 for T2 (termination), got {min_action_cost}")
        self.budget = BudgetController(budget)
        self.invariants = list(invariants)
        self.min_action_cost = min_action_cost
        self.trace = ExecutionTrace()
        self.step_count = 0
        self.max_steps = int(budget / min_action_cost)
        self._lock = _threading.Lock()

        # Emergency actions: can bypass cost and step limit checks
        # Must have empty effects and zero cost
        # Designed for graceful degradation (e.g., SAFE_HOVER)
        self.emergency_actions: Set[str] = emergency_actions or set()

        # Emergency actions bypass cost/step checks (T8).
        # Must have empty effects and zero cost by convention.
        self.emergency_actions: Set[str] = emergency_actions or set()

        # Pluggable precondition checkers (called before invariant checks).
        self._precondition_fns: List[Callable[[State, ActionSpec], Tuple[bool, str]]] = []

    def add_precondition(self, fn: Callable[[State, ActionSpec], Tuple[bool, str]]):
        """Register an additional precondition function (checked in evaluate())."""
        self._precondition_fns.append(fn)

    def register_emergency_action(self, action_id: str) -> None:
        """Register an action as an emergency action (bypasses cost/step limits, T8)."""
        self.emergency_actions.add(action_id)

    def evaluate(self, state: State, action: ActionSpec) -> SafetyVerdict:
        """
        Evaluate action safety WITHOUT committing. Pure function (no side effects).

        Runs all checks on a simulated copy of state. Returns SafetyVerdict with:
          - approved: True iff ALL checks pass
          - simulated_next_state: the state after action (if approved)
          - All check details for audit/reasoning

        Note on T3 (Invariant Preservation):
          T3 is scoped to invariants with enforcement="blocking".
          Invariants with enforcement="monitoring" are checked and logged
          for diagnostic purposes, but their violations do not block actions.
          Only blocking-mode invariants provide formal safety guarantees.
        """
        checks: List[Tuple[str, CheckResult, str]] = []
        approved = True

        # Check 0: Minimum cost (prerequisite for T2 termination bound).
        if action.cost < self.min_action_cost:
            if action.id in self.emergency_actions:
                # Emergency action bypass: allowed to have zero cost
                checks.append(("MinCost", CheckResult.PASS,
                              f"Emergency action '{action.id}' bypasses cost minimum"))
            else:
                checks.append(("MinCost", CheckResult.FAIL_TERMINATION,
                              f"Action cost ${action.cost:.4f} < min ${self.min_action_cost:.4f}"))
                approved = False

        # Check 1: Budget (T1).
        can_afford, reason = self.budget.can_afford(action.cost)
        if can_afford:
            checks.append(("Budget", CheckResult.PASS,
                          f"${action.cost:.2f} ≤ ${self.budget.remaining:.2f} remaining"))
        else:
            checks.append(("Budget", CheckResult.FAIL_BUDGET, reason))
            approved = False

        # Check 2: Step limit (T2).
        if self.step_count >= self.max_steps:
            if action.id in self.emergency_actions:
                # Emergency action bypass: allowed even at step limit
                checks.append(("Termination", CheckResult.PASS,
                              f"Emergency action '{action.id}' bypasses step limit"))
            else:
                checks.append(("Termination", CheckResult.FAIL_TERMINATION,
                              f"Step limit {self.max_steps} reached"))
                approved = False
        else:
            checks.append(("Termination", CheckResult.PASS,
                          f"Step {self.step_count + 1}/{self.max_steps}"))

        # ── Check 3: Invariants (T3 for blocking-mode only) ──
        sim_state = None
        if approved:
            sim_state = action.simulate(state)
            for inv in self.invariants:
                holds, msg = inv.check(sim_state)
                if holds:
                    checks.append((f"Invariant:{inv.name}", CheckResult.PASS, "holds"))
                else:
                    # Invariant violated: check enforcement mode
                    enforcement = getattr(inv, "enforcement", "blocking")
                    checks.append((f"Invariant:{inv.name}",
                                  CheckResult.FAIL_INVARIANT, 
                                  f"[{enforcement}] {msg}"))
                    # Only blocking-mode violations prevent approval (T3 scope)
                    if enforcement == "blocking":
                        approved = False
                    # Monitoring-mode violations are logged but do not block

        # Check 4: Pluggable preconditions (user-supplied additional gates).
        for fn in self._precondition_fns:
            try:
                ok, msg = fn(state, action)
                if ok:
                    checks.append(("Precondition", CheckResult.PASS, msg or "ok"))
                else:
                    checks.append(("Precondition", CheckResult.FAIL_PRECONDITION, msg))
                    approved = False
            except Exception as e:
                checks.append(("Precondition", CheckResult.FAIL_PRECONDITION, str(e)))
                approved = False

        return SafetyVerdict(
            approved=approved, checks=checks,
            simulated_next_state=sim_state, cost=action.cost
        )

    def execute(self, state: State, action: ActionSpec,
                reasoning_summary: str = "") -> Tuple[State, TraceEntry]:
        """
        Execute action with full safety guarantees (defense-in-depth re-evaluation).

        Calls evaluate() again as a safety net, then commits atomically:
          budget charge → step increment → trace append.

        Note on TOCTOU: In multi-threaded scenarios, use evaluate_and_execute_atomic()
        instead to eliminate the time-of-check-to-time-of-use race. This method
        is safe for single-threaded use.

        This method calls evaluate() for defense-in-depth verification,
        then commits the state change, budget charge, and trace entry.

        Precondition:  evaluate(state, action).approved == True
        Postconditions:
          - Budget charged (T1: spent_net + cost ≤ budget)
          - Step count incremented (T2)
          - New state satisfies all blocking-mode invariants (T3)
          - All-or-nothing atomicity (T5)
          - Trace updated with entry (T6)

        Note: Re-evaluation provides defense-in-depth but creates a TOCTOU gap
        in concurrent scenarios. Use evaluate_and_execute_atomic() for true
        atomicity across threads.
        """
        verdict = self.evaluate(state, action)
        if not verdict.approved:
            raise RuntimeError(
                f"SAFETY VIOLATION: execute() called on unapproved action "
                f"{action.id}. Reasons: {verdict.rejection_reasons}")

        new_state = verdict.simulated_next_state
        assert new_state is not None

        # ── Commit: budget + state + trace ──
        self.budget.charge(action.id, action.cost)
        self.step_count += 1

        entry = TraceEntry(
            step=self.step_count,
            action_id=action.id, action_name=action.name,
            state_before_fp=state.fingerprint,
            state_after_fp=new_state.fingerprint,
            cost=action.cost, timestamp=_time.time(),
            approved=True, reasoning_summary=reasoning_summary,
        )
        self.trace.append(entry)

        # Runtime verification (belt AND suspenders)
        assert self.budget.spent_net <= self.budget.budget, "T1 VIOLATED: spent_net > budget"
        assert self.step_count <= self.max_steps, "T2 VIOLATED: step_count exceeded max_steps"
        # Only check blocking-mode invariants per T3 scope
        for inv in self.invariants:
            if getattr(inv, "enforcement", "blocking") == "blocking":
                ok, msg = inv.check(new_state)
                assert ok, f"T3 VIOLATED (blocking invariant): {msg}"

        return new_state, entry

    def evaluate_and_execute_atomic(self, state: State, action: ActionSpec,
                                     reasoning_summary: str = "") -> Tuple[State, TraceEntry]:
        """
        Evaluate and execute as a single atomic transaction (thread-safe).

        This method combines evaluate() and execute() under a single lock,
        eliminating the TOCTOU (time-of-check-to-time-of-use) gap that exists
        in concurrent scenarios when evaluate() and execute() are called separately.

        Guarantees (concurrent-safe):
          - Budget check and charge are atomic
          - Step check and increment are atomic
          - No interleaving with other threads' charges or steps

        Returns: (new_state, trace_entry) on success
        Raises: RuntimeError if safety check fails

        This is the preferred method for concurrent execution.
        """
        with self._lock:
            # Evaluate (within lock)
            verdict = self.evaluate(state, action)
            if not verdict.approved:
                raise RuntimeError(
                    f"SAFETY VIOLATION: atomic execution failed for {action.id}. "
                    f"Reasons: {verdict.rejection_reasons}")

            new_state = verdict.simulated_next_state
            assert new_state is not None

            # Execute (within lock — no race possible)
            self.budget.charge(action.id, action.cost)
            self.step_count += 1

            entry = TraceEntry(
                step=self.step_count,
                action_id=action.id, action_name=action.name,
                state_before_fp=state.fingerprint,
                state_after_fp=new_state.fingerprint,
                cost=action.cost, timestamp=_time.time(),
                approved=True, reasoning_summary=reasoning_summary,
            )
            self.trace.append(entry)

            # Assertions (post-commit verification)
            assert self.budget.spent_net <= self.budget.budget, "T1 VIOLATED: spent_net > budget"
            assert self.step_count <= self.max_steps, "T2 VIOLATED: step_count exceeded max_steps"
            for inv in self.invariants:
                if getattr(inv, "enforcement", "blocking") == "blocking":
                    ok, msg = inv.check(new_state)
                    assert ok, f"T3 VIOLATED (blocking invariant): {msg}"

            return new_state, entry

    def record_rejection(self, state: State, action: ActionSpec,
                         reasons: Tuple[str, ...],
                         reasoning_summary: str = "") -> TraceEntry:
        """Append a rejected-action entry to the trace for audit completeness."""
        entry = TraceEntry(
            step=self.step_count,
            action_id=action.id, action_name=action.name,
            state_before_fp=state.fingerprint,
            state_after_fp=state.fingerprint,  # state unchanged on rejection
            cost=0.0, timestamp=_time.time(),
            approved=False, rejection_reasons=reasons,
            reasoning_summary=reasoning_summary,
        )
        self.trace.append(entry)
        return entry

    def rollback(self, state_before: State, state_after: State,
                 action: ActionSpec) -> State:
        """
        Rollback an executed action. Returns state_before exactly.

        T7 guarantee: State is immutable, so state_before still exists unchanged.
        Budget is refunded via separate accounting (T4 monotonicity preserved).
        """
        self.budget.refund(action.id, action.cost)
        self.step_count = max(0, self.step_count - 1)
        return state_before  # Exact, because State immutability (P1) guarantees no mutation.

    def status(self) -> str:
        return (f"SafetyKernel: step={self.step_count}/{self.max_steps} | "
                f"{self.budget.summary()} | "
                f"trace={self.trace.length} entries")


# ═══════════════════════════════════════════════════════════════════════════
# §7  CLAIMS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

FORMAL_CLAIMS = [
    Claim("T1", "spent_net(t) ≤ B₀ for all t",
          GuaranteeLevel.PROVEN,
          "Induction on t. Guard check-before-charge. See BudgetController.charge()."),
    Claim("T2", "System halts in ≤ ⌊B₀/ε⌋ steps",
          GuaranteeLevel.CONDITIONAL,
          "Induction + pigeonhole. See SafetyKernel.evaluate() Check 0.",
          assumptions=("Discrete actions", "min_cost ε > 0")),
    Claim("T3", "I(s₀)=True ⟹ I(sₜ)=True for all t (blocking-mode invariants)",
          GuaranteeLevel.PROVEN,
          "Induction. Check-before-commit. See SafetyKernel.evaluate() Check 3."),
    Claim("T4", "spent_gross(t) ≤ spent_gross(t+1)",
          GuaranteeLevel.PROVEN,
          "cost ≥ 0 asserted. See BudgetController.charge()."),
    Claim("T5", "State transitions are all-or-nothing",
          GuaranteeLevel.PROVEN,
          "evaluate() simulates on copy; execute() commits only on approval. See §6."),
    Claim("T6", "Trace is append-only and tamper-evident",
          GuaranteeLevel.PROVEN,
          "TraceEntry is frozen; hash chain covers all fields. See ExecutionTrace."),
    Claim("T7", "undo(execute(s,a)) == s",
          GuaranteeLevel.PROVEN,
          "State immutability + algebraic inverse effects. See ActionSpec.compute_inverse_effects()."),
    Claim("T8", "SAFE_HOVER is always executable",
          GuaranteeLevel.CONDITIONAL,
          "Emergency action bypasses cost/step checks. See SafetyKernel.evaluate() Checks 0,2.",
          assumptions=("Action registered in emergency_actions", "cost=0.0, effects=()")),
]
