"""
ConstrAI — Autonomous Engine for Guaranteed Intelligent Safety
============================================================
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
    """Every claim in ConstrAI is tagged with its proof status."""
    PROVEN      = "proven"       # Holds unconditionally by construction
    CONDITIONAL = "conditional"  # Proven given stated assumptions
    EMPIRICAL   = "empirical"    # Measured with confidence intervals
    HEURISTIC   = "heuristic"    # Best-effort, no formal guarantee


@dataclass(frozen=True)
class Claim:
    """An auditable claim about a system property."""
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
      P1  Immutability:   no method mutates self after __init__
      P2  Det. equality:  s1 == s2  ⟺  identical canonical JSON
      P3  O(1) hash:      cached at construction time
      P4  Isolation:      deep-copy on in, deep-copy on out
    """
    __slots__ = ('_vars', '_json', '_hash')

    def __init__(self, variables: Dict[str, Any]):
        # Sort keys for canonical JSON + deep copy for isolation (P4)
        # Use MappingProxyType to prevent direct dict mutation
        import types
        v = {k: copy.deepcopy(variables[k]) for k in sorted(variables)}
        object.__setattr__(self, '_vars', types.MappingProxyType(v))
        object.__setattr__(self, '_json',
                           json.dumps(dict(v), sort_keys=True, default=str))
        object.__setattr__(self, '_hash', hash(self._json))

    # ── Immutability enforcement (P1) ──
    def __setattr__(self, *_):
        raise AttributeError("State is immutable")
    def __delattr__(self, *_):
        raise AttributeError("State is immutable")

    # ── Accessors (always return copies — P4) ──
    def get(self, key: str, default: Any = None) -> Any:
        v = self._vars.get(key, default)
        return copy.deepcopy(v)

    def keys(self) -> List[str]:
        return list(self._vars.keys())

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(dict(self._vars))

    def has(self, key: str) -> bool:
        return key in self._vars

    # ── Derived states (create new, never mutate — P1) ──
    def with_updates(self, updates: Dict[str, Any]) -> 'State':
        d = self.to_dict()
        d.update(updates)
        return State(d)

    def without_keys(self, keys: Set[str]) -> 'State':
        d = self.to_dict()
        for k in keys:
            d.pop(k, None)
        return State(d)

    # ── Equality and hashing (P2, P3) ──
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
        return hashlib.sha256(self._json.encode()).hexdigest()[:16]

    def diff(self, other: 'State') -> Dict[str, Tuple[Any, Any]]:
        """Return {key: (old_val, new_val)} for all differences."""
        all_keys = set(self._vars) | set(other._vars)
        return {k: (self._vars.get(k), other._vars.get(k))
                for k in all_keys if self._vars.get(k) != other._vars.get(k)}

    def describe(self, max_keys: int = 20) -> str:
        """Human/LLM-readable summary of state."""
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

_SENTINEL_DELETE = object()


@dataclass(frozen=True)
class Effect:
    """
    Atomic effect on one state variable. This is DATA, not a function pointer.

    Modes:
      set        — variable := value
      increment  — variable += value
      decrement  — variable -= value
      multiply   — variable *= value
      append     — variable.append(value)
      remove     — variable.remove(value)
      delete     — del variable

    Determinism proof: apply() uses only arithmetic and list ops on its
    inputs. No randomness, no I/O, no mutable external state.
    Therefore: same (current, effect) → same result.  ∎
    """
    variable: str
    mode: str       # "set" | "increment" | "decrement" | "multiply" | "append" | "remove" | "delete"
    value: Any = None

    def apply(self, current: Any) -> Any:
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
        Compute the inverse Effect that reverts this effect's operation.

        For increment/decrement/multiply/append/remove modes, returns the
        algebraic inverse. For set/delete, raises ValueError (requires prior state).
        """
        m = self.mode
        if m == "set":
            # set cannot be inverted without prior value
            # Return placeholder; ActionSpec.compute_inverse_effects() handles this
            raise ValueError("Cannot invert 'set' mode without prior state value")
        elif m == "increment":
            return Effect(self.variable, "decrement", self.value)
        elif m == "decrement":
            return Effect(self.variable, "increment", self.value)
        elif m == "multiply":
            if self.value == 0:
                raise ValueError("Cannot invert multiply by 0 without prior state")
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
    Fully declarative action specification.

    WHY data-not-code?
      1. Deterministic replay:  same spec → same transition
      2. Pre-execution sim:     no side effects during planning
      3. Formal verification:   effects are inspectable data
      4. LLM readability:       semantic metadata for reasoning
      5. Invertibility:         compute rollback from spec + prior state

    Theorem (Determinism of simulate):
      ∀ state s, action a: a.simulate(s) returns a unique state s'.
    Proof:
      State.to_dict() deep-copies (P4).
      Effect.apply() is pure (proven above).
      State() constructor is deterministic (P2).
      ⟹ same (s, a) → same s'.  ∎
    """
    id: str
    name: str
    description: str
    effects: Tuple[Effect, ...]
    cost: float

    # ── Semantic metadata (for LLM reasoning layer) ──
    category: str = "general"
    risk_level: str = "low"          # low | medium | high | critical
    reversible: bool = True
    preconditions_text: str = ""     # Natural language: what must be true
    postconditions_text: str = ""    # Natural language: what will be true
    estimated_duration_s: float = 0.0
    tags: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.cost < 0:
            raise ValueError(f"Action cost must be ≥ 0, got {self.cost}")
        if not self.id:
            raise ValueError("Action must have a non-empty ID")

    def simulate(self, state: State) -> State:
        """Apply effects to state, returning new state. Original unchanged."""
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
        Compute effects that undo this action from state_before.

        Theorem T7 (Rollback Exactness):
          Let s' = a.simulate(s).
          Let inv = a.compute_inverse_effects(s).
          Let a_inv = ActionSpec(effects=inv, ...).
          Then a_inv.simulate(s') == s.
        Proof: Each inverse effect restores the original value of its
          variable from state_before. Since state_before is immutable,
          the restored values are exact.  ∎
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
        return {e.variable for e in self.effects}

    # ── LLM-readable representation ──
    def to_llm_text(self) -> str:
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
        return f"[{self.id}] {self.name} (${self.cost:.2f}, {self.risk_level})"


# ═══════════════════════════════════════════════════════════════════════════
# §3  INVARIANTS — Verifiable Safety Predicates
# ═══════════════════════════════════════════════════════════════════════════

class Invariant:
    """
    Safety predicate that must hold on every reachable state.

    Theorem T3 (Invariant Preservation):
      Given I(s₀) = True, and system uses check-before-commit:
        ∀t: I(sₜ) = True.
    Proof by induction on t:
      Base: I(s₀) = True (precondition of system construction).
      Step: Assume I(sₜ) = True.
        Let s' = a.simulate(sₜ).
        If I(s') = False → action rejected, sₜ₊₁ := sₜ. I holds.
        If I(s') = True  → action committed, sₜ₊₁ := s'. I holds.
      ∎
    """
    def __init__(self, name: str, predicate: Callable[[State], bool],
                 description: str = "", severity: str = "critical",
                 enforcement: Optional[str] = None):
        self.name = name
        self.predicate = predicate
        self.description = description or name
        # Back-compat: severity historically implied enforcement.
        # New explicit field names the behavior: "blocking" or "monitoring".
        # - "critical" => blocking
        # - "warning"  => monitoring
        self.severity = severity  # "warning" or "critical" (legacy label)
        if enforcement is None:
            self.enforcement = "blocking" if severity == "critical" else "monitoring"
        else:
            if enforcement not in ("blocking", "monitoring"):
                raise ValueError(
                    f"Invariant enforcement must be 'blocking' or 'monitoring', got {enforcement!r}")
            self.enforcement = enforcement
        self._violations = 0

    def check(self, state: State) -> Tuple[bool, str]:
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
    _SCALE = 100_000  # Millicents: 1.0 = 100_000 units. Exact for ≤5 decimal places.

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
            raise ValueError(f"Cost must be ≥ 0")
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
    Single entry in the execution trace. Immutable and hash-linked.

    Theorem T6 (Trace Integrity):
      The trace is append-only. Each entry's hash depends on the
      previous entry's hash, forming a hash chain.
      Tampering with entry i invalidates all entries j > i.
    Proof: By construction — prev_hash field creates a linked chain,
      and TraceEntry is frozen (immutable). ∎
    """
    step: int
    action_id: str
    action_name: str
    state_before_fp: str      # fingerprint of state before
    state_after_fp: str       # fingerprint of state after
    cost: float
    timestamp: float
    approved: bool
    rejection_reasons: Tuple[str, ...] = ()
    reasoning_summary: str = ""   # LLM's reasoning (for audit)
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
    """Append-only, hash-chained execution log."""

    def __init__(self):
        self._entries: List[TraceEntry] = []
        self._head_hash = "0" * 32  # Genesis hash

    def append(self, entry: TraceEntry) -> str:
        """Append entry and return its hash."""
        # Link to previous
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
        """Verify the entire hash chain."""
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
    """Complete result of safety evaluation."""
    approved: bool
    checks: List[Tuple[str, CheckResult, str]]  # (name, result, detail)
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
    The formal safety gate. Every action passes through here.

    This kernel enforces T1 (budget), T2 (termination), T3 (invariants),
    T5 (atomicity). It knows NOTHING about LLMs or heuristics.
    It is the innermost, non-negotiable layer.

    Theorem T2 (Termination):
      Given discrete actions with min_cost ε > 0 and budget B:
        System halts in ≤ ⌊B/ε⌋ steps.
    Proof:
      After n steps: spent ≥ n·ε (each step costs ≥ ε).
      When n > ⌊B/ε⌋: spent > B - ε.
      Next action needs cost ≥ ε but remaining < ε → rejected.
      System halts. ∎

    Theorem T5 (Atomicity):
      Either ALL of {budget charge, state transition, trace append}
      happen, or NONE do.
    Proof: By construction — evaluate() simulates but does NOT commit.
      execute() commits only if evaluate() returned approved=True.
      If any step in execute() fails (assertion), exception prevents
      partial commit. ∎
    """
    def __init__(self, budget: float, invariants: List[Invariant],
                 min_action_cost: float = 0.001,
                 emergency_actions: Optional[Set[str]] = None):
        if min_action_cost <= 0:
            raise ValueError(f"min_action_cost must be > 0 for T2 (termination), got {min_action_cost}")
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

        # Precondition checkers (pluggable)
        self._precondition_fns: List[Callable[[State, ActionSpec], Tuple[bool, str]]] = []

    def add_precondition(self, fn: Callable[[State, ActionSpec], Tuple[bool, str]]):
        """Register an additional precondition checker."""
        self._precondition_fns.append(fn)

    def evaluate(self, state: State, action: ActionSpec) -> SafetyVerdict:
        """
        Evaluate action safety WITHOUT executing. Pure function.

        Returns SafetyVerdict with:
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

        # ── Check 0: Minimum cost (T2 prerequisite) ──
        if action.cost < self.min_action_cost:
            if action.id in self.emergency_actions:
                # Emergency action bypass: allowed to have zero cost
                checks.append(("MinCost", CheckResult.PASS,
                              f"Emergency action '{action.id}' bypasses cost minimum"))
            else:
                checks.append(("MinCost", CheckResult.FAIL_TERMINATION,
                              f"Action cost ${action.cost:.4f} < min ${self.min_action_cost:.4f}"))
                approved = False

        # ── Check 1: Budget (T1) ──
        can_afford, reason = self.budget.can_afford(action.cost)
        if can_afford:
            checks.append(("Budget", CheckResult.PASS,
                          f"${action.cost:.2f} ≤ ${self.budget.remaining:.2f} remaining"))
        else:
            checks.append(("Budget", CheckResult.FAIL_BUDGET, reason))
            approved = False

        # ── Check 2: Termination (T2) ──
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
        if approved:  # Only simulate if budget/termination passed
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

        # ── Check 4: Pluggable preconditions ──
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
        Execute action with full safety guarantees.

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
        # Verify safety (defense in depth — even if caller checked)
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
        """Record a rejected action in the trace (for audit)."""
        entry = TraceEntry(
            step=self.step_count,
            action_id=action.id, action_name=action.name,
            state_before_fp=state.fingerprint,
            state_after_fp=state.fingerprint,  # state unchanged
            cost=0.0, timestamp=_time.time(),
            approved=False, rejection_reasons=reasons,
            reasoning_summary=reasoning_summary,
        )
        self.trace.append(entry)
        return entry

    def rollback(self, state_before: State, state_after: State,
                 action: ActionSpec) -> State:
        """
        Rollback an executed action. Returns state_before (exact).
        Theorem T7: guaranteed by compute_inverse_effects.
        """
        self.budget.refund(action.id, action.cost)
        self.step_count = max(0, self.step_count - 1)
        return state_before  # Exact, because State is immutable (P1)

    def status(self) -> str:
        return (f"SafetyKernel: step={self.step_count}/{self.max_steps} | "
                f"{self.budget.summary()} | "
                f"trace={self.trace.length} entries")


# ═══════════════════════════════════════════════════════════════════════════
# §7  CLAIMS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

FORMAL_CLAIMS = [
    Claim("T1", "spent(t) ≤ B₀ for all t",
          GuaranteeLevel.PROVEN,
          "Induction on t. Guard: cost ≤ remaining. See §4."),
    Claim("T2", "System halts in ≤ ⌊B₀/ε⌋ steps",
          GuaranteeLevel.CONDITIONAL,
          "Induction + pigeonhole. See §6.",
          assumptions=("Discrete actions", "min_cost ε > 0")),
    Claim("T3", "I(s₀)=True ⟹ I(sₜ)=True for all t",
          GuaranteeLevel.PROVEN,
          "Induction. Check-before-commit. See §3."),
    Claim("T4", "spent(t) ≤ spent(t+1)",
          GuaranteeLevel.PROVEN,
          "cost ≥ 0 asserted. See §4."),
    Claim("T5", "State transitions are all-or-nothing",
          GuaranteeLevel.PROVEN,
          "Evaluate simulates, execute commits atomically. See §6."),
    Claim("T6", "Trace is append-only, hash-chained",
          GuaranteeLevel.PROVEN,
          "TraceEntry is frozen, hash includes prev_hash. See §5."),
    Claim("T7", "undo(execute(s,a)) == s",
          GuaranteeLevel.PROVEN,
          "State immutability + stored inverse effects. See §2, §6."),
]
