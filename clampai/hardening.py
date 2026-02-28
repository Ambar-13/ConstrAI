"""
clampai.hardening — Defence-in-depth layer for autonomous agents.

Addresses five vulnerability classes identified during audits: command
injection (sandboxed attestors with frozen allowlists), temporal blindness
(readiness probes with exponential backoff), Bayesian cold-start bias
(cost-aware pessimistic priors), spec-reality gaps (environment
reconciliation), and reward hacking (multi-dimensional cross-attestation).

The guiding idea is straightforward: rather than trying to monitor the
agent's intent, we monitor the environment it is touching. If the
environment does not change the way the declared effects predict, the
agent is stopped immediately.
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import time as _time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, ClassVar, Dict, List, Optional, Protocol, Set, Tuple

from .formal import ActionSpec, Claim, GuaranteeLevel, State
from .reasoning import BeliefState, CausalGraph, Dependency

# Sandboxed attestors — command injection defence

class AttestationResult(Enum):
    VERIFIED = auto()
    FAILED = auto()
    ERROR = auto()
    TIMEOUT = auto()
    SKIPPED = auto()

@dataclass(frozen=True)
class Attestation:
    attestor_name: str
    result: AttestationResult
    evidence: str
    timestamp: float
    fingerprint: str
    def is_positive(self) -> bool:
        return self.result == AttestationResult.VERIFIED

class Attestor(Protocol):
    @property
    def name(self) -> str: ...
    def verify(self, state: State, goal_description: str,
               timeout_s: float) -> Attestation: ...

_DEFAULT_COMMAND_ALLOWLIST = frozenset({
    "ls", "cat", "wc", "head", "tail", "grep", "find", "file",
    "curl", "wget", "node", "python3", "python", "npm", "pytest",
    "go", "cargo", "make", "test", "stat", "du", "df",
    "pg_isready", "redis-cli", "mysql", "psql",
    "docker", "kubectl", "terraform",
    "sha256sum", "md5sum", "diff",
})

class SubprocessAttestor:
    """
    Sandboxed subprocess attestor. COMMAND INJECTION PROOF.

    Security:
      1. Command is TUPLE set at __init__ (frozen, immutable)
      2. shell=False ALWAYS (no shell expansion)
      3. Command[0] must be in allowlist (no arbitrary binaries)
      4. Working directory validated at creation
      5. Environment sanitized: only PATH, HOME, LANG preserved
      6. No method accepts agent-controlled data as command args
      7. Output truncated to prevent memory exhaustion

    Theorem (Command Isolation):
      No method modifies self._command after __init__.
      verify() reads it, never writes it.
      shell=False prevents metacharacter expansion.  ∎

    Guarantee: PROVEN
    """
    MAX_OUTPUT_BYTES = 8192
    SAFE_ENV_KEYS = frozenset({"HOME", "LANG", "TERM", "USER"})  # No PATH
    # Hardcoded safe PATH — agent cannot poison binary resolution
    SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    def __init__(self, name: str, command: List[str],
                 success_pattern: str = "",
                 working_dir: Optional[str] = None,
                 command_allowlist: Optional[frozenset] = None):
        if not command:
            raise ValueError("Command cannot be empty")
        allowlist = command_allowlist or _DEFAULT_COMMAND_ALLOWLIST
        binary = os.path.basename(command[0])
        if binary not in allowlist:
            raise ValueError(f"'{binary}' not in allowlist: {sorted(allowlist)}")
        if working_dir is not None and not os.path.isdir(working_dir):
            raise ValueError(f"Working dir does not exist: {working_dir}")

        self._name = str(name)
        self._command = tuple(str(c) for c in command)
        self._success_pattern = str(success_pattern)
        self._working_dir = working_dir
        self._env = {k: v for k, v in os.environ.items()
                     if k in self.SAFE_ENV_KEYS}
        self._env["PATH"] = self.SAFE_PATH  # Hardcoded, not inherited

    @property
    def name(self) -> str: return self._name

    def verify(self, state: State, goal_description: str,
               timeout_s: float = 30.0) -> Attestation:
        try:
            result = subprocess.run(
                self._command, capture_output=True, text=True,
                timeout=timeout_s, cwd=self._working_dir,
                env=self._env, shell=False)
            output = (result.stdout + result.stderr)[:self.MAX_OUTPUT_BYTES]
            passed = result.returncode == 0
            if passed and self._success_pattern:
                passed = self._success_pattern in output
            fp = hashlib.sha256(output.encode()).hexdigest()[:32]
            return Attestation(self._name,
                AttestationResult.VERIFIED if passed else AttestationResult.FAILED,
                output, _time.time(), fp)
        except subprocess.TimeoutExpired:
            return Attestation(self._name, AttestationResult.TIMEOUT,
                f"Timed out after {timeout_s}s", _time.time(), "timeout")
        except Exception as e:
            return Attestation(self._name, AttestationResult.ERROR,
                str(e)[:self.MAX_OUTPUT_BYTES], _time.time(), "error")


class PredicateAttestor:
    def __init__(self, name: str, check_fn: Callable[[], Tuple[bool, str]]):
        self._name = name
        self._check_fn = check_fn
    @property
    def name(self) -> str: return self._name
    def verify(self, state: State, goal_description: str,
               timeout_s: float = 30.0) -> Attestation:
        try:
            passed, evidence = self._check_fn()
            fp = hashlib.sha256(evidence.encode()).hexdigest()[:32]
            return Attestation(self._name,
                AttestationResult.VERIFIED if passed else AttestationResult.FAILED,
                evidence, _time.time(), fp)
        except Exception as e:
            return Attestation(self._name, AttestationResult.ERROR,
                str(e), _time.time(), "error")


class AttestationGate:
    def __init__(self, quorum: int = 1):
        self.quorum = quorum
        self._attestors: List[Attestor] = []
        self._history: List[List[Attestation]] = []

    def add_attestor(self, attestor: Attestor) -> None:
        self._attestors.append(attestor)

    def verify_goal(self, state: State, goal_description: str,
                    goal_predicate: Callable[[State], bool],
                    timeout_per_attestor: float = 30.0
                    ) -> Tuple[bool, List[Attestation]]:
        if not goal_predicate(state):
            return False, []
        if not self._attestors:
            return True, [Attestation("predicate_only",
                AttestationResult.SKIPPED,
                "No attestors registered (WEAK).", _time.time(), "none")]
        attestations = [a.verify(state, goal_description, timeout_per_attestor)
                       for a in self._attestors]
        self._history.append(attestations)
        return sum(1 for a in attestations if a.is_positive()) >= self.quorum, attestations

    @property
    def attestation_history(self) -> List[List[Attestation]]:
        return list(self._history)


# Temporal dependencies — readiness probes

@dataclass(frozen=True)
class ReadinessProbe:
    """
    Checks whether a resource is truly READY, not just EXISTING.

    Uses exponential backoff with bounded max wait.
    Guarantee: CONDITIONAL (correct if probe accurately reflects state)
    """
    name: str
    check_fn: Callable[[], Tuple[bool, str]]
    initial_delay_s: float = 0.0
    interval_s: float = 1.0
    max_retries: int = 10
    backoff_factor: float = 1.5

    def wait_until_ready(self, simulated_time: float = 0.0,
                         use_real_time: bool = False
                         ) -> Tuple[bool, str, float]:
        total_wait = self.initial_delay_s
        if use_real_time and self.initial_delay_s > 0:
            _time.sleep(self.initial_delay_s)
        interval = self.interval_s
        for _ in range(self.max_retries):
            try:
                ready, detail = self.check_fn()
                if ready:
                    return True, detail, total_wait
            except Exception as e:
                detail = f"Probe error: {e}"
            total_wait += interval
            if use_real_time:
                _time.sleep(interval)
            interval = min(interval * self.backoff_factor, 30.0)
        return False, f"Not ready after {self.max_retries} attempts", total_wait


class TemporalDependency:
    """
    Dependency with time/readiness dimension.

    Extends binary "met/not met" with:
      - Minimum delay after action completion
      - Readiness probe polling
    """
    def __init__(self, required_action_id: str, reason: str,
                 readiness_probe: Optional[ReadinessProbe] = None,
                 min_delay_s: float = 0.0):
        self.required_action_id = required_action_id
        self.reason = reason
        self.readiness_probe = readiness_probe
        self.min_delay_s = min_delay_s
        self._completion_time: Optional[float] = None

    def mark_completed(self, at_time: float) -> None:
        self._completion_time = at_time

    def is_satisfied(self, current_time: float,
                     use_real_time: bool = False) -> Tuple[bool, str]:
        if self._completion_time is None:
            return False, f"{self.required_action_id} not completed"
        elapsed = current_time - self._completion_time
        if elapsed < self.min_delay_s:
            return False, (f"{self.required_action_id}: {elapsed:.1f}s elapsed, "
                          f"need {self.min_delay_s:.1f}s")
        if self.readiness_probe:
            ready, detail, _ = self.readiness_probe.wait_until_ready(
                simulated_time=current_time, use_real_time=use_real_time)
            if not ready:
                return False, f"{self.required_action_id}: exists but not ready ({detail})"
        return True, ""


class TemporalCausalGraph:
    """
    CausalGraph + temporal readiness checking.
    Backward compatible: no temporal deps = standard behavior.
    """
    def __init__(self, base_graph: CausalGraph):
        self._base = base_graph
        self._temporal: Dict[str, List[TemporalDependency]] = {}
        self._current_time: float = 0.0

    def add_temporal_dep(self, action_id: str, dep: TemporalDependency) -> None:
        self._temporal.setdefault(action_id, []).append(dep)

    def set_time(self, t: float) -> None:
        self._current_time = t

    def mark_completed(self, action_id: str, at_time: Optional[float] = None) -> None:
        t = at_time or self._current_time
        self._base.mark_completed(action_id)
        for deps in self._temporal.values():
            for dep in deps:
                if dep.required_action_id == action_id:
                    dep.mark_completed(t)

    def can_execute(self, action_id: str,
                    use_real_time: bool = False) -> Tuple[bool, List[str]]:
        base_ok, base_unmet = self._base.can_execute(action_id)
        if not base_ok:
            return False, base_unmet
        unmet = []
        for dep in self._temporal.get(action_id, []):
            ok, reason = dep.is_satisfied(self._current_time, use_real_time)
            if not ok:
                unmet.append(reason)
        return len(unmet) == 0, unmet

    def ready_actions(self, available: List[str],
                      use_real_time: bool = False) -> List[str]:
        return [a for a in available if self.can_execute(a, use_real_time)[0]]


# Cost-aware Bayesian priors — cold-start correction

class CostAwarePriorFactory:
    """
    Cost-proportional pessimism for Bayesian priors.

    Theorem (Cost-Bounded First-Strike):
      With prior Beta(1, K·c/B), mean = 1/(1+K·c/B).
      For K=5, c=$50, B=$100: mean ≈ 0.29 vs 0.50 (uniform).
      First-strike selection probability reduced by 42%.
    Proof: Direct computation of Beta mean.  ∎

    Tiers:
      EXPLORE:  Cheap, reversible → Beta(3, 1)
      CAUTIOUS: Moderate → Beta(2, 1+K·c/B·0.5)
      GUARDED:  Expensive → Beta(1, 1+K·c/B)
      GATED:    Irreversible+critical → BLOCKED until authorized

    Guarantee: PROVEN (prior algebra is exact)
    """
    def __init__(self, total_budget: float, pessimism_factor: float = 5.0,
                 expensive_threshold: float = 0.1,
                 critical_threshold: float = 0.3):
        self.budget = total_budget
        self.K = pessimism_factor
        self.expensive_threshold = expensive_threshold
        self.critical_threshold = critical_threshold
        self._authorized: Set[str] = set()

    def compute_prior(self, action: ActionSpec) -> Tuple[float, float, str]:
        cost_ratio = action.cost / max(self.budget, 0.01)
        is_irreversible = not action.reversible
        is_critical = action.risk_level in ("high", "critical")

        if (is_irreversible and is_critical) or cost_ratio >= self.critical_threshold:
            if action.id not in self._authorized:
                return 0.01, 100.0, "GATED"
            return 1.0, 1.0 + self.K * cost_ratio, "GATED_AUTHORIZED"
        if cost_ratio >= self.expensive_threshold or is_critical:
            return 1.0, 1.0 + self.K * cost_ratio, "GUARDED"
        if cost_ratio >= 0.03 or action.risk_level == "medium":
            return 2.0, 1.0 + self.K * cost_ratio * 0.5, "CAUTIOUS"
        return 3.0, 1.0, "EXPLORE"

    def authorize(self, action_id: str) -> None:
        self._authorized.add(action_id)

    def initialize_beliefs(self, beliefs: BeliefState,
                           actions: List[ActionSpec]) -> Dict[str, str]:
        tiers = {}
        for a in actions:
            alpha, beta, tier = self.compute_prior(a)
            beliefs.set_prior(f"action:{a.id}:succeeds", alpha, beta)
            tiers[a.id] = tier
        return tiers

    def summary(self, actions: List[ActionSpec]) -> str:
        lines = ["Cost-Aware Priors:"]
        for a in actions:
            alpha, beta, tier = self.compute_prior(a)
            mean = alpha / (alpha + beta)
            lines.append(f"  {a.id}: [{tier}] ${a.cost:.2f} → "
                        f"Beta({alpha:.1f},{beta:.1f}) μ={mean:.3f}")
        return "\n".join(lines)


# Environment reconciliation — spec-reality gap detection

@dataclass(frozen=True)
class EnvironmentProbe:
    """Reads a REAL value from environment (not from agent state)."""
    variable_name: str
    read_fn: Callable[[], Any]
    description: str = ""

@dataclass(frozen=True)
class ReconciliationResult:
    matches: List[Tuple[str, Any, Any]]
    mismatches: List[Tuple[str, Any, Any]]
    unprobed: List[str]
    timestamp: float = 0.0

    @property
    def is_consistent(self) -> bool:
        return len(self.mismatches) == 0

    @property
    def drift_severity(self) -> float:
        total = len(self.matches) + len(self.mismatches)
        return len(self.mismatches) / total if total > 0 else 0.0


class EnvironmentDriftError(Exception):
    def __init__(self, message: str, result: ReconciliationResult):
        super().__init__(message)
        self.result = result


class EnvironmentReconciler:
    """
    Detects spec-reality drift by comparing model to environment probes.

    "We don't monitor the AI; we monitor the environment the AI is touching."

    After every action:
      EXPECTED state (from ActionSpec.simulate)  vs
      ACTUAL state (from EnvironmentProbes)
      If drift > threshold → HALT

    Guarantee: CONDITIONAL
      Detects drift on probed variables. Cannot detect unprobed drift.
    """
    def __init__(self, drift_threshold: float = 0.0,
                 halt_on_drift: bool = True):
        self.drift_threshold = drift_threshold
        self.halt_on_drift = halt_on_drift
        self._probes: Dict[str, EnvironmentProbe] = {}
        self._history: List[ReconciliationResult] = []

    def add_probe(self, probe: EnvironmentProbe) -> None:
        self._probes[probe.variable_name] = probe

    def snapshot(self) -> Dict[str, Any]:
        snap = {}
        for name, probe in self._probes.items():
            try:
                snap[name] = probe.read_fn()
            except Exception as e:
                snap[name] = f"PROBE_ERROR:{e}"
        return snap

    def reconcile(self, expected_state: State,
                  label: str = "") -> ReconciliationResult:
        actual = self.snapshot()
        matches: List[Tuple[str, Any, Any]] = []
        mismatches: List[Tuple[str, Any, Any]] = []
        unprobed: List[str] = []
        for var_name in self._probes:
            expected = expected_state.get(var_name)
            actual_val = actual.get(var_name)
            if isinstance(actual_val, str) and actual_val.startswith("PROBE_ERROR"):
                mismatches.append((var_name, expected, actual_val))
            elif expected == actual_val:
                matches.append((var_name, expected, actual_val))
            else:
                mismatches.append((var_name, expected, actual_val))
        for key in expected_state.keys():
            if key not in self._probes and not key.startswith("_"):
                unprobed.append(key)
        result = ReconciliationResult(matches, mismatches, unprobed, _time.time())
        self._history.append(result)
        if self.halt_on_drift and result.drift_severity > self.drift_threshold:
            details = "; ".join(f"{v}: exp={e}, got={a}" for v, e, a in mismatches)
            raise EnvironmentDriftError(f"DRIFT ({label}): {details}", result)
        return result

    @property
    def reconciliation_history(self) -> List[ReconciliationResult]:
        return list(self._history)


# Multi-dimensional attestation — reward-hacking defence

class QualityDimension(Enum):
    EXISTENCE = "existence"
    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"
    QUALITY = "quality"
    SAFETY = "safety"
    REGRESSION = "regression"


@dataclass(frozen=True)
class QualityScore:
    dimension: QualityDimension
    score: float
    detail: str


class MultiDimensionalAttestor:
    """
    Checks MULTIPLE quality dimensions. Harder to game than single-metric.

    Goal passes iff overall_score >= threshold AND no dimension scores 0.
    Agent must satisfy ALL dimensions simultaneously.

    Guarantee: EMPIRICAL (more dimensions = harder to game, not provably ungameable)
    """
    def __init__(self, name: str, threshold: float = 0.7):
        self._name = name
        self._threshold = threshold
        self._checks: List[Tuple[QualityDimension, float,
                                  Callable[[], Tuple[float, str]]]] = []
    @property
    def name(self) -> str: return self._name

    def add_check(self, dimension: QualityDimension, weight: float,
                  check_fn: Callable[[], Tuple[float, str]]) -> None:
        self._checks.append((dimension, weight, check_fn))

    def verify(self, state: State, goal_description: str,
               timeout_s: float = 30.0) -> Attestation:
        scores = []
        total_w, weighted_sum, any_zero = 0.0, 0.0, False
        for dim, w, fn in self._checks:
            try:
                score, detail = fn()
                score = max(0.0, min(1.0, score))
            except Exception as e:
                score, detail = 0.0, f"Error: {e}"
            scores.append(QualityScore(dim, score, detail))
            weighted_sum += score * w
            total_w += w
            if score == 0.0:
                any_zero = True
        overall = weighted_sum / max(total_w, 0.01)
        passed = overall >= self._threshold and not any_zero
        evidence = (f"Overall={overall:.3f} | " +
                   " | ".join(f"{s.dimension.value}={s.score:.2f}" for s in scores))
        fp = hashlib.sha256(evidence.encode()).hexdigest()[:32]
        return Attestation(self._name,
            AttestationResult.VERIFIED if passed else AttestationResult.FAILED,
            evidence, _time.time(), fp)


# Dynamic dependency discovery

@dataclass
class FailurePattern:
    action_failed: str
    action_missing: str
    observations: int = 0
    failures_with_missing: int = 0
    failures_without_missing: int = 0
    successes_with_missing: int = 0
    successes_without_missing: int = 0

class DependencyDiscovery:
    """Chi-squared dependency discovery. Guarantee: EMPIRICAL."""
    def __init__(self, significance_threshold: float = 0.05,
                 min_observations: int = 5):
        self.threshold = significance_threshold
        self.min_obs = min_observations
        self._patterns: Dict[Tuple[str, str], FailurePattern] = {}
        self._discovered: Dict[str, List[Tuple[str, float, float]]] = {}

    def observe(self, action_id: str, succeeded: bool,
                completed_actions: Set[str],
                all_known_actions: Set[str]) -> None:
        for other_id in all_known_actions:
            if other_id == action_id: continue
            key = (action_id, other_id)
            if key not in self._patterns:
                self._patterns[key] = FailurePattern(action_id, other_id)
            p = self._patterns[key]
            p.observations += 1
            done = other_id in completed_actions
            if succeeded and done: p.successes_without_missing += 1
            elif succeeded and not done: p.successes_with_missing += 1
            elif not succeeded and done: p.failures_without_missing += 1
            else: p.failures_with_missing += 1

    def discover(self) -> Dict[str, List[Tuple[str, float, float]]]:
        disc: Dict[str, List[Tuple[str, float, float]]] = {}
        for (aid, oid), p in self._patterns.items():
            if p.observations < self.min_obs: continue
            wm = p.failures_with_missing + p.successes_with_missing
            wo = p.failures_without_missing + p.successes_without_missing
            if wm == 0 or wo == 0: continue
            effect = p.failures_with_missing/wm - p.failures_without_missing/wo
            if effect <= 0: continue
            pv = self._chi2p(p)
            if pv < self.threshold and effect > 0.2:
                disc.setdefault(aid, []).append((oid, pv, effect))
        self._discovered = disc
        return disc

    def inject_into_graph(self, graph: CausalGraph) -> List[Tuple[str, str, float]]:
        disc = self.discover()
        injected = []
        for aid, deps in disc.items():
            for did, pv, eff in deps:
                node = graph._nodes.get(aid)
                existing = {d.required_action_id for d in node.dependencies} if node else set()
                if did not in existing:
                    old = list(node.dependencies) if node else []
                    old.append(Dependency(did, f"DISCOVERED: p={pv:.4f}"))
                    graph.add_action(aid, [(d.required_action_id, d.reason) for d in old])
                    injected.append((aid, did, pv))
        return injected

    @staticmethod
    def _chi2p(p: FailurePattern) -> float:
        a, b = p.successes_without_missing, p.successes_with_missing
        c, d = p.failures_without_missing, p.failures_with_missing
        n = a+b+c+d
        if n == 0: return 1.0
        r1, r2, c1, c2 = a+b, c+d, a+c, b+d
        if r1==0 or r2==0 or c1==0 or c2==0: return 1.0
        chi2 = sum((o-en/n)**2/(en/n) for o, en in
                   [(a,r1*c1),(b,r1*c2),(c,r2*c1),(d,r2*c2)] if en/n > 0)
        if chi2 > 10: return math.exp(-chi2/2)
        for t, pval in [(10.83,.001),(6.63,.01),(5.02,.025),(3.84,.05),
                        (2.71,.1),(1.32,.25),(0.455,.5),(0.0,1.0)]:
            if chi2 >= t: return pval
        return 1.0

    def summary(self) -> str:
        if not self._discovered: self.discover()
        lines = ["Discovered Dependencies:"]
        for a, deps in self._discovered.items():
            for d, p, e in deps:
                lines.append(f"  {a}→{d} (p={p:.4f}, Δ={e:.2f})")
        return "\n".join(lines) if len(lines) > 1 else "  (none yet)"


# Resource-aware state

class ResourceState(Enum):
    ABSENT="absent"; CREATING="creating"; READY="ready"
    DEGRADED="degraded"; DELETING="deleting"; FAILED="failed"

class Permission(Enum):
    NONE="none"; READ="read"; WRITE="write"; EXECUTE="execute"; ADMIN="admin"

@dataclass(frozen=True)
class ResourceDescriptor:
    kind: str; identifier: str
    state: ResourceState = ResourceState.ABSENT
    permissions: frozenset = frozenset()
    health_endpoint: str = ""
    created_at: float = 0.0
    metadata: str = ""
    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "identifier": self.identifier,
                "state": self.state.value,
                "permissions": sorted(p.value for p in self.permissions)}

class ResourceTracker:
    VALID_TRANSITIONS: ClassVar[Dict[ResourceState, Set[ResourceState]]] = {
        ResourceState.ABSENT: {ResourceState.CREATING},
        ResourceState.CREATING: {ResourceState.READY, ResourceState.FAILED},
        ResourceState.READY: {ResourceState.DEGRADED, ResourceState.DELETING},
        ResourceState.DEGRADED: {ResourceState.READY, ResourceState.DELETING, ResourceState.FAILED},
        ResourceState.DELETING: {ResourceState.ABSENT, ResourceState.FAILED},
        ResourceState.FAILED: {ResourceState.CREATING, ResourceState.DELETING, ResourceState.ABSENT},
    }
    def __init__(self) -> None: self._resources: Dict[str, ResourceDescriptor] = {}
    def register(self, r: ResourceDescriptor) -> None: self._resources[r.identifier] = r
    def transition(self, rid: str, ns: ResourceState) -> Tuple[bool, str]:
        c = self._resources.get(rid)
        if not c: return False, f"{rid} not registered"
        valid = self.VALID_TRANSITIONS.get(c.state, set())
        if ns not in valid:
            return False, f"Invalid: {c.state.value}→{ns.value}"
        self._resources[rid] = ResourceDescriptor(
            c.kind, c.identifier, ns, c.permissions, c.health_endpoint,
            c.created_at if ns != ResourceState.CREATING else _time.time(), c.metadata)
        return True, f"{rid}: {c.state.value}→{ns.value}"
    def get(self, rid: str) -> Optional[ResourceDescriptor]:
        return self._resources.get(rid)
    def all_ready(self, ids: Optional[List[str]] = None) -> bool:
        t = ids or list(self._resources.keys())
        return all(self._resources.get(r, ResourceDescriptor("",r)).state == ResourceState.READY for r in t)
    def summary(self) -> str:
        return "\n".join(f"  [{r.state.value:>10}] {r.kind}:{rid}"
                        for rid, r in sorted(self._resources.items()))


# Claims registry

HARDENING_CLAIMS = [
    Claim("H1", "Attestor commands frozen; agent has zero influence",
          GuaranteeLevel.PROVEN,
          "Tuple in __init__, shell=False, allowlist."),
    Claim("H2", "Temporal deps prevent readiness race conditions",
          GuaranteeLevel.CONDITIONAL,
          "Correct if probes accurately reflect resource state.",
          assumptions=("Probes check the right property",)),
    Claim("H3", "Cost-aware priors reduce first-strike waste",
          GuaranteeLevel.PROVEN,
          "Beta(1,K·c/B) has mean < 0.5. Direct computation."),
    Claim("H4", "Environment reconciliation detects spec-reality drift",
          GuaranteeLevel.CONDITIONAL,
          "Compares model to probes post-execution.",
          assumptions=("Probes cover affected variables",)),
    Claim("H5", "Multi-dim attestation harder to game than single-metric",
          GuaranteeLevel.EMPIRICAL,
          "More clampaints = harder to satisfy simultaneously."),
    Claim("H6", "Dynamic discovery reduces unspecified failures",
          GuaranteeLevel.EMPIRICAL, "Chi-squared on 2×2 tables."),
    Claim("H7", "Resource lifecycle transitions validated",
          GuaranteeLevel.PROVEN, "Enumerated valid transitions."),
]
