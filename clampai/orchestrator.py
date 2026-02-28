"""
clampai.orchestrator — Main execution loop for autonomous agents.

Ties the safety kernel (formal.py) and reasoning engine (reasoning.py)
into a complete run-to-completion loop. Each iteration computes action
values, constructs a structured LLM prompt, verifies the proposed action
against T1–T7, commits the state transition, and updates Bayesian beliefs
from the outcome. When the LLM fails or proposes an invalid action, the
orchestrator falls back to the highest-value READY action.
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .active_hjb_barrier import ActiveHJBBarrier, RecoveryStrategy, choose_recovery_strategy
from .formal import (
    ActionSpec,
    AsyncSafetyKernel,
    Invariant,
    SafetyKernel,
    State,
)
from .gradient_tracker import GradientTracker

# ── Advanced Math (T7, Gradients, Active HJB) ──
from .inverse_algebra import InverseAlgebra, RollbackRecord
from .reasoning import (
    ActionValue,
    ActionValueComputer,
    BeliefState,
    CausalGraph,
    LLMAdapter,
    MockLLMAdapter,
    ReasoningRequest,
    ReasoningResponse,
    compute_state_delta,
    estimate_tokens,
    parse_llm_response,
    should_skip_llm,
)
from .reference_monitor import (
    CaptureBasin,
    ReferenceMonitor,
)
from .verification_log import ProofRecord, ProofStep, write_proof

# Outcome observation

class OutcomeType(Enum):
    SUCCESS = auto()
    PARTIAL = auto()
    FAILURE = auto()
    UNEXPECTED = auto()


@dataclass(frozen=True)
class Outcome:
    """Observed outcome of an action execution."""
    action_id: str
    outcome_type: OutcomeType
    actual_state: State
    expected_state: State
    details: str = ""

    @property
    def succeeded(self) -> bool:
        return self.outcome_type == OutcomeType.SUCCESS

    @property
    def state_matches_expected(self) -> bool:
        return self.actual_state == self.expected_state


# Progress monitor — stuck detection

class ProgressMonitor:
    """
    Tracks goal progress and detects when system is stuck.

    Stuck detection (EMPIRICAL):
      If progress hasn't improved in `patience` steps AND
      budget utilization > 50%, flag as stuck.

    This is a HEURISTIC — it can have false positives.
    The LLM gets the stuck flag as input and decides what to do.
    The formal layer doesn't care about stuck — it only cares about
    budget/invariants/termination.
    """
    def __init__(self, patience: int = 5):
        self.patience = patience
        self._history: List[Tuple[int, float]] = []  # (step, progress)

    def record(self, step: int, progress: float) -> None:
        self._history.append((step, progress))

    @property
    def current_progress(self) -> float:
        return self._history[-1][1] if self._history else 0.0

    @property
    def is_stuck(self) -> bool:
        if len(self._history) < self.patience:
            return False
        recent = [p for _, p in self._history[-self.patience:]]
        return max(recent) - min(recent) < 0.01  # No meaningful change

    @property
    def progress_rate(self) -> float:
        """Progress per step (moving average)."""
        if len(self._history) < 2:
            return 0.0
        recent = self._history[-min(5, len(self._history)):]
        dp = recent[-1][1] - recent[0][1]
        ds = recent[-1][0] - recent[0][0]
        return dp / max(ds, 1)

    def estimated_steps_to_goal(self) -> Optional[int]:
        """Estimate steps remaining to reach 100% progress."""
        rate = self.progress_rate
        if rate <= 0:
            return None
        remaining = 1.0 - self.current_progress
        return max(1, int(remaining / rate))

    def to_llm_text(self) -> str:
        lines = [f"Progress: {self.current_progress:.1%}"]
        if self.is_stuck:
            lines.append("⚠ STUCK: No progress in last {self.patience} steps")
        rate = self.progress_rate
        if rate > 0:
            est = self.estimated_steps_to_goal()
            lines.append(f"Rate: {rate:.3f}/step, ~{est} steps to completion")
        return " | ".join(lines)


# Task definition

@dataclass
class TaskDefinition:
    """
    Everything needed to define an autonomous task.

    This is the PUBLIC API for using ClampAI. You define:
      - goal: what to achieve (natural language)
      - initial_state: starting world state
      - available_actions: what the agent can do
      - invariants: what must ALWAYS be true
      - budget: maximum resource spend
      - goal_predicate: formal success criterion
      - dependencies: which actions require which others
      - priors: initial beliefs about action success rates
    """
    goal: str
    initial_state: State
    available_actions: List[ActionSpec]
    invariants: List[Invariant]
    budget: float
    goal_predicate: Callable[[State], bool]
    goal_progress_fn: Optional[Callable[[State], float]] = None
    min_action_cost: float = 0.001
    dependencies: Optional[Dict[str, List[Tuple[str, str]]]] = None
    priors: Optional[Dict[str, Tuple[float, float]]] = None  # key → (α, β)
    max_retries_per_action: int = 3
    max_consecutive_failures: int = 5
    stuck_patience: int = 5
    system_prompt: str = ""
    risk_aversion: float = 1.0
    # Prompt-token optimization (Integral Sensitivity Filter)
    sensitivity_threshold: float = 0.05
    max_prompt_state_keys: int = 20
    proof_path: str = ""  # if set, write a .clampai_proof JSON record
    capture_basins: Optional[List['CaptureBasin']] = None  # Advanced: HJB reachability basins


# Execution result

class TerminationReason(Enum):
    GOAL_ACHIEVED = "goal_achieved"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STEP_LIMIT = "step_limit"
    LLM_STOP = "llm_requested_stop"
    STUCK = "stuck_detected"
    MAX_FAILURES = "max_consecutive_failures"
    ERROR = "unrecoverable_error"


@dataclass
class ExecutionResult:
    """Complete result of an ClampAI execution run."""
    goal_achieved: bool
    termination_reason: TerminationReason
    final_state: State
    total_cost: float
    total_steps: int
    goal_progress: float
    execution_time_s: float
    trace_length: int
    beliefs_summary: str
    budget_summary: str
    errors: List[str] = field(default_factory=list)

    # Detailed metrics
    actions_attempted: int = 0
    actions_succeeded: int = 0
    actions_rejected_safety: int = 0
    actions_rejected_reasoning: int = 0
    rollbacks: int = 0
    llm_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'goal_achieved': self.goal_achieved,
            'termination_reason': self.termination_reason.value,
            'total_cost': self.total_cost,
            'total_steps': self.total_steps,
            'goal_progress': self.goal_progress,
            'execution_time_s': self.execution_time_s,
            'actions_attempted': self.actions_attempted,
            'actions_succeeded': self.actions_succeeded,
            'actions_rejected_safety': self.actions_rejected_safety,
            'actions_rejected_reasoning': self.actions_rejected_reasoning,
            'rollbacks': self.rollbacks,
            'llm_calls': self.llm_calls,
            'errors': self.errors,
        }

    def summary(self) -> str:
        status = "GOAL ACHIEVED" if self.goal_achieved else "GOAL NOT ACHIEVED"
        return (
            f"\n{'='*60}\n"
            f"  ClampAI Execution Summary\n"
            f"{'='*60}\n"
            f"  {status}\n"
            f"  Reason: {self.termination_reason.value}\n"
            f"  Progress: {self.goal_progress:.1%}\n"
            f"  Cost: ${self.total_cost:.2f}\n"
            f"  Steps: {self.total_steps}\n"
            f"  Time: {self.execution_time_s:.2f}s\n"
            f"  Actions: {self.actions_succeeded}/{self.actions_attempted} succeeded\n"
            f"  Safety rejections: {self.actions_rejected_safety}\n"
            f"  LLM calls: {self.llm_calls}\n"
            f"  Rollbacks: {self.rollbacks}\n"
            f"  Errors: {len(self.errors)}\n"
            f"{'='*60}"
        )


# Orchestrator — main execution loop

class Orchestrator:
    """
    The main ClampAI execution engine.

    Architecture:
      ┌────────────────────────────────────────────┐
      │            Orchestrator (this)              │
      │  ┌──────────────────────────────────────┐  │
      │  │   LLM Adapter (pluggable)            │  │
      │  │   ↕ structured prompts/responses      │  │
      │  ├──────────────────────────────────────┤  │
      │  │   Reasoning Engine                    │  │
      │  │   • Belief State (Bayesian)           │  │
      │  │   • Causal Graph (DAG)                │  │
      │  │   • Action Value (info-theoretic)     │  │
      │  │   ↕ computed analysis                 │  │
      │  ├──────────────────────────────────────┤  │
      │  │   Safety Kernel (FORMAL)              │  │
      │  │   • Budget (T1)                       │  │
      │  │   • Termination (T2)                  │  │
      │  │   • Invariants (T3)                   │  │
      │  │   • Atomicity (T5)                    │  │
      │  │   • Trace (T6)                        │  │
      │  └──────────────────────────────────────┘  │
      └────────────────────────────────────────────┘

    The formal layer CANNOT be bypassed by the LLM.
    The LLM CANNOT be bypassed by the formal layer.
    Both must agree for any action to execute.
    """

    def __init__(self, task: TaskDefinition, llm: Optional[LLMAdapter] = None):
        self.task = task
        self.llm = llm or MockLLMAdapter()

        # ── Formal Layer ──
        self.kernel = SafetyKernel(
            budget=task.budget,
            invariants=task.invariants,
            min_action_cost=task.min_action_cost,
        )

        # ── Attestation / proof steps (optional) ──
        self._proof_steps: List[ProofStep] = []

        # ── Reference Monitor (Authoritative Enforcement) ──
        self.monitor = ReferenceMonitor(
            ifc_enabled=True,
            cbf_enabled=True,
            hjb_enabled=True,
        )
        # Configure CBF for budget resource
        self.monitor.add_cbf(
            h=lambda s: (task.budget - self.kernel.budget.spent) / max(task.budget, 1.0),
            alpha=0.1  # 10% decay per step when approaching limit
        )

        # ── Advanced Math: Gradient Tracker (T7 Safety Margins) ──
        self.gradient_tracker = GradientTracker(task.invariants)

        # ── Advanced Math: Active HJB Barrier ──
        self.hjb_barrier = ActiveHJBBarrier(
            basins=task.capture_basins or [],
            max_lookahead=min(5, int(task.budget / task.min_action_cost // 2))
        )

        # ── State history for rollback (T7) ──
        self._rollback_records: List[RollbackRecord] = []

        # ── Reasoning Layer ──
        self.beliefs = BeliefState()
        self.causal_graph = CausalGraph()
        self.value_computer = ActionValueComputer(
            risk_aversion=task.risk_aversion)
        self.progress_monitor = ProgressMonitor(
            patience=task.stuck_patience)

        # ── Metrics ──
        self._actions_attempted = 0
        self._actions_succeeded = 0
        self._actions_rejected_safety = 0
        self._actions_rejected_reasoning = 0
        self._rollbacks = 0
        self._llm_calls = 0
        self._llm_prompt_tokens = 0
        self._llm_output_tokens = 0
        self._last_prompt_state = None
        self._consecutive_failures = 0
        self._errors = []

        # ── State ──
        self.current_state = task.initial_state
        self.action_map = {a.id: a for a in task.available_actions}
        self._state_history = [task.initial_state]

        # ── Initialize causal graph ──
        if task.dependencies:
            for action_id, deps in task.dependencies.items():
                self.causal_graph.add_action(action_id, deps)
        # Also add actions without explicit deps
        for a in task.available_actions:
            if a.id not in (task.dependencies or {}):
                self.causal_graph.add_action(a.id)

        # ── Initialize priors ──
        if task.priors:
            for key, (alpha, beta) in task.priors.items():
                self.beliefs.set_prior(key, alpha, beta)

    def _compute_progress(self) -> float:
        if self.task.goal_progress_fn:
            return self.task.goal_progress_fn(self.current_state)
        if self.task.goal_predicate(self.current_state):
            return 1.0
        return 0.0

    def _get_available_actions(self) -> List[ActionSpec]:
        """Filter actions to those affordable AND not already completed."""
        available = []
        for action in self.task.available_actions:
            can_afford, _ = self.kernel.budget.can_afford(action.cost)
            if not can_afford:
                continue
            # Skip actions already completed (in causal graph)
            # Unless they're designed to be repeatable (no deps tracking)
            if action.id in self.causal_graph._completed:
                # Check if this action's effects would still change state
                sim = action.simulate(self.current_state)
                if sim == self.current_state:
                    continue  # No-op, skip it
            available.append(action)
        return available

    def _compute_action_values(self, actions: List[ActionSpec]) -> List[ActionValue]:
        progress = self._compute_progress()
        steps_left = self.kernel.max_steps - self.kernel.step_count
        return [
            self.value_computer.compute(
                action=a, state=self.current_state,
                beliefs=self.beliefs,
                budget_remaining=self.kernel.budget.remaining,
                goal_progress=progress,
                steps_remaining=steps_left,
            )
            for a in actions
        ]

    def _build_history_summary(self, last_n: int = 5) -> str:
        entries = self.kernel.trace.last_n(last_n)
        if not entries:
            return "(no actions taken yet)"
        lines = []
        for e in entries:
            icon = "ok" if e.approved else "--"
            lines.append(f"  [{icon}] {e.action_name} (${e.cost:.2f})")
            if e.reasoning_summary:
                lines.append(f"      → {e.reasoning_summary[:100]}")
        return "\n".join(lines)

    def _ask_llm(self, available: List[ActionSpec],
                 values: List[ActionValue]) -> ReasoningResponse:
        """Build structured prompt, query LLM, parse response."""

        # Dominant-strategy skip: if one action clearly dominates, don't pay LLM.
        if should_skip_llm(values):
            best = sorted(values, key=lambda v: v.value_score, reverse=True)[0]
            return ReasoningResponse(
                chosen_action_id=best.action_id,
                reasoning="Dominant-strategy skip: selected top-valued action without LLM call.",
                expected_outcome="Progress",
                risk_assessment="Low (large value gap)",
                alternative_considered="Second-best action had much lower value score.",
                should_stop=False,
                stop_reason="",
                raw_response="",
                parse_errors=[],
            )

        # ── Integral Sensitivity Filter (prompt saliency pruning) ──
        # Compute a local integrated sensitivity score per state key:
        #   S(k) = Σ_{a in available} 𝟙[k ∈ affected(a)] · |V(a)|
        # Then keep keys with S(k) above threshold (plus underscore keys).
        state_for_prompt = self.current_state
        sensitivity_scores: Dict[str, float] = {}
        try:
            d = self.current_state.to_dict()
            sensitivity_scores = {k: 0.0 for k in d.keys()}
            value_by_id = {v.action_id: v.value_score for v in values}
            for a in available:
                w = abs(float(value_by_id.get(a.id, 0.0)))
                for k in a.affected_variables():
                    if k in sensitivity_scores:
                        sensitivity_scores[k] += w

            # Always keep internal/meta keys
            for k in list(sensitivity_scores.keys()):
                if k.startswith("_"):
                    sensitivity_scores[k] = float("inf")

            threshold = float(getattr(self.task, "sensitivity_threshold", 0.05) or 0.05)
            max_keys = int(getattr(self.task, "max_prompt_state_keys", 20) or 20)
            kept = [
                k for k, s in sensitivity_scores.items()
                if s == float("inf") or s >= threshold
            ]
            if len(kept) > max_keys:
                kept = sorted(kept, key=lambda kk: sensitivity_scores.get(kk, 0.0), reverse=True)[:max_keys]
            if kept:
                state_for_prompt = State({k: d[k] for k in kept if k in d})
        except Exception:
            # If anything goes wrong, fall back to full state.
            state_for_prompt = self.current_state

        request = ReasoningRequest(
            goal=self.task.goal,
            state=state_for_prompt,
            available_actions=available,
            action_values=values,
            beliefs=self.beliefs,
            causal_graph=self.causal_graph,
            safety_kernel=self.kernel,
            history_summary=self._build_history_summary(),
        )
        prompt = request.to_prompt()

        # Record prompt token reduction (best-effort).
        try:
            full_req = ReasoningRequest(
                goal=self.task.goal,
                state=self.current_state,
                available_actions=available,
                action_values=values,
                beliefs=self.beliefs,
                causal_graph=self.causal_graph,
                safety_kernel=self.kernel,
                history_summary=self._build_history_summary(),
            )
            full_prompt = full_req.to_prompt()
            self._llm_prompt_tokens += estimate_tokens(prompt)
            self._llm_prompt_tokens += estimate_tokens(self.task.system_prompt or ClampAI_SYSTEM_PROMPT)
            # Track savings as a negative "virtual" token count in errors log for now.
            saved = max(0, estimate_tokens(full_prompt) - estimate_tokens(prompt))
            if saved:
                self._errors.append(f"prompt_savings_tokens={saved}")
        except Exception:
            self._llm_prompt_tokens += estimate_tokens(prompt)
            self._llm_prompt_tokens += estimate_tokens(self.task.system_prompt or ClampAI_SYSTEM_PROMPT)

        # Delta-state append (token saver). Safety enforcement still sees full state.
        if self._last_prompt_state is not None:
            delta = compute_state_delta(self._last_prompt_state, self.current_state)
            prompt = prompt + "\n\n## STATE DELTA (since last decision)\n" + json.dumps(delta, indent=2, default=str)
        self._last_prompt_state = self.current_state

        t0 = _time.time()
        raw = self.llm.complete(
            prompt=prompt,
            system_prompt=self.task.system_prompt or ClampAI_SYSTEM_PROMPT,
            temperature=0.3,
        )
        latency = (_time.time() - t0) * 1000
        self._llm_calls += 1
        self._llm_output_tokens += estimate_tokens(raw)

        valid_ids = set(a.id for a in available)
        response = parse_llm_response(raw, valid_ids)
        response.latency_ms = latency
        return response

    def _execute_action(self, action: ActionSpec,
                        reasoning: str) -> Tuple[bool, str]:
        """
        Execute an action through the COMPLETE safety gauntlet.

        Flow:
          1. Gradient Tracker: Compute safety margins
          2. Active HJB Barrier: Check reachability → force Safe Hover if critical
          3. Reference Monitor: M1–M5 (IFC, CBF, QP, HJB, composition)
          4. Formal Kernel: T1–T7 (budget, termination, invariants, atomicity)
          5. Inverse Algebra: Record rollback metadata for recovery

        Returns (success, message).
        """
        self._actions_attempted += 1

        # ── Step 0: Compute Safety Gradients (Formal Margin Analysis) ──
        gradient_report = self.gradient_tracker.compute_gradients(self.current_state)
        should_hover_by_gradient, grad_msg = self.gradient_tracker.should_trigger_safe_hover(
            gradient_report
        )
        if should_hover_by_gradient:
            self._errors.append(f"Safety gradient triggered warning: {grad_msg}")

        # ── Step 1: Active HJB Reachability Barrier ──
        available_next = self.task.available_actions
        hjb_safe, hjb_check = self.hjb_barrier.check_and_enforce(
            self.current_state,
            action,
            available_next,
            current_step=self.kernel.step_count,
            max_steps=self.kernel.max_steps
        )

        if not hjb_safe:
            # HJB violation: force safe hover or rollback
            self._actions_rejected_safety += 1
            self.kernel.record_rejection(
                self.current_state, action,
                (hjb_check.recommendation,), reasoning)
            self.beliefs.observe(f"action:{action.id}:succeeds", False)

            # Decide recovery strategy
            recovery = choose_recovery_strategy(
                hjb_check,
                self.kernel.step_count,
                self.kernel.max_steps,
                is_reversible_available=(len(self._rollback_records) > 0)
            )

            # Attempt rollback if possible
            if recovery == RecoveryStrategy.ROLLBACK_ONE and self._rollback_records:
                record = self._rollback_records.pop()
                self.current_state = record.apply_rollback(self.current_state)
                self.kernel.rollback(record.state_before_snapshot, self.current_state, action)
                self._rollbacks += 1
                return True, f"Rollback triggered by HJB: {hjb_check.recommendation}"

            # Otherwise Safe Hover
            self._proof_steps.append(ProofStep(
                step_index=self.kernel.step_count,
                action_id=action.id,
                action_name=action.name,
                approved=False,
                reason="hjb_barrier",
                monitor_reason=hjb_check.recommendation,
                kernel_reasons=[],
                prompt_tokens_est=self._llm_prompt_tokens,
                output_tokens_est=self._llm_output_tokens,
            ))
            return False, f"HJB barrier: {hjb_check.recommendation}"

        # ── Step 2: Reference Monitor (Authoritative Enforcement) ──
        # Theorem M0: enforce() returns (safe, reason, repaired_action)
        monitor_safe, monitor_msg, repaired = self.monitor.enforce(
            action, self.current_state, available_next
        )

        if not monitor_safe:
            self._actions_rejected_safety += 1
            self.kernel.record_rejection(
                self.current_state, action,
                (monitor_msg,), reasoning)
            self.beliefs.observe(f"action:{action.id}:succeeds", False)
            self._proof_steps.append(ProofStep(
                step_index=self.kernel.step_count,
                action_id=action.id,
                action_name=action.name,
                approved=False,
                reason="monitor_reject",
                monitor_reason=monitor_msg,
                kernel_reasons=[],
                prompt_tokens_est=self._llm_prompt_tokens,
                output_tokens_est=self._llm_output_tokens,
            ))
            return False, f"Monitor rejected: {monitor_msg}"

        # Use repaired action if monitor produced one
        exec_action = repaired if repaired is not None else action

        # ── Step 3: Formal Safety Kernel (T1-T7) ──
        verdict = self.kernel.evaluate(self.current_state, exec_action)
        if not verdict.approved:
            self._actions_rejected_safety += 1
            self.kernel.record_rejection(
                self.current_state, exec_action,
                verdict.rejection_reasons, reasoning)
            self.beliefs.observe(f"action:{exec_action.id}:succeeds", False)
            self._proof_steps.append(ProofStep(
                step_index=self.kernel.step_count,
                action_id=exec_action.id,
                action_name=exec_action.name,
                approved=False,
                reason="kernel_reject",
                monitor_reason=monitor_msg,
                kernel_reasons=list(verdict.rejection_reasons),
                prompt_tokens_est=self._llm_prompt_tokens,
                output_tokens_est=self._llm_output_tokens,
            ))
            return False, f"Kernel rejected: {verdict.rejection_reasons}"

        # ── Step 4: Execute (atomic — T5) ──
        try:
            new_state, _trace_entry = self.kernel.execute(
                self.current_state, exec_action, reasoning)
        except Exception as e:
            self._errors.append(f"Execute error: {e}")
            self.beliefs.observe(f"action:{exec_action.id}:succeeds", False)
            self._proof_steps.append(ProofStep(
                step_index=self.kernel.step_count,
                action_id=exec_action.id,
                action_name=exec_action.name,
                approved=False,
                reason="execute_error",
                monitor_reason=monitor_msg,
                kernel_reasons=[str(e)],
                prompt_tokens_est=self._llm_prompt_tokens,
                output_tokens_est=self._llm_output_tokens,
            ))
            return False, f"Execution error: {e}"

        # ── Step 5: Record Rollback (T7 Inverse Algebra) ──
        # Compute and store inverse effects for future recovery
        try:
            rollback_record = InverseAlgebra.make_rollback_record(
                exec_action,
                self.current_state,
                new_state,
                timestamp=_time.time()
            )
            self._rollback_records.append(rollback_record)
        except Exception as e:
            self._errors.append(f"Rollback record error: {e}")
            # Non-fatal; execution still succeeds

        # ── Step 6: Observe outcome ──
        self.current_state = new_state
        self._state_history.append(new_state)
        self._actions_succeeded += 1
        self._consecutive_failures = 0

        self._proof_steps.append(ProofStep(
            step_index=self.kernel.step_count,
            action_id=exec_action.id,
            action_name=exec_action.name,
            approved=True,
            reason="approved",
            monitor_reason=monitor_msg,
            kernel_reasons=[],
            prompt_tokens_est=self._llm_prompt_tokens,
            output_tokens_est=self._llm_output_tokens,
        ))

        # Update beliefs
        self.beliefs.observe(f"action:{exec_action.id}:succeeds", True)
        self.causal_graph.mark_completed(exec_action.id)

        # Update progress
        progress = self._compute_progress()
        self.progress_monitor.record(self.kernel.step_count, progress)
        # Store progress in state for visibility
        self.current_state = self.current_state.with_updates(
            {'_progress': progress})

        return True, f"Executed {action.name}, progress={progress:.1%}"

    def run(self) -> ExecutionResult:
        """
        Main execution loop. Returns when done.

        Loop invariants (maintained every iteration):
          - self.kernel.budget.spent ≤ self.kernel.budget.budget  (T1)
          - self.kernel.step_count ≤ self.kernel.max_steps        (T2)
          - All invariants hold on self.current_state              (T3)
        """
        t_start = _time.time()

        # Verify initial invariants (blocking-mode only — monitoring invariants
        # never block execution, even if the initial state violates them).
        for inv in self.task.invariants:
            if inv.enforcement != "blocking":
                continue
            ok, msg = inv.check(self.current_state)
            if not ok:
                return self._make_result(
                    TerminationReason.ERROR, t_start,
                    errors=[f"Initial state violates invariant: {msg}"])

        while True:
            # ── Check goal ──
            if self.task.goal_predicate(self.current_state):
                return self._make_result(
                    TerminationReason.GOAL_ACHIEVED, t_start)

            # ── Check termination conditions ──
            if self.kernel.step_count >= self.kernel.max_steps:
                return self._make_result(
                    TerminationReason.STEP_LIMIT, t_start)

            if self._consecutive_failures >= self.task.max_consecutive_failures:
                return self._make_result(
                    TerminationReason.MAX_FAILURES, t_start)

            # ── Get available actions ──
            available = self._get_available_actions()
            if not available:
                return self._make_result(
                    TerminationReason.BUDGET_EXHAUSTED, t_start)

            # ── Compute values (formal analysis) ──
            values = self._compute_action_values(available)

            # ── Ask LLM to reason ──
            try:
                response = self._ask_llm(available, values)
            except Exception as e:
                self._errors.append(f"LLM error: {e}")
                # Fallback: use highest-value action
                response = self._fallback_selection(available, values)

            # ── Handle LLM stop request ──
            if response.should_stop:
                return self._make_result(
                    TerminationReason.LLM_STOP, t_start)

            # ── Validate LLM choice ──
            if not response.is_valid:
                self._errors.append(f"Invalid LLM response: {response.parse_errors}")
                self._consecutive_failures += 1
                response = self._fallback_selection(available, values)
                if response.should_stop:
                    return self._make_result(
                        TerminationReason.ERROR, t_start,
                        errors=self._errors)

            # ── Execute chosen action ──
            action = self.action_map.get(response.chosen_action_id)
            if action is None:
                self._errors.append(f"Action {response.chosen_action_id} not found")
                self._consecutive_failures += 1
                continue

            success, msg = self._execute_action(action, response.reasoning)
            if not success:
                self._consecutive_failures += 1

            # ── Check stuck ──
            if self.progress_monitor.is_stuck:
                # Don't terminate immediately — let LLM know it's stuck
                # via the progress monitor text in next iteration
                if self._consecutive_failures > self.task.stuck_patience:
                    return self._make_result(
                        TerminationReason.STUCK, t_start)

    def _fallback_selection(self, available: List[ActionSpec],
                            values: List[ActionValue]) -> ReasoningResponse:
        """When LLM fails, select highest-value READY action."""
        ready_ids = set(self.causal_graph.ready_actions(
            [a.id for a in available]))

        ready_values = [v for v in values if v.action_id in ready_ids]
        if not ready_values:
            return ReasoningResponse(
                chosen_action_id="", reasoning="No ready actions",
                expected_outcome="", risk_assessment="",
                alternative_considered="", should_stop=True,
                stop_reason="No ready actions available")

        best = max(ready_values, key=lambda v: v.value_score)
        return ReasoningResponse(
            chosen_action_id=best.action_id,
            reasoning=f"Fallback: selected highest-value action {best.action_id}",
            expected_outcome="", risk_assessment="",
            alternative_considered="", should_stop=False,
            stop_reason="")

    def _make_result(self, reason: TerminationReason,
                     t_start: float,
                     errors: Optional[List[str]] = None) -> ExecutionResult:
        progress = self._compute_progress()
        # Optional proof artifact (LLM-independent)
        if getattr(self.task, "proof_path", ""):
            try:
                trace_ok, head_hash = self.kernel.trace.verify_integrity()
                record = ProofRecord(
                    version="0.1",
                    created_at=_time.time(),
                    goal=self.task.goal,
                    budget=float(self.task.budget),
                    trace_hash_head=head_hash if trace_ok else "",
                    steps=list(self._proof_steps),
                )
                write_proof(self.task.proof_path, record)
            except Exception as e:
                self._errors.append(f"Proof write error: {e}")
        return ExecutionResult(
            goal_achieved=(reason == TerminationReason.GOAL_ACHIEVED),
            termination_reason=reason,
            final_state=self.current_state,
            total_cost=self.kernel.budget.spent,
            total_steps=self.kernel.step_count,
            goal_progress=progress,
            execution_time_s=_time.time() - t_start,
            trace_length=self.kernel.trace.length,
            beliefs_summary=self.beliefs.summary(),
            budget_summary=self.kernel.budget.summary(),
            errors=errors or self._errors,
            actions_attempted=self._actions_attempted,
            actions_succeeded=self._actions_succeeded,
            actions_rejected_safety=self._actions_rejected_safety,
            actions_rejected_reasoning=self._actions_rejected_reasoning,
            rollbacks=self._rollbacks,
            llm_calls=self._llm_calls,
        )


# Async execution engine


class AsyncOrchestrator:
    """
    Native-async execution engine. Drop-in async replacement for Orchestrator.

    Uses AsyncSafetyKernel so competing coroutines yield to the event loop
    on the locked commit step rather than blocking OS threads. All T1–T8
    guarantees are preserved: the async lock serialises budget charge,
    step increment, and trace append exactly as the threading lock does in
    the synchronous kernel.

    Usage::

        from clampai import AsyncOrchestrator, TaskDefinition
        from clampai.adapters import AsyncAnthropicAdapter
        import anthropic

        engine = AsyncOrchestrator(
            task,
            llm=AsyncAnthropicAdapter(anthropic.AsyncAnthropic()),
        )
        result = await engine.run()

    Multiple independent ``AsyncOrchestrator`` instances can share a single
    ``AsyncSafetyKernel`` for multi-agent budget enforcement — pass the kernel
    as the ``kernel`` keyword argument to override the auto-created one.

    Guarantee: T1–T8 PROVEN (same kernel logic as Orchestrator).
    """

    def __init__(
        self,
        task: TaskDefinition,
        llm: Optional[LLMAdapter] = None,
        *,
        kernel: Optional[AsyncSafetyKernel] = None,
    ) -> None:
        self.task = task
        self.llm = llm or MockLLMAdapter()

        self.kernel: AsyncSafetyKernel = kernel or AsyncSafetyKernel(
            budget=task.budget,
            invariants=task.invariants,
            min_action_cost=task.min_action_cost,
        )

        self._proof_steps: List[ProofStep] = []

        self.monitor = ReferenceMonitor(
            ifc_enabled=True,
            cbf_enabled=True,
            hjb_enabled=True,
        )
        self.monitor.add_cbf(
            h=lambda s: (task.budget - self.kernel.budget.spent) / max(task.budget, 1.0),
            alpha=0.1,
        )

        self.gradient_tracker = GradientTracker(task.invariants)
        self.hjb_barrier = ActiveHJBBarrier(
            basins=task.capture_basins or [],
            max_lookahead=min(5, int(task.budget / task.min_action_cost // 2)),
        )

        self._rollback_records: List[RollbackRecord] = []

        self.beliefs = BeliefState()
        self.causal_graph = CausalGraph()
        self.value_computer = ActionValueComputer(risk_aversion=task.risk_aversion)
        self.progress_monitor = ProgressMonitor(patience=task.stuck_patience)

        self._actions_attempted = 0
        self._actions_succeeded = 0
        self._actions_rejected_safety = 0
        self._actions_rejected_reasoning = 0
        self._rollbacks = 0
        self._llm_calls = 0
        self._llm_prompt_tokens = 0
        self._llm_output_tokens = 0
        self._last_prompt_state = None
        self._consecutive_failures = 0
        self._errors: List[str] = []

        self.current_state = task.initial_state
        self.action_map = {a.id: a for a in task.available_actions}
        self._state_history = [task.initial_state]

        if task.dependencies:
            for action_id, deps in task.dependencies.items():
                self.causal_graph.add_action(action_id, deps)
        for a in task.available_actions:
            if a.id not in (task.dependencies or {}):
                self.causal_graph.add_action(a.id)

        if task.priors:
            for key, (alpha, beta) in task.priors.items():
                self.beliefs.set_prior(key, alpha, beta)

    def _compute_progress(self) -> float:
        if self.task.goal_progress_fn:
            return self.task.goal_progress_fn(self.current_state)
        if self.task.goal_predicate(self.current_state):
            return 1.0
        return 0.0

    def _get_available_actions(self) -> List[ActionSpec]:
        available = []
        for action in self.task.available_actions:
            can_afford, _ = self.kernel.budget.can_afford(action.cost)
            if not can_afford:
                continue
            if action.id in self.causal_graph._completed:
                sim = action.simulate(self.current_state)
                if sim == self.current_state:
                    continue
            available.append(action)
        return available

    def _compute_action_values(self, actions: List[ActionSpec]) -> List[ActionValue]:
        progress = self._compute_progress()
        steps_left = self.kernel.max_steps - self.kernel.step_count
        return [
            self.value_computer.compute(
                action=a,
                state=self.current_state,
                beliefs=self.beliefs,
                budget_remaining=self.kernel.budget.remaining,
                goal_progress=progress,
                steps_remaining=steps_left,
            )
            for a in actions
        ]

    def _build_history_summary(self, last_n: int = 5) -> str:
        entries = self.kernel.trace.last_n(last_n)
        if not entries:
            return "(no actions taken yet)"
        lines = []
        for e in entries:
            icon = "ok" if e.approved else "--"
            lines.append(f"  [{icon}] {e.action_name} (${e.cost:.2f})")
            if e.reasoning_summary:
                lines.append(f"      → {e.reasoning_summary[:100]}")
        return "\n".join(lines)

    async def _ask_llm(
        self,
        available: List[ActionSpec],
        values: List[ActionValue],
    ) -> ReasoningResponse:
        """Build structured prompt, query LLM asynchronously, parse response."""

        if should_skip_llm(values):
            best = sorted(values, key=lambda v: v.value_score, reverse=True)[0]
            return ReasoningResponse(
                chosen_action_id=best.action_id,
                reasoning="Dominant-strategy skip: selected top-valued action without LLM call.",
                expected_outcome="Progress",
                risk_assessment="Low (large value gap)",
                alternative_considered="Second-best action had much lower value score.",
                should_stop=False,
                stop_reason="",
                raw_response="",
                parse_errors=[],
            )

        state_for_prompt = self.current_state
        sensitivity_scores: Dict[str, float] = {}
        try:
            d = self.current_state.to_dict()
            sensitivity_scores = {k: 0.0 for k in d.keys()}
            value_by_id = {v.action_id: v.value_score for v in values}
            for a in available:
                w = abs(float(value_by_id.get(a.id, 0.0)))
                for k in a.affected_variables():
                    if k in sensitivity_scores:
                        sensitivity_scores[k] += w
            for k in list(sensitivity_scores.keys()):
                if k.startswith("_"):
                    sensitivity_scores[k] = float("inf")
            threshold = float(getattr(self.task, "sensitivity_threshold", 0.05) or 0.05)
            max_keys = int(getattr(self.task, "max_prompt_state_keys", 20) or 20)
            kept = [
                k for k, s in sensitivity_scores.items()
                if s == float("inf") or s >= threshold
            ]
            if len(kept) > max_keys:
                kept = sorted(
                    kept, key=lambda kk: sensitivity_scores.get(kk, 0.0), reverse=True
                )[:max_keys]
            if kept:
                state_for_prompt = State({k: d[k] for k in kept if k in d})
        except Exception:
            state_for_prompt = self.current_state

        request = ReasoningRequest(
            goal=self.task.goal,
            state=state_for_prompt,
            available_actions=available,
            action_values=values,
            beliefs=self.beliefs,
            causal_graph=self.causal_graph,
            safety_kernel=self.kernel._kernel,
            history_summary=self._build_history_summary(),
        )
        prompt = request.to_prompt()

        try:
            full_req = ReasoningRequest(
                goal=self.task.goal,
                state=self.current_state,
                available_actions=available,
                action_values=values,
                beliefs=self.beliefs,
                causal_graph=self.causal_graph,
                safety_kernel=self.kernel._kernel,
                history_summary=self._build_history_summary(),
            )
            full_prompt = full_req.to_prompt()
            self._llm_prompt_tokens += estimate_tokens(prompt)
            self._llm_prompt_tokens += estimate_tokens(
                self.task.system_prompt or ClampAI_SYSTEM_PROMPT
            )
            saved = max(0, estimate_tokens(full_prompt) - estimate_tokens(prompt))
            if saved:
                self._errors.append(f"prompt_savings_tokens={saved}")
        except Exception:
            self._llm_prompt_tokens += estimate_tokens(prompt)
            self._llm_prompt_tokens += estimate_tokens(
                self.task.system_prompt or ClampAI_SYSTEM_PROMPT
            )

        if self._last_prompt_state is not None:
            delta = compute_state_delta(self._last_prompt_state, self.current_state)
            prompt = (
                prompt
                + "\n\n## STATE DELTA (since last decision)\n"
                + json.dumps(delta, indent=2, default=str)
            )
        self._last_prompt_state = self.current_state

        t0 = _time.time()
        raw = await self.llm.acomplete(
            prompt=prompt,
            system_prompt=self.task.system_prompt or ClampAI_SYSTEM_PROMPT,
            temperature=0.3,
        )
        latency = (_time.time() - t0) * 1000
        self._llm_calls += 1
        self._llm_output_tokens += estimate_tokens(raw)

        valid_ids = set(a.id for a in available)
        response = parse_llm_response(raw, valid_ids)
        response.latency_ms = latency
        return response

    async def _execute_action(
        self,
        action: ActionSpec,
        reasoning: str,
    ) -> Tuple[bool, str]:
        """
        Execute an action through the complete safety gauntlet, asynchronously.

        Steps:
          1. Gradient Tracker: safety margin analysis
          2. Active HJB Barrier: reachability check
          3. Reference Monitor: IFC, CBF, QP, composition (M1–M5)
          4. AsyncSafetyKernel.evaluate(): fast pre-check without lock
          5. AsyncSafetyKernel.execute_atomic(): locked evaluate-and-commit (T1–T7)
          6. Inverse Algebra: rollback metadata (T7)

        Returns ``(success, message)``.
        """
        self._actions_attempted += 1

        gradient_report = self.gradient_tracker.compute_gradients(self.current_state)
        should_hover_by_gradient, grad_msg = self.gradient_tracker.should_trigger_safe_hover(
            gradient_report
        )
        if should_hover_by_gradient:
            self._errors.append(f"Safety gradient triggered warning: {grad_msg}")

        available_next = self.task.available_actions
        hjb_safe, hjb_check = self.hjb_barrier.check_and_enforce(
            self.current_state,
            action,
            available_next,
            current_step=self.kernel.step_count,
            max_steps=self.kernel.max_steps,
        )

        if not hjb_safe:
            self._actions_rejected_safety += 1
            self.kernel.record_rejection(
                self.current_state, action, (hjb_check.recommendation,), reasoning
            )
            self.beliefs.observe(f"action:{action.id}:succeeds", False)

            recovery = choose_recovery_strategy(
                hjb_check,
                self.kernel.step_count,
                self.kernel.max_steps,
                is_reversible_available=(len(self._rollback_records) > 0),
            )

            if recovery == RecoveryStrategy.ROLLBACK_ONE and self._rollback_records:
                record = self._rollback_records.pop()
                self.current_state = record.apply_rollback(self.current_state)
                self.kernel.rollback(record.state_before_snapshot, self.current_state, action)
                self._rollbacks += 1
                return True, f"Rollback triggered by HJB: {hjb_check.recommendation}"

            self._proof_steps.append(ProofStep(
                step_index=self.kernel.step_count,
                action_id=action.id,
                action_name=action.name,
                approved=False,
                reason="hjb_barrier",
                monitor_reason=hjb_check.recommendation,
                kernel_reasons=[],
                prompt_tokens_est=self._llm_prompt_tokens,
                output_tokens_est=self._llm_output_tokens,
            ))
            return False, f"HJB barrier: {hjb_check.recommendation}"

        monitor_safe, monitor_msg, repaired = self.monitor.enforce(
            action, self.current_state, available_next
        )

        if not monitor_safe:
            self._actions_rejected_safety += 1
            self.kernel.record_rejection(
                self.current_state, action, (monitor_msg,), reasoning
            )
            self.beliefs.observe(f"action:{action.id}:succeeds", False)
            self._proof_steps.append(ProofStep(
                step_index=self.kernel.step_count,
                action_id=action.id,
                action_name=action.name,
                approved=False,
                reason="monitor_reject",
                monitor_reason=monitor_msg,
                kernel_reasons=[],
                prompt_tokens_est=self._llm_prompt_tokens,
                output_tokens_est=self._llm_output_tokens,
            ))
            return False, f"Monitor rejected: {monitor_msg}"

        exec_action = repaired if repaired is not None else action

        # Fast pre-check (no lock). execute_atomic re-evaluates under asyncio.Lock.
        verdict = await self.kernel.evaluate(self.current_state, exec_action)
        if not verdict.approved:
            self._actions_rejected_safety += 1
            self.kernel.record_rejection(
                self.current_state, exec_action, verdict.rejection_reasons, reasoning
            )
            self.beliefs.observe(f"action:{exec_action.id}:succeeds", False)
            self._proof_steps.append(ProofStep(
                step_index=self.kernel.step_count,
                action_id=exec_action.id,
                action_name=exec_action.name,
                approved=False,
                reason="kernel_reject",
                monitor_reason=monitor_msg,
                kernel_reasons=list(verdict.rejection_reasons),
                prompt_tokens_est=self._llm_prompt_tokens,
                output_tokens_est=self._llm_output_tokens,
            ))
            return False, f"Kernel rejected: {verdict.rejection_reasons}"

        # Atomic evaluate-and-commit under asyncio.Lock (T1, T5, T6).
        try:
            new_state, _trace_entry = await self.kernel.execute_atomic(
                self.current_state, exec_action, reasoning
            )
        except RuntimeError as e:
            # TOCTOU: another coroutine consumed budget between evaluate and execute_atomic.
            self._actions_rejected_safety += 1
            self.kernel.record_rejection(
                self.current_state, exec_action, (str(e),), reasoning
            )
            self.beliefs.observe(f"action:{exec_action.id}:succeeds", False)
            self._errors.append(f"Execute error: {e}")
            self._proof_steps.append(ProofStep(
                step_index=self.kernel.step_count,
                action_id=exec_action.id,
                action_name=exec_action.name,
                approved=False,
                reason="execute_error",
                monitor_reason=monitor_msg,
                kernel_reasons=[str(e)],
                prompt_tokens_est=self._llm_prompt_tokens,
                output_tokens_est=self._llm_output_tokens,
            ))
            return False, f"Execution error: {e}"

        try:
            rollback_record = InverseAlgebra.make_rollback_record(
                exec_action,
                self.current_state,
                new_state,
                timestamp=_time.time(),
            )
            self._rollback_records.append(rollback_record)
        except Exception as e:
            self._errors.append(f"Rollback record error: {e}")

        self.current_state = new_state
        self._state_history.append(new_state)
        self._actions_succeeded += 1
        self._consecutive_failures = 0

        self._proof_steps.append(ProofStep(
            step_index=self.kernel.step_count,
            action_id=exec_action.id,
            action_name=exec_action.name,
            approved=True,
            reason="approved",
            monitor_reason=monitor_msg,
            kernel_reasons=[],
            prompt_tokens_est=self._llm_prompt_tokens,
            output_tokens_est=self._llm_output_tokens,
        ))

        self.beliefs.observe(f"action:{exec_action.id}:succeeds", True)
        self.causal_graph.mark_completed(exec_action.id)

        progress = self._compute_progress()
        self.progress_monitor.record(self.kernel.step_count, progress)
        self.current_state = self.current_state.with_updates({"_progress": progress})

        return True, f"Executed {action.name}, progress={progress:.1%}"

    async def run(self) -> ExecutionResult:
        """
        Async main execution loop. Await this to run to completion.

        Loop invariants maintained every iteration:
          - kernel.budget.spent ≤ kernel.budget.budget          (T1)
          - kernel.step_count  ≤ kernel.max_steps               (T2)
          - All blocking invariants hold on current_state        (T3)

        Returns:
            An :class:`ExecutionResult` describing the outcome.
        """
        t_start = _time.time()

        for inv in self.task.invariants:
            if inv.enforcement != "blocking":
                continue
            ok, msg = inv.check(self.current_state)
            if not ok:
                return self._make_result(
                    TerminationReason.ERROR, t_start,
                    errors=[f"Initial state violates invariant: {msg}"],
                )

        while True:
            if self.task.goal_predicate(self.current_state):
                return self._make_result(TerminationReason.GOAL_ACHIEVED, t_start)

            if self.kernel.step_count >= self.kernel.max_steps:
                return self._make_result(TerminationReason.STEP_LIMIT, t_start)

            if self._consecutive_failures >= self.task.max_consecutive_failures:
                return self._make_result(TerminationReason.MAX_FAILURES, t_start)

            available = self._get_available_actions()
            if not available:
                return self._make_result(TerminationReason.BUDGET_EXHAUSTED, t_start)

            values = self._compute_action_values(available)

            try:
                response = await self._ask_llm(available, values)
            except Exception as e:
                self._errors.append(f"LLM error: {e}")
                response = self._fallback_selection(available, values)

            if response.should_stop:
                return self._make_result(TerminationReason.LLM_STOP, t_start)

            if not response.is_valid:
                self._errors.append(f"Invalid LLM response: {response.parse_errors}")
                self._consecutive_failures += 1
                response = self._fallback_selection(available, values)
                if response.should_stop:
                    return self._make_result(
                        TerminationReason.ERROR, t_start, errors=self._errors
                    )

            action = self.action_map.get(response.chosen_action_id)
            if action is None:
                self._errors.append(f"Action {response.chosen_action_id} not found")
                self._consecutive_failures += 1
                continue

            success, _msg = await self._execute_action(action, response.reasoning)
            if not success:
                self._consecutive_failures += 1

            if self.progress_monitor.is_stuck:
                if self._consecutive_failures > self.task.stuck_patience:
                    return self._make_result(TerminationReason.STUCK, t_start)

    def _fallback_selection(
        self,
        available: List[ActionSpec],
        values: List[ActionValue],
    ) -> ReasoningResponse:
        ready_ids = set(self.causal_graph.ready_actions([a.id for a in available]))
        ready_values = [v for v in values if v.action_id in ready_ids]
        if not ready_values:
            return ReasoningResponse(
                chosen_action_id="",
                reasoning="No ready actions",
                expected_outcome="",
                risk_assessment="",
                alternative_considered="",
                should_stop=True,
                stop_reason="No ready actions available",
            )
        best = max(ready_values, key=lambda v: v.value_score)
        return ReasoningResponse(
            chosen_action_id=best.action_id,
            reasoning=f"Fallback: selected highest-value action {best.action_id}",
            expected_outcome="",
            risk_assessment="",
            alternative_considered="",
            should_stop=False,
            stop_reason="",
        )

    def _make_result(
        self,
        reason: TerminationReason,
        t_start: float,
        errors: Optional[List[str]] = None,
    ) -> ExecutionResult:
        progress = self._compute_progress()
        if getattr(self.task, "proof_path", ""):
            try:
                trace_ok, head_hash = self.kernel.trace.verify_integrity()
                record = ProofRecord(
                    version="0.1",
                    created_at=_time.time(),
                    goal=self.task.goal,
                    budget=float(self.task.budget),
                    trace_hash_head=head_hash if trace_ok else "",
                    steps=list(self._proof_steps),
                )
                write_proof(self.task.proof_path, record)
            except Exception as e:
                self._errors.append(f"Proof write error: {e}")
        return ExecutionResult(
            goal_achieved=(reason == TerminationReason.GOAL_ACHIEVED),
            termination_reason=reason,
            final_state=self.current_state,
            total_cost=self.kernel.budget.spent,
            total_steps=self.kernel.step_count,
            goal_progress=progress,
            execution_time_s=_time.time() - t_start,
            trace_length=self.kernel.trace.length,
            beliefs_summary=self.beliefs.summary(),
            budget_summary=self.kernel.budget.summary(),
            errors=errors or self._errors,
            actions_attempted=self._actions_attempted,
            actions_succeeded=self._actions_succeeded,
            actions_rejected_safety=self._actions_rejected_safety,
            actions_rejected_reasoning=self._actions_rejected_reasoning,
            rollbacks=self._rollbacks,
            llm_calls=self._llm_calls,
        )


# Default system prompt

ClampAI_SYSTEM_PROMPT = """You are an AI agent operating within the ClampAI safety framework.

YOUR ROLE:
You select actions to achieve a goal. You are given formal analysis
(action values, beliefs, dependencies, budget) and must make a rational
decision.

CLAMPAINTS YOU CANNOT OVERRIDE:
- Budget limits are enforced by the safety kernel
- Invariants are enforced by the safety kernel
- Step limits are enforced by the safety kernel
- You can only choose from the READY actions listed

YOUR DECISION PROCESS:
1. Read the goal and current progress
2. Review the action value analysis (it's already computed for you)
3. Consider dependencies — don't pick BLOCKED actions
4. Consider beliefs — prefer actions with high success probability
5. Consider budget — don't waste money on low-value actions
6. Select the best action and explain your reasoning

RESPOND WITH VALID JSON ONLY.
"""
