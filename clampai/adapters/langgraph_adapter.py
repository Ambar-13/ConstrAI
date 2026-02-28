"""
clampai.adapters.langgraph_adapter — LangGraph safety node integration.

Wraps ClampAI budget enforcement and invariant checking into LangGraph-compatible
callables. Every node executes only after the safety kernel has approved the
action; if the budget is exhausted or an invariant is violated the node raises
before the wrapped function is invoked.

    from clampai.adapters import clampai_node, budget_guard, invariant_guard
    from clampai.invariants import no_delete_invariant, rate_limit_invariant

    @clampai_node(budget=100.0, cost_per_step=2.0)
    def research_node(state: dict) -> dict:
        return {"result": "..."}

    # Pre-built guard nodes:
    graph.add_node("budget_check", budget_guard(budget=100.0))
    graph.add_node("safety_check", invariant_guard([
        no_delete_invariant("audit_log"),
        rate_limit_invariant("api_calls", 50),
    ]))

Safety guarantees on every node call:

- T1 (Budget Safety): the budget charge happens atomically inside
  ``SafetyKernel.execute_atomic``; the node function is never invoked if the
  remaining budget would go negative.
- T3 (Invariant Preservation): invariants are checked against the current state
  before the node runs. Blocking-mode invariants halt the graph on violation.
- T5 (Atomicity): the budget debit and trace append are all-or-nothing.

LangGraph nodes are plain Python callables ``(state: dict) -> dict``.
``SafetyNode.__call__`` satisfies this interface directly.

Requires: No additional pip install — uses only ``clampai.formal``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Sequence

from clampai.formal import ActionSpec, Invariant, SafetyKernel, State


class ClampAISafetyError(RuntimeError):
    """Raised when ClampAI blocks a LangGraph node."""


class ClampAIBudgetError(ClampAISafetyError):
    """Raised when a node is blocked because the budget is exhausted."""


class ClampAIInvariantError(ClampAISafetyError):
    """Raised when a node is blocked because an invariant is violated."""


class SafetyNode:
    """
    A LangGraph-compatible callable that enforces ClampAI budget and invariants.

    Each call to this node:

    1. Checks all blocking invariants against the current state.
    2. Charges ``cost`` against the running budget (T1, T5).
    3. Runs the wrapped function if all checks pass.
    4. Returns the function's ``dict`` result unchanged.

    The node holds a ``SafetyKernel`` that persists across calls, so budget
    depletion is tracked across the lifetime of the graph execution.  Call
    ``reset()`` to restore the budget between runs.

    Args:
        fn:
            The LangGraph node function ``(state: dict) -> dict``.
        budget:
            Total budget available for all calls to this node.
        invariants:
            Sequence of ``Invariant`` objects checked before each call.
        cost:
            Budget charged per invocation.
        action_id:
            Identifier used in the execution trace.  Defaults to ``fn.__name__``.

    Example::

        node = SafetyNode(my_fn, budget=50.0, cost=1.0)
        graph.add_node("my_node", node)
    """

    def __init__(
        self,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        budget: float,
        invariants: Sequence[Invariant] = (),
        cost: float = 1.0,
        action_id: Optional[str] = None,
    ) -> None:
        self.fn = fn
        self.budget = budget
        self.invariants: List[Invariant] = list(invariants)
        self.cost = cost
        _default_id: str = str(getattr(fn, "__name__", None) or "node")
        self.action_id: str = action_id if action_id is not None else _default_id

        self._kernel = SafetyKernel(self.budget, self.invariants)
        self._action = ActionSpec(
            id=self.action_id,
            name=self.action_id,
            description=f"LangGraph node: {self.action_id}",
            effects=(),
            cost=cost,
            reversible=False,
        )

    # Public interface

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node with full safety enforcement.

        Args:
            state: The LangGraph state dict.

        Returns:
            The dict returned by the wrapped node function.

        Raises:
            ClampAIBudgetError: If the budget is exhausted.
            ClampAIInvariantError: If a blocking invariant is violated.
            ClampAISafetyError: For any other safety violation.
        """
        clampai_state = State(state)

        try:
            self._kernel.evaluate_and_execute_atomic(clampai_state, self._action)
        except Exception as exc:
            msg = str(exc)
            if any(kw in msg.lower() for kw in ("budget", "afford", "exceeded", "insufficient")):
                raise ClampAIBudgetError(
                    f"[ClampAI] Node '{self.action_id}' blocked — budget exhausted: {exc}"
                ) from exc
            raise ClampAIInvariantError(
                f"[ClampAI] Node '{self.action_id}' blocked — invariant violated: {exc}"
            ) from exc

        return self.fn(state)

    def reset(self) -> None:
        """Recreate the kernel with the original budget and invariants."""
        self._kernel = SafetyKernel(self.budget, self.invariants)

    @property
    def budget_remaining(self) -> float:
        """Budget remaining in the kernel's BudgetController."""
        return self._kernel.budget.remaining

    @property
    def step_count(self) -> int:
        """Number of times this node has been successfully invoked."""
        return self._kernel.step_count

    def __repr__(self) -> str:
        return (
            f"SafetyNode({self.fn.__name__!r}, budget={self.budget}, "
            f"cost={self.cost}, invariants={len(self.invariants)})"
        )


# Decorator

def clampai_node(
    budget: float = 100.0,
    *,
    cost_per_step: float = 1.0,
    invariants: Sequence[Invariant] = (),
    action_id: Optional[str] = None,
) -> Callable[[Callable], SafetyNode]:
    """
    Decorator that wraps a LangGraph node with ClampAI safety enforcement.

    Equivalent to ``SafetyNode(fn, budget=budget, cost=cost_per_step, ...)``,
    but written as a decorator for ergonomics.  The resulting object is a
    ``SafetyNode`` (callable) with the original function's ``__name__`` and
    ``__doc__`` preserved.

    Args:
        budget:
            Total budget for this node across all invocations.
        cost_per_step:
            Budget charged per call.
        invariants:
            Invariants checked before each invocation.
        action_id:
            Trace identifier.  Defaults to the decorated function's name.

    Example::

        @clampai_node(budget=200.0, cost_per_step=2.0)
        def summarise_node(state: dict) -> dict:
            return {"summary": "..."}
    """

    def decorator(fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> SafetyNode:
        node = SafetyNode(
            fn,
            budget=budget,
            invariants=invariants,
            cost=cost_per_step,
            action_id=action_id or fn.__name__,
        )
        functools.update_wrapper(node, fn)
        return node

    return decorator


# Pre-built guard nodes

def budget_guard(
    budget: float,
    *,
    cost_per_step: float = 1.0,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create a LangGraph node that enforces a budget cap.

    Each invocation charges ``cost_per_step`` against ``budget``.  When the
    budget is exhausted the node raises ``ClampAIBudgetError`` (HTTP analogue:
    429 Too Many Requests).  The node is a pass-through: it returns an empty
    dict so LangGraph state is unchanged.

    The returned callable persists budget state across calls.  To reset between
    graph runs, reassign the node by calling ``budget_guard(...)`` again.

    Args:
        budget: Total budget available across all invocations.
        cost_per_step: Budget charged per call.

    Example::

        graph.add_node("budget_gate", budget_guard(budget=50.0))
    """
    kernel = SafetyKernel(budget, [])
    _action = ActionSpec(
        id="budget_gate",
        name="Budget Gate",
        description="ClampAI budget enforcement node",
        effects=(),
        cost=cost_per_step,
        reversible=False,
    )

    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            kernel.evaluate_and_execute_atomic(State(state), _action)
        except Exception as exc:
            raise ClampAIBudgetError(
                f"[ClampAI] Budget exhausted: {exc}"
            ) from exc
        return {}

    _node.__name__ = "budget_guard"
    _node.__doc__ = f"ClampAI budget gate (budget={budget}, cost={cost_per_step})"
    return _node


def invariant_guard(
    invariants: Sequence[Invariant],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create a LangGraph node that checks invariants on the current state.

    This node enforces **only** invariants — it charges no budget.  Use it to
    add safety checkpoints anywhere in a graph without affecting budget tracking.
    The node is a pass-through: it returns an empty dict so LangGraph state is
    unchanged.

    Blocking-mode invariants raise ``ClampAIInvariantError`` on violation.
    Monitoring-mode invariants are checked but do not raise.

    Args:
        invariants: Sequence of ``Invariant`` objects to check.

    Example::

        graph.add_node("data_guard", invariant_guard([
            no_delete_invariant("audit_log"),
            pii_guard_invariant("user_output"),
        ]))
    """
    _invs: List[Invariant] = list(invariants)

    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        s = State(state)
        violations: List[str] = []
        for inv in _invs:
            ok, msg = inv.check(s)
            if not ok and getattr(inv, "enforcement", "blocking") == "blocking":
                violations.append(f"'{inv.name}': {msg}")
        if violations:
            raise ClampAIInvariantError(
                "[ClampAI] Invariant violation(s): " + "; ".join(violations)
            )
        return {}

    _node.__name__ = "invariant_guard"
    _node.__doc__ = (
        f"ClampAI invariant gate ({len(_invs)} invariant(s): "
        f"{', '.join(getattr(i, 'name', '?') for i in _invs)})"
    )
    return _node


__all__ = [
    "ClampAIBudgetError",
    "ClampAIInvariantError",
    "ClampAISafetyError",
    "SafetyNode",
    "budget_guard",
    "clampai_node",
    "invariant_guard",
]
