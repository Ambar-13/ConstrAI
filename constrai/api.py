"""
constrai/api.py — @safe: zero-config safety for any callable.

Wraps any function with a ConstrAI SafetyKernel so that budget (T1) and
invariant (T3) guarantees are enforced on every call without modifying
the underlying function.

Usage:
    from constrai import safe, Invariant

    @safe(budget=50.0, cost_per_call=5.0)
    def call_llm(prompt: str) -> str:
        return openai_client.chat(prompt)

    # Raises SafetyViolation once 10 calls exhaust the $50 budget.
    # The wrapped function exposes .kernel and .audit_log for introspection.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from .formal import (
    ActionSpec,
    Effect,
    Invariant,
    SafetyKernel,
    SafetyVerdict,
    State,
)

F = TypeVar("F", bound=Callable[..., Any])


class SafetyViolation(RuntimeError):
    """Raised when a @safe decorated call is blocked by the kernel.

    Attributes:
        verdict: The SafetyVerdict that triggered the block, or None if the
            kernel raised before producing a structured verdict.
    """

    def __init__(self, reason: str, verdict: Optional[SafetyVerdict] = None) -> None:
        super().__init__(reason)
        self.verdict = verdict


class _SafeWrapper:
    """Runtime wrapper installed by @safe.

    Thread-safe: concurrent calls are serialized through a per-wrapper lock
    so that evaluate → execute is atomic with respect to internal state.

    Not intended for direct instantiation. Use safe() instead.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        budget: float,
        cost_per_call: float,
        invariants: Sequence[Invariant],
        state_fn: Optional[Callable[[], Dict[str, Any]]],
        action_name: Optional[str],
    ) -> None:
        self._fn = fn
        self._cost_per_call = cost_per_call
        self._state_fn = state_fn
        self._action_name = action_name or fn.__name__
        self._budget = budget
        self._lock = threading.Lock()

        # Internal state for invariant evaluation (call_count, total_cost).
        # The kernel tracks budget independently via BudgetController.
        self._state = State({"call_count": 0, "total_cost": 0.0})
        self._kernel = SafetyKernel(budget=budget, invariants=list(invariants))

        functools.update_wrapper(self, fn)

    @property
    def kernel(self) -> SafetyKernel:
        """The underlying SafetyKernel for introspection and testing."""
        return self._kernel

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Structured audit log of all approved calls. JSON-serializable."""
        return [
            {
                "step": entry.step,
                "action": entry.action_id,
                "cost": entry.cost,
                "timestamp": entry.timestamp,
                "approved": entry.approved,
            }
            for entry in self._kernel.trace.entries
        ]

    def reset(self) -> None:
        """Reset call history and reinstate the full budget.

        Useful between test cases or logical sessions without needing to
        recreate the decorated function.
        """
        with self._lock:
            self._state = State({"call_count": 0, "total_cost": 0.0})
            self._kernel = SafetyKernel(
                budget=self._budget,
                invariants=self._kernel.invariants,
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            current_count = int(self._state.get("call_count", 0))
            current_cost = float(self._state.get("total_cost", 0.0))

            # Build evaluation state: merge external state with internal tracking
            # so invariants can reference call_count, total_cost, and user fields.
            merged: Dict[str, Any] = {}
            if self._state_fn is not None:
                try:
                    merged.update(self._state_fn())
                except Exception:
                    pass
            merged["call_count"] = current_count
            merged["total_cost"] = current_cost
            check_state = State(merged)

            action = ActionSpec(
                id=f"{self._action_name}_{current_count + 1}",
                name=self._action_name,
                description=f"Call #{current_count + 1} to {self._fn.__qualname__}",
                effects=(
                    Effect("call_count", "increment", 1),
                    Effect("total_cost", "increment", self._cost_per_call),
                ),
                cost=self._cost_per_call,
            )

            # Evaluate first to surface a structured verdict in the error message.
            verdict = self._kernel.evaluate(check_state, action)
            if not verdict.approved:
                reasons = (
                    "; ".join(verdict.rejection_reasons)
                    if verdict.rejection_reasons
                    else "kernel rejected (no reasons recorded)"
                )
                raise SafetyViolation(
                    f"Call to '{self._action_name}' blocked by ConstrAI: {reasons}",
                    verdict,
                )

            # Execute commits atomically: budget charge + step + trace.
            new_state, _ = self._kernel.execute(check_state, action)
            self._state = State({
                "call_count": new_state.get("call_count", current_count + 1),
                "total_cost": new_state.get(
                    "total_cost", current_cost + self._cost_per_call
                ),
            })

        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<safe({self._fn.__qualname__!r}) "
            f"calls={self._state.get('call_count', 0)} "
            f"remaining=${self._kernel.budget.remaining:.2f}>"
        )


def safe(
    budget: float,
    *,
    cost_per_call: float = 1.0,
    invariants: Sequence[Invariant] = (),
    state_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    action_name: Optional[str] = None,
) -> Callable[[F], _SafeWrapper]:
    """Make any function safe by enforcing budget and invariant guarantees.

    Wraps a function so that every invocation is vetted by a SafetyKernel
    before execution. Budget (T1) and invariants (T3) are enforced; violations
    raise SafetyViolation instead of reaching the underlying function.

    Args:
        budget: Total cost budget across all calls. Raises SafetyViolation
            once cumulative cost would exceed this value (T1 guarantee).
        cost_per_call: Cost charged per invocation. Defaults to 1.0.
        invariants: Additional blocking invariants evaluated before each call
            (T3 guarantee). Combine with the factory functions in
            ``constrai.invariants`` for common patterns such as rate limiting,
            resource ceilings, and value ranges.
        state_fn: Optional zero-argument callable returning a dict of
            additional state variables for invariant evaluation. If omitted,
            only ``call_count`` and ``total_cost`` are available to invariants.
        action_name: Override the action name used in traces and error messages.
            Defaults to the decorated function's ``__name__``.

    Returns:
        A wrapper that preserves the original function's signature and
        docstring, exposing ``.kernel`` (SafetyKernel) and ``.audit_log``
        (list[dict]) properties, and a ``.reset()`` method.

    Raises:
        SafetyViolation: On budget exceeded or invariant violation.

    Example:
        >>> from constrai import safe, Invariant
        >>> @safe(budget=20.0, cost_per_call=5.0)
        ... def expensive_call(x: int) -> int:
        ...     return x * 2
        >>> expensive_call(3)
        6
        >>> # After 4 calls (cumulative cost = 20.0), the 5th raises SafetyViolation.
    """
    def decorator(fn: F) -> _SafeWrapper:
        return _SafeWrapper(
            fn=fn,
            budget=budget,
            cost_per_call=cost_per_call,
            invariants=invariants,
            state_fn=state_fn,
            action_name=action_name,
        )

    return decorator


# Alias for backwards-compatibility in case anything already imports constrai_safe.
constrai_safe = safe
