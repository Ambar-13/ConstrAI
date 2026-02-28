"""
clampai.testing — Test utilities for verifying ClampAI safety properties.

Zero runtime dependencies beyond clampai itself.  All helpers are pure Python
and work with any test runner (pytest, unittest, or plain ``assert``).

Usage::

    from clampai.testing import SafetyHarness, make_state, make_action
    from clampai import rate_limit_invariant

    def test_budget_blocks_overspend():
        with SafetyHarness(budget=5.0) as h:
            h.assert_allowed(make_state(), make_action("cheap", cost=3.0))
            h.assert_blocked(make_state(), make_action("expensive", cost=10.0))
            h.assert_budget_remaining(5.0)  # nothing was executed, just evaluated

    def test_invariant_blocks_after_threshold():
        inv = rate_limit_invariant("calls", 2)
        with SafetyHarness(budget=100.0, invariants=[inv]) as h:
            s = make_state(calls=0)
            s = h.execute(s, make_action("a", cost=1.0, calls=1))
            s = h.execute(s, make_action("b", cost=1.0, calls=2))
            h.assert_blocked(s, make_action("c", cost=1.0, calls=3),
                             reason_contains="calls")

Guarantee: HEURISTIC — test helper only; not part of the formal safety proof.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence, Set

from .formal import ActionSpec, Effect, Invariant, SafetyKernel, State


def make_state(**vars: Any) -> State:
    """Return an immutable :class:`State` from keyword arguments.

    Args:
        **vars: State variable names and their initial values.

    Returns:
        A new :class:`State` containing the given variables.

    Example::

        s = make_state(files_created=0, human_approved=False)
    """
    return State(dict(vars))


def make_action(
    name: str,
    cost: float = 1.0,
    *,
    reversible: bool = True,
    tags: Sequence[str] = (),
    **effects: Any,
) -> ActionSpec:
    """Return an :class:`ActionSpec` with auto-generated id and set-effects.

    Each keyword argument beyond ``cost``, ``reversible``, and ``tags``
    becomes an ``Effect(key, "set", value)`` on the action.

    Args:
        name: Human-readable action name.  A unique ``id`` is generated
            automatically as ``"<name>-<short-uuid>"``.
        cost: Resource cost charged when the action is executed.
            Default: ``1.0``.
        reversible: Whether the action supports rollback (T7).
            Default: ``True``.
        tags: Tuple of string tags for grouping and filtering.
        **effects: Each keyword argument is converted to
            ``Effect(key, "set", value)``.

    Returns:
        A new frozen :class:`ActionSpec`.

    Example::

        a = make_action("send_email", cost=2.0, emails_sent=1)
        # effects: Effect("emails_sent", "set", 1)
    """
    action_effects = tuple(
        Effect(key, "set", value) for key, value in effects.items()
    )
    action_id = f"{name}-{uuid.uuid4().hex[:8]}"
    return ActionSpec(
        id=action_id,
        name=name,
        description=f"Test action: {name}",
        effects=action_effects,
        cost=cost,
        reversible=reversible,
        tags=tuple(tags),
    )


class SafetyHarness:
    """Context manager wrapping a :class:`SafetyKernel` with assertion helpers.

    Designed for use in pytest or any ``assert``-based test suite.
    The harness creates a fresh kernel on entry and exposes a small set of
    fluent assertion methods that produce informative ``AssertionError``
    messages on failure.

    Args:
        budget: Total resource budget passed to the underlying
            :class:`SafetyKernel`.
        invariants: Safety invariants to enforce.
        min_action_cost: Minimum action cost (T2 termination parameter).
        emergency_actions: Action IDs that bypass cost/step limits (T8).

    Usage::

        with SafetyHarness(budget=10.0, invariants=[my_inv]) as h:
            h.assert_allowed(state, action)
            h.assert_blocked(state, bad_action, reason_contains="budget")
            new_state = h.execute(state, action)
            h.assert_budget_remaining(8.0)
            h.assert_step_count(1)
    """

    def __init__(
        self,
        budget: float,
        invariants: Sequence[Invariant] = (),
        *,
        min_action_cost: float = 0.001,
        emergency_actions: Optional[Set[str]] = None,
    ) -> None:
        self._budget = budget
        self._invariants = list(invariants)
        self._min_action_cost = min_action_cost
        self._emergency_actions = emergency_actions or set()
        self._kernel = self._make_kernel()

    def _make_kernel(self) -> SafetyKernel:
        return SafetyKernel(
            self._budget,
            self._invariants,
            min_action_cost=self._min_action_cost,
            emergency_actions=set(self._emergency_actions),
        )

    @property
    def kernel(self) -> SafetyKernel:
        """The underlying :class:`SafetyKernel` for direct inspection."""
        return self._kernel

    def assert_allowed(
        self,
        state: State,
        action: ActionSpec,
        msg: Optional[str] = None,
    ) -> None:
        """Assert that ``action`` is approved by the kernel on ``state``.

        Args:
            state: State to evaluate against.
            action: Action to check.
            msg: Optional prefix for the AssertionError message.

        Raises:
            AssertionError: If the action is NOT approved.
        """
        verdict = self._kernel.evaluate(state, action)
        if not verdict.approved:
            reasons = "; ".join(verdict.rejection_reasons) or "no detail"
            prefix = f"{msg}: " if msg else ""
            raise AssertionError(
                f"{prefix}Expected action '{action.name}' to be ALLOWED "
                f"but it was BLOCKED. Reasons: {reasons}"
            )

    def assert_blocked(
        self,
        state: State,
        action: ActionSpec,
        *,
        reason_contains: Optional[str] = None,
        msg: Optional[str] = None,
    ) -> None:
        """Assert that ``action`` is blocked by the kernel on ``state``.

        Args:
            state: State to evaluate against.
            action: Action to check.
            reason_contains: Optional substring that must appear somewhere
                in the rejection reasons.  Useful for asserting which specific
                check blocked the action (e.g. ``"budget"`` or ``"rate_limit"``).
            msg: Optional prefix for the AssertionError message.

        Raises:
            AssertionError: If the action IS approved (not blocked), or if
                ``reason_contains`` is given but not found in the reasons.
        """
        verdict = self._kernel.evaluate(state, action)
        prefix = f"{msg}: " if msg else ""
        if verdict.approved:
            raise AssertionError(
                f"{prefix}Expected action '{action.name}' to be BLOCKED "
                f"but it was APPROVED."
            )
        if reason_contains is not None:
            all_reasons = " ".join(verdict.rejection_reasons)
            if reason_contains.lower() not in all_reasons.lower():
                raise AssertionError(
                    f"{prefix}Action '{action.name}' was blocked, but "
                    f"reasons do not contain {reason_contains!r}. "
                    f"Actual reasons: {all_reasons!r}"
                )

    def assert_budget_remaining(
        self, expected: float, tol: float = 0.01
    ) -> None:
        """Assert that the remaining budget is approximately ``expected``.

        Args:
            expected: Expected remaining budget value.
            tol: Absolute tolerance for floating-point comparison.

        Raises:
            AssertionError: If ``|remaining - expected| > tol``.
        """
        actual = self._kernel.budget.remaining
        if abs(actual - expected) > tol:
            raise AssertionError(
                f"Expected budget remaining ≈ {expected:.4f} (±{tol}), "
                f"got {actual:.4f}"
            )

    def assert_step_count(self, expected: int) -> None:
        """Assert that the kernel's step count equals ``expected``.

        Args:
            expected: Expected step count.

        Raises:
            AssertionError: If ``kernel.step_count != expected``.
        """
        actual = self._kernel.step_count
        if actual != expected:
            raise AssertionError(
                f"Expected step_count == {expected}, got {actual}"
            )

    def execute(
        self,
        state: State,
        action: ActionSpec,
        reasoning: str = "",
    ) -> State:
        """Evaluate and execute ``action`` atomically; return the new state.

        Args:
            state: Current state.
            action: Action to execute.
            reasoning: Optional reasoning summary for the trace entry.

        Returns:
            The new :class:`State` after the action's effects are applied.

        Raises:
            RuntimeError: If the action is blocked by the kernel.
        """
        new_state, _ = self._kernel.evaluate_and_execute_atomic(
            state, action, reasoning
        )
        return new_state

    def reset(self) -> None:
        """Recreate the kernel, restoring the original budget and step count."""
        self._kernel = self._make_kernel()

    def __enter__(self) -> "SafetyHarness":
        return self

    def __exit__(self, *_: Any) -> None:
        pass
