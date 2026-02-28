"""
clampai.adapters.langchain_callback — LangChain callback-based safety enforcement.

Wraps ANY LangChain agent or chain with ClampAI budget and invariant enforcement
using the standard LangChain callback interface. No changes to the agent are
required — attach the handler when calling the agent.

    from clampai.adapters import ClampAICallbackHandler
    from clampai.invariants import pii_guard_invariant, rate_limit_invariant

    handler = ClampAICallbackHandler(
        budget=50.0,
        cost_per_action=2.0,
        invariants=[
            pii_guard_invariant("tool_input"),
            rate_limit_invariant("tool_calls", 20),
        ],
    )

    # Standard LangChain agent (unchanged):
    result = agent_executor.invoke(
        {"input": "Summarise my emails"},
        config={"callbacks": [handler]},
    )

    print(handler.budget_remaining)   # budget after the run
    print(handler.actions_blocked)    # number of blocked tool calls

Two enforcement points:

- **on_agent_action**: fired before EACH tool call. Budget is charged and
  invariants are checked here. If the budget is exhausted or an invariant is
  violated the handler raises ``ClampAICallbackError``, which surfaces through
  the agent as a tool execution error and halts the run.

State passed to invariants is built from the agent action:

    {
        "tool":        "search",           # name of the tool being invoked
        "tool_input":  "quarterly report", # tool input coerced to str
        "tool_calls":  3,                  # running count of tool calls
        "budget_spent": 6.0,              # net budget spent so far
    }

Extra fields can be injected via ``state_fn``:

    handler = ClampAICallbackHandler(
        budget=50.0,
        state_fn=lambda action: {"user_id": current_user()},
    )

Safety guarantees on every tool call:

- T1 (Budget Safety): budget is charged atomically inside
  ``SafetyKernel.evaluate_and_execute_atomic``; the tool is never called if
  the remaining budget would go negative.
- T3 (Invariant Preservation): blocking invariants are checked on the state
  built from the agent action before any tool executes.
- T5 (Atomicity): budget debit and trace append are all-or-nothing.

Requires: pip install 'clampai[langchain]'  (langchain-core>=0.3)
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

try:
    from langchain_core.callbacks.base import BaseCallbackHandler  # type: ignore[import]
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Stub used when langchain-core is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


from clampai.formal import ActionSpec, Invariant, SafetyKernel, State


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required for ClampAICallbackHandler. "
            "Install it with: pip install 'clampai[langchain]'"
        )


class ClampAICallbackError(RuntimeError):
    """Raised by the callback handler when ClampAI blocks a tool call."""


class ClampAICallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangChain callback handler that enforces ClampAI budget and invariants.

    Attach to any LangChain agent or chain via the ``callbacks`` parameter.
    No modifications to the agent are required.

    Each tool call (agent action) is evaluated by a ``SafetyKernel`` before
    the tool executes. Calls that exceed the budget or violate a blocking
    invariant raise ``ClampAICallbackError``, halting the agent.

    Args:
        budget:
            Total budget available for all tool calls in this session.
        cost_per_action:
            Budget charged per tool call (default 1.0).
        invariants:
            ``Invariant`` objects checked against every tool call's state.
        state_fn:
            Optional callable ``(action) -> dict`` to inject extra fields
            into the state checked by invariants. Called with the LangChain
            ``AgentAction`` object.
        raise_on_block:
            If True (default), raise ``ClampAICallbackError`` when blocked.
            If False, silently allow the tool to run (monitoring-only mode).

    Example::

        handler = ClampAICallbackHandler(budget=50.0, cost_per_action=2.0)
        result = agent_executor.invoke(
            {"input": "Search for Q4 results"},
            config={"callbacks": [handler]},
        )
        print(f"Budget used: {50.0 - handler.budget_remaining:.2f}")
    """

    raise_error = True  # LangChain BaseCallbackHandler compatibility

    def __init__(
        self,
        budget: float,
        *,
        cost_per_action: float = 1.0,
        invariants: Sequence[Invariant] = (),
        state_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        raise_on_block: bool = True,
    ) -> None:
        _require_langchain()
        super().__init__()
        self._budget = budget
        self._cost = cost_per_action
        self._invariants: List[Invariant] = list(invariants)
        self._state_fn = state_fn
        self._raise_on_block = raise_on_block

        self._kernel = SafetyKernel(budget, self._invariants)
        self._tool_call_count = 0
        self._actions_blocked = 0
        self._lock = threading.Lock()

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Called before a tool is invoked. Enforces budget and invariants.

        Raises:
            ClampAICallbackError: If the budget is exhausted or a blocking
                invariant is violated (and ``raise_on_block=True``).
        """
        with self._lock:
            self._tool_call_count += 1
            tool_name = getattr(action, "tool", str(action))
            raw_input = getattr(action, "tool_input", "")
            input_str = str(raw_input) if not isinstance(raw_input, str) else raw_input

            state_data: Dict[str, Any] = {
                "tool": tool_name,
                "tool_input": input_str,
                "tool_calls": self._tool_call_count,
                "budget_spent": self._kernel.budget.spent_net,
            }

            if self._state_fn is not None:
                try:
                    extra = self._state_fn(action)
                    if extra and isinstance(extra, dict):
                        state_data.update(extra)
                except Exception:
                    pass

            agent_action = ActionSpec(
                id=f"tool_call_{self._tool_call_count}",
                name=tool_name,
                description=f"LangChain tool call: {tool_name}",
                effects=(),
                cost=self._cost,
                reversible=False,
            )

            try:
                self._kernel.evaluate_and_execute_atomic(State(state_data), agent_action)
            except RuntimeError as exc:
                self._actions_blocked += 1
                if self._raise_on_block:
                    raise ClampAICallbackError(
                        f"[ClampAI] Tool call '{tool_name}' blocked: {exc}"
                    ) from exc

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """No-op: primary enforcement is in on_agent_action."""
        pass

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """No-op: tool completed normally."""
        pass

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """No-op: tool raised an error (already handled)."""
        pass

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """No-op: agent finished normally."""
        pass

    @property
    def budget_remaining(self) -> float:
        """Budget remaining across all tool calls in this session."""
        return self._kernel.budget.remaining

    @property
    def actions_blocked(self) -> int:
        """Number of tool calls that were blocked by ClampAI."""
        return self._actions_blocked

    @property
    def tool_calls_made(self) -> int:
        """Total number of tool calls attempted (including blocked ones)."""
        return self._tool_call_count

    @property
    def step_count(self) -> int:
        """Number of tool calls that passed the safety check."""
        return self._kernel.step_count

    def reset(self) -> None:
        """
        Recreate the kernel with the original budget and invariants.

        Use between agent runs or test cases where the budget should start fresh.
        """
        with self._lock:
            self._kernel = SafetyKernel(self._budget, self._invariants)
            self._tool_call_count = 0
            self._actions_blocked = 0

    def __repr__(self) -> str:
        return (
            f"ClampAICallbackHandler(budget={self._budget}, "
            f"cost_per_action={self._cost}, "
            f"invariants={len(self._invariants)})"
        )


__all__ = [
    "ClampAICallbackError",
    "ClampAICallbackHandler",
]
