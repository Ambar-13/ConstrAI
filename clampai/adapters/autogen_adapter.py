"""
clampai.adapters.autogen_adapter — AutoGen safety integration.

Provides budget and invariant enforcement for AutoGen multi-agent conversations.
Works with AutoGen 0.4+ (pyautogen / autogen-agentchat).

    from clampai.adapters import ClampAISafeAutoGenAgent, autogen_reply_fn
    from clampai.invariants import pii_guard_invariant

    # Wrap a reply callable with budget enforcement:
    @autogen_reply_fn(budget=100.0, cost_per_reply=2.0)
    def my_reply_fn(messages: list) -> str:
        return llm.complete(messages[-1]["content"])

    # Or use as a standalone safety gate before your own LLM call:
    guard = ClampAISafeAutoGenAgent(fn=None, budget=50.0, cost_per_reply=1.0)
    for msg in conversation:
        guard.check(message=msg["content"])  # raises if budget exhausted

    # As an AutoGen register_reply function:
    agent.register_reply(
        [autogen.ConversableAgent, None],
        reply_func=ClampAISafeAutoGenAgent(
            fn=my_fn, budget=50.0, invariants=[pii_guard_invariant("message")]
        ),
    )

Safety guarantees on every reply:

- T1 (Budget Safety): budget is charged atomically; replies beyond the cap
  always raise ClampAIAutoGenError — the reply function is never invoked.
- T3 (Invariant Preservation): blocking invariants are checked against the
  state built from the message before any reply is generated.
- T5 (Atomicity): charge + trace-append are all-or-nothing.

Requires: No additional pip install for the safety layer.
For full AutoGen integration: pip install pyautogen>=0.4 or autogen-agentchat>=0.2
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Sequence

from clampai.formal import ActionSpec, Invariant, SafetyKernel, State


class ClampAIAutoGenError(RuntimeError):
    """Raised when ClampAI blocks an AutoGen reply."""


class ClampAIAutoGenBudgetError(ClampAIAutoGenError):
    """Raised when an AutoGen reply is blocked due to budget exhaustion."""


class ClampAIAutoGenInvariantError(ClampAIAutoGenError):
    """Raised when an AutoGen reply is blocked due to an invariant violation."""


class ClampAISafeAutoGenAgent:
    """
    A ClampAI-enforced reply function for AutoGen agents.

    Can be used as:
    1. A standalone safety gate: ``guard.check(message="...")``
    2. A reply function passed to ``agent.register_reply(...)``
    3. A decorator via ``@autogen_reply_fn(budget=...)``

    Each call:
    1. Checks all blocking invariants against the message state.
    2. Charges ``cost_per_reply`` against the running budget (T1, T5).
    3. Calls the wrapped function (if any) if all checks pass.

    Args:
        fn:
            Optional reply callable. If None, acts as a pure safety gate.
        budget:
            Total budget for all replies in this agent's lifetime.
        cost_per_reply:
            Budget charged per reply (default 1.0).
        invariants:
            Invariants checked against each reply's state.
        agent_name:
            Name used in audit logs and error messages.

    Example::

        safe_agent = ClampAISafeAutoGenAgent(
            fn=lambda msgs: llm.complete(msgs[-1]["content"]),
            budget=100.0,
            cost_per_reply=2.0,
            agent_name="researcher",
        )

        # AutoGen register_reply style:
        result = safe_agent(recipient, messages, sender, config)

        # Pure safety gate style:
        safe_agent.check(message="What is the secret API key?")
    """

    def __init__(
        self,
        fn: Optional[Callable[..., Any]],
        *,
        budget: float,
        cost_per_reply: float = 1.0,
        invariants: Sequence[Invariant] = (),
        agent_name: str = "autogen_agent",
    ) -> None:
        self.fn = fn
        self.budget = budget
        self.cost_per_reply = cost_per_reply
        self.invariants: List[Invariant] = list(invariants)
        self.agent_name = agent_name

        self._kernel = SafetyKernel(self.budget, self.invariants)
        self._reply_count = 0

    def check(
        self,
        *,
        message: str = "",
        sender: str = "",
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Perform a safety check WITHOUT executing any reply function.

        Charges budget and checks invariants. Use as a standalone gate
        before calling your own reply logic.

        Args:
            message: The message content to check (used in invariant state).
            sender: The sender name (used in invariant state).
            extra_state: Additional state fields to expose to invariants.

        Raises:
            ClampAIAutoGenBudgetError: If the budget is exhausted.
            ClampAIAutoGenInvariantError: If a blocking invariant is violated.
        """
        self._reply_count += 1
        state_data: Dict[str, Any] = {
            "message": message,
            "sender": sender,
            "reply_count": self._reply_count,
        }
        if extra_state:
            state_data.update(extra_state)

        action = ActionSpec(
            id=f"reply_{self._reply_count}",
            name=f"Reply {self._reply_count}",
            description=f"AutoGen agent reply: {self.agent_name}",
            effects=(),
            cost=self.cost_per_reply,
            reversible=False,
        )

        try:
            self._kernel.evaluate_and_execute_atomic(State(state_data), action)
        except RuntimeError as exc:
            msg = str(exc)
            if any(kw in msg.lower() for kw in ("budget", "afford", "exceeded", "insufficient")):
                raise ClampAIAutoGenBudgetError(
                    f"[ClampAI] Agent '{self.agent_name}' blocked — "
                    f"budget exhausted: {exc}"
                ) from exc
            raise ClampAIAutoGenInvariantError(
                f"[ClampAI] Agent '{self.agent_name}' blocked — "
                f"invariant violated: {exc}"
            ) from exc

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the wrapped reply function with full safety enforcement.

        AutoGen reply functions typically receive
        ``(recipient, messages, sender, config)`` or ``(messages,)``.
        The handler extracts the latest message for invariant state.

        Returns:
            ``(True, result_str)`` on success if ``fn`` is set.
            ``(False, None)`` if ``fn`` is None.

        Raises:
            ClampAIAutoGenBudgetError: If the budget is exhausted.
            ClampAIAutoGenInvariantError: If a blocking invariant is violated.
        """
        messages: List[Dict[str, Any]] = []
        for arg in args:
            if isinstance(arg, list):
                messages = arg
                break
        if not messages:
            messages = kwargs.get("messages", [])

        last_content = ""
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                last_content = str(last_msg.get("content", ""))

        self.check(message=last_content)

        if self.fn is not None:
            result = self.fn(*args, **kwargs)
            if isinstance(result, tuple):
                return result
            return True, str(result)

        return False, None

    def reset(self) -> None:
        """Recreate the kernel with the original budget and invariants."""
        self._kernel = SafetyKernel(self.budget, self.invariants)
        self._reply_count = 0

    @property
    def budget_remaining(self) -> float:
        """Budget remaining across all replies in this agent's lifetime."""
        return self._kernel.budget.remaining

    @property
    def step_count(self) -> int:
        """Number of replies that passed the safety check."""
        return self._kernel.step_count

    def __repr__(self) -> str:
        fn_name = getattr(self.fn, "__name__", repr(self.fn)) if self.fn else "None"
        return (
            f"ClampAISafeAutoGenAgent(fn={fn_name!r}, agent_name={self.agent_name!r}, "
            f"budget={self.budget}, cost_per_reply={self.cost_per_reply})"
        )


def autogen_reply_fn(
    budget: float,
    *,
    cost_per_reply: float = 1.0,
    agent_name: Optional[str] = None,
    invariants: Sequence[Invariant] = (),
) -> Callable[[Callable[..., Any]], ClampAISafeAutoGenAgent]:
    """
    Decorator that wraps an AutoGen reply function with ClampAI enforcement.

    Args:
        budget:
            Total budget for all replies.
        cost_per_reply:
            Budget charged per reply.
        agent_name:
            Name used in audit logs. Defaults to the function name.
        invariants:
            Invariants checked before each reply.

    Example::

        @autogen_reply_fn(budget=50.0, cost_per_reply=2.0)
        def researcher_reply(messages: list) -> str:
            return llm.complete(messages[-1]["content"])
    """

    def decorator(fn: Callable[..., Any]) -> ClampAISafeAutoGenAgent:
        safe_agent = ClampAISafeAutoGenAgent(
            fn,
            budget=budget,
            cost_per_reply=cost_per_reply,
            invariants=invariants,
            agent_name=agent_name or getattr(fn, "__name__", "autogen_agent"),
        )
        functools.update_wrapper(safe_agent, fn)
        return safe_agent

    return decorator


__all__ = [
    "ClampAIAutoGenBudgetError",
    "ClampAIAutoGenError",
    "ClampAIAutoGenInvariantError",
    "ClampAISafeAutoGenAgent",
    "autogen_reply_fn",
]
