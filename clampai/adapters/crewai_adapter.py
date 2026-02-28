"""
clampai.adapters.crewai_adapter — CrewAI safety integration.

Wraps CrewAI tools and agent steps with ClampAI budget and invariant enforcement.
Every tool call or task step is evaluated by a SafetyKernel before execution.

    from clampai.adapters import ClampAISafeCrewTool, ClampAICrewCallback
    from clampai.invariants import pii_guard_invariant, string_length_invariant

    # Wrap a callable as a ClampAI-enforced CrewAI tool:
    @safe_crew_tool(budget=100.0, cost=2.0)
    def search_web(query: str) -> str:
        \"\"\"Search the web for information.\"\"\"
        return do_search(query)

    # Or wrap an existing callable directly:
    safe_search = ClampAISafeCrewTool(
        func=search_web,
        name="web_search",
        description="Search the web for information",
        budget=50.0,
        cost=2.0,
        invariants=[pii_guard_invariant("query")],
    )

    # As a CrewAI step/task callback (monitors all agent steps):
    callback = ClampAICrewCallback(budget=200.0, cost_per_step=1.0)
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        step_callback=callback.step_callback,
        task_callback=callback.task_callback,
    )

Safety guarantees on every tool call or step:

- T1 (Budget Safety): budget is charged atomically; calls beyond the cap
  always raise ClampAICrewError — the tool is never invoked.
- T3 (Invariant Preservation): blocking invariants are checked against the
  state built from the tool arguments before any execution.
- T5 (Atomicity): charge + trace-append are all-or-nothing.

Requires: No additional pip install for the safety layer.
For full CrewAI integration: pip install crewai>=0.28
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Sequence

from clampai.formal import ActionSpec, Invariant, SafetyKernel, State


class ClampAICrewError(RuntimeError):
    """Raised when ClampAI blocks a CrewAI tool call or agent step."""


class ClampAICrewBudgetError(ClampAICrewError):
    """Raised when a CrewAI call is blocked due to budget exhaustion."""


class ClampAICrewInvariantError(ClampAICrewError):
    """Raised when a CrewAI call is blocked due to an invariant violation."""


class ClampAISafeCrewTool:
    """
    A CrewAI-compatible callable wrapped with ClampAI safety enforcement.

    Behaves as a plain callable ``(*args, **kwargs) -> Any``, satisfying
    CrewAI's tool interface. Can be used as a function tool directly.

    Each call:
    1. Checks all blocking invariants against the call's state.
    2. Charges ``cost`` against the running budget (T1, T5).
    3. Runs the wrapped function if all checks pass.
    4. Returns the function's result unchanged.

    Args:
        func:
            The callable to wrap. Must accept the same arguments you pass
            to the tool.
        name:
            Tool name (used in audit logs and error messages).
        description:
            Human-readable description of what the tool does.
        budget:
            Total budget available for all calls to this tool instance.
        cost:
            Budget charged per call (default 1.0).
        invariants:
            Sequence of ``Invariant`` objects checked before each call.

    Example::

        safe_tool = ClampAISafeCrewTool(
            func=lambda q: search(q),
            name="search",
            description="Search the web",
            budget=50.0,
            cost=2.0,
            invariants=[string_length_invariant("query", 500)],
        )
        result = safe_tool("latest AI news")
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str = "",
        description: str = "",
        budget: float,
        cost: float = 1.0,
        invariants: Sequence[Invariant] = (),
    ) -> None:
        self.func = func
        self.name = name or getattr(func, "__name__", "crew_tool")
        self.description = description
        self.budget = budget
        self.cost = cost
        self.invariants: List[Invariant] = list(invariants)

        self._kernel = SafetyKernel(self.budget, self.invariants)
        self._call_count = 0
        self._action = ActionSpec(
            id=self.name,
            name=self.name,
            description=description or f"CrewAI tool: {self.name}",
            effects=(),
            cost=cost,
            reversible=False,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool with full ClampAI safety enforcement.

        Args:
            *args: Positional arguments forwarded to the wrapped function.
            **kwargs: Keyword arguments forwarded to the wrapped function.

        Returns:
            The result of the wrapped function.

        Raises:
            ClampAICrewBudgetError: If the budget is exhausted.
            ClampAICrewInvariantError: If a blocking invariant is violated.
        """
        self._call_count += 1

        state_data: Dict[str, Any] = {"call_count": self._call_count}
        if args:
            state_data["query"] = str(args[0])
        state_data.update({k: str(v) for k, v in kwargs.items()})

        clampai_state = State(state_data)

        try:
            self._kernel.evaluate_and_execute_atomic(clampai_state, self._action)
        except RuntimeError as exc:
            msg = str(exc)
            if any(kw in msg.lower() for kw in ("budget", "afford", "exceeded", "insufficient")):
                raise ClampAICrewBudgetError(
                    f"[ClampAI] Tool '{self.name}' blocked — budget exhausted: {exc}"
                ) from exc
            raise ClampAICrewInvariantError(
                f"[ClampAI] Tool '{self.name}' blocked — invariant violated: {exc}"
            ) from exc

        return self.func(*args, **kwargs)

    def reset(self) -> None:
        """Recreate the kernel with the original budget and invariants."""
        self._kernel = SafetyKernel(self.budget, self.invariants)
        self._call_count = 0

    @property
    def budget_remaining(self) -> float:
        """Budget remaining in the kernel's BudgetController."""
        return self._kernel.budget.remaining

    @property
    def step_count(self) -> int:
        """Number of times this tool has been successfully invoked."""
        return self._kernel.step_count

    def __repr__(self) -> str:
        return (
            f"ClampAISafeCrewTool(name={self.name!r}, budget={self.budget}, "
            f"cost={self.cost}, invariants={len(self.invariants)})"
        )


class ClampAICrewCallback:
    """
    CrewAI step/task callback enforcing ClampAI budget and invariants.

    Attach to a ``Crew`` instance via ``step_callback`` and ``task_callback``.
    Budget is charged on every agent step; invariants are checked against the
    step output.

    Args:
        budget:
            Total budget for all steps in the crew run.
        cost_per_step:
            Budget charged per agent step (default 1.0).
        invariants:
            Invariants checked against each step's output state.

    Example::

        callback = ClampAICrewCallback(budget=200.0, cost_per_step=1.0)

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            step_callback=callback.step_callback,
            task_callback=callback.task_callback,
        )
        crew.kickoff()

        print(f"Steps taken: {callback.step_count}")
        print(f"Budget remaining: {callback.budget_remaining:.2f}")
    """

    def __init__(
        self,
        budget: float,
        *,
        cost_per_step: float = 1.0,
        invariants: Sequence[Invariant] = (),
    ) -> None:
        self._budget = budget
        self._cost = cost_per_step
        self._invariants: List[Invariant] = list(invariants)

        self._kernel = SafetyKernel(self._budget, self._invariants)
        self._step_num = 0
        self._task_num = 0

    def step_callback(self, step_output: Any) -> None:
        """
        Called after each agent step. Charges budget and checks invariants.

        Args:
            step_output: The output from the agent step (CrewAI AgentAction).

        Raises:
            ClampAICrewBudgetError: If the budget is exhausted.
            ClampAICrewInvariantError: If a blocking invariant is violated.
        """
        self._step_num += 1

        state_data: Dict[str, Any] = {
            "step_number": self._step_num,
            "task_number": self._task_num,
        }
        if step_output is not None:
            output_str = str(step_output)
            state_data["step_output"] = output_str[:2000]

        action = ActionSpec(
            id=f"step_{self._step_num}",
            name=f"Agent Step {self._step_num}",
            description="CrewAI agent step",
            effects=(),
            cost=self._cost,
            reversible=False,
        )

        try:
            self._kernel.evaluate_and_execute_atomic(State(state_data), action)
        except RuntimeError as exc:
            msg = str(exc)
            if any(kw in msg.lower() for kw in ("budget", "afford", "exceeded", "insufficient")):
                raise ClampAICrewBudgetError(
                    f"[ClampAI] Step {self._step_num} blocked — budget exhausted: {exc}"
                ) from exc
            raise ClampAICrewInvariantError(
                f"[ClampAI] Step {self._step_num} blocked — invariant violated: {exc}"
            ) from exc

    def task_callback(self, task_output: Any) -> None:
        """
        Called after each task completes. Records task completion.

        Args:
            task_output: The output from the completed task.
        """
        self._task_num += 1

    def reset(self) -> None:
        """Recreate the kernel with the original budget and invariants."""
        self._kernel = SafetyKernel(self._budget, self._invariants)
        self._step_num = 0
        self._task_num = 0

    @property
    def budget_remaining(self) -> float:
        """Budget remaining across all steps in this crew run."""
        return self._kernel.budget.remaining

    @property
    def step_count(self) -> int:
        """Number of steps that passed the safety check."""
        return self._kernel.step_count

    def __repr__(self) -> str:
        return (
            f"ClampAICrewCallback(budget={self._budget}, "
            f"cost_per_step={self._cost}, "
            f"invariants={len(self._invariants)})"
        )


def safe_crew_tool(
    budget: float,
    *,
    cost: float = 1.0,
    name: Optional[str] = None,
    description: str = "",
    invariants: Sequence[Invariant] = (),
) -> Callable[[Callable[..., Any]], ClampAISafeCrewTool]:
    """
    Decorator that wraps a function as a ClampAI-enforced CrewAI tool.

    Args:
        budget:
            Total budget for all calls to this tool.
        cost:
            Budget charged per call.
        name:
            Tool name. Defaults to the decorated function's name.
        description:
            Tool description. Defaults to the function's docstring.
        invariants:
            Invariants checked before each call.

    Example::

        @safe_crew_tool(budget=100.0, cost=2.0)
        def search_web(query: str) -> str:
            \"\"\"Search the web for information.\"\"\"
            return do_search(query)
    """

    def decorator(func: Callable[..., Any]) -> ClampAISafeCrewTool:
        tool_name = name or getattr(func, "__name__", "crew_tool")
        tool_desc = description or (func.__doc__ or "").strip()
        tool = ClampAISafeCrewTool(
            func,
            name=tool_name,
            description=tool_desc,
            budget=budget,
            cost=cost,
            invariants=invariants,
        )
        functools.update_wrapper(tool, func)
        return tool

    return decorator


__all__ = [
    "ClampAICrewBudgetError",
    "ClampAICrewCallback",
    "ClampAICrewError",
    "ClampAICrewInvariantError",
    "ClampAISafeCrewTool",
    "safe_crew_tool",
]
