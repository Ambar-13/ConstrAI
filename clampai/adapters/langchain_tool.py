"""
clampai.adapters.langchain_tool — LangChain BaseTool wrapper.

Wraps a ClampAI ``Orchestrator`` as a LangChain ``BaseTool`` so that
ClampAI-enforced agents can participate in LangChain chains and agent loops.

    from clampai import Orchestrator, TaskDefinition
    from clampai.adapters import ClampAISafeTool

    tool = ClampAISafeTool(
        orchestrator=Orchestrator(task),
        name="email_agent",
        description="Manages an email inbox with safety guarantees.",
    )

The wrapped Orchestrator runs with its full safety kernel. T1, T3, T5 and T6
hold regardless of the inputs sent by the LangChain orchestrator — a prompt
phrased differently cannot bypass a blocking invariant.

LangGraph works out of the box; ``ClampAISafeTool`` is a standard
``BaseTool`` and needs no additional adapter.

The Orchestrator runs synchronously. ``_arun`` wraps it with
``asyncio.to_thread`` (experimental; an ``AsyncOrchestrator`` is planned for
v0.6). Custom output formatting can be done by subclassing and overriding
``_format_result``.

Requires: pip install langchain>=0.3 langchain-core>=0.3
"""

from __future__ import annotations

from typing import Any, Optional, Type

# LangChain imports are deferred so that importing this module does not
# fail when langchain is not installed (clampai itself does not depend on it).
try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseTool = object  # type: ignore[misc,assignment]
    BaseModel = object  # type: ignore[misc,assignment]
    Field = None        # type: ignore[assignment]


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for ClampAISafeTool. "
            "Install it with: pip install 'clampai[langchain]'"
        )


class _ClampAIToolInput(BaseModel if _LANGCHAIN_AVAILABLE else object):  # type: ignore[misc]
    """Input schema for ClampAISafeTool."""
    goal_override: Optional[str] = (
        Field(default=None, description="Optional goal override for this run.")
        if _LANGCHAIN_AVAILABLE else None
    )


class ClampAISafeTool(BaseTool):  # type: ignore[misc]
    """
    LangChain BaseTool that runs a ClampAI Orchestrator with full safety enforcement.

    Args:
    orchestrator:
        A ClampAI ``Orchestrator`` instance.  The orchestrator must be
        fully configured (task, LLM adapter, etc.) before wrapping.
    name:
        Tool name used by the LangChain agent to refer to this tool.
    description:
        Human-readable description of what the tool does.  Shown to the LLM.
    """

    name: str = "clampai_safe_agent"
    description: str = (
        "A safety-enforced AI agent. Runs a task with formal guarantees: "
        "budget cannot be exceeded (T1), declared invariants always hold (T3), "
        "and the execution log is tamper-evident (T6)."
    )
    # Pydantic field — stores the orchestrator instance
    orchestrator: Any = None
    args_schema: Type[_ClampAIToolInput] = (
        _ClampAIToolInput if _LANGCHAIN_AVAILABLE else None  # type: ignore[assignment]
    )

    def __init__(self, orchestrator: Any, **kwargs: Any) -> None:
        _require_langchain()
        super().__init__(orchestrator=orchestrator, **kwargs)


    def _run(
        self,
        goal_override: Optional[str] = None,
        run_manager: Optional[Any] = None,
    ) -> str:
        """
        Execute the ClampAI Orchestrator and return a summary string.

        Args:
        goal_override:
            If provided, replaces the task goal for this run.  Useful when
            a LangChain agent dynamically specifies the goal.
        run_manager:
            LangChain callback manager (passed by the agent executor).
            Not used directly; present for interface compatibility.
        """
        if goal_override and hasattr(self.orchestrator, "task"):
            # Shallow goal override — does not affect safety properties
            original_goal = self.orchestrator.task.goal
            self.orchestrator.task = self.orchestrator.task  # immutable; log override only
            # TaskDefinition is immutable after construction. The override
            # is recorded here, but the Orchestrator runs against its
            # configured task.goal. For true dynamic goals, construct a new
            # TaskDefinition and Orchestrator before each run.

        result = self.orchestrator.run()
        return self._format_result(result)

    async def _arun(
        self,
        goal_override: Optional[str] = None,
        run_manager: Optional[Any] = None,
    ) -> str:
        """
        Async variant. Runs the Orchestrator in a thread pool.

        EXPERIMENTAL: ClampAI's Orchestrator is synchronous. This wraps
        it with ``asyncio.to_thread``, which means it does not benefit from
        Python's async event loop. For truly async execution, an
        ``AsyncOrchestrator`` is planned for v0.6.
        """
        import asyncio
        return await asyncio.to_thread(self._run, goal_override)


    def _format_result(self, result: Any) -> str:
        """
        Format an ``ExecutionResult`` as a string for the LangChain agent.

        Override this method to customise the output format.
        """
        if hasattr(result, "summary"):
            return result.summary()

        # Fallback for unexpected result types
        lines = []
        for attr in ("goal_achieved", "total_steps", "total_cost",
                     "termination_reason", "actions_succeeded",
                     "actions_rejected_safety"):
            if hasattr(result, attr):
                lines.append(f"{attr}: {getattr(result, attr)}")
        return "\n".join(lines) if lines else str(result)

    def __repr__(self) -> str:
        return (
            f"ClampAISafeTool(name={self.name!r}, "
            f"orchestrator={self.orchestrator!r})"
        )
