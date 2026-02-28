"""
clampai/adapters/mcp_server.py — ClampAI safety wrapper for MCP tool servers.

Provides SafeMCPServer, a drop-in enhancement for FastMCP that routes every
tool call through a shared SafetyKernel before the handler executes. All tools
registered on one server share a single budget (T1) and a common set of
invariants (T3), making the safety boundary visible at the server boundary
rather than scattered across individual tool implementations.

The ``mcp`` package is a soft dependency: importing this module succeeds
without it installed. A helpful ImportError is raised only when
SafeMCPServer is instantiated.

Usage:
    from clampai.adapters.mcp_server import SafeMCPServer
    from clampai import Invariant, no_action_after_flag_invariant

    server = SafeMCPServer(
        "my-agent-server",
        budget=500.0,
        cost_per_tool=10.0,
        invariants=[
            Invariant(
                "no_bulk_delete",
                lambda s: s.get("delete_count", 0) < 5,
                "No more than 5 deletes per session",
            ),
        ],
    )

    @server.tool()
    def send_email(to: str, subject: str, body: str) -> str:
        return f"Sent '{subject}' to {to}"

    @server.tool(cost=50.0)  # expensive operation charged differently
    def train_model(dataset_path: str) -> str:
        return "Training started"

    if __name__ == "__main__":
        server.run()
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..api import SafetyViolation
from ..formal import (
    ActionSpec,
    Effect,
    Invariant,
    SafetyKernel,
    State,
)


class SafeMCPServer:
    """A ClampAI-guarded wrapper around FastMCP.

    All tools registered via ``@server.tool()`` share a single SafetyKernel.
    The shared kernel enforces:
      - Budget (T1): cumulative cost across ALL tool calls
      - Invariants (T3): server-wide predicates checked before every call
      - Termination (T2): max calls bounded by budget / min_action_cost
      - Trace (T6): tamper-evident audit log of every approved call

    Per-tool invariants (passed to the individual ``@server.tool()`` decorator)
    are checked BEFORE the shared kernel, allowing fine-grained per-tool
    policies on top of the server-wide baseline.

    Args:
        name: MCP server name (passed through to FastMCP).
        budget: Shared budget across all tool calls (T1).
        cost_per_tool: Default cost per tool invocation. Individual tools
            can override this with the ``cost`` argument on ``@server.tool()``.
        invariants: Server-wide invariants checked before every tool call (T3).
        min_action_cost: Minimum allowed cost for the T2 termination bound.

    Raises:
        ImportError: If the ``mcp`` package is not installed when constructing
            a SafeMCPServer instance.
    """

    def __init__(
        self,
        name: str,
        budget: float = 1000.0,
        cost_per_tool: float = 1.0,
        invariants: Sequence[Invariant] = (),
        min_action_cost: float = 0.001,
    ) -> None:
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            raise ImportError(
                "The 'mcp' package is required for SafeMCPServer. "
                "Install it with: pip install mcp"
            ) from exc

        self._mcp = FastMCP(name)
        self._budget = budget
        self._cost_per_tool = cost_per_tool
        self._server_invariants = list(invariants)
        self._lock = threading.Lock()

        # Shared kernel: all tools draw from the same budget and must satisfy
        # the same server-wide invariants.
        self._kernel = SafetyKernel(
            budget=budget,
            invariants=list(invariants),
            min_action_cost=min_action_cost,
        )

        # Shared state for kernel evaluation (tracks total_tool_calls).
        self._state = State({"total_tool_calls": 0, "total_cost": 0.0})

        # Registry for introspection.
        self._tool_registry: Dict[str, Callable[..., Any]] = {}
        self._tool_costs: Dict[str, float] = {}

    @property
    def kernel(self) -> SafetyKernel:
        """The shared SafetyKernel for introspection."""
        return self._kernel

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Structured audit log of all approved tool calls. JSON-serializable."""
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

    @property
    def budget_remaining(self) -> float:
        """Remaining budget across all tools in this server."""
        return self._kernel.budget.remaining

    def tool(
        self,
        *,
        cost: Optional[float] = None,
        invariants: Optional[Sequence[Invariant]] = None,
        name: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a ClampAI-guarded MCP tool.

        Args:
            cost: Per-call cost for this tool. Defaults to the server's
                ``cost_per_tool``.
            invariants: Per-tool invariants checked before this specific tool
                runs. Layered on top of server-wide invariants.
            name: Override the tool name exposed to MCP clients. Defaults
                to the function's ``__name__``.

        Returns:
            A decorator that wraps the function and registers it with FastMCP.
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            effective_cost = cost if cost is not None else self._cost_per_tool
            tool_name = name or fn.__name__
            per_tool_invariants = list(invariants or [])

            @functools.wraps(fn)
            def safe_handler(*args: Any, **kwargs: Any) -> Any:
                with self._lock:
                    current_calls = int(self._state.get("total_tool_calls", 0))
                    current_cost = float(self._state.get("total_cost", 0.0))

                    eval_state = State({
                        "total_tool_calls": current_calls,
                        "total_cost": current_cost,
                    })

                    # Check per-tool invariants before the shared kernel.
                    for inv in per_tool_invariants:
                        ok, msg = inv.check(eval_state)
                        if not ok:
                            raise SafetyViolation(
                                f"Tool '{tool_name}' blocked by per-tool invariant: {msg}"
                            )

                    action = ActionSpec(
                        id=f"{tool_name}_{current_calls + 1}",
                        name=tool_name,
                        description=f"MCP tool call #{current_calls + 1} to {fn.__qualname__}",
                        effects=(
                            Effect("total_tool_calls", "increment", 1),
                            Effect("total_cost", "increment", effective_cost),
                        ),
                        cost=effective_cost,
                    )

                    verdict = self._kernel.evaluate(eval_state, action)
                    if not verdict.approved:
                        reasons = (
                            "; ".join(verdict.rejection_reasons)
                            if verdict.rejection_reasons
                            else "kernel rejected"
                        )
                        raise SafetyViolation(
                            f"Tool '{tool_name}' blocked by ClampAI: {reasons}"
                        )

                    new_state, _ = self._kernel.execute(eval_state, action)
                    self._state = State({
                        "total_tool_calls": new_state.get(
                            "total_tool_calls", current_calls + 1
                        ),
                        "total_cost": new_state.get(
                            "total_cost", current_cost + effective_cost
                        ),
                    })

                return fn(*args, **kwargs)

            # Register with FastMCP under the tool name.
            self._mcp.tool(name=tool_name)(safe_handler)
            self._tool_registry[tool_name] = safe_handler
            self._tool_costs[tool_name] = effective_cost

            return safe_handler

        return decorator

    def run(self, **kwargs: Any) -> None:
        """Start the MCP server (delegates to FastMCP.run)."""
        self._mcp.run(**kwargs)

    def summary(self) -> str:
        """Human-readable summary of server state."""
        lines = [
            f"SafeMCPServer — {len(self._tool_registry)} tools registered",
            f"  Budget: ${self._kernel.budget.remaining:.2f} remaining "
            f"of ${self._budget:.2f}",
            f"  Tool calls executed: {self._kernel.step_count}",
        ]
        for tool_name, tool_cost in self._tool_costs.items():
            lines.append(f"  • {tool_name} (${tool_cost:.2f}/call)")
        return "\n".join(lines)
