"""
examples/multi_agent_shared_kernel.py — Concurrent agents sharing one kernel.

Demonstrates eight Orchestrators running in separate threads against a single
SafetyKernel. All T1–T8 theorems hold across threads because
``evaluate_and_execute_atomic()`` serialises check + charge + commit under
one ``threading.Lock``. This is the supported multi-agent pattern for v0.x.

For the clampaints around multi-process coordination, see
docs/MULTI_AGENT_ARCHITECTURE.md.
"""

import threading
import time
from typing import List, Tuple

from clampai import (
    ActionSpec,
    Effect,
    Invariant,
    SafetyKernel,
    State,
)


def make_shared_kernel(total_budget: float) -> SafetyKernel:
    """
    Create a single SafetyKernel to be shared across all agents in this process.

    The shared budget B0=total_budget is enforced globally: the sum of all
    charges from all threads will never exceed total_budget (T1: Budget Safety).
    """
    return SafetyKernel(
        budget=total_budget,
        invariants=[
            Invariant(
                name="concurrent_task_limit",
                predicate=lambda s: s.get("active_tasks", 0) <= 5,
                description="No more than 5 tasks may be active at once across all agents",
                enforcement="blocking",
            ),
            Invariant(
                name="total_cost_reasonable",
                predicate=lambda s: s.get("total_cost_accumulated", 0.0) <= 1000.0,
                description="Accumulated cost must stay under 1000 units",
                enforcement="blocking",
            ),
        ],
        min_action_cost=1.0,
    )


class SimpleAgent:
    """
    A minimal agent that proposes and executes actions via a shared kernel.

    In a real deployment this would wrap an Orchestrator and an LLM adapter.
    Here it runs a fixed sequence of actions for demonstration purposes.
    """

    def __init__(self, agent_id: str, kernel: SafetyKernel, initial_state: State):
        self.agent_id = agent_id
        self.kernel = kernel
        self.state = initial_state
        self.log: List[str] = []

    def _make_start_action(self) -> ActionSpec:
        return ActionSpec(
            id=f"start_{self.agent_id}",
            name=f"Agent {self.agent_id}: start task",
            description="Increment the shared active_tasks counter",
            effects=(
                Effect("active_tasks", "increment", 1),
                Effect("total_cost_accumulated", "increment", 10.0),
            ),
            cost=10.0,
        )

    def _make_finish_action(self) -> ActionSpec:
        return ActionSpec(
            id=f"finish_{self.agent_id}",
            name=f"Agent {self.agent_id}: finish task",
            description="Decrement the shared active_tasks counter",
            effects=(
                Effect("active_tasks", "decrement", 1),
            ),
            cost=5.0,
        )

    def run(self) -> None:
        """Execute start -> (work) -> finish sequence via shared kernel."""
        start_action = self._make_start_action()

        # evaluate_and_execute_atomic is thread-safe: acquires Lock, checks,
        # charges, commits, releases Lock. No other thread can interleave.
        try:
            new_state, entry = self.kernel.evaluate_and_execute_atomic(
                self.state, start_action,
                reasoning_summary=f"Agent {self.agent_id} starting work"
            )
            self.state = new_state
            self.log.append(f"[{self.agent_id}] START approved: {entry.action_id}")
        except RuntimeError as e:
            self.log.append(f"[{self.agent_id}] START rejected: {e}")
            return

        # Simulate doing work
        time.sleep(0.01)

        finish_action = self._make_finish_action()
        try:
            new_state, entry = self.kernel.evaluate_and_execute_atomic(
                self.state, finish_action,
                reasoning_summary=f"Agent {self.agent_id} finishing work"
            )
            self.state = new_state
            self.log.append(f"[{self.agent_id}] FINISH approved: {entry.action_id}")
        except RuntimeError as e:
            self.log.append(f"[{self.agent_id}] FINISH rejected: {e}")


def run_shared_kernel_demo(num_agents: int = 8, total_budget: float = 100.0) -> None:
    """
    Launch num_agents agents concurrently against a single shared kernel.

    Budget: 100 units. Each agent START costs 10 units (budget enforced globally).
    Invariant: no more than 5 concurrent active_tasks.

    Expected outcome:
    - At most 5 agents can hold an active task simultaneously (T3: Invariant Safety)
    - Total spend across all agents never exceeds 100 units (T1: Budget Safety)
    - Some agents will be rejected due to budget exhaustion or invariant violation
    """
    kernel = make_shared_kernel(total_budget)
    initial_state = State({"active_tasks": 0, "total_cost_accumulated": 0.0})

    agents = [
        SimpleAgent(agent_id=f"A{i}", kernel=kernel, initial_state=initial_state)
        for i in range(num_agents)
    ]

    threads = [threading.Thread(target=agent.run) for agent in agents]

    print(f"Launching {num_agents} agents with shared budget={total_budget}")
    print("Each START action costs 10.0 units; max concurrent tasks = 5")
    print()

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("Agent logs")
    for agent in agents:
        for line in agent.log:
            print(line)

    print()
    print("Kernel status")
    print(kernel.status())

    print()
    print("Trace integrity check")
    valid, msg = kernel.trace.verify_integrity()
    print(f"Trace valid: {valid} — {msg}")

    print()
    print("Budget summary")
    print(kernel.budget.summary())

    # Verify T1: net spend never exceeded budget
    assert kernel.budget.spent_net <= total_budget, (
        f"T1 VIOLATED: spent_net={kernel.budget.spent_net} > budget={total_budget}"
    )
    print(f"T1 verified: spent_net ({kernel.budget.spent_net}) <= budget ({total_budget})")


if __name__ == "__main__":
    run_shared_kernel_demo(num_agents=8, total_budget=100.0)
