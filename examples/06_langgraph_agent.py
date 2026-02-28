"""
examples/06_langgraph_agent.py — ClampAI + LangGraph safety enforcement.

Demonstrates how to add provable budget enforcement and invariant checking
to a LangGraph graph using the ClampAI LangGraph adapter.

Three patterns shown:
  1. @clampai_node decorator — wraps a node function, budget depletes over calls
  2. budget_guard() — standalone guard node that blocks the graph when exhausted
  3. invariant_guard() — standalone node that blocks on invariant violations

Run with:
    pip install clampai[langgraph]
    python examples/06_langgraph_agent.py
"""

from __future__ import annotations

from clampai.adapters import (
    ClampAIBudgetError,
    ClampAIInvariantError,
    budget_guard,
    clampai_node,
    invariant_guard,
)
from clampai.invariants import rate_limit_invariant, value_range_invariant

# Pattern 1: @clampai_node decorator
#
# Budget is tracked per-node. After 3 calls (budget=6.0, cost=2.0) the
# node raises ClampAIBudgetError and the graph should route to an error node.

@clampai_node(budget=6.0, cost_per_step=2.0)
def research_node(state: dict) -> dict:
    """Simulate a research step that costs 2 budget units per call."""
    query = state.get("query", "default query")
    api_calls = state.get("api_calls", 0) + 1
    print(f"  [research_node] step {api_calls}: processing '{query}'")
    return {"api_calls": api_calls, "result": f"research result for: {query}"}


@clampai_node(budget=10.0, cost_per_step=1.0)
def summarise_node(state: dict) -> dict:
    """Summarise the research result (costs 1 budget unit)."""
    result = state.get("result", "")
    print(f"  [summarise_node] summarising: {result[:40]}...")
    return {"summary": f"Summary: {result[:60]}"}


def demo_clampai_node() -> None:
    print("\n--- Pattern 1: @clampai_node decorator ---")
    state: dict = {"query": "AI safety research", "api_calls": 0}

    for i in range(1, 5):
        try:
            state = research_node(state)
            print(f"  budget_remaining={research_node.budget_remaining:.1f}")
        except ClampAIBudgetError as e:
            print(f"  BLOCKED at call {i}: {e}")
            break

    print(f"  Total successful calls: {research_node.step_count}")


# Pattern 2: budget_guard() as a standalone guard node
#
# budget_guard returns an empty dict (pass-through) or raises ClampAIBudgetError.
# Use it as a gate before expensive operations in your graph.

def demo_budget_guard() -> None:
    print("\n--- Pattern 2: budget_guard() ---")

    # budget=3.0, cost=1.0 → allows 3 calls then blocks
    guard = budget_guard(budget=3.0, cost_per_step=1.0)

    for i in range(1, 6):
        try:
            result = guard({"step": i})
            print(f"  Call {i}: PASSED → {result}")
        except ClampAIBudgetError as e:
            print(f"  Call {i}: BLOCKED — {e}")


# Pattern 3: invariant_guard() — block on invariant violations
#
# invariant_guard checks invariants on the current state without charging budget.
# Use it as a safety checkpoint anywhere in the graph.

def demo_invariant_guard() -> None:
    print("\n--- Pattern 3: invariant_guard() ---")

    guard = invariant_guard([
        rate_limit_invariant("api_calls", 5),
        value_range_invariant("confidence", 0.0, 1.0),
    ])

    test_states = [
        {"api_calls": 3, "confidence": 0.8},   # passes
        {"api_calls": 7, "confidence": 0.8},   # fails: api_calls > 5
        {"api_calls": 3, "confidence": 1.5},   # fails: confidence out of range
        {"api_calls": 2, "confidence": 0.95},  # passes
    ]

    for state in test_states:
        try:
            guard(state)
            print(f"  State {state} → PASSED")
        except ClampAIInvariantError as e:
            print(f"  State {state} → BLOCKED — {e}")


# Pattern 4: Using @clampai_node with invariants

@clampai_node(
    budget=20.0,
    cost_per_step=1.0,
    invariants=[
        rate_limit_invariant("tool_calls", 5),
        value_range_invariant("error_rate", 0.0, 0.1),
    ],
)
def api_node(state: dict) -> dict:
    """API call node with both budget AND invariant enforcement."""
    tool_calls = state.get("tool_calls", 0) + 1
    error_rate = state.get("error_rate", 0.0)
    print(f"  [api_node] tool_calls={tool_calls}, error_rate={error_rate:.2f}")
    return {"tool_calls": tool_calls, "output": f"api_result_{tool_calls}"}


def demo_invariant_node() -> None:
    print("\n--- Pattern 4: @clampai_node with invariants ---")

    # Normal calls
    state: dict = {"tool_calls": 0, "error_rate": 0.02}
    for _ in range(3):
        state = api_node(state)

    # This should fail: tool_calls will exceed 5
    state = {"tool_calls": 5, "error_rate": 0.02}
    try:
        state = api_node(state)
    except ClampAIInvariantError as e:
        print(f"  BLOCKED (invariant): {e}")

    # This should fail: error_rate too high
    state = {"tool_calls": 2, "error_rate": 0.99}
    try:
        state = api_node(state)
    except ClampAIInvariantError as e:
        print(f"  BLOCKED (invariant): {e}")

    print(f"  Successful calls: {api_node.step_count}")


# Main

def main() -> None:
    print("ClampAI + LangGraph: safety enforcement examples")
    print("=" * 55)

    demo_clampai_node()
    demo_budget_guard()
    demo_invariant_guard()
    demo_invariant_node()

    print("\nAll patterns demonstrated. Integrate these nodes into any")
    print("LangGraph StateGraph — they are plain (state: dict) -> dict callables.")


if __name__ == "__main__":
    main()
