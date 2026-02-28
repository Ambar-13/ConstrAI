"""
examples/02_budget_enforcement.py — Budget enforcement (Theorem T1).

Demonstrates how the SafetyKernel tracks cumulative cost and automatically
blocks actions that would exceed the budget. Also shows @safe as the
zero-config alternative for wrapping individual functions.

Run:
    python examples/02_budget_enforcement.py
"""

from clampai import (
    ActionSpec,
    Effect,
    Invariant,
    SafetyKernel,
    SafetyViolation,
    State,
    safe,
)


def demo_kernel_budget() -> None:
    print("=== Part 1: Direct kernel budget tracking ===\n")

    kernel = SafetyKernel(budget=25.0, invariants=[])
    state = State({"files_processed": 0})

    action = ActionSpec(
        id="process_file",
        name="Process File",
        description="Process one file (costs $5)",
        effects=(Effect("files_processed", "increment", 1),),
        cost=5.0,
    )

    for i in range(6):  # 6 * $5 = $30 > $25 budget
        verdict = kernel.evaluate(state, action)
        if verdict.approved:
            state, _ = kernel.execute(state, action)
            print(f"  Call {i + 1}: approved  — remaining ${kernel.budget.remaining:.2f}")
        else:
            reasons = "; ".join(verdict.rejection_reasons)
            print(f"  Call {i + 1}: BLOCKED   — {reasons}")

    print(f"\n  Final files processed: {state.get('files_processed')}")
    print(f"  Total spent: ${kernel.budget.spent_net:.2f}")


def demo_decorator_budget() -> None:
    print("\n=== Part 2: @safe decorator ===\n")

    @safe(budget=15.0, cost_per_call=5.0)
    def call_api(endpoint: str) -> dict:
        """Simulate an API call."""
        return {"endpoint": endpoint, "status": 200}

    for i in range(4):  # 4 * $5 = $20 > $15 budget
        try:
            result = call_api(f"/api/data/{i}")
            print(f"  Call {i + 1}: success — {result}")
        except SafetyViolation as exc:
            print(f"  Call {i + 1}: BLOCKED — {exc}")

    print(f"\n  Audit log entries: {len(call_api.audit_log)}")
    print(f"  Budget remaining:  ${call_api.kernel.budget.remaining:.2f}")


def demo_budget_plus_invariants() -> None:
    print("\n=== Part 3: Budget + invariants together ===\n")

    @safe(
        budget=100.0,
        cost_per_call=10.0,
        invariants=[
            Invariant(
                "rate_limit",
                lambda s: s.get("call_count", 0) < 3,
                "Rate limit: max 3 calls (invariant fires before budget)",
            ),
        ],
    )
    def expensive_model_call(prompt: str) -> str:
        return f"Response to: {prompt[:20]}..."

    for i in range(5):
        try:
            result = expensive_model_call(f"Prompt number {i}")
            print(f"  Call {i + 1}: success — {result}")
        except SafetyViolation as exc:
            print(f"  Call {i + 1}: BLOCKED — {exc}")


if __name__ == "__main__":
    demo_kernel_budget()
    demo_decorator_budget()
    demo_budget_plus_invariants()
    print("\nTheorem T1 (Budget Safety) demonstrated.")
