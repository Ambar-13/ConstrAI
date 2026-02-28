"""
examples/05_safe_patterns.py — @safe patterns for real-world use.

The @safe function is the fastest path to safety for existing code.
Wrap any function — LLM calls, tool executions, API requests — without
restructuring your codebase into TaskDefinition/Orchestrator.

This example shows four patterns:
  1. Basic wrapping with budget and rate limiting
  2. State-aware invariants via state_fn
  3. Chained safe functions (pipeline pattern)
  4. Resetting the wrapper between sessions

Run:
    python examples/05_safe_patterns.py
"""

from __future__ import annotations

from typing import Any

from clampai import (
    Invariant,
    SafetyViolation,
    rate_limit_invariant,
    safe,
    value_range_invariant,
)


def demo_basic_wrapping() -> None:
    print("=== Pattern 1: Basic budget + rate limit ===\n")

    @safe(
        budget=30.0,
        cost_per_call=5.0,
        invariants=[
            rate_limit_invariant("call_count", 5),
        ],
        action_name="llm_complete",
    )
    def llm_complete(prompt: str) -> str:
        """Simulate an LLM completion call."""
        return f"[LLM response to: {prompt[:30]}...]"

    prompts = [
        "What is the capital of France?",
        "Explain gradient descent.",
        "Write a haiku about safety.",
        "How does a transformer work?",
        "What is a control barrier function?",
        "This call should be rate-limited",  # 6th call, rate limit = 5
        "This call should be budget-limited",
    ]

    for prompt in prompts:
        try:
            response = llm_complete(prompt)
            print(f"  ok: {response}")
        except SafetyViolation as exc:
            print(f"  BLOCKED: {exc}")

    print(f"\n  Approved calls: {len(llm_complete.audit_log)}")
    print(f"  Budget remaining: ${llm_complete.kernel.budget.remaining:.2f}")


def demo_state_fn() -> None:
    print("\n=== Pattern 2: state_fn for external state ===\n")

    _session_state: dict[str, Any] = {
        "user_tier": "free",
        "daily_calls_made": 0,
    }

    def get_session_state() -> dict[str, Any]:
        return dict(_session_state)

    @safe(
        budget=1000.0,
        cost_per_call=1.0,
        invariants=[
            Invariant(
                "free_tier_daily_limit",
                lambda s: not (s.get("user_tier") == "free") or s.get("call_count", 0) < 3,
                "Free tier is limited to 3 calls per day",
            ),
        ],
        state_fn=get_session_state,
    )
    def premium_api_call(endpoint: str) -> dict:
        _session_state["daily_calls_made"] += 1
        return {"endpoint": endpoint, "status": 200}

    for i in range(4):
        try:
            result = premium_api_call(f"/premium/v1/data/{i}")
            print(f"  ok: Call {i + 1}: {result}")
        except SafetyViolation as exc:
            print(f"  FAIL: Call {i + 1}: BLOCKED — {exc}")

    print("\n  Upgrading to paid tier...")
    _session_state["user_tier"] = "paid"
    premium_api_call.reset()

    for i in range(2):
        result = premium_api_call(f"/premium/v1/data/paid_{i}")
        print(f"  ok: Paid call {i + 1}: {result}")


def demo_chained_pipeline() -> None:
    print("\n=== Pattern 3: Chained safe functions ===\n")

    @safe(budget=20.0, cost_per_call=5.0, action_name="fetch_data")
    def fetch_data(source: str) -> list[str]:
        return [f"record_{i}" for i in range(10)]

    @safe(
        budget=30.0,
        cost_per_call=3.0,
        invariants=[
            value_range_invariant("total_cost", 0, 25),
        ],
        action_name="transform_data",
    )
    def transform_data(records: list[str]) -> list[str]:
        return [r.upper() for r in records]

    @safe(budget=10.0, cost_per_call=8.0, action_name="send_report")
    def send_report(data: list[str]) -> str:
        return f"Report sent with {len(data)} records"


    try:
        raw = fetch_data("s3://my-bucket/data")
        print(f"  ok: fetch_data: {len(raw)} records")

        transformed = transform_data(raw)
        print(f"  ok: transform_data: {len(transformed)} records")

        result = send_report(transformed)
        print(f"  ok: send_report: {result}")

    except SafetyViolation as exc:
        print(f"  FAIL: Pipeline blocked: {exc}")

    print(f"\n  fetch_data budget remaining:     ${fetch_data.kernel.budget.remaining:.2f}")
    print(f"  transform_data budget remaining: ${transform_data.kernel.budget.remaining:.2f}")
    print(f"  send_report budget remaining:    ${send_report.kernel.budget.remaining:.2f}")


def demo_reset() -> None:
    print("\n=== Pattern 4: Reset between sessions ===\n")

    @safe(budget=10.0, cost_per_call=4.0)
    def session_action(user_id: str) -> str:
        return f"Action for {user_id}"

    print("  Session 1:")
    for i in range(3):  # 3rd call exceeds budget (cost=12 > 10)
        try:
            result = session_action(f"user_{i}")
            print(f"    ok: {result}")
        except SafetyViolation as exc:
            print(f"    FAIL: {exc}")

    session_action.reset()
    print(f"\n  After reset — budget: ${session_action.kernel.budget.remaining:.2f}")

    print("  Session 2:")
    for i in range(2):
        result = session_action(f"user_{i}")
        print(f"    ok: {result}")
    print(f"  Remaining: ${session_action.kernel.budget.remaining:.2f}")


if __name__ == "__main__":
    demo_basic_wrapping()
    demo_state_fn()
    demo_chained_pipeline()
    demo_reset()
    print("\n@safe patterns demonstrated.")
