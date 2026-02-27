"""
examples/03_invariants.py — Invariant enforcement (Theorem T3).

Invariants are formal safety predicates evaluated on the projected next state
before any action is committed. ConstrAI ships 21 pre-built invariant factory
functions for common patterns. This example shows:

  • Writing custom invariants
  • Using factory functions from constrai.invariants
  • Blocking vs monitoring enforcement modes
  • Critical severity — actions that violate a critical invariant are
    rejected even if they would otherwise pass budget and other checks

Run:
    python examples/03_invariants.py
"""

from constrai import (
    ActionSpec,
    Effect,
    Invariant,
    SafetyKernel,
    State,
    email_safety_invariant,
    no_delete_invariant,
    rate_limit_invariant,
    resource_ceiling_invariant,
)


def try_action(kernel: SafetyKernel, state: State, action: ActionSpec) -> State:
    """Evaluate an action, print the result, and return the updated state."""
    verdict = kernel.evaluate(state, action)
    symbol = "ok" if verdict.approved else "FAIL"
    print(f"  {symbol} {action.name}")
    if not verdict.approved:
        for r in verdict.rejection_reasons:
            print(f"      blocked: {r}")
        return state
    new_state, _ = kernel.execute(state, action)
    return new_state


def demo_custom_invariants() -> None:
    print("=== Custom invariants ===\n")

    kernel = SafetyKernel(
        budget=100.0,
        invariants=[
            Invariant(
                "no_production_without_tests",
                lambda s: not s.get("in_production", False) or s.get("tests_passing", False),
                "Must have passing tests before deploying to production",
                severity="critical",
            ),
            Invariant(
                "component_ceiling",
                lambda s: s.get("components_built", 0) <= 5,
                "Maximum 5 components per session",
            ),
        ],
    )

    state = State({"in_production": False, "tests_passing": False, "components_built": 0})

    build = ActionSpec(
        id="build",
        name="Build Component",
        description="Add a component",
        effects=(Effect("components_built", "increment", 1),),
        cost=5.0,
    )
    for _ in range(3):
        state = try_action(kernel, state, build)

    # Deploy WITHOUT tests — blocked by critical invariant
    deploy_no_tests = ActionSpec(
        id="deploy_bad",
        name="Deploy (no tests)",
        description="Deploy before tests pass",
        effects=(Effect("in_production", "set", True),),
        cost=10.0,
    )
    state = try_action(kernel, state, deploy_no_tests)

    run_tests = ActionSpec(
        id="tests",
        name="Run Tests",
        description="Pass the test suite",
        effects=(Effect("tests_passing", "set", True),),
        cost=5.0,
    )
    state = try_action(kernel, state, run_tests)

    deploy_ok = ActionSpec(
        id="deploy_ok",
        name="Deploy (tests passing)",
        description="Deploy after tests pass",
        effects=(Effect("in_production", "set", True),),
        cost=10.0,
    )
    state = try_action(kernel, state, deploy_ok)
    print(f"\n  Final state: in_production={state.get('in_production')}, "
          f"tests_passing={state.get('tests_passing')}")


def demo_factory_invariants() -> None:
    print("\n=== Pre-built invariant factories ===\n")

    kernel = SafetyKernel(
        budget=200.0,
        invariants=[
            rate_limit_invariant("api_calls", 3),
            resource_ceiling_invariant("memory_mb", ceiling=512),
            no_delete_invariant("critical_records"),
            email_safety_invariant(),
        ],
    )

    state = State({
        "api_calls": 0,
        "memory_mb": 100,
        "critical_records": [],
        "emails_deleted": 0,
    })

    api_call = ActionSpec(
        id="api",
        name="API Call",
        description="Make one API call",
        effects=(Effect("api_calls", "increment", 1),),
        cost=1.0,
    )
    for i in range(4):  # 4th blocked by rate limit
        state = try_action(kernel, state, api_call)

    oom = ActionSpec(
        id="oom",
        name="Load Large Model",
        description="Load a model that exceeds memory",
        effects=(Effect("memory_mb", "set", 1024),),
        cost=5.0,
    )
    state = try_action(kernel, state, oom)

    append_record = ActionSpec(
        id="append",
        name="Add Record",
        description="Add a critical record (allowed)",
        effects=(Effect("critical_records", "append", "record_1"),),
        cost=2.0,
    )
    state = try_action(kernel, state, append_record)  # append is fine

    delete_all = ActionSpec(
        id="delete_all",
        name="Delete ALL Records",
        description="Wipe critical_records",
        effects=(Effect("critical_records", "set", []),),
        cost=2.0,
    )
    state = try_action(kernel, state, delete_all)  # blocked: no_delete_invariant


def demo_monitoring_invariants() -> None:
    print("\n=== Monitoring mode invariants (log only) ===\n")

    kernel = SafetyKernel(
        budget=100.0,
        invariants=[
            Invariant(
                "performance_degraded",
                lambda s: s.get("latency_ms", 0) < 200,
                "Response latency is above threshold (monitoring only)",
                enforcement="monitoring",  # logs but does not block
            ),
            Invariant(
                "hard_limit",
                lambda s: s.get("latency_ms", 0) < 5000,
                "Response latency is critically high (blocking)",
                severity="critical",
            ),
        ],
    )

    state = State({"latency_ms": 0})

    slow_action = ActionSpec(
        id="slow",
        name="Slow Request",
        description="Request that degrades latency",
        effects=(Effect("latency_ms", "set", 300),),  # exceeds monitoring threshold
        cost=5.0,
    )
    print("  Monitoring invariant violated (300ms > 200ms threshold):")
    state = try_action(kernel, state, slow_action)
    print(f"  Action was still approved (monitoring mode). latency={state.get('latency_ms')}")

    critical_slow = ActionSpec(
        id="critical_slow",
        name="Critical Slow Request",
        description="Request that hits the hard limit",
        effects=(Effect("latency_ms", "set", 6000),),  # exceeds blocking threshold
        cost=5.0,
    )
    print("\n  Blocking invariant violated (6000ms > 5000ms hard limit):")
    try_action(kernel, state, critical_slow)


if __name__ == "__main__":
    demo_custom_invariants()
    demo_factory_invariants()
    demo_monitoring_invariants()
    print("\nTheorem T3 (Invariant Preservation) demonstrated.")
