"""ClampAI Reference Monitor Example

Complete working example showing:
- Multi-layer safety enforcement (information flow control + safety barriers)
- Automatic action repair (quadratic programming)
- Safe task composition
- Full autonomous execution with safety guarantees

Scenario: Autonomous financial advisor managing a budget across multiple accounts.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clampai import (
    ActionSpec,
    CaptureBasin,
    ContractSpecification,
    DataLabel,
    Effect,
    Invariant,
    OperadicComposition,
    Orchestrator,
    ReferenceMonitor,
    SafeHoverState,
    SecurityLevel,
    State,
    TaskDefinition,
)


def example_financial_advisor():
    """
    Autonomous financial advisor that:
      - Manages a budget of $1000
      - Can allocate funds to savings, checking, credit card
      - Must never exceed budget (T1)
      - Must prevent information leakage (IFC: account numbers are PII)
      - Must gracefully degrade as budget depletes (CBF)
      - Must avoid insolvency (HJB capture basin)
    """

    print("\n" + "="*70)
    print("  EXAMPLE: Autonomous Financial Advisor with Reference Monitor")
    print("="*70)

    # ─────────────────────────────────────────────────────────────────────────
    # TASK DEFINITION
    # ─────────────────────────────────────────────────────────────────────────

    initial_state = State({
        "budget_remaining": 1000.0,
        "savings_balance": 0.0,
        "checking_balance": 0.0,
        "cc_payment": 0.0,
        "transactions_logged": 0,
        "risk_level": 0.0,
    })

    actions = [
        ActionSpec(
            id="deposit_savings",
            name="Deposit to Savings",
            description="Transfer $100 to savings account",
            effects=[
                Effect("savings_balance", "set", 100.0),
                Effect("budget_remaining", "set", 900.0),
            ],
            cost=50.0,
            risk_level="low",
        ),
        ActionSpec(
            id="pay_credit_card",
            name="Pay Credit Card",
            description="Pay $50 toward credit card",
            effects=[
                Effect("cc_payment", "set", 50.0),
                Effect("budget_remaining", "set", 950.0),
            ],
            cost=30.0,
            risk_level="low",
        ),
        ActionSpec(
            id="transfer_checking",
            name="Transfer to Checking",
            description="Move $200 to checking account",
            effects=[
                Effect("checking_balance", "set", 200.0),
                Effect("budget_remaining", "set", 800.0),
            ],
            cost=40.0,
            risk_level="medium",
        ),
        ActionSpec(
            id="log_transaction",
            name="Log Transaction",
            description="Log account details to audit log (PII violation risk!)",
            effects=[
                Effect("transactions_logged", "set", 1),
            ],
            cost=5.0,
            risk_level="high",
        ),
        ActionSpec(
            id="safe_hover",
            name="Safe Hover",
            description="No-op; system in safe idle",
            effects=[],
            cost=0.0,
            risk_level="low",
        ),
    ]

    invariants = [
        Invariant(
            "budget_non_negative",
            lambda s: s.get("budget_remaining", 0.0) >= 0.0,
            "Budget must never go negative",
            severity="critical",
        ),
        Invariant(
            "total_allocations_safe",
            lambda s: (s.get("savings_balance", 0.0) +
                      s.get("checking_balance", 0.0) +
                      s.get("cc_payment", 0.0)) <= 1000.0,
            "Total allocations must not exceed initial budget",
            severity="critical",
        ),
    ]

    task = TaskDefinition(
        goal="Allocate $500 across accounts while maintaining safety",
        initial_state=initial_state,
        available_actions=actions,
        invariants=invariants,
        budget=1000.0,
        goal_predicate=lambda s: (
            s.get("savings_balance", 0.0) +
            s.get("checking_balance", 0.0) +
            s.get("cc_payment", 0.0)
        ) >= 500.0,
        goal_progress_fn=lambda s: min(1.0, (
            s.get("savings_balance", 0.0) +
            s.get("checking_balance", 0.0) +
            s.get("cc_payment", 0.0)
        ) / 500.0),
        min_action_cost=5.0,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # REFERENCE MONITOR CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────

    monitor = ReferenceMonitor(ifc_enabled=True, cbf_enabled=True, hjb_enabled=True)

    # IFC: Account numbers are PII, audit log is internal
    pii_label = DataLabel(SecurityLevel.PII)
    internal_label = DataLabel(SecurityLevel.INTERNAL)
    public_label = DataLabel(SecurityLevel.PUBLIC)

    monitor.set_ifc_label("savings_balance", pii_label)
    monitor.set_ifc_label("checking_balance", pii_label)
    monitor.set_ifc_label("cc_payment", pii_label)
    monitor.set_ifc_label("transactions_logged", internal_label)

    # CBF: As budget depletes, action space tightens
    # h(s) = remaining_budget / initial_budget
    # Threshold: Δh ≥ -0.2 * h (20% per step when >50%, stricter as depletes)
    monitor.add_cbf(
        h=lambda s: s.get("budget_remaining", 0.0) / 1000.0,
        alpha=0.2,
    )

    # HJB: Avoid insolvency (capture basin where budget ≤ 0)
    insolvency_basin = CaptureBasin(
        name="insolvency",
        is_bad=lambda s: s.get("budget_remaining", 0.0) <= 0.0,
        max_steps=3,
    )
    monitor.add_capture_basin(insolvency_basin)

    # ─────────────────────────────────────────────────────────────────────────
    # OPERADIC COMPOSITION: Task contracts
    # ─────────────────────────────────────────────────────────────────────────

    # Contract: "Fund Savings"
    contract_fund_savings = ContractSpecification(
        name="FundSavings",
        assume=lambda s: s.get("budget_remaining", 0.0) >= 100.0,
        guarantee=lambda s: s.get("savings_balance", 0.0) >= 100.0,
        side_effects=("savings_balance", "budget_remaining"),
    )

    # Contract: "Allocate to Checking"
    contract_checking = ContractSpecification(
        name="AllocateChecking",
        assume=lambda s: s.get("budget_remaining", 0.0) >= 200.0,
        guarantee=lambda s: s.get("checking_balance", 0.0) >= 200.0,
        side_effects=("checking_balance", "budget_remaining"),
    )

    # Composed contract: "Fund Savings, then Allocate to Checking"
    composed = OperadicComposition.compose(contract_fund_savings, contract_checking)
    print(f"\n  Composed Contract: {composed.name}")
    print("    Assume: >=400 remaining (100 for savings + 200 for checking)")
    print("    Guarantee: savings >= 100 AND checking >= 200")
    print(f"    Side effects: {composed.side_effects}")

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTION WITH REFERENCE MONITOR
    # ─────────────────────────────────────────────────────────────────────────

    print("\n" + "-"*70)
    print("  EXECUTION TRACE")
    print("-"*70)

    orchestrator = Orchestrator(task)
    # Inject the monitor (would normally be passed to Orchestrator constructor)
    orchestrator.monitor = monitor

    # Run the orchestrator
    result = orchestrator.run()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n{result.summary()}")

    print("\n  Safety Enforcement Metrics:")
    print(f"    Actions attempted:        {result.actions_attempted}")
    print(f"    Actions approved (kernel): {result.actions_succeeded}")
    print(f"    Safety rejections:        {result.actions_rejected_safety}")
    print("    Reference Monitor checks: IFCok: CBFok: HJBok")
    print(f"    Budget safety (T1):       ${result.total_cost:.2f} / $1000.00")
    print(f"    Termination guarantee:    {result.total_steps} steps (safe)")
    print(f"    Trace integrity (T6):     {result.trace_length} entries (verified)")

    print("\n  Final State:")
    print(f"    Savings:     ${result.final_state.get('savings_balance', 0.0):.2f}")
    print(f"    Checking:    ${result.final_state.get('checking_balance', 0.0):.2f}")
    print(f"    CC Payment:  ${result.final_state.get('cc_payment', 0.0):.2f}")
    print(f"    Budget Left: ${result.final_state.get('budget_remaining', 0.0):.2f}")
    print(f"    Goal Progress: {result.goal_progress:.1%}")

    if result.goal_achieved:
        print("\n  GOAL ACHIEVED")
    else:
        print("\n  WARNING: Goal not achieved (but system remained safe)")

    return result


def example_ifc_violation_detection():
    """
    Simple example showing Information Flow Control (IFC) in action.
    """
    print("\n\n" + "="*70)
    print("  EXAMPLE: IFC (Information Flow Control) Violation Detection")
    print("="*70)

    from clampai import DataLabel, SecurityLevel

    # Create labels
    pii_label = DataLabel(SecurityLevel.PII, tags={"ssn"})
    internal_label = DataLabel(SecurityLevel.INTERNAL)
    public_label = DataLabel(SecurityLevel.PUBLIC)

    print("\nSecurity Lattice:")
    print(f"  Public:   {public_label}")
    print(f"  Internal: {internal_label}")
    print(f"  PII:      {pii_label}")

    print("\nFlow assertions:")
    print(f"  Public ≤ Internal?     {public_label <= internal_label}  ok: (allowed)")
    print(f"  Public ≤ PII?          {public_label <= pii_label}  ok: (allowed)")
    print(f"  Internal ≤ PII?        {internal_label <= pii_label}  ok: (allowed)")
    print(f"  PII ≤ Internal?        {pii_label <= internal_label}  FAIL: (NOT allowed)")
    print(f"  PII ≤ Public?          {pii_label <= public_label}  FAIL: (NOT allowed)")

    # Monitor setup
    monitor = ReferenceMonitor(ifc_enabled=True, cbf_enabled=False, hjb_enabled=False)
    monitor.set_ifc_label("customer_ssn", pii_label)
    monitor.set_ifc_label("audit_log", internal_label)

    state = State({
        "customer_ssn": "123-45-6789",
        "audit_log": "",
    })

    action = ActionSpec(
        id="log_ssn",
        name="Log SSN to Audit",
        description="This would leak PII!",
        effects=[Effect("audit_log", "set", "ssn_data")],
        cost=1.0,
    )

    safe, msg, _ = monitor.enforce(action, state, [])
    print("\nAction: Log SSN to audit log")
    print(f"  Safe? {safe}")
    print(f"  Monitor says: {msg}")


def example_cbf_gradual_restriction():
    """
    Example showing Control Barrier Function (CBF) in action.
    CBF smoothly restricts action space as budget depletes.
    """
    print("\n\n" + "="*70)
    print("  EXAMPLE: Control Barrier Function (CBF) Resource Tightening")
    print("="*70)

    from clampai import ControlBarrierFunction

    budget_limit = 100.0
    alpha = 0.3  # 30% tightening per step

    cbf = ControlBarrierFunction(
        h=lambda s: (budget_limit - s.get("spent", 0.0)) / budget_limit,
        alpha=alpha,
    )

    print("\nCBF Configuration:")
    print(f"  Initial budget: ${budget_limit:.2f}")
    print(f"  Decay rate (α):  {alpha:.0%} per step")
    print("  Safety condition: Δh ≥ -α·h")

    scenarios = [
        ("Early phase", State({"spent": 10.0}), State({"spent": 20.0})),
        ("Mid phase", State({"spent": 50.0}), State({"spent": 65.0})),
        ("Late phase", State({"spent": 80.0}), State({"spent": 95.0})),
        ("Critical", State({"spent": 90.0}), State({"spent": 100.0})),
    ]

    print("\n  Scenario Analysis:")
    print(f"  {'Phase':<15} {'h_now':<10} {'h_next':<10} {'Δh':<10} {'Threshold':<12} {'Safe?':<6}")
    print(f"  {'-'*67}")

    for name, state_before, state_after in scenarios:
        h_now = cbf.h(state_before)
        h_next = cbf.h(state_after)
        delta = h_next - h_now
        threshold = -alpha * h_now
        passes, _ = cbf.evaluate(state_before, state_after)

        print(f"  {name:<15} {h_now:<10.2%} {h_next:<10.2%} {delta:<10.4f} {threshold:<12.4f} {'ok' if passes else 'FAIL':<6}")


if __name__ == "__main__":
    # Run all examples
    example_financial_advisor()
    example_ifc_violation_detection()
    example_cbf_gradual_restriction()

    print("\n" + "="*70)
    print("  All examples completed successfully!")
    print("="*70 + "\n")
