"""
examples/01_hello_safety.py — Your first ClampAI safety check.

The simplest possible usage: create a SafetyKernel with a budget and one
invariant, then evaluate a proposed action. The kernel either approves it
or explains exactly why it was rejected.

Run:
    python examples/01_hello_safety.py
"""

from clampai import (
    ActionSpec,
    Effect,
    Invariant,
    SafetyKernel,
    State,
)


def main() -> None:
    state = State({
        "emails_sent": 0,
        "approved": False,
    })

    kernel = SafetyKernel(
        budget=10.0,
        invariants=[
            Invariant(
                "human_must_approve",
                lambda s: not s.get("emails_sent", 0) > 0 or s.get("approved", False),
                "Cannot send emails without human approval",
                severity="critical",
            ),
            Invariant(
                "email_cap",
                lambda s: s.get("emails_sent", 0) <= 3,
                "No more than 3 emails per session",
            ),
        ],
    )

    dangerous_action = ActionSpec(
        id="send_email_unapproved",
        name="Send Email",
        description="Send an email before getting human approval",
        effects=(Effect("emails_sent", "increment", 1),),
        cost=1.0,
    )

    verdict = kernel.evaluate(state, dangerous_action)
    print("=== Evaluation 1: Send email WITHOUT approval ===")
    print(f"  Approved: {verdict.approved}")
    if not verdict.approved:
        for reason in verdict.rejection_reasons:
            print(f"  Blocked: {reason}")

    approved_state = State({
        "emails_sent": 0,
        "approved": True,
    })

    verdict2 = kernel.evaluate(approved_state, dangerous_action)
    print("\n=== Evaluation 2: Send email WITH approval ===")
    print(f"  Approved: {verdict2.approved}")

    if verdict2.approved:
        new_state, entry = kernel.execute(approved_state, dangerous_action)
        print(f"  emails_sent is now: {new_state.get('emails_sent')}")
        print(f"  Budget remaining:   ${kernel.budget.remaining:.2f}")
        print(f"  Trace step:         {entry.step}")

    print("\nT1 (budget) + T3 (invariants) enforced successfully.")


if __name__ == "__main__":
    main()
