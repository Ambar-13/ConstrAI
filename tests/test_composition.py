"""Test suite for task composition and safety enforcement.

Tests verify that safety checks don't conflict during composition:
- Budget repair (QP) may modify an action
- But information flow control (IFC) is still enforced afterward

This ensures safety layers work together, not against each other.
"""

from __future__ import annotations

from constrai import (
    ActionSpec,
    DataLabel,
    Effect,
    Invariant,
    LabelledData,
    Orchestrator,
    SecurityLevel,
    State,
    TaskDefinition,
)


def test_repair_does_not_bypass_ifc():
    # State is deepcopy'd at construction time, so keep values plain.
    # We'll still exercise IFC by passing labelled data through the action
    # payload that the ReferenceMonitor inspects.
    pii = LabelledData(value="ssn=123-45-6789", label=DataLabel(SecurityLevel.PII))
    s0 = State({
        "budget": 100.0,
        "public_log": "",
    })

    # This action:
    # - costs too much (forces QP repair / projection)
    # - tries to copy PII into a public sink (must be rejected by IFC)
    leak_and_overspend = ActionSpec(
        id="leak_and_overspend",
        name="Leak and Overspend",
        description="Attempts to leak PII and overspend.",
        effects=(
            Effect("public_log", "set", pii),
        ),
        cost=10_000.0,
        risk_level="critical",
        reversible=False,
    )

    # Goal irrelevant; we just care that the orchestrator blocks.
    task = TaskDefinition(
        goal="Do not leak",
        initial_state=s0,
        available_actions=[leak_and_overspend],
        invariants=[Invariant("noop", lambda s: True)],
        budget=100.0,
        goal_predicate=lambda s: False,
        max_consecutive_failures=1,
    )

    orch = Orchestrator(task)
    # Encode the policy: public_log is PUBLIC, so PII payload must not flow there.
    orch.monitor.set_ifc_label("public_log", DataLabel(SecurityLevel.PUBLIC))
    ok, msg = orch._execute_action(leak_and_overspend, reasoning="test")

    assert ok is False
    # Must be rejected by monitor (IFC) not just kernel.
    assert "Monitor rejected" in msg
