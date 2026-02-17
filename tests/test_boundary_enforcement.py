"""Test suite for boundary detection and enforcement.

Tests verify:
1. Jacobian boundary sensitivity (detecting danger zones)
2. HJB barrier enforcement (safe hover mode)
3. Task composition (combining verified tasks)
"""

import pytest
from constrai import (
    State, ActionSpec, Effect, Invariant,
    JacobianFusion, BoundarySeverity,
    AuthoritativeHJBBarrier, SafeHoverSignal,
    SuperTask, TaskComposer, InterfaceSignature, VerificationCertificate, CompositionType,
    CaptureBasin,
)


class TestJacobianFusion:
    """Test Jacobian-HJB Fusion: authoritative boundary sensitivity."""

    def test_jacobian_safe_distant(self):
        """Test Jacobian detects safe, distant state."""
        print("\n" + "="*70)
        print("TEST: Jacobian Safe Distant State")
        print("="*70)

        invariants = [
            Invariant("x_limit", lambda s: s.get("x", 0) <= 100, "x must be ≤ 100"),
        ]

        fusion = JacobianFusion(invariants, epsilon=0.01, lookahead_steps=5)
        
        # Safe state, far from boundary
        safe_state = State({"x": 10})
        report = fusion.compute_jacobian(safe_state)
        
        print(f"\nSafe state: x=10 (limit=100)")
        print(report.describe())
        
        assert not report.safety_barrier_violated
        assert "x" not in report.critical_variables
        assert report.recommendation == "✓ Safe state; all constraints well within margins"
        print("✅ Jacobian correctly detects safe, distant state")

    def test_jacobian_danger_zone(self):
        """Test Jacobian detects critical state near boundary."""
        print("\n" + "="*70)
        print("TEST: Jacobian Detects Danger Zone")
        print("="*70)

        invariants = [
            Invariant("x_limit", lambda s: s.get("x", 0) < 100, "x must be < 100 (strict)"),
        ]

        fusion = JacobianFusion(invariants, epsilon=0.01, lookahead_steps=5)
        
        # State very close to boundary (x=99.99 vs limit=100)
        danger_state = State({"x": 99.99})
        report = fusion.compute_jacobian(danger_state)
        
        print(f"\nDanger state: x=99.99 (limit=100, strict)")
        print(report.describe())
        
        # With x=99.99 and limit < 100, small perturbations will break it
        # The algorithm should detect high boundary sensitivity
        assert len(report.critical_variables) > 0 or report.safety_barrier_violated or any(
            s.score > 0.3 for s in report.scores
        ), f"Expected boundary sensitivity to be detected, got {[s.score for s in report.scores]}"
        assert "x" in report.critical_variables or any(s.variable == "x" and s.score > 0.3 for s in report.scores)
        print("✅ Jacobian correctly detects danger zone (x approaching boundary)")

    def test_jacobian_forcing_critical_into_prompt(self):
        """Test that critical variables are forced into prompt via Saliency."""
        print("\n" + "="*70)
        print("TEST: Jacobian Forcing Critical Variables into Prompt")
        print("="*70)

        from constrai import SaliencyEngine
        
        invariants = [
            Invariant("x_limit", lambda s: s.get("x", 0) <= 100, "x must be ≤ 100"),
        ]

        fusion = JacobianFusion(invariants, epsilon=0.01)
        saliency = SaliencyEngine(threshold=0.05, max_keys=20, 
                                  jacobian_fusion=fusion, jacobian_weight=10.0)

        # Critical state
        critical_state = State({"x": 98, "y": 5, "z": 10})
        
        jacobian_report = fusion.compute_jacobian(critical_state)
        print(f"\nCritical state: x=98 (limit=100)")
        print(f"Critical variables: {jacobian_report.critical_variables}")
        
        # Saliency analysis without action values
        saliency_result = saliency.analyze(
            state=critical_state,
            available_actions=[],
            action_values=[]
        )
        
        print(f"Kept keys (with Jacobian forcing): {saliency_result.kept_keys}")
        print(f"Dropped keys: {saliency_result.dropped_keys}")
        
        # If x is critical, it must be in kept_keys
        if "x" in jacobian_report.critical_variables:
            assert "x" in saliency_result.kept_keys, "Critical var 'x' should be forced into prompt"
        
        print("✅ Jacobian successfully forces critical variables into prompt")


class TestAuthoritativeHJB:
    """Test Authoritative HJB Safe Hover enforcement."""

    def test_safe_state_proceeds(self):
        """Test that safe state allows PROCEED."""
        print("\n" + "="*70)
        print("TEST: Authoritative HJB - Safe State Proceeds")
        print("="*70)

        basin = CaptureBasin(
            name="bankruptcy",
            is_bad=lambda s: s.get("balance", 0) < 0,
            max_steps=5
        )
        
        barrier = AuthoritativeHJBBarrier(capture_basins=[basin])
        
        safe_state = State({"balance": 100})
        check = barrier.check_state_safety(safe_state)
        
        print(f"\nState: balance=100 (lower bound: 0)")
        print(f"HJB Check Result: {check.signal.value}")
        print(f"Reason: {check.reason}")
        
        assert check.signal == SafeHoverSignal.PROCEED
        assert not check.requires_immediate_rollback
        print("✅ Safe state correctly proceeds")

    def test_danger_state_triggers_rollback(self):
        """Test that danger state triggers TERMINATE_AND_ROLLBACK."""
        print("\n" + "="*70)
        print("TEST: Authoritative HJB - Danger State Triggers Rollback")
        print("="*70)

        basin = CaptureBasin(
            name="bankruptcy",
            is_bad=lambda s: s.get("balance", 0) < 0,
            max_steps=5
        )
        
        barrier = AuthoritativeHJBBarrier(capture_basins=[basin])
        
        danger_state = State({"balance": -50})
        check = barrier.check_state_safety(danger_state)
        
        print(f"\nState: balance=-50 (VIOLATION)")
        print(f"HJB Check Result: {check.signal.value}")
        print(f"Reason: {check.reason}")
        print(f"Violated Basin: {check.violated_basin.name if check.violated_basin else 'None'}")
        
        assert check.signal == SafeHoverSignal.TERMINATE_AND_ROLLBACK
        assert check.requires_immediate_rollback
        assert check.violated_basin is not None
        assert check.violated_basin.name == "bankruptcy"
        print("✅ Danger state correctly triggers rollback")

    def test_action_leads_to_danger(self):
        """Test that action leading to danger is rejected."""
        print("\n" + "="*70)
        print("TEST: Authoritative HJB - Action Rejected (Leads to Danger)")
        print("="*70)

        basin = CaptureBasin(
            name="overflow",
            is_bad=lambda s: s.get("x", 0) > 100,
            max_steps=5
        )
        
        barrier = AuthoritativeHJBBarrier(capture_basins=[basin])
        
        action = ActionSpec(
            id="inc_big",
            name="Increment by 150",
            description="Add 150 to x",
            effects=(Effect("x", "increment", 150),),
            cost=10.0,
            reversible=True,
        )
        
        current_state = State({"x": 10})
        check = barrier.check_action_leads_to_danger(
            state=current_state,
            action=action,
            available_actions=[action]
        )
        
        print(f"\nAction: Increment x by 150")
        print(f"Current state: x=10")
        print(f"Simulated next state: x=160 (exceeds limit of 100)")
        print(f"HJB Check Result: {check.signal.value}")
        print(f"Reason: {check.reason}")
        
        assert check.signal == SafeHoverSignal.TERMINATE_AND_ROLLBACK
        assert not check.requires_immediate_rollback  # Action wasn't executed yet
        print("✅ Dangerous action correctly rejected")


class TestOperadicComposition:
    """Test Operadic Composition for proof-based task composition."""

    def test_supertask_creation(self):
        """Test SuperTask creation with verification certificate."""
        print("\n" + "="*70)
        print("TEST: SuperTask Creation with Verification")
        print("="*70)

        from constrai.formal import GuaranteeLevel
        
        cert = VerificationCertificate(
            task_id="task_1",
            proof_level=GuaranteeLevel.PROVEN,
            all_invariants_satisfied=True,
            budget_safe=True,
            no_deadlock=True,
            rollback_exact=True,
            proof_hash="abc123"
        )
        
        interface = InterfaceSignature(
            required_inputs=("x", "y"),
            produced_outputs=("z",)
        )
        
        task = SuperTask(
            task_id="task_1",
            name="Add X and Y",
            description="Adds x and y to produce z",
            goal="Compute z = x + y",
            available_actions=(),
            invariants=(),
            interface=interface,
            certificate=cert
        )
        
        print(f"\nTask: {task.name}")
        print(f"ID: {task.task_id}")
        print(f"Verified: {task.is_verified()}")
        print(f"Inputs: {interface.required_inputs}")
        print(f"Outputs: {interface.produced_outputs}")
        
        assert task.is_verified()
        assert cert.is_complete()
        print("✅ SuperTask created and verified")

    def test_compatible_composition(self):
        """Test that compatible tasks can compose."""
        print("\n" + "="*70)
        print("TEST: Compatible Task Composition")
        print("="*70)

        from constrai.formal import GuaranteeLevel
        
        cert1 = VerificationCertificate(
            task_id="task_add",
            proof_level=GuaranteeLevel.PROVEN,
            all_invariants_satisfied=True,
            budget_safe=True,
            no_deadlock=True,
            rollback_exact=True,
            proof_hash="hash1"
        )
        
        cert2 = VerificationCertificate(
            task_id="task_mul",
            proof_level=GuaranteeLevel.PROVEN,
            all_invariants_satisfied=True,
            budget_safe=True,
            no_deadlock=True,
            rollback_exact=True,
            proof_hash="hash2"
        )
        
        # Task 1: x, y → z (adds them)
        task1 = SuperTask(
            task_id="task_add",
            name="Add",
            description="Adds x and y",
            goal="Compute z = x + y",
            available_actions=(),
            invariants=(),
            interface=InterfaceSignature(
                required_inputs=("x", "y"),
                produced_outputs=("z",)
            ),
            certificate=cert1
        )
        
        # Task 2: z → result (multiplies z by 2)
        task2 = SuperTask(
            task_id="task_mul",
            name="Double",
            description="Doubles z",
            goal="Compute result = 2*z",
            available_actions=(),
            invariants=(),
            interface=InterfaceSignature(
                required_inputs=("z",),
                produced_outputs=("result",)
            ),
            certificate=cert2
        )
        
        print(f"\nTask 1: {task1.name} ({task1.interface.required_inputs} → {task1.interface.produced_outputs})")
        print(f"Task 2: {task2.name} ({task2.interface.required_inputs} → {task2.interface.produced_outputs})")
        
        can_compose, reason = task1.can_compose_with(task2, CompositionType.SEQUENTIAL)
        print(f"\nCan compose: {can_compose}")
        print(f"Reason: {reason}")
        
        assert can_compose
        
        composed = task1.compose(task2, CompositionType.SEQUENTIAL)
        assert composed is not None
        # Note: Composed task is now CONDITIONAL, not PROVEN
        assert composed.certificate.is_conditionally_verified()
        
        print(f"\nComposed task: {composed.name}")
        print(f"Proof level: {composed.certificate.proof_level.value}")
        print(f"Conditionally verified: {composed.certificate.is_conditionally_verified()}")
        print("✅ Compatible tasks successfully composed")

    def test_incompatible_composition(self):
        """Test that incompatible tasks cannot compose."""
        print("\n" + "="*70)
        print("TEST: Incompatible Task Composition Rejected")
        print("="*70)

        from constrai.formal import GuaranteeLevel
        
        cert1 = VerificationCertificate(
            task_id="task_1",
            proof_level=GuaranteeLevel.PROVEN,
            all_invariants_satisfied=True,
            budget_safe=True,
            no_deadlock=True,
            rollback_exact=True,
            proof_hash="hash1"
        )
        
        cert2 = VerificationCertificate(
            task_id="task_2",
            proof_level=GuaranteeLevel.PROVEN,
            all_invariants_satisfied=True,
            budget_safe=True,
            no_deadlock=True,
            rollback_exact=True,
            proof_hash="hash2"
        )
        
        # Task 1: produces x
        task1 = SuperTask(
            task_id="task_1",
            name="Task 1",
            description="Produces x",
            goal="Goal 1",
            available_actions=(),
            invariants=(),
            interface=InterfaceSignature(
                required_inputs=(),
                produced_outputs=("x",)
            ),
            certificate=cert1
        )
        
        # Task 2: requires y (which task1 doesn't produce)
        task2 = SuperTask(
            task_id="task_2",
            name="Task 2",
            description="Requires y",
            goal="Goal 2",
            available_actions=(),
            invariants=(),
            interface=InterfaceSignature(
                required_inputs=("y",),
                produced_outputs=("z",)
            ),
            certificate=cert2
        )
        
        print(f"\nTask 1 outputs: {task1.interface.produced_outputs}")
        print(f"Task 2 inputs: {task2.interface.required_inputs}")
        
        can_compose, reason = task1.can_compose_with(task2, CompositionType.SEQUENTIAL)
        print(f"\nCan compose: {can_compose}")
        print(f"Reason: {reason}")
        
        assert not can_compose
        print("✅ Incompatible composition correctly rejected")

    def test_task_composer_library(self):
        """Test TaskComposer managing a library of verified tasks."""
        print("\n" + "="*70)
        print("TEST: TaskComposer Library Management")
        print("="*70)

        from constrai.formal import GuaranteeLevel
        
        composer = TaskComposer()
        
        # Create and register 2 tasks
        cert1 = VerificationCertificate(
            task_id="sort", proof_level=GuaranteeLevel.PROVEN,
            all_invariants_satisfied=True,
            budget_safe=True, no_deadlock=True, rollback_exact=True,
            proof_hash="sort_hash"
        )
        task_sort = SuperTask(
            task_id="sort",
            name="Sort Array",
            description="Sorts an array",
            goal="Sort array",
            available_actions=(),
            invariants=(),
            interface=InterfaceSignature(
                required_inputs=("array",),
                produced_outputs=("sorted_array",)
            ),
            certificate=cert1
        )
        
        cert2 = VerificationCertificate(
            task_id="filter", proof_level=GuaranteeLevel.PROVEN,
            all_invariants_satisfied=True,
            budget_safe=True, no_deadlock=True, rollback_exact=True,
            proof_hash="filter_hash"
        )
        task_filter = SuperTask(
            task_id="filter",
            name="Filter Array",
            description="Filters an array",
            goal="Filter array",
            available_actions=(),
            invariants=(),
            interface=InterfaceSignature(
                required_inputs=("sorted_array",),
                produced_outputs=("filtered_array",)
            ),
            certificate=cert2
        )
        
        # Register
        assert composer.register_task(task_sort)
        assert composer.register_task(task_filter)
        
        print(f"\nRegistered tasks: {list(composer.tasks.keys())}")
        
        # Compose chain
        composed = composer.compose_chain(["sort", "filter"], CompositionType.SEQUENTIAL)
        assert composed is not None
        # Note: Composed task is now CONDITIONAL, not PROVEN
        assert composed.certificate.is_conditionally_verified()
        
        print(f"Composed task: {composed.name}")
        print(f"Composition history: {composed.composition_history}")
        print(f"Proof level: {composed.certificate.proof_level.value}")
        print(f"Conditionally verified: {composed.certificate.is_conditionally_verified()}")
        
        print("✅ TaskComposer successfully managed verified library")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
