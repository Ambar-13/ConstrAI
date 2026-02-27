"""Tests for constrai.operadic_composition — proof-based task composition.

Covers InterfaceSignature, VerificationCertificate, SuperTask, and TaskComposer
including composition logic, certificate propagation, and the operadic transfer
principle (Theorems OC-1, OC-2).
"""
from __future__ import annotations

from typing import Tuple

import pytest

from constrai.formal import ActionSpec, Effect, GuaranteeLevel, Invariant, State
from constrai.operadic_composition import (
    CompositionType,
    InterfaceSignature,
    SuperTask,
    TaskComposer,
    VerificationCertificate,
)

# ─── helpers ─────────────────────────────────────────────────────────────────

def _action(aid: str, cost: float = 1.0) -> ActionSpec:
    return ActionSpec(
        id=aid,
        name=aid,
        description="",
        effects=(Effect("x", "increment", 1),),
        cost=cost,
    )


def _cert(task_id: str, *,
          proof_level: GuaranteeLevel = GuaranteeLevel.PROVEN,
          complete: bool = True) -> VerificationCertificate:
    return VerificationCertificate(
        task_id=task_id,
        proof_level=proof_level,
        all_invariants_satisfied=complete,
        budget_safe=complete,
        no_deadlock=complete,
        rollback_exact=complete,
        proof_hash=f"hash_{task_id}",
    )


def _iface(inputs: Tuple[str, ...] = (),
           outputs: Tuple[str, ...] = ()) -> InterfaceSignature:
    return InterfaceSignature(
        required_inputs=inputs,
        produced_outputs=outputs,
    )


def _supertask(tid: str, *,
               inputs: Tuple[str, ...] = (),
               outputs: Tuple[str, ...] = (),
               actions=None,
               complete: bool = True) -> SuperTask:
    actions = actions or (_action(tid + "_act"),)
    return SuperTask(
        task_id=tid,
        name=tid,
        description=f"task {tid}",
        goal=f"goal of {tid}",
        available_actions=tuple(actions),
        invariants=(Invariant("inv", lambda s: True, "always true"),),
        interface=_iface(inputs, outputs),
        certificate=_cert(tid, complete=complete),
    )


# ─── InterfaceSignature.compatible_with ──────────────────────────────────────

class TestInterfaceSignatureCompatible:
    def test_compatible_when_outputs_cover_inputs(self):
        a = _iface(inputs=(), outputs=("x", "y"))
        b = _iface(inputs=("x",), outputs=("z",))
        assert a.compatible_with(b) is True

    def test_not_compatible_when_outputs_miss_inputs(self):
        a = _iface(inputs=(), outputs=("x",))
        b = _iface(inputs=("y",), outputs=())
        assert a.compatible_with(b) is False

    def test_not_compatible_when_forbidden_conflict(self):
        a = InterfaceSignature(
            required_inputs=(),
            produced_outputs=("x",),
            forbidden_variables=("conflict",),
        )
        b = InterfaceSignature(
            required_inputs=(),
            produced_outputs=(),
            forbidden_variables=("conflict",),
        )
        assert a.compatible_with(b) is False

    def test_compatible_no_overlap_forbidden(self):
        a = InterfaceSignature(
            required_inputs=(),
            produced_outputs=("x",),
            forbidden_variables=("a_forbidden",),
        )
        b = InterfaceSignature(
            required_inputs=("x",),
            produced_outputs=("y",),
            forbidden_variables=("b_forbidden",),
        )
        assert a.compatible_with(b) is True


# ─── InterfaceSignature preconditions/postconditions ─────────────────────────

class TestInterfaceSignaturePrePost:
    def test_check_preconditions_all_pass(self):
        iface = InterfaceSignature(
            required_inputs=("x",),
            produced_outputs=(),
            precondition_predicates=(lambda s: s.get("x", 0) > 0,),
        )
        ok, msg = iface.check_preconditions(State({"x": 5}))
        assert ok is True
        assert "satisfied" in msg

    def test_check_preconditions_one_fails(self):
        iface = InterfaceSignature(
            required_inputs=(),
            produced_outputs=(),
            precondition_predicates=(lambda s: False,),
        )
        ok, msg = iface.check_preconditions(State({}))
        assert ok is False
        assert "Precondition 0" in msg

    def test_check_preconditions_raises(self):
        def bad(s):
            raise ValueError("boom")
        iface = InterfaceSignature(
            required_inputs=(),
            produced_outputs=(),
            precondition_predicates=(bad,),
        )
        ok, msg = iface.check_preconditions(State({}))
        assert ok is False
        assert "exception" in msg

    def test_check_postconditions_all_pass(self):
        iface = InterfaceSignature(
            required_inputs=(),
            produced_outputs=("y",),
            postcondition_predicates=(lambda s: s.get("y", 0) > 0,),
        )
        ok, _msg = iface.check_postconditions(State({"y": 1}))
        assert ok is True

    def test_check_postconditions_one_fails(self):
        iface = InterfaceSignature(
            required_inputs=(),
            produced_outputs=(),
            postcondition_predicates=(lambda s: False,),
        )
        ok, msg = iface.check_postconditions(State({}))
        assert ok is False
        assert "Postcondition 0" in msg

    def test_check_postconditions_raises(self):
        def bad(s):
            raise RuntimeError("post boom")
        iface = InterfaceSignature(
            required_inputs=(),
            produced_outputs=(),
            postcondition_predicates=(bad,),
        )
        ok, msg = iface.check_postconditions(State({}))
        assert ok is False
        assert "exception" in msg

    def test_no_predicates_returns_satisfied(self):
        iface = _iface()
        ok, _ = iface.check_preconditions(State({}))
        assert ok is True
        ok2, _ = iface.check_postconditions(State({}))
        assert ok2 is True


# ─── VerificationCertificate ─────────────────────────────────────────────────

class TestVerificationCertificate:
    def test_is_complete_proven_all_true(self):
        c = _cert("t")
        assert c.is_complete() is True

    def test_is_complete_false_when_not_proven(self):
        c = _cert("t", proof_level=GuaranteeLevel.CONDITIONAL)
        assert c.is_complete() is False

    def test_is_complete_false_when_any_false(self):
        c = VerificationCertificate(
            task_id="t",
            proof_level=GuaranteeLevel.PROVEN,
            all_invariants_satisfied=True,
            budget_safe=False,
            no_deadlock=True,
            rollback_exact=True,
            proof_hash="h",
        )
        assert c.is_complete() is False

    def test_is_conditionally_verified_proven(self):
        c = _cert("t", proof_level=GuaranteeLevel.PROVEN)
        assert c.is_conditionally_verified() is True

    def test_is_conditionally_verified_conditional(self):
        c = _cert("t", proof_level=GuaranteeLevel.CONDITIONAL)
        assert c.is_conditionally_verified() is True

    def test_is_conditionally_verified_heuristic_false(self):
        c = _cert("t", proof_level=GuaranteeLevel.HEURISTIC)
        assert c.is_conditionally_verified() is False


# ─── SuperTask.can_compose_with ───────────────────────────────────────────────

class TestSuperTaskCanComposeWith:
    def test_source_not_verified(self):
        a = _supertask("a", outputs=("x",), complete=False)
        b = _supertask("b", inputs=("x",))
        ok, reason = a.can_compose_with(b, CompositionType.SEQUENTIAL)
        assert ok is False
        assert "a" in reason

    def test_target_not_verified(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("x",), complete=False)
        ok, reason = a.can_compose_with(b, CompositionType.SEQUENTIAL)
        assert ok is False
        assert "b" in reason

    def test_sequential_interface_mismatch(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("y",))  # b needs y, but a only produces x
        ok, reason = a.can_compose_with(b, CompositionType.SEQUENTIAL)
        assert ok is False
        assert "mismatch" in reason.lower()

    def test_sequential_compatible(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("x",))
        ok, _ = a.can_compose_with(b, CompositionType.SEQUENTIAL)
        assert ok is True

    def test_parallel_variable_conflict(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", outputs=("x",))
        ok, reason = a.can_compose_with(b, CompositionType.PARALLEL)
        assert ok is False
        assert "conflict" in reason.lower()

    def test_parallel_no_conflict(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", outputs=("y",))
        ok, _ = a.can_compose_with(b, CompositionType.PARALLEL)
        assert ok is True


# ─── SuperTask.compose ────────────────────────────────────────────────────────

class TestSuperTaskCompose:
    def test_sequential_compose_returns_supertask(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("x",), outputs=("z",))
        composed = a.compose(b, CompositionType.SEQUENTIAL)
        assert composed is not None
        assert composed.composition_type == CompositionType.SEQUENTIAL

    def test_compose_returns_none_on_incompatible(self, capsys):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("y",))
        composed = a.compose(b, CompositionType.SEQUENTIAL)
        assert composed is None

    def test_parallel_compose(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", outputs=("y",))
        composed = a.compose(b, CompositionType.PARALLEL)
        assert composed is not None
        assert "x" in composed.interface.produced_outputs
        assert "y" in composed.interface.produced_outputs

    def test_conditional_compose(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", outputs=("x",))
        # CONDITIONAL type uses sequential interface logic
        composed = a.compose(b, CompositionType.CONDITIONAL)
        assert composed is not None
        assert composed.composition_type == CompositionType.CONDITIONAL

    def test_composed_certificate_is_conditional(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("x",))
        composed = a.compose(b, CompositionType.SEQUENTIAL)
        assert composed is not None
        assert composed.certificate.proof_level == GuaranteeLevel.CONDITIONAL

    def test_composed_history_contains_both_ids(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("x",))
        composed = a.compose(b, CompositionType.SEQUENTIAL)
        assert composed is not None
        assert "a" in composed.composition_history
        assert "b" in composed.composition_history

    def test_overlapping_invariants_recorded_in_assumptions(self):
        shared_inv = Invariant("shared", lambda s: True, "shared")
        a = SuperTask(
            task_id="a", name="a", description="",
            goal="g", available_actions=(_action("act_a"),),
            invariants=(shared_inv,),
            interface=_iface(outputs=("x",)),
            certificate=_cert("a"),
        )
        b = SuperTask(
            task_id="b", name="b", description="",
            goal="g", available_actions=(_action("act_b"),),
            invariants=(shared_inv,),
            interface=_iface(inputs=("x",)),
            certificate=_cert("b"),
        )
        composed = a.compose(b, CompositionType.SEQUENTIAL)
        assert composed is not None
        assert any("shared" in asmp for asmp in composed.certificate.assumptions)

    def test_overlapping_action_ids_noted_in_assumptions(self):
        act = _action("shared_act")
        a = _supertask("a", outputs=("x",), actions=(act,))
        b = _supertask("b", inputs=("x",), actions=(act,))
        composed = a.compose(b, CompositionType.SEQUENTIAL)
        assert composed is not None
        assert any("shared_act" in asmp for asmp in composed.certificate.assumptions)

    def test_repeat_compose(self):
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", outputs=("y",))
        composed = a.compose(b, CompositionType.REPEAT)
        assert composed is not None


# ─── TaskComposer ─────────────────────────────────────────────────────────────

class TestTaskComposer:
    def test_register_verified_task(self):
        c = TaskComposer()
        t = _supertask("a", outputs=("x",))
        ok = c.register_task(t)
        assert ok is True
        assert "a" in c.tasks

    def test_register_duplicate_task(self, capsys):
        c = TaskComposer()
        t = _supertask("a")
        c.register_task(t)
        ok = c.register_task(t)
        assert ok is False

    def test_register_unverified_task(self, capsys):
        c = TaskComposer()
        t = _supertask("a", complete=False)
        ok = c.register_task(t)
        assert ok is False

    def test_compose_chain_empty_returns_none(self):
        c = TaskComposer()
        assert c.compose_chain([]) is None

    def test_compose_chain_first_not_found(self, capsys):
        c = TaskComposer()
        assert c.compose_chain(["missing"]) is None

    def test_compose_chain_second_not_found(self, capsys):
        c = TaskComposer()
        a = _supertask("a", outputs=("x",))
        c.register_task(a)
        result = c.compose_chain(["a", "missing"])
        assert result is None

    def test_compose_chain_incompatible(self, capsys):
        c = TaskComposer()
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("y",))  # mismatch
        c.register_task(a)
        c.register_task(b)
        result = c.compose_chain(["a", "b"])
        assert result is None

    def test_compose_chain_two_tasks(self):
        c = TaskComposer()
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("x",))
        c.register_task(a)
        c.register_task(b)
        result = c.compose_chain(["a", "b"])
        assert result is not None

    def test_compose_chain_single_task(self):
        c = TaskComposer()
        a = _supertask("a", outputs=("x",))
        c.register_task(a)
        result = c.compose_chain(["a"])
        assert result is not None
        assert result.task_id == "a"

    def test_get_composed_tasks_depth_zero(self):
        c = TaskComposer()
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", outputs=("y",))
        c.register_task(a)
        c.register_task(b)
        tasks = c.get_composed_tasks(depth=0)
        assert len(tasks) == 2

    def test_get_composed_tasks_depth_one(self):
        c = TaskComposer()
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", outputs=("y",))
        c.register_task(a)
        c.register_task(b)
        tasks = c.get_composed_tasks(depth=1)
        assert len(tasks) == 2

    def test_get_composed_tasks_depth_two(self):
        c = TaskComposer()
        a = _supertask("a", outputs=("x",))
        b = _supertask("b", inputs=("x",), outputs=("y",))
        c.register_task(a)
        c.register_task(b)
        tasks = c.get_composed_tasks(depth=2)
        # a→b is compatible; b→a is not. At least one composition.
        assert len(tasks) >= 1
