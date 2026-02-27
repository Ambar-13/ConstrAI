"""
ConstrAI Test Suite — Continued

S12 — State immutability.
S13 — Monte Carlo statistical validation.
"""
import json
import math
import random
import time

from constrai import (
    ActionSpec,
    Belief,
    BeliefState,
    BudgetController,
    CausalGraph,
    Effect,
    Invariant,
    MockLLMAdapter,
    Orchestrator,
    SafetyKernel,
    State,
    TaskDefinition,
    TerminationReason,
)

# S12 — State immutability

class TestS12StateImmutability:

    def test_s12_1_external_mutation_blocked(self):
        original_dict = {"list": [1, 2, 3], "dict": {"a": 1}}
        s = State(original_dict)
        original_dict["list"].append(4)
        assert s.get("list") == [1, 2, 3], "S12.1 external mutation blocked"

    def test_s12_2_get_returns_copy(self):
        s = State({"list": [1, 2, 3]})
        lst = s.get("list")
        lst.append(99)
        assert s.get("list") == [1, 2, 3], "S12.2 get returns copy"

    def test_s12_3_setattr_blocked(self):
        s = State({"x": 1})
        try:
            s._vars = {}
            assert False, "S12.3 setattr should raise"
        except AttributeError:
            pass

    def test_s12_4_has_and_keys(self):
        s = State({"a": 1, "b": 2})
        assert s.has("a"), "S12.4 has('a')"
        assert not s.has("z"), "S12.4 not has('z')"
        keys = s.keys()
        assert "a" in keys and "b" in keys, "S12.4 keys() contains 'a' and 'b'"

    def test_s12_5_fingerprint_deterministic(self):
        s1 = State({"x": 1, "y": 2})
        s2 = State({"x": 1, "y": 2})
        assert s1.fingerprint == s2.fingerprint, "S12.5 fingerprint deterministic"

    def test_s12_6_fingerprint_changes_with_state(self):
        s1 = State({"x": 1})
        s2 = State({"x": 2})
        assert s1.fingerprint != s2.fingerprint, "S12.6 fingerprint changes with value"


# S13 — Monte Carlo statistical validation

def _make_random_task(seed: int) -> TaskDefinition:
    """Generate a random but solvable task."""
    rng = random.Random(seed)
    n_steps = rng.randint(3, 8)
    budget = n_steps * 5.0 + rng.uniform(5, 20)

    actions = []
    for i in range(n_steps):
        actions.append(ActionSpec(
            id=f"step_{i}", name=f"Step {i}",
            description=f"Perform step {i}",
            effects=(Effect("progress", "increment", 1),),
            cost=rng.uniform(1.0, 4.0),
            risk_level=rng.choice(["low", "low", "medium"]),
        ))

    deps = {}
    for i in range(n_steps):
        if i == 0:
            deps[f"step_{i}"] = []
        else:
            deps[f"step_{i}"] = [(f"step_{i-1}", f"Need step {i-1}")]

    return TaskDefinition(
        goal=f"Complete {n_steps} steps",
        initial_state=State({"progress": 0}),
        available_actions=actions,
        invariants=[
            Invariant("bound", lambda s, n=n_steps+5: s.get("progress", 0) <= n),
        ],
        budget=budget,
        goal_predicate=lambda s, n=n_steps: s.get("progress", 0) >= n,
        goal_progress_fn=lambda s, n=n_steps: min(1.0, s.get("progress", 0) / n),
        dependencies=deps,
        min_action_cost=0.5,
        max_consecutive_failures=10,
    )


class TestMC13MonteCarlo:

    N_TRIALS = 1000

    def test_mc13_1_budget_safe_all_trials(self):
        """T1: spent ≤ budget in every trial (100% required)."""
        violations = 0
        for seed in range(self.N_TRIALS):
            task = _make_random_task(seed)
            engine = Orchestrator(task, llm=MockLLMAdapter())
            result = engine.run()
            if result.total_cost > task.budget:
                violations += 1
        assert violations == 0, \
            f"MC13.1 Budget safe: {self.N_TRIALS - violations}/{self.N_TRIALS} " \
            f"({violations} violations)"

    def test_mc13_2_invariants_safe_all_trials(self):
        """T3: blocking invariants hold on final state in every trial."""
        violations = 0
        for seed in range(self.N_TRIALS):
            task = _make_random_task(seed)
            engine = Orchestrator(task, llm=MockLLMAdapter())
            result = engine.run()
            for inv in task.invariants:
                if inv.enforcement == "blocking":
                    holds, _ = inv.check(result.final_state)
                    if not holds:
                        violations += 1
                        break
        assert violations == 0, \
            f"MC13.2 Invariants safe: {self.N_TRIALS - violations}/{self.N_TRIALS} " \
            f"({violations} violations)"

    def test_mc13_3_all_trials_terminate(self):
        """T2: every trial terminates (returns a result)."""
        terminated = 0
        for seed in range(self.N_TRIALS):
            task = _make_random_task(seed)
            engine = Orchestrator(task, llm=MockLLMAdapter())
            result = engine.run()
            terminated += 1  # If we got here, it terminated
        assert terminated == self.N_TRIALS, \
            f"MC13.3 All terminated: {terminated}/{self.N_TRIALS}"

    def test_mc13_4_trace_integrity_all_trials(self):
        """T6: hash chain valid in every trial."""
        failures = 0
        for seed in range(self.N_TRIALS):
            task = _make_random_task(seed)
            engine = Orchestrator(task, llm=MockLLMAdapter())
            engine.run()
            ok, _ = engine.kernel.trace.verify_integrity()
            if not ok:
                failures += 1
        assert failures == 0, \
            f"MC13.4 Trace integrity: {self.N_TRIALS - failures}/{self.N_TRIALS} " \
            f"({failures} failures)"

    def test_mc13_5_goal_rate_above_threshold(self):
        """Empirical: goal should be achieved in >80% of solvable tasks."""
        goal_achieved = 0
        for seed in range(self.N_TRIALS):
            task = _make_random_task(seed)
            engine = Orchestrator(task, llm=MockLLMAdapter())
            result = engine.run()
            if result.goal_achieved:
                goal_achieved += 1

        goal_rate = goal_achieved / self.N_TRIALS
        # Wilson score 95% CI lower bound
        n, p = self.N_TRIALS, goal_rate
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        lower_ci = center - margin

        assert goal_rate > 0.80, \
            f"MC13.5 Goal rate {goal_rate:.1%} > 80% " \
            f"(95% CI lower bound: {lower_ci:.3f})"


if __name__ == "__main__":
    # Allow running this file directly for quick local checks.
    import pytest
    pytest.main([__file__, "-v"])
