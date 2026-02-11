"""
ConstrAI Test Suite — Continued
§12 State Immutability
§13 Monte Carlo Statistical Validation
"""
import json, math, random, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constrai import (
    State, Effect, ActionSpec, Invariant,
    SafetyKernel, BudgetController,
    Belief, BeliefState, CausalGraph,
    TaskDefinition, Orchestrator, TerminationReason, MockLLMAdapter,
)

from tests.test_constrai import TestResults

T = TestResults()


# ═══════════════════════════════════════════════════════════════════════════
# §12  STATE IMMUTABILITY
# ═══════════════════════════════════════════════════════════════════════════

def test_state_immutability():
    print("\n§12  State Immutability")
    print("-" * 40)

    s = State({"list": [1, 2, 3], "dict": {"a": 1}})

    # External mutation shouldn't affect state
    original_dict = {"list": [1, 2, 3], "dict": {"a": 1}}
    s2 = State(original_dict)
    original_dict["list"].append(4)
    T.check("S12.1 external mutation blocked", s2.get("list") == [1, 2, 3])

    # Get returns copies
    lst = s.get("list")
    lst.append(99)
    T.check("S12.2 get returns copy", s.get("list") == [1, 2, 3])

    # setattr blocked
    try:
        s._vars = {}
        T.check("S12.3 setattr blocked", False)
    except AttributeError:
        T.check("S12.3 setattr blocked", True)

    assert T.summary()


# ═══════════════════════════════════════════════════════════════════════════
# §13  MONTE CARLO STATISTICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def test_monte_carlo_validation():
    print("\n§13  Monte Carlo Validation (1000 runs)")
    print("-" * 40)

    N_TRIALS = 1000
    results = {
        'budget_safe': 0,
        'invariant_safe': 0,
        'terminated': 0,
        'goal_achieved': 0,
        'costs': [],
        'steps': [],
        'trace_valid': 0,
    }

    t0 = time.time()
    for trial in range(N_TRIALS):
        task = make_random_task(trial)
        engine = Orchestrator(task)
        result = engine.run()

        # T1: Budget Safety
        if result.total_cost <= task.budget:
            results['budget_safe'] += 1

        # T3: Invariant check on final state
        all_inv_ok = all(inv.check(result.final_state)[0] for inv in task.invariants)
        if all_inv_ok:
            results['invariant_safe'] += 1

        # T2: Termination (it returned, so it terminated)
        results['terminated'] += 1

        # T6: Trace integrity
        if engine.kernel.trace.verify_integrity()[0]:
            results['trace_valid'] += 1

        if result.goal_achieved:
            results['goal_achieved'] += 1

        results['costs'].append(result.total_cost)
        results['steps'].append(result.total_steps)

    elapsed = time.time() - t0

    T.check(f"MC13.1 Budget safe: {results['budget_safe']}/{N_TRIALS}",
            results['budget_safe'] == N_TRIALS)
    T.check(f"MC13.2 Invariants safe: {results['invariant_safe']}/{N_TRIALS}",
            results['invariant_safe'] == N_TRIALS)
    T.check(f"MC13.3 All terminated: {results['terminated']}/{N_TRIALS}",
            results['terminated'] == N_TRIALS)
    T.check(f"MC13.4 Trace integrity: {results['trace_valid']}/{N_TRIALS}",
            results['trace_valid'] == N_TRIALS)

    goal_rate = results['goal_achieved'] / N_TRIALS
    mean_cost = sum(results['costs']) / N_TRIALS
    mean_steps = sum(results['steps']) / N_TRIALS

    T.check(f"MC13.5 Goal rate: {goal_rate:.1%} (>80%)", goal_rate > 0.80)
    print(f"       Mean cost: ${mean_cost:.2f}")
    print(f"       Mean steps: {mean_steps:.1f}")
    print(f"       Time: {elapsed:.2f}s ({elapsed/N_TRIALS*1000:.2f}ms/trial)")

    # Wilson score confidence interval for goal rate
    n = N_TRIALS
    p = goal_rate
    z = 1.96  # 95% CI
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    print(f"       Goal rate 95% CI: [{center-margin:.3f}, {center+margin:.3f}]")

    assert T.summary()

def make_random_task(seed: int) -> TaskDefinition:
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


if __name__ == "__main__":
    # Allow running this file directly for quick local checks.
    test_state_immutability()
    test_monte_carlo_validation()
