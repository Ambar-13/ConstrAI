"""
ConstrAI Test Suite — All Theorems + Integration + Adversarial
"""
import json, math, random, time, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constrai import (
    State, Effect, ActionSpec, Invariant,
    SafetyKernel, SafetyVerdict, CheckResult,
    BudgetController, ExecutionTrace, TraceEntry,
    Belief, BeliefState, CausalGraph, ActionValueComputer,
    ReasoningRequest, ReasoningResponse, parse_llm_response,
    MockLLMAdapter,
    TaskDefinition, Orchestrator, ExecutionResult, TerminationReason,
    ProgressMonitor,
)

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            msg = f"  ✗ {name}: {detail}"
            self.errors.append(msg)
            print(msg)
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\n  FAILURES:")
            for e in self.errors:
                print(f"    {e}")
        print(f"{'='*60}")
        return self.failed == 0

T = TestResults()

# ═══ §1 T1: Budget Safety ═══
print("\n§1  T1: Budget Safety")
print("-" * 40)

bc = BudgetController(100.0)
ok, _ = bc.can_afford(50.0)
T.check("T1.1 can afford within budget", ok)
bc.charge("a1", 50.0)
ok, _ = bc.can_afford(60.0)
T.check("T1.2 cannot afford over budget", not ok)
bc.charge("a2", 50.0)
ok, _ = bc.can_afford(0.01)
T.check("T1.3 cannot afford at limit", not ok)
T.check("T1.4 spent == budget", bc.spent == 100.0)

kernel = SafetyKernel(budget=10.0, invariants=[], min_action_cost=1.0)
state = State({"x": 0})
expensive = ActionSpec(id="big", name="Big", description="Costly",
                       effects=(Effect("x", "increment", 1),), cost=15.0)
v = kernel.evaluate(state, expensive)
T.check("T1.5 kernel rejects over-budget", not v.approved)

try:
    BudgetController(100.0).charge("bad", -5.0)
    T.check("T1.6 negative cost rejected", False)
except (ValueError, AssertionError):
    T.check("T1.6 negative cost rejected", True)

# ═══ §2 T2: Termination ═══
print("\n§2  T2: Termination")
print("-" * 40)

kernel2 = SafetyKernel(budget=5.0, invariants=[], min_action_cost=1.0)
T.check("T2.1 max_steps = floor(B/ε)", kernel2.max_steps == 5)

state = State({"n": 0})
action = ActionSpec(id="inc", name="Inc", description="+1",
                    effects=(Effect("n", "increment", 1),), cost=1.0)
count = 0
while count < 100:
    v = kernel2.evaluate(state, action)
    if not v.approved:
        break
    state, _ = kernel2.execute(state, action)
    count += 1

T.check("T2.2 terminated at 5", count == 5)
T.check("T2.3 budget exhausted", kernel2.budget.remaining == 0.0)

# ═══ §3 T3: Invariant Preservation ═══
print("\n§3  T3: Invariant Preservation")
print("-" * 40)

inv_max5 = Invariant("max_5", lambda s: (s.get("n", 0)) <= 5, "n ≤ 5")
kernel3 = SafetyKernel(budget=100.0, invariants=[inv_max5], min_action_cost=0.1)
state3 = State({"n": 0})
inc = ActionSpec(id="inc", name="Inc", description="+1",
                 effects=(Effect("n", "increment", 1),), cost=1.0)

for i in range(5):
    v = kernel3.evaluate(state3, inc)
    T.check(f"T3.{i+1} increment {i+1} allowed", v.approved)
    state3, _ = kernel3.execute(state3, inc)

v = kernel3.evaluate(state3, inc)
T.check("T3.6 6th increment blocked", not v.approved)
T.check("T3.7 n still 5", state3.get("n") == 5)

bad_inv = Invariant("bad", lambda s: 1/0, "Raises")
kernel_bad = SafetyKernel(budget=100.0, invariants=[bad_inv])
v = kernel_bad.evaluate(State({}), inc)
T.check("T3.8 exception = violation", not v.approved)

monitor_only = Invariant(
    "monitor_only",
    lambda s: False,
    "Should be logged but not block",
    severity="warning",  # back-compat: warning => monitoring
)
kernel_monitor = SafetyKernel(budget=100.0, invariants=[monitor_only])
v = kernel_monitor.evaluate(State({"n": 0}), inc)
T.check("T3.9 monitoring invariant does not block", v.approved)
T.check(
    "T3.10 monitoring invariant recorded as FAIL_INVARIANT",
    any(result == CheckResult.FAIL_INVARIANT for _, result, _ in v.checks),
)

# ═══ §4 T4: Monotone Spend ═══
print("\n§4  T4: Monotone Spend")
print("-" * 40)

bc4 = BudgetController(100.0)
prev = 0.0
for i in range(10):
    c = random.uniform(0.5, 5.0)
    if bc4.remaining >= c:
        bc4.charge(f"a{i}", c)
        T.check(f"T4.{i+1} monotone", bc4.spent >= prev)
        prev = bc4.spent

# ═══ §5 T5: Atomicity ═══
print("\n§5  T5: Atomicity")
print("-" * 40)

kernel5 = SafetyKernel(budget=100.0, invariants=[
    Invariant("positive", lambda s: s.get("x", 0) >= 0, "x ≥ 0")
])
state5 = State({"x": 5})
bad_action = ActionSpec(id="neg", name="Neg", description="x → -10",
                        effects=(Effect("x", "set", -10),), cost=1.0)
v = kernel5.evaluate(state5, bad_action)
T.check("T5.1 bad action rejected", not v.approved)
T.check("T5.2 state unchanged", state5.get("x") == 5)
T.check("T5.3 budget unchanged", kernel5.budget.spent == 0.0)

# ═══ §6 T6: Trace Integrity ═══
print("\n§6  T6: Trace Integrity")
print("-" * 40)

kernel6 = SafetyKernel(budget=100.0, invariants=[])
state6 = State({"step": 0})
for i in range(5):
    a = ActionSpec(id=f"a{i}", name=f"Step {i}", description="",
                   effects=(Effect("step", "set", i+1),), cost=1.0)
    state6, _ = kernel6.execute(state6, a)
ok, msg = kernel6.trace.verify_integrity()
T.check("T6.1 hash chain valid", ok, msg)
T.check("T6.2 trace length", kernel6.trace.length == 5)

# ═══ §7 T7: Rollback Exactness ═══
print("\n§7  T7: Rollback Exactness")
print("-" * 40)

kernel7 = SafetyKernel(budget=100.0, invariants=[])
orig = State({"x": 42, "y": [1, 2, 3], "z": "hello"})
action7 = ActionSpec(id="mod", name="Modify", description="",
                     effects=(Effect("x", "set", 999), Effect("y", "append", 4),
                              Effect("z", "delete")), cost=5.0)
modified, _ = kernel7.execute(orig, action7)
T.check("T7.1 state modified", modified != orig)
restored = kernel7.rollback(orig, modified, action7)
T.check("T7.2 rollback exact", restored == orig)
T.check("T7.3 x restored", restored.get("x") == 42)
T.check("T7.4 y restored", restored.get("y") == [1, 2, 3])
T.check("T7.5 z restored", restored.get("z") == "hello")

# ═══ §8 Reasoning Layer ═══
print("\n§8  Reasoning Layer")
print("-" * 40)

b = Belief()
T.check("R8.1 uniform prior", abs(b.mean - 0.5) < 0.01)
b = b.observe(True).observe(True).observe(True)
T.check("R8.2 belief up on success", b.mean > 0.7)

bs = BeliefState()
for _ in range(10):
    bs.observe("test:x", False)
T.check("R8.3 belief down on failure", bs.get("test:x").mean < 0.15)

cg = CausalGraph()
cg.add_action("build", [])
cg.add_action("test", [("build", "need build")])
cg.add_action("deploy", [("test", "need test")])
ok, unmet = cg.can_execute("deploy")
T.check("R8.4 deploy blocked", not ok)
cg.mark_completed("build")
cg.mark_completed("test")
ok, _ = cg.can_execute("deploy")
T.check("R8.5 deploy unblocked", ok)
T.check("R8.6 no cycles", not cg.has_cycle())

valid = {"a1", "a2", "a3"}
good = json.dumps({"chosen_action_id": "a1", "reasoning": "ok",
    "expected_outcome": "x", "risk_assessment": "l",
    "alternative_considered": "a2", "should_stop": False, "stop_reason": ""})
r = parse_llm_response(good, valid)
T.check("R8.7 valid parse", r.is_valid)
r2 = parse_llm_response(json.dumps({"chosen_action_id": "fake", "should_stop": False}), valid)
T.check("R8.8 hallucination caught", not r2.is_valid)
r3 = parse_llm_response("not json", valid)
T.check("R8.9 malformed caught", not r3.is_valid)
r4 = parse_llm_response(f"```json\n{good}\n```", valid)
T.check("R8.10 markdown unwrap", r4.is_valid)

# ═══ §9 Orchestrator Full Integration ═══
print("\n§9  Orchestrator Integration")
print("-" * 40)

task = TaskDefinition(
    goal="Build a website with 3 pages",
    initial_state=State({"pages": 0, "tested": False, "deployed": False}),
    available_actions=[
        ActionSpec(id="create_page", name="Create Page", description="Create HTML page",
                   effects=(Effect("pages", "increment", 1),), cost=2.0),
        ActionSpec(id="test", name="Run Tests", description="Test site",
                   effects=(Effect("tested", "set", True),), cost=3.0,
                   preconditions_text="pages >= 3"),
        ActionSpec(id="deploy", name="Deploy", description="Deploy to prod",
                   effects=(Effect("deployed", "set", True),), cost=5.0,
                   risk_level="medium", preconditions_text="tested"),
    ],
    invariants=[Invariant("max_pages", lambda s: s.get("pages", 0) <= 10)],
    budget=50.0,
    goal_predicate=lambda s: s.get("deployed", False) is True,
    goal_progress_fn=lambda s: min(1.0, (
        min(s.get("pages", 0), 3) / 3 * 0.6 +
        (0.2 if s.get("tested", False) else 0.0) +
        (0.2 if s.get("deployed", False) else 0.0))),
    dependencies={
        "create_page": [],
        "test": [("create_page", "Need pages")],
        "deploy": [("test", "Must test")],
    },
    min_action_cost=1.0,
)

engine = Orchestrator(task)
result = engine.run()
T.check("O9.1 goal achieved", result.goal_achieved)
T.check("O9.2 budget safe", result.total_cost <= 50.0)
T.check("O9.3 progress > 0", result.goal_progress > 0)
T.check("O9.4 trace valid", engine.kernel.trace.verify_integrity()[0])
print(f"       Cost: ${result.total_cost:.2f}, Steps: {result.total_steps}")

# ═══ §10 Domain Tests ═══
print("\n§10  Domain Tests")
print("-" * 40)

ds_task = TaskDefinition(
    goal="Train ML model",
    initial_state=State({"data_loaded": False, "cleaned": False,
                         "trained": False, "evaluated": False, "accuracy": 0.0}),
    available_actions=[
        ActionSpec(id="load", name="Load", description="Load data",
                   effects=(Effect("data_loaded", "set", True),), cost=1.0),
        ActionSpec(id="clean", name="Clean", description="Clean data",
                   effects=(Effect("cleaned", "set", True),), cost=2.0),
        ActionSpec(id="train", name="Train", description="Train model",
                   effects=(Effect("trained", "set", True), Effect("accuracy", "set", 0.92)),
                   cost=10.0, risk_level="medium"),
        ActionSpec(id="eval", name="Evaluate", description="Evaluate model",
                   effects=(Effect("evaluated", "set", True),), cost=3.0),
    ],
    invariants=[Invariant("acc", lambda s: s.get("accuracy", 0) <= 1.0)],
    budget=50.0,
    goal_predicate=lambda s: s.get("evaluated", False),
    dependencies={"load": [], "clean": [("load", "need data")],
                  "train": [("clean", "need clean data")],
                  "eval": [("train", "need model")]},
)
ds_r = Orchestrator(ds_task).run()
T.check("D10.1 DS pipeline", ds_r.goal_achieved)
T.check("D10.2 DS cost", ds_r.total_cost <= 20.0)

devops_task = TaskDefinition(
    goal="Deploy service",
    initial_state=State({"vpc": False, "db": False, "app": False, "healthy": False}),
    available_actions=[
        ActionSpec(id="vpc", name="VPC", description="Create VPC",
                   effects=(Effect("vpc", "set", True),), cost=5.0),
        ActionSpec(id="db", name="DB", description="Create DB",
                   effects=(Effect("db", "set", True),), cost=8.0),
        ActionSpec(id="app", name="App", description="Deploy app",
                   effects=(Effect("app", "set", True),), cost=5.0, risk_level="high"),
        ActionSpec(id="health", name="Health", description="Check health",
                   effects=(Effect("healthy", "set", True),), cost=1.0),
    ],
    invariants=[],
    budget=50.0,
    goal_predicate=lambda s: s.get("healthy", False),
    dependencies={"vpc": [], "db": [("vpc", "need VPC")],
                  "app": [("vpc", "need VPC"), ("db", "need DB")],
                  "health": [("app", "need app")]},
)
devops_r = Orchestrator(devops_task).run()
T.check("D10.3 DevOps done", devops_r.goal_achieved)
T.check("D10.4 DevOps cost", devops_r.total_cost == 19.0)

# ═══ §11 Adversarial ═══
print("\n§11  Adversarial Tests")
print("-" * 40)

class GreedyLLM:
    def complete(self, prompt, **kw):
        for line in prompt.split('\n'):
            if "(id=" in line and "[READY]" in line:
                aid = line.split("(id=")[1].split(")")[0]
                return json.dumps({"chosen_action_id": aid, "reasoning": "SPEND",
                    "expected_outcome": "", "risk_assessment": "",
                    "alternative_considered": "", "should_stop": False, "stop_reason": ""})
        return json.dumps({"should_stop": True, "stop_reason": "nothing",
            "chosen_action_id": "", "reasoning": "", "expected_outcome": "",
            "risk_assessment": "", "alternative_considered": ""})

greedy_task = TaskDefinition(
    goal="Impossible",
    initial_state=State({"x": 0}),
    available_actions=[ActionSpec(id=f"w{i}", name=f"Waste {i}", description="",
        effects=(Effect("x", "increment", 1),), cost=3.0) for i in range(100)],
    invariants=[Invariant("bound", lambda s: s.get("x", 0) <= 50)],
    budget=20.0,
    goal_predicate=lambda s: s.get("x", 0) >= 999,
    min_action_cost=1.0,
)
gr = Orchestrator(greedy_task, llm=GreedyLLM()).run()
T.check("A11.1 budget safe vs greedy", gr.total_cost <= 20.0)
T.check("A11.2 invariant vs greedy", gr.final_state.get("x", 0) <= 50)

class GarbageLLM:
    def complete(self, *a, **kw):
        return "CHAOS 🔥🔥🔥 undefined is not a function"

gt = TaskDefinition(
    goal="Build", initial_state=State({"built": False}),
    available_actions=[ActionSpec(id="build", name="Build", description="",
        effects=(Effect("built", "set", True),), cost=1.0)],
    invariants=[], budget=10.0,
    goal_predicate=lambda s: s.get("built", False),
    max_consecutive_failures=3,
)
gr2 = Orchestrator(gt, llm=GarbageLLM()).run()
T.check("A11.3 survives garbage LLM", True)
T.check("A11.4 budget safe vs garbage", gr2.total_cost <= 10.0)

class HalluciLLM:
    def complete(self, *a, **kw):
        return json.dumps({"chosen_action_id": "FAKE_42", "reasoning": "",
            "expected_outcome": "", "risk_assessment": "",
            "alternative_considered": "", "should_stop": False, "stop_reason": ""})

gr3 = Orchestrator(gt, llm=HalluciLLM()).run()
T.check("A11.5 handles hallucinations", True)
T.check("A11.6 budget safe vs hallucinator", gr3.total_cost <= 10.0)

# ═══ §12 State Immutability ═══
print("\n§12  State Immutability")
print("-" * 40)

s = State({"list": [1, 2, 3], "dict": {"a": 1}})
ext = {"list": [1, 2, 3]}
s2 = State(ext)
ext["list"].append(4)
T.check("S12.1 external mutation blocked", s2.get("list") == [1, 2, 3])
lst = s.get("list")
lst.append(99)
T.check("S12.2 get returns copy", s.get("list") == [1, 2, 3])
try:
    s._vars = {}
    T.check("S12.3 setattr blocked", False)
except AttributeError:
    T.check("S12.3 setattr blocked", True)

# ═══ §13 Monte Carlo ═══
print("\n§13  Monte Carlo Validation (1000 runs)")
print("-" * 40)

def make_random_task(seed):
    rng = random.Random(seed)
    n_steps = rng.randint(3, 8)
    budget = n_steps * 5.0 + rng.uniform(5, 20)
    actions = [
        ActionSpec(id=f"s{i}", name=f"Step {i}", description=f"Step {i}",
                   effects=(Effect("prog", "increment", 1),),
                   cost=rng.uniform(1.0, 4.0))
        for i in range(n_steps)
    ]
    deps = {f"s{i}": ([(f"s{i-1}", "seq")] if i > 0 else []) for i in range(n_steps)}
    return TaskDefinition(
        goal=f"Do {n_steps} steps", initial_state=State({"prog": 0}),
        available_actions=actions,
        invariants=[Invariant("bnd", lambda s, m=n_steps+5: s.get("prog", 0) <= m)],
        budget=budget,
        goal_predicate=lambda s, n=n_steps: s.get("prog", 0) >= n,
        goal_progress_fn=lambda s, n=n_steps: min(1.0, s.get("prog", 0) / n),
        dependencies=deps, min_action_cost=0.5, max_consecutive_failures=10,
    )

N = 1000
mc = {"budget": 0, "inv": 0, "term": 0, "goal": 0, "trace": 0, "costs": [], "steps": []}
t0 = time.time()
for i in range(N):
    t = make_random_task(i)
    r = Orchestrator(t).run()
    if r.total_cost <= t.budget: mc["budget"] += 1
    if all(inv.check(r.final_state)[0] for inv in t.invariants): mc["inv"] += 1
    mc["term"] += 1
    e = Orchestrator(t)  # need fresh engine for trace check
    e_r = e.run()
    if e.kernel.trace.verify_integrity()[0]: mc["trace"] += 1
    if r.goal_achieved: mc["goal"] += 1
    mc["costs"].append(r.total_cost)
    mc["steps"].append(r.total_steps)

elapsed = time.time() - t0
T.check(f"MC13.1 Budget safe {mc['budget']}/{N}", mc["budget"] == N)
T.check(f"MC13.2 Invariants safe {mc['inv']}/{N}", mc["inv"] == N)
T.check(f"MC13.3 Terminated {mc['term']}/{N}", mc["term"] == N)
T.check(f"MC13.4 Trace valid {mc['trace']}/{N}", mc["trace"] == N)

gr = mc["goal"] / N
T.check(f"MC13.5 Goal rate {gr:.1%} (>80%)", gr > 0.80)
print(f"       Mean cost: ${sum(mc['costs'])/N:.2f}")
print(f"       Mean steps: {sum(mc['steps'])/N:.1f}")
print(f"       Time: {elapsed:.2f}s ({elapsed/N*1000:.2f}ms/trial)")

# CI
z = 1.96
d = 1 + z**2/N
c = (gr + z**2/(2*N)) / d
m = z * math.sqrt((gr*(1-gr) + z**2/(4*N))/N) / d
print(f"       Goal rate 95% CI: [{c-m:.3f}, {c+m:.3f}]")

def _run_suite() -> bool:
    """Run the historical script-style test suite and return pass/fail.

    This file originally functioned as a standalone script.
    Under pytest, the module is imported, so we must not call sys.exit()
    at import time.
    """
    return T.summary()


def test_script_suite_passes():
    assert _run_suite(), "Script-style suite reported failures; see captured output above."


if __name__ == "__main__":
    all_ok = _run_suite()
    sys.exit(0 if all_ok else 1)
