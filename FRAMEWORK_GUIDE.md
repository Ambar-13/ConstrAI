# ConstrAI Framework Guide

## What ConstrAI Does

ConstrAI is a safety framework for AI agents. It ensures that agents can't break rules, exceed budgets, or cause harm - no matter what the LLM tries to do.

**The key insight**: Safety happens at the execution level, not the prompt level. We let the LLM think freely, but we verify every action before it runs.

---

## The Three Safety Layers

### Layer 1: Formal Safety Kernel

The foundation that everything else builds on.

#### State (Immutable)

States are snapshots of the world. You can't modify them directly - any change creates a new state.

```python
from constrai import State

# Create
state = State({"balance": 100, "deployed": False})

# Read
balance = state.get("balance")  # 100

# "Modify" (creates new state)
new_state = state.with_updates({"balance": 95})
# Original state is still {"balance": 100}
```

**Why this matters**: If state is immutable, undo is trivial. We just go back to the previous state.

#### Effects (Declarative)

Actions don't contain code - they describe what changes:

```python
from constrai import Effect

Effect("balance", "decrement", 50)      # balance -= 50
Effect("status", "set", "deployed")     # status = "deployed"
Effect("log", "append", "done")         # log.append("done")
Effect("counter", "increment", 1)       # counter += 1
```

**Why this matters**: We can verify effects without running code. No surprises, no side effects.

#### Invariants (Always True)

Safety rules that must hold after every action:

```python
from constrai import Invariant

Invariant(
    name="budget_safe",
    predicate=lambda s: s.get("balance", 0) >= 0,
    description="Balance must never go negative"
)
```

**How it works**: Before accepting an action, we simulate it on a copy of the state. If the invariant fails, we reject the action. State is never modified.

### Layer 2: Reference Monitor

Enforces constraints across multiple dimensions.

#### Information Flow Control (IFC)

Prevents data from flowing to places it shouldn't:

```python
from constrai import DataLabel, SecurityLevel

# Mark sensitive data
secret_label = DataLabel(SecurityLevel.SECRET)
public_label = DataLabel(SecurityLevel.PUBLIC)

# Monitor ensures:
# - Secret data can flow to Secret variables ✓
# - Secret data cannot flow to Public variables ✗
```

#### Resource Barriers

Prevents overconsumption via Control Barrier Functions:

```python
from constrai import ControlBarrierFunction

def budget_h(state):
    """Barrier function: how much budget margin do we have?"""
    remaining = state.get("budget", 0)
    return remaining / 100.0  # Normalized to [0,1]

barrier = ControlBarrierFunction(h=budget_h, alpha=0.1)
# alpha=0.1 means: budget can decay 10% per step, but no faster
```

**What it does**: Prevents actions that would erode the resource boundary too fast.

#### Capture Basins (Forbidden Regions)

Defines areas of state space to avoid:

```python
from constrai import CaptureBasin

CaptureBasin(
    name="bankruptcy",
    is_bad=lambda s: s.get("balance", 0) < 0,
    max_steps=5  # Will reach bad region within 5 steps?
)
```

**Prevention**: System actively rejects actions that would lead into these regions.

### Layer 3: Execution Flow

The orchestrator ties it all together:

```
1. Agent proposes action
   ↓
2. Boundary check
   "Is any variable getting close to a limit?"
   ↓
3. Barrier enforcement
   "Would this enter a forbidden region?"
   ↓
4. Reference monitor
   "Information flow? Resource limits? Can we repair it?"
   ↓
5. Simulate on copy
   "Does invariant hold after this action?"
   ↓
6. Execute (if all pass)
   "Actually apply the effects"
   ↓
7. Record for undo
   "Save what we did, in case we need to reverse it"
```

If any check fails → action is rejected, state unchanged.

---

## Current Features (v0.3.0)

### Boundary Detection

Detects when variables are approaching constraint violations:

```python
from constrai import JacobianFusion

jacobian = JacobianFusion()
report = jacobian.compute_gradients(state)

# Check which variables are near limits
for variable, severity in report.severity_scores.items():
    if severity > 0.7:  # Getting close
        print(f"Warning: {variable} is near its limit")
```

**Severity levels**:
- 0.0-0.3: Safe, plenty of margin
- 0.3-0.6: Caution, monitor this variable
- 0.6-0.9: Warning, very close to limit
- 0.9-1.0: Critical, one small step from violation

### Enforcement Barriers

Actively prevents dangerous actions:

```python
from constrai import AuthoritativeHJBBarrier

barrier = AuthoritativeHJBBarrier()

# Check if action would be dangerous
check = barrier.check_action_leads_to_danger(
    state=current_state,
    action=proposed_action,
    available_actions=all_actions
)

if not check.safe:
    # Action is rejected, state unchanged
    print(f"Action rejected: {check.reason}")
```

### Task Composition

Combine verified subtasks safely:

```python
from constrai import SuperTask, TaskComposer

# Create verified subtasks
prepare = SuperTask(
    id="prepare",
    task_definition=prep_task,
    interface=InterfaceSignature(
        required_inputs=("config",),
        produced_outputs=("ready",)
    ),
    verified=True
)

deploy = SuperTask(
    id="deploy",
    task_definition=deploy_task,
    interface=InterfaceSignature(
        required_inputs=("ready",),
        produced_outputs=("deployed",)
    ),
    verified=True
)

# Compose them - automatically verified!
combined = prepare.compose(deploy)

if combined:
    result = orchestrator.execute_task(combined)
```

---

## The 7 Theorems

ConstrAI proves these theorems by construction:

| Theorem | What It Guarantees | How |
|---------|-------------------|-----|
| T1: Budget Safety | You cannot spend more than your budget | Check before commit |
| T2: Termination | Execution will eventually stop | Budget runs out → no more actions |
| T3: Invariant Preservation | Safety rules always hold | Simulate before commit |
| T4: Monotone Resources | Spending only increases | Non-negative cost assertion |
| T5: Atomicity | Rejected actions don't change anything | Simulate on copy only |
| T6: Trace Integrity | Execution log can't be tampered with | Cryptographic hash chain |
| T7: Rollback Exactness | Undo restores exact prior state | Immutable state + inverse effects |

See `docs/THEOREMS.md` for detailed proofs.

---

## Working Example

```python
from constrai import (
    State, Effect, ActionSpec, Invariant, CaptureBasin,
    TaskDefinition, Orchestrator
)

# 1. Define actions
spend_action = ActionSpec(
    id="spend",
    name="Spend money",
    effects=(Effect("balance", "decrement", 50),),
    cost=1.0
)

invest_action = ActionSpec(
    id="invest",
    name="Invest money",
    effects=(Effect("invested", "increment", 100),),
    cost=2.0
)

# 2. Define safety rules
safety_rule = Invariant(
    "no_negative_balance",
    lambda s: s.get("balance", 0) >= 0,
    "Balance must never go negative"
)

danger_zone = CaptureBasin(
    "bankruptcy",
    is_bad=lambda s: s.get("balance", 0) < 0,
    max_steps=2
)

# 3. Create task
task = TaskDefinition(
    goal="Invest exactly 100 units",
    initial_state=State({"balance": 150, "invested": 0}),
    available_actions=[spend_action, invest_action],
    invariants=[safety_rule],
    capture_basins=[danger_zone],
    budget=10.0,  # Can't spend more than 10 tokens
    goal_predicate=lambda s: s.get("invested", 0) == 100
)

# 4. Run with LLM
orch = Orchestrator(task, llm_client=your_llm)
result = orch.execute_task(max_steps=30)

print(f"Goal achieved: {result.goal_achieved}")
print(f"Steps: {result.step_count}")
print(f"Budget used: {result.spent}")

# Safety guarantees:
# ✓ Balance never went negative (T3: Invariant Preservation)
# ✓ Never entered bankruptcy zone (Barrier enforcement)
# ✓ Never exceeded 10-token budget (T1: Budget Safety)
# ✓ Can undo any action that happened (T7: Rollback Exactness)
```

---

## API Quick Reference

### State Management

```python
from constrai import State

s = State({"key": value})
s.get("key")                          # Read
s.with_updates({"key": new_value})    # Update (immutable)
s.without_keys(["key"])               # Remove
```

### Actions

```python
from constrai import Effect, ActionSpec

effect = Effect(variable, mode, value)
# Modes: set, increment, decrement, multiply, append, remove, delete

action = ActionSpec(
    id="action_id",
    name="Human name",
    description="What does it do?",
    effects=(effect1, effect2),
    cost=1.0,
    risk_level="low"  # low, medium, high, critical
)

# Simulate without executing
next_state = action.simulate(current_state)
```

### Safety Rules

```python
from constrai import Invariant, CaptureBasin, ControlBarrierFunction

# Rule that must always hold
invariant = Invariant("name", predicate, "description")

# Region to avoid
basin = CaptureBasin("name", is_bad, max_steps=5)

# Smooth constraint boundary
barrier = ControlBarrierFunction(h=function, alpha=0.1)
```

### Execution

```python
from constrai import TaskDefinition, Orchestrator

task = TaskDefinition(
    goal="What should happen",
    initial_state=State({...}),
    available_actions=[...],
    invariants=[...],
    capture_basins=[...],
    budget=100.0,
    goal_predicate=lambda s: condition,
    dependencies={...}
)

orch = Orchestrator(task, llm_client)
result = orch.execute_task(max_steps=50)
```

---

## Key Properties

1. **Non-bypassable** — LLM cannot talk its way past the safety kernel
2. **Immutable state** — Every change creates a new snapshot
3. **Simulate before commit** — Nothing real changes until verification passes
4. **Reversible** — Every action can be undone via T7 rollback
5. **Formal guarantees** — All 7 theorems proven by construction
6. **No token overhead** — Safety happens at execution layer, not in prompts

---

## When to Use ConstrAI

✓ When you need hard safety guarantees  
✓ When actions cost money or have side effects  
✓ When you need formal verification  
✓ When you can't afford mistakes

✗ When you need extreme performance  
✗ When you have no constraints  
✗ When safety requirements are vague

---

## Next Steps

1. Read `README.md` for overview
2. Check `docs/THEOREMS.md` for formal proofs
3. Review `docs/API.md` for complete API
4. Look at `examples/` for working code
5. See `docs/VULNERABILITIES.md` for limitations

All code is tested: `python -m pytest tests/ -v` (44/44 passing)
