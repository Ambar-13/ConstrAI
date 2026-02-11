"""
ConstrAI — Autonomous Engine for Guaranteed Intelligent Safety
============================================================

A framework for safe autonomous AI agent execution with:
  - Mathematically proven safety guarantees (budget, termination, invariants)
  - Structured LLM reasoning (not free-form guessing)  
  - Bayesian belief tracking
  - Information-theoretic action valuation
  - Hash-chained audit trails

Quick Start:
    from constrai import TaskDefinition, State, ActionSpec, Effect, Invariant, Orchestrator

    task = TaskDefinition(
        goal="Build a website",
        initial_state=State({"files_created": 0}),
        available_actions=[
            ActionSpec(id="create_html", name="Create HTML", 
                      description="Create index.html",
                      effects=(Effect("files_created", "increment", 1),),
                      cost=2.0),
        ],
        invariants=[
            Invariant("max_files", lambda s: (s.get("files_created", 0)) <= 100,
                     "No more than 100 files"),
        ],
        budget=50.0,
        goal_predicate=lambda s: s.get("files_created", 0) >= 5,
    )
    
    engine = Orchestrator(task)
    result = engine.run()
    print(result.summary())
"""

# ── Formal Layer (proven) ──
from .formal import (
    State,
    Effect,
    ActionSpec,
    Invariant,
    SafetyKernel,
    SafetyVerdict,
    CheckResult,
    BudgetController,
    ExecutionTrace,
    TraceEntry,
    GuaranteeLevel,
    Claim,
    FORMAL_CLAIMS,
)

# ── Reasoning Layer ──
from .reasoning import (
    Belief,
    BeliefState,
    CausalGraph,
    Dependency,
    ActionValue,
    ActionValueComputer,
    ReasoningRequest,
    ReasoningResponse,
    parse_llm_response,
    LLMAdapter,
    MockLLMAdapter,
    REASONING_CLAIMS,
)

# ── Saliency & Prompt Optimization ──
from .saliency import (
    SaliencyEngine,
    SaliencyResult,
)

# ── Reference Monitor (Formal Enforcement) ──
from .reference_monitor import (
    SecurityLevel,
    DataLabel,
    LabelledData,
    ControlBarrierFunction,
    QPProjector,
    CaptureBasin,
    ContractSpecification,
    OperadicComposition,
    ReferenceMonitor,
    SafeHoverState,
)

# ── Advanced Math (T7, Gradients, Active HJB) ──
from .inverse_algebra import (
    InverseEffect,
    RollbackRecord,
    InverseAlgebra,
    action_with_inverse_guarantee,
)

from .gradient_tracker import (
    SensitivityLevel,
    GradientScore,
    GradientReport,
    GradientTracker,
    PerInvariantBudget,
)

from .active_hjb_barrier import (
    SafetyBarrierViolation,
    HJBBarrierCheck,
    ActiveHJBBarrier,
    RecoveryStrategy,
    choose_recovery_strategy,
)

# ── Orchestrator ──
from .orchestrator import (
    TaskDefinition,
    Orchestrator,
    ExecutionResult,
    TerminationReason,
    OutcomeType,
    Outcome,
    ProgressMonitor,
    ConstrAI_SYSTEM_PROMPT,
)

# ── Hardening Layer ──
from .hardening import (
    AttestationGate,
    AttestationResult,
    Attestation,
    Attestor,
    SubprocessAttestor,
    PredicateAttestor,
    DependencyDiscovery,
    FailurePattern,
    ResourceTracker,
    ResourceDescriptor,
    ResourceState,
    Permission,
    # v2 additions
    ReadinessProbe,
    TemporalDependency,
    TemporalCausalGraph,
    CostAwarePriorFactory,
    EnvironmentProbe,
    EnvironmentReconciler,
    EnvironmentDriftError,
    ReconciliationResult,
    MultiDimensionalAttestor,
    QualityDimension,
    QualityScore,
    HARDENING_CLAIMS,
)

# ── Advanced Math: Absolute Rigor ──
from .jacobian_fusion import (
    JacobianFusion,
    JacobianScore,
    JacobianReport,
    BoundarySeverity,
)

from .safe_hover import (
    AuthoritativeHJBBarrier,
    SafeHoverSignal,
    HJBEnforcementCheck,
)

from .operadic_composition import (
    SuperTask,
    TaskComposer,
    InterfaceSignature,
    VerificationCertificate,
    CompositionType,
)

__version__ = "0.3.0"
__all__ = [
    # Formal
    "State", "Effect", "ActionSpec", "Invariant",
    "SafetyKernel", "SafetyVerdict", "CheckResult",
    "BudgetController", "ExecutionTrace", "TraceEntry",
    "GuaranteeLevel", "Claim", "FORMAL_CLAIMS",
    # Reasoning
    "Belief", "BeliefState", "CausalGraph", "Dependency",
    "ActionValue", "ActionValueComputer",
    "ReasoningRequest", "ReasoningResponse", "parse_llm_response",
    "LLMAdapter", "MockLLMAdapter", "REASONING_CLAIMS",
    # Saliency
    "SaliencyEngine", "SaliencyResult",
    # Reference Monitor
    "SecurityLevel", "DataLabel", "LabelledData",
    "ControlBarrierFunction", "QPProjector",
    "CaptureBasin", "ContractSpecification", "OperadicComposition",
    "ReferenceMonitor", "SafeHoverState",
    # Orchestrator
    "TaskDefinition", "Orchestrator", "ExecutionResult",
    "TerminationReason", "OutcomeType", "Outcome",
    "ProgressMonitor", "ConstrAI_SYSTEM_PROMPT",
    # Advanced Math: Absolute Rigor
    "JacobianFusion", "JacobianScore", "JacobianReport", "BoundarySeverity",
    "AuthoritativeHJBBarrier", "SafeHoverSignal", "HJBEnforcementCheck",
    "SuperTask", "TaskComposer", "InterfaceSignature", "VerificationCertificate", "CompositionType",
]
