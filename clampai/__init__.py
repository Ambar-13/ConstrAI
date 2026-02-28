"""
ClampAI — Formal safety framework for autonomous AI agents.

The safety kernel enforces eight provable guarantees (T1–T8) on every action
before it reaches the real world. These include budget safety, bounded
termination, invariant preservation, atomicity, and tamper-evident tracing.
All guarantees are derived from first principles in THEOREMS.md.

Quick start:
    from clampai import TaskDefinition, State, ActionSpec, Effect, Invariant, Orchestrator

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

# Safety kernel (T1–T8 theorems)
from .active_hjb_barrier import (
    ActiveHJBBarrier,
    HJBBarrierCheck,
    RecoveryStrategy,
    SafetyBarrierViolation,
    choose_recovery_strategy,
)

# Simple API — @safe for zero-config safety
from .api import SafetyViolation, _SafeWrapper, clampai_safe, safe
from .formal import (
    FORMAL_CLAIMS,
    ActionSpec,
    AsyncSafetyKernel,
    BudgetController,
    CheckResult,
    Claim,
    Effect,
    ExecutionTrace,
    GuaranteeLevel,
    Invariant,
    MetricsBackend,
    NoOpMetrics,
    ProcessSharedBudgetController,
    RejectionFormatter,
    SafetyKernel,
    SafetyVerdict,
    State,
    TraceEntry,
)
from .gradient_tracker import (
    GradientReport,
    GradientScore,
    GradientTracker,
    PerInvariantBudget,
    SensitivityLevel,
)

# Hardening layer
from .hardening import (
    HARDENING_CLAIMS,
    Attestation,
    AttestationGate,
    AttestationResult,
    Attestor,
    CostAwarePriorFactory,
    DependencyDiscovery,
    EnvironmentDriftError,
    EnvironmentProbe,
    EnvironmentReconciler,
    FailurePattern,
    MultiDimensionalAttestor,
    Permission,
    PredicateAttestor,
    QualityDimension,
    QualityScore,
    ReadinessProbe,
    ReconciliationResult,
    ResourceDescriptor,
    ResourceState,
    ResourceTracker,
    SubprocessAttestor,
    TemporalCausalGraph,
    TemporalDependency,
)

# Pre-built invariant library
from .invariants import (
    allowed_values_invariant,
    api_call_limit_invariant,
    custom_invariant,
    email_safety_invariant,
    file_operation_limit_invariant,
    human_approval_gate_invariant,
    json_schema_invariant,
    list_length_invariant,
    max_retries_invariant,
    monotone_decreasing_invariant,
    monotone_increasing_invariant,
    no_action_after_flag_invariant,
    no_delete_invariant,
    no_duplicate_ids_invariant,
    no_regex_match_invariant,
    no_sensitive_substring_invariant,
    non_empty_invariant,
    pii_guard_invariant,
    rate_limit_invariant,
    read_only_keys_invariant,
    required_fields_invariant,
    resource_ceiling_invariant,
    string_length_invariant,
    time_window_rate_invariant,
    value_range_invariant,
)

# Inverse algebra (T7 exact rollback), gradient analysis, active HJB barrier
from .inverse_algebra import (
    InverseAlgebra,
    InverseEffect,
    RollbackRecord,
    action_with_inverse_guarantee,
)

# Jacobian fusion, safe hover enforcement, operadic composition
from .jacobian_fusion import (
    BoundarySeverity,
    JacobianFusion,
    JacobianReport,
    JacobianScore,
)
from .operadic_composition import (
    CompositionType,
    InterfaceSignature,
    SuperTask,
    TaskComposer,
    VerificationCertificate,
)

# Orchestrator
from .orchestrator import (
    AsyncOrchestrator,
    ClampAI_SYSTEM_PROMPT,
    ExecutionResult,
    Orchestrator,
    Outcome,
    OutcomeType,
    ProgressMonitor,
    TaskDefinition,
    TerminationReason,
)

# Reasoning layer
from .reasoning import (
    REASONING_CLAIMS,
    ActionValue,
    ActionValueComputer,
    Belief,
    BeliefState,
    CausalGraph,
    Dependency,
    LLMAdapter,
    MockLLMAdapter,
    ReasoningRequest,
    ReasoningResponse,
    parse_llm_response,
)

# Reference monitor (IFC, CBF, QP projection)
from .reference_monitor import (
    CaptureBasin,
    ContractSpecification,
    ControlBarrierFunction,
    DataLabel,
    LabelledData,
    OperadicComposition,
    QPProjector,
    ReferenceMonitor,
    SafeHoverState,
    SecurityLevel,
)
from .safe_hover import (
    AuthoritativeHJBBarrier,
    HJBEnforcementCheck,
    SafeHoverSignal,
)

# Saliency and prompt optimization
from .saliency import (
    SaliencyEngine,
    SaliencyResult,
)

# Test utilities
from .testing import SafetyHarness, make_action, make_state

__version__ = "1.0.1"
__all__ = [
    "FORMAL_CLAIMS",
    "HARDENING_CLAIMS",
    "REASONING_CLAIMS",
    # Safety kernel — Layer 0
    "ActionSpec",
    # Reasoning engine — Layer 1
    "ActionValue",
    "ActionValueComputer",
    # Active HJB barrier
    "ActiveHJBBarrier",
    # Native-async kernel and orchestrator
    "AsyncOrchestrator",
    "AsyncSafetyKernel",
    # Hardening — Layer 3
    "Attestation",
    "AttestationGate",
    "AttestationResult",
    "Attestor",
    # Safe hover enforcement
    "AuthoritativeHJBBarrier",
    "Belief",
    "BeliefState",
    # Jacobian fusion
    "BoundarySeverity",
    "BudgetController",
    # Reference monitor
    "CaptureBasin",
    "CausalGraph",
    "CheckResult",
    "Claim",
    # Orchestrator — Layer 2
    "ClampAI_SYSTEM_PROMPT",
    # Operadic composition
    "CompositionType",
    "ContractSpecification",
    "ControlBarrierFunction",
    "CostAwarePriorFactory",
    "DataLabel",
    "Dependency",
    "DependencyDiscovery",
    "Effect",
    "EnvironmentDriftError",
    "EnvironmentProbe",
    "EnvironmentReconciler",
    "ExecutionResult",
    "ExecutionTrace",
    "FailurePattern",
    # Gradient tracker
    "GradientReport",
    "GradientScore",
    "GradientTracker",
    "GuaranteeLevel",
    "HJBBarrierCheck",
    "HJBEnforcementCheck",
    "InterfaceSignature",
    "Invariant",
    # Inverse algebra (T7 exact rollback)
    "InverseAlgebra",
    "InverseEffect",
    "JacobianFusion",
    "JacobianReport",
    "JacobianScore",
    "LLMAdapter",
    "LabelledData",
    "MetricsBackend",
    "MockLLMAdapter",
    "MultiDimensionalAttestor",
    "NoOpMetrics",
    "OperadicComposition",
    "Orchestrator",
    "Outcome",
    "OutcomeType",
    "PerInvariantBudget",
    "Permission",
    "PredicateAttestor",
    # Distributed multi-agent budget
    "ProcessSharedBudgetController",
    "ProgressMonitor",
    "QPProjector",
    "QualityDimension",
    "QualityScore",
    "ReadinessProbe",
    "ReasoningRequest",
    "ReasoningResponse",
    "ReconciliationResult",
    "RecoveryStrategy",
    "ReferenceMonitor",
    "RejectionFormatter",
    "ResourceDescriptor",
    "ResourceState",
    "ResourceTracker",
    "RollbackRecord",
    "SafeHoverSignal",
    "SafeHoverState",
    "SafetyBarrierViolation",
    "SafetyHarness",
    "SafetyKernel",
    "SafetyVerdict",
    # Simple API
    "SafetyViolation",
    # Saliency engine
    "SaliencyEngine",
    "SaliencyResult",
    "SecurityLevel",
    "SensitivityLevel",
    "State",
    "SubprocessAttestor",
    "SuperTask",
    "TaskComposer",
    "TaskDefinition",
    "TemporalCausalGraph",
    "TemporalDependency",
    "TerminationReason",
    "TraceEntry",
    "VerificationCertificate",
    "action_with_inverse_guarantee",
    # Pre-built invariant library
    "allowed_values_invariant",
    "api_call_limit_invariant",
    "choose_recovery_strategy",
    "clampai_safe",
    "custom_invariant",
    "email_safety_invariant",
    "file_operation_limit_invariant",
    "human_approval_gate_invariant",
    "json_schema_invariant",
    "list_length_invariant",
    # Test utilities
    "make_action",
    "make_state",
    "max_retries_invariant",
    "monotone_decreasing_invariant",
    "monotone_increasing_invariant",
    "no_action_after_flag_invariant",
    "no_delete_invariant",
    "no_duplicate_ids_invariant",
    "no_regex_match_invariant",
    "no_sensitive_substring_invariant",
    "non_empty_invariant",
    "parse_llm_response",
    "pii_guard_invariant",
    "rate_limit_invariant",
    "read_only_keys_invariant",
    "required_fields_invariant",
    "resource_ceiling_invariant",
    "safe",
    "string_length_invariant",
    "time_window_rate_invariant",
    "value_range_invariant",
]
