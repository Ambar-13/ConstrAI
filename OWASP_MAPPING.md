# ConstrAI — OWASP LLM Top 10 (2025) Compliance Mapping

This document maps each risk in the **OWASP Top 10 for Large Language Model
Applications (2025)** to the specific ConstrAI theorems and components that
mitigate it, together with the residual risk and recommended configuration.

The ConstrAI eight theorems are: **T1** (Budget Safety), **T2** (Bounded
Termination), **T3** (Invariant Preservation), **T4** (Monotone Resource Consumption),
**T5** (Atomic Execution), **T6** (Tamper-Evident Trace), **T7** (Exact
Rollback), **T8** (Emergency Halt).

---

## LLM01: Prompt Injection

**Risk**: Malicious content in user input or retrieved documents tricks the
LLM into taking unintended actions (e.g., exfiltrating data, bypassing
authorization, executing arbitrary commands).

**ConstrAI mitigation**:

| Component | How it helps |
|---|---|
| **T3 Invariants** | Block any action whose *projected next state* would violate a predicate — regardless of what the LLM claimed as justification. The kernel has no language model; it cannot be persuaded by text. |
| **AttestationGate** (`constrai/hardening.py`) | Requires cryptographic attestation from designated attestors before high-risk actions execute. An injected instruction cannot produce a valid attestation. |
| **ReferenceMonitor** (`constrai/reference_monitor.py`) | IFC (Information Flow Control) labels prevent data exfiltration paths from being approved even when the LLM believes the action is safe. |
| **T6 Trace** | Every approved action is logged with a hash-chained fingerprint; forensic analysis can identify the injection point after the fact. |

**Recommended invariants**:
```python
from constrai import no_sensitive_substring_invariant, human_approval_gate_invariant

kernel = SafetyKernel(
    budget=100.0,
    invariants=[
        # Block actions containing credential patterns
        no_sensitive_substring_invariant("output_text", ["password", "api_key", "Bearer "]),
        # Require human approval before any network egress
        human_approval_gate_invariant("network_egress_approved"),
    ],
)
```

**Residual risk**: ConstrAI cannot prevent the LLM from *reasoning* about
injected content. It only blocks the resulting *action* if it violates a
declared invariant. Invariants must cover all sensitive action classes.

---

## LLM02: Sensitive Information Disclosure

**Risk**: The LLM inadvertently reveals PII, credentials, proprietary data,
or system internals in its responses or tool outputs.

**ConstrAI mitigation**:

| Component | How it helps |
|---|---|
| **IFC Labels** (`SecurityLevel`, `DataLabel`) | Data items carry secrecy labels (PUBLIC, CONFIDENTIAL, SECRET, TOP_SECRET). The ReferenceMonitor enforces a no-read-up / no-write-down lattice; high-label data cannot flow to low-label outputs. |
| **T3 Invariants** | `no_sensitive_substring_invariant` blocks actions whose output fields contain forbidden patterns (SSNs, card numbers, key material). |
| **SaliencyEngine** | Identifies which state variables most influenced a proposed action, enabling post-hoc audits of potential disclosure paths. |

**Recommended configuration**:
```python
from constrai import no_sensitive_substring_invariant
from constrai.reference_monitor import SecurityLevel, DataLabel, LabelledData

# Label sensitive data at ingestion time
pii_record = LabelledData(
    data={"name": "Alice", "ssn": "123-45-6789"},
    label=DataLabel(level=SecurityLevel.CONFIDENTIAL),
)

# Invariant: output must not contain SSN-pattern text
kernel = SafetyKernel(
    budget=100.0,
    invariants=[
        no_sensitive_substring_invariant("agent_response", [r"\d{3}-\d{2}-\d{4}"]),
    ],
)
```

**Residual risk**: Pattern-based matching has false negatives (encoded,
paraphrased, or translated content). Combine with output-layer classifiers
for defense in depth.

---

## LLM03: Supply Chain Vulnerabilities

**Risk**: Poisoned models, malicious plugins, or compromised training data
introduce backdoors or unsafe behaviors.

**ConstrAI mitigation**:

| Component | How it helps |
|---|---|
| **AttestationGate** | Cryptographically verifies the integrity of each action before execution. An adversarially fine-tuned model that proposes unsafe actions still cannot execute them if invariants are violated. |
| **T3 Invariants** | The safety predicate layer is independent of the model weights. Even a fully compromised model cannot override a blocking invariant. |
| **T6 Hash-chained Trace** | Provides a tamper-evident audit log. Anomalous action sequences (e.g., sudden shifts toward sensitive operations) are detectable via the trace. |

**Residual risk**: ConstrAI cannot verify model provenance or detect
weight-level backdoors. Supply-chain mitigations must include model signing
(e.g., MLflow model registry with hash verification) at the ML-ops layer.

---

## LLM04: Data and Model Poisoning

**Risk**: Training or fine-tuning data is manipulated to alter model behavior,
inject biases, or create exploitable patterns at inference time.

**ConstrAI mitigation**:

| Component | How it helps |
|---|---|
| **Bayesian beliefs** (`BeliefState`) | Cross-checks the LLM's proposed action against its prior success rate. A poisoned model producing anomalous action sequences will trigger low-confidence flags. Part of the Reasoning Engine (Layer 1), not a formal theorem. |
| **T3 Invariants** | Regardless of how the model was trained, actions violating safety predicates are blocked at runtime. |
| **CausalGraph** | Models dependencies between actions. Actions inconsistent with the declared causal graph are rejected before reaching the kernel. |

**Residual risk**: Subtle poisoning that produces plausible-looking actions
within invariant bounds cannot be detected by ConstrAI alone. Runtime monitoring
(T6 trace anomaly detection) supplements but does not replace training-time
integrity checks.

---

## LLM05: Improper Output Handling

**Risk**: LLM outputs are passed unsanitized to downstream systems (SQL
interpreters, OS shells, JavaScript contexts), enabling injection attacks.

**ConstrAI mitigation**:

| Component | How it helps |
|---|---|
| **T3 Invariants** | `no_regex_match_invariant` and `no_sensitive_substring_invariant` can reject actions whose output fields contain injection patterns (SQL keywords, shell metacharacters). |
| **ActionSpec effects model** | Actions declare *typed effects* on named state variables. The kernel enforces that only declared effects are applied — an LLM cannot inject arbitrary state mutations by embedding them in output text. |
| **T5 Atomicity** | State changes happen only through the effects model, never through raw output interpretation. |

**Recommended pattern**:
```python
from constrai import no_regex_match_invariant

kernel = SafetyKernel(
    budget=100.0,
    invariants=[
        # Block SQL injection patterns in generated queries
        no_regex_match_invariant("generated_sql", r"(DROP|DELETE|TRUNCATE)\s+TABLE"),
        # Block shell injection in generated commands
        no_regex_match_invariant("shell_cmd", r"[;&|`$]"),
    ],
)
```

**Residual risk**: Invariants must explicitly enumerate the downstream
interpreters and their injection patterns. Unknown downstream consumers
require additional sandboxing at the execution layer.

---

## LLM06: Excessive Agency

**Risk**: LLM agents are granted more permissions, capabilities, or
autonomy than necessary for the task, amplifying the blast radius of
any error or compromise.

**ConstrAI mitigation — primary coverage**:

| Component | How it helps |
|---|---|
| **T1 Budget Safety** | Hard upper bound on total resource expenditure. No action sequence can spend more than the declared budget, regardless of how many steps the agent takes. |
| **T2 Bounded Termination** | The agent halts in ≤ ⌊budget / min_action_cost⌋ steps — a provable upper bound on autonomy. |
| **T3 Invariants** | Capability restrictions encoded as invariants: `no_delete_invariant`, `read_only_keys_invariant`, `allowed_values_invariant`, etc. The agent cannot exceed declared permissions. |
| **T8 Emergency Halt** | Emergency actions bypass cost checks and immediately terminate the session — a guaranteed exit ramp even when budget is exhausted. |
| **`@constrai_safe` decorator** | Minimum viable constraint: wraps any function with budget + invariants in one decorator. Suitable for capability-bounded tool calls. |

**This is ConstrAI's core strength.** The framework was designed specifically
to address LLM06 through provable bounds rather than heuristic filtering.

```python
# Minimal LLM06 mitigation — wrap every agent tool
@constrai_safe(
    budget=50.0,           # T1: $50 total, then halts
    cost_per_call=5.0,     # T2: max 10 calls
    invariants=[
        no_delete_invariant("records"),         # T3: no deletion
        no_delete_invariant("config"),          # T3: config must not be removed
    ],
)
def agent_tool(input: str) -> str:
    ...
```

**Residual risk**: None within the declared invariant scope. Invariants that
are too permissive (e.g., `lambda s: True`) provide no protection.

---

## LLM07: System Prompt Leakage

**Risk**: The LLM's system prompt (containing instructions, credentials, or
proprietary context) is extracted by adversarial users.

**ConstrAI mitigation**:

| Component | How it helps |
|---|---|
| **T3 Invariants** | Block any action that would write system prompt content to user-visible output fields. |
| **IFC Labels** | Label the system prompt as CONFIDENTIAL; IFC prevents it from flowing into PUBLIC-labelled response fields. |

**Note**: ConstrAI operates at the action/state level, not at the
token/prompt level. Mitigating prompt leakage requires invariants that
explicitly model the system prompt as a protected state variable.

**Residual risk**: If the system prompt is not modelled as a state variable
(which is the common case for most deployments), ConstrAI provides no
protection. This risk requires prompt-level defenses (separate system/user
context isolation, output classifiers).

---

## LLM08: Vector and Embedding Weaknesses

**Risk**: Vector store poisoning, adversarial embeddings, or retrieval
manipulation cause the LLM to retrieve and act on malicious content.

**ConstrAI mitigation**:

| Component | How it helps |
|---|---|
| **T3 Invariants** | Retrieved content passes through invariant checks before being acted upon. A retrieved malicious instruction that produces a dangerous proposed action is blocked at the kernel. |
| **Bayesian beliefs** (`BeliefState`) | Bayesian priors on action success rates flag anomalous action proposals that may result from retrieval poisoning. Part of the Reasoning Engine, not a formal theorem. |

**Residual risk**: ConstrAI does not inspect vector store contents or
embedding distances. Poisoned retrievals that produce plausible-looking
actions within invariant bounds pass through. Mitigate at the retrieval
layer with content hashing, source attribution, and anomaly scoring.

---

## LLM09: Misinformation

**Risk**: The LLM generates confidently stated but factually incorrect
content that is subsequently acted upon.

**ConstrAI mitigation**:

| Component | How it helps |
|---|---|
| **Bayesian beliefs** (`BeliefState`) | Maintains calibrated uncertainty estimates per action. Actions proposed with low confidence are flagged before execution. Part of the Reasoning Engine, not a formal theorem. |
| **T3 Invariants** | Domain-specific invariants can enforce factual constraints: `value_range_invariant` ensures numeric outputs (e.g., dosages, financial figures) remain in plausible ranges. |
| **Human approval gates** | `human_approval_gate_invariant` requires explicit human sign-off before high-stakes actions derived from LLM-generated facts are executed. |

**Residual risk**: ConstrAI cannot verify factual accuracy as a general
capability. Mitigations are limited to structural constraints on outputs.
Factual verification requires retrieval-augmented generation with cited
sources and human review workflows.

---

## LLM10: Unbounded Consumption

**Risk**: Denial-of-service through resource exhaustion: the LLM is induced
to generate excessively long outputs, make unlimited API calls, or consume
disproportionate compute.

**ConstrAI mitigation — strong coverage**:

| Component | How it helps |
|---|---|
| **T1 Budget Safety** | Hard cap on cumulative cost. No sequence of actions can exhaust more than the declared budget, regardless of adversarial intent. |
| **T2 Bounded Termination** | Provable maximum step count = ⌊budget / min_action_cost⌋. The agent cannot loop indefinitely. |
| **`rate_limit_invariant`** | Explicit per-window call rate limits, complementing the T1/T2 resource bounds. |
| **`resource_ceiling_invariant`** | Hard ceiling on named resource metrics (tokens generated, API calls, memory usage). |
| **T8 Emergency Halt** | If a runaway agent exceeds its step budget, emergency actions guarantee a clean exit. |

```python
from constrai import rate_limit_invariant, resource_ceiling_invariant

kernel = SafetyKernel(
    budget=10.0,           # T1: hard cost cap
    min_action_cost=0.1,   # T2: max 100 steps (10.0/0.1)
    invariants=[
        rate_limit_invariant("api_calls", max_count=20),
        resource_ceiling_invariant("tokens_generated", ceiling=100_000),
    ],
)
```

**Residual risk**: Budget is denominated in abstract "cost units." Token-level
rate limiting requires mapping token counts to cost units in the ActionSpec
effects, which is the integrator's responsibility.

---

## Coverage Summary

| OWASP Risk | ConstrAI Coverage | Strength |
|---|---|---|
| LLM01 Prompt Injection | T3, AttestationGate, IFC, T6 | Partial (invariants must be declared) |
| LLM02 Sensitive Disclosure | IFC, T3, SaliencyEngine | Partial (pattern-based) |
| LLM03 Supply Chain | AttestationGate, T3, T6 | Partial (runtime only) |
| LLM04 Data Poisoning | Bayesian beliefs, T3, CausalGraph | Partial (runtime anomaly detection) |
| LLM05 Output Handling | T3, T5, effects model | Strong (typed effects prevent raw injection) |
| LLM06 Excessive Agency | T1, T2, T3, T8 | **Strong — primary design goal** |
| LLM07 System Prompt Leakage | T3, IFC | Weak (requires explicit modelling) |
| LLM08 Vector Weaknesses | T3, Bayesian beliefs | Partial (upstream retrieval not covered) |
| LLM09 Misinformation | Bayesian beliefs, T3, human gates | Partial (structural constraints only) |
| LLM10 Unbounded Consumption | T1, T2, rate_limit, ceiling | **Strong — full provable bounds** |

ConstrAI is **strongest** on LLM06 (Excessive Agency) and LLM10 (Unbounded
Consumption) — the two risks most amenable to formal, provable bounds.

---

## See Also

- `THEOREMS.md` — formal proofs of T1–T8
- `MATHEMATICAL_COMPLIANCE.md` — full mathematical derivations
- `docs/REFERENCE_MONITOR.md` — IFC and CBF documentation
- `SECURITY.md` — responsible disclosure policy
- [OWASP LLM Top 10 (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
