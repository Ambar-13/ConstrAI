"""
clampai.invariants — Pre-built safety invariant library.

Twenty ready-to-use ``Invariant`` constructors covering the most common
AI agent safety patterns. All predicates are pure (no I/O, no randomness,
no global state), run in O(1) or O(n) with small bounded n, and are
fail-safe: exceptions propagate as violations rather than letting unsafe
states through.

Usage::

    from clampai.invariants import (
        no_delete_invariant,
        rate_limit_invariant,
        human_approval_gate_invariant,
        no_sensitive_substring_invariant,
    )

    kernel = SafetyKernel(
        budget=10.0,
        invariants=[
            no_delete_invariant("user_data"),
            rate_limit_invariant("api_calls", max_count=20),
            human_approval_gate_invariant("human_approved"),
        ],
    )

Guarantee: Every invariant in this library is ``PROVEN`` in the
``blocking`` enforcement mode — they hold on every reachable state
provided the kernel uses check-before-commit (which SafetyKernel does).
"""
from __future__ import annotations

import re
import time as _time
from typing import Any, Callable, Collection, Dict, List, Optional, Set

from .formal import Invariant, State

# Resource and budget limits

def rate_limit_invariant(
    key: str,
    max_count: int,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` exceeds ``max_count``.

    Common uses: API call counters, retry counts, request-per-minute limits.

    Example::

        rate_limit_invariant("api_calls", max_count=100)
        # → blocks actions that would push api_calls > 100
    """
    _sug = (
        suggestion
        or f"'{key}' has reached its limit of {max_count}. "
           f"Wait for the counter to reset, batch operations, or increase the allowed limit."
    )
    return Invariant(
        f"rate_limit:{key}",
        lambda s: (s.get(key, 0) or 0) <= max_count,
        description=f"'{key}' must not exceed {max_count} (rate limit)",
        enforcement=enforcement,
        suggestion=_sug,
    )


def resource_ceiling_invariant(
    key: str,
    ceiling: float,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` (numeric) exceeds ``ceiling``.

    Common uses: memory limits, CPU quotas, token budgets, file-size caps.

    Example::

        resource_ceiling_invariant("tokens_used", ceiling=50_000)
    """
    _sug = (
        suggestion
        or f"'{key}' has exceeded its ceiling of {ceiling}. "
           f"Free up resources or increase the allowed ceiling before proceeding."
    )
    return Invariant(
        f"resource_ceiling:{key}",
        lambda s: float(s.get(key, 0) or 0) <= ceiling,
        description=f"'{key}' must not exceed {ceiling}",
        enforcement=enforcement,
        suggestion=_sug,
    )


def value_range_invariant(
    key: str,
    min_val: float,
    max_val: float,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` is outside ``[min_val, max_val]``.

    Common uses: temperature controls, confidence thresholds, score bounds.

    Example::

        value_range_invariant("confidence", 0.0, 1.0)
        # → confidence must stay in [0, 1]
    """
    _sug = (
        suggestion
        or f"'{key}' is outside the valid range [{min_val}, {max_val}]. "
           f"Clamp or recompute the value before attempting this action."
    )
    return Invariant(
        f"value_range:{key}",
        lambda s: min_val <= float(s.get(key, min_val) or min_val) <= max_val,
        description=f"'{key}' must be in [{min_val}, {max_val}]",
        enforcement=enforcement,
        suggestion=_sug,
    )


def max_retries_invariant(
    key: str,
    limit: int,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` (retry counter) exceeds ``limit``.

    Identical to ``rate_limit_invariant`` in logic, but named for the
    common "retry exhaustion" pattern so intent is clear in audit logs.

    Example::

        max_retries_invariant("fetch_retries", limit=3)
    """
    _sug = (
        suggestion
        or f"'{key}' has reached the retry limit of {limit}. "
           f"Escalate to a human, choose an alternative action, or abort the task."
    )
    return Invariant(
        f"max_retries:{key}",
        lambda s: (s.get(key, 0) or 0) <= limit,
        description=f"Retry count '{key}' must not exceed {limit}",
        enforcement=enforcement,
        suggestion=_sug,
    )


# Data protection

def no_delete_invariant(
    key: str,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any action that removes or falsifies ``state[key]``.

    Considers ``None``, ``False``, ``0``, ``""`` and missing key as violations.
    Protects critical data from accidental or adversarial deletion.

    Example::

        no_delete_invariant("user_session")
        # → any action that sets user_session=None is blocked
    """
    _sug = (
        suggestion
        or f"'{key}' has been deleted or set to a falsy value. "
           f"Restore it to a valid, non-empty value before proceeding."
    )
    return Invariant(
        f"no_delete:{key}",
        lambda s: bool(s.get(key, None)),
        description=f"'{key}' must remain present and non-empty",
        enforcement=enforcement,
        suggestion=_sug,
    )


def read_only_keys_invariant(
    keys: Collection[str],
    initial_state: State,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any action that changes the value of any key in ``keys``.

    Compares against values frozen in ``initial_state`` at construction time.
    Useful for protecting configuration, credentials references, or schema
    fields that should never change during a run.

    Example::

        read_only_keys_invariant(["db_schema_version"], initial_state=s0)
    """
    frozen = {k: initial_state.get(k) for k in keys}
    key_list = list(keys)
    _sug = (
        suggestion
        or f"Fields {key_list} are read-only and cannot be modified during this run. "
           f"If a change is required, construct a new task with an updated initial state."
    )
    return Invariant(
        f"read_only:{','.join(sorted(key_list))}",
        lambda s: all(s.get(k) == frozen[k] for k in key_list),
        description=f"Fields {key_list} must remain unchanged",
        enforcement=enforcement,
        suggestion=_sug,
    )


def required_fields_invariant(
    fields: Collection[str],
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where one or more ``fields`` are absent or ``None``.

    Ensures the state always carries the minimum required structure.

    Example::

        required_fields_invariant(["task_id", "user_id", "session_token"])
    """
    field_list = list(fields)
    _sug = (
        suggestion
        or f"One or more required fields {field_list} are missing or None. "
           f"Populate all required fields before attempting this action."
    )
    return Invariant(
        f"required_fields:{','.join(sorted(field_list))}",
        lambda s: all(s.get(f) is not None for f in field_list),
        description=f"Fields {field_list} must all be present",
        enforcement=enforcement,
        suggestion=_sug,
    )


def no_sensitive_substring_invariant(
    key: str,
    forbidden: Collection[str],
    *,
    case_sensitive: bool = False,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` (string) contains any forbidden substring.

    Common uses: prevent API keys, passwords, or PII patterns from appearing
    in LLM-visible output fields.

    Example::

        no_sensitive_substring_invariant(
            "llm_output",
            ["password", "api_key", "sk-", "Bearer "],
        )
    """
    forbidden_list = list(forbidden)
    if not case_sensitive:
        _forbidden_lower = [f.lower() for f in forbidden_list]

        def _pred(s: State, fl: List[str] = _forbidden_lower) -> bool:
            val = str(s.get(key, "") or "")
            val_lower = val.lower()
            return not any(sub in val_lower for sub in fl)
    else:
        def _pred(s: State, fl: List[str] = forbidden_list) -> bool:
            val = str(s.get(key, "") or "")
            return not any(sub in val for sub in fl)

    _sug = (
        suggestion
        or f"'{key}' contains a forbidden string from the list: {forbidden_list}. "
           f"Strip or redact the sensitive content before writing to this field."
    )
    return Invariant(
        f"no_sensitive_substring:{key}",
        _pred,
        description=f"'{key}' must not contain forbidden substrings",
        enforcement=enforcement,
        suggestion=_sug,
    )


def no_regex_match_invariant(
    key: str,
    pattern: str,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` (string) matches a regex pattern.

    Useful for catching structured secrets (credit card numbers, SSNs, JWTs).

    Example::

        # Block JWT tokens from leaking into output
        no_regex_match_invariant("response_text", r"eyJ[A-Za-z0-9+/=]{20,}")
    """
    _compiled = re.compile(pattern)
    _sug = (
        suggestion
        or f"'{key}' matches the forbidden pattern '{pattern}'. "
           f"Redact or sanitise the value before proceeding."
    )
    return Invariant(
        f"no_regex:{key}",
        lambda s: not _compiled.search(str(s.get(key, "") or "")),
        description=f"'{key}' must not match pattern {pattern!r}",
        enforcement=enforcement,
        suggestion=_sug,
    )


# Access control and human oversight

def human_approval_gate_invariant(
    approval_key: str,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block all actions unless ``state[approval_key]`` is truthy.

    Implements a human-in-the-loop gate: the agent cannot proceed until a
    human (or an authoritative upstream process) sets the approval flag.

    Usage pattern::

        state = state.with_updates({"human_approved": True})
        # Now the kernel will allow actions past this invariant

    Example::

        human_approval_gate_invariant("human_approved")
        # → all actions blocked until state["human_approved"] is truthy
    """
    _sug = (
        suggestion
        or f"Human approval is required before this action can proceed. "
           f"Set state['{approval_key}'] = True after obtaining explicit user confirmation."
    )
    return Invariant(
        f"human_approval_gate:{approval_key}",
        lambda s: bool(s.get(approval_key, False)),
        description=f"Human approval required ('{approval_key}' must be truthy)",
        enforcement=enforcement,
        suggestion=_sug,
    )


def no_action_after_flag_invariant(
    flag_key: str,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block all actions once ``state[flag_key]`` becomes truthy.

    Use for hard-stop conditions: task completion, error states, or
    explicit abort flags.  Once the flag is set, no further actions
    are permitted.

    Example::

        no_action_after_flag_invariant("task_complete")
        # → once task_complete=True, no further actions allowed
    """
    _sug = (
        suggestion
        or f"The flag '{flag_key}' is set — the agent has reached a terminal state. "
           f"No further actions are permitted. Start a new task to continue."
    )
    return Invariant(
        f"no_action_after_flag:{flag_key}",
        lambda s: not bool(s.get(flag_key, False)),
        description=f"No actions allowed after '{flag_key}' is set",
        enforcement=enforcement,
        suggestion=_sug,
    )


def allowed_values_invariant(
    key: str,
    allowed: Collection[Any],
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` is not in the ``allowed`` set.

    Useful for state-machine transitions, enum fields, and configuration
    values that must stay within a known-safe set.

    Example::

        allowed_values_invariant("deployment_env", {"staging", "prod"})
        # → state["deployment_env"] must be "staging" or "prod"
    """
    allowed_set: Set[Any] = set(allowed)
    _sug = (
        suggestion
        or f"'{key}' has an unexpected value. "
           f"Choose one of the permitted values: {sorted(str(v) for v in allowed_set)}."
    )
    return Invariant(
        f"allowed_values:{key}",
        lambda s: s.get(key) in allowed_set,
        description=f"'{key}' must be one of {sorted(str(v) for v in allowed_set)}",
        enforcement=enforcement,
        suggestion=_sug,
    )


# Progress and monotonicity

def monotone_increasing_invariant(
    key: str,
    initial_value: float = 0.0,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any action that would decrease ``state[key]``.

    Uses a closure over the last-seen maximum to enforce monotone increase.
    Suitable for progress counters, completion percentages, and step counts.

    This invariant is stateful: it tracks the last-seen maximum internally.
    Guarantee: CONDITIONAL — monotonicity holds for the lifetime of this
    Invariant object; constructing a fresh object resets the baseline.

    Example::

        monotone_increasing_invariant("tasks_completed")
    """
    # Mutable container so lambda can write to it
    _max = [initial_value]

    def _pred(s: State, m: List[float] = _max) -> bool:
        val = float(s.get(key, initial_value) or initial_value)
        if val >= m[0]:
            m[0] = val
            return True
        return False  # Decreased — violation

    _sug = (
        suggestion
        or f"'{key}' decreased below its previous maximum. "
           f"The value must only increase over time. Revert to the last valid value or abort."
    )
    return Invariant(
        f"monotone_increasing:{key}",
        _pred,
        description=f"'{key}' must be non-decreasing",
        enforcement=enforcement,
        suggestion=_sug,
    )


def monotone_decreasing_invariant(
    key: str,
    initial_value: float = float('inf'),
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any action that would increase ``state[key]``.

    Suitable for debt counters, backlog lengths, and cost accumulators that
    should only decrease over time.

    This invariant is stateful; the same caveat as ``monotone_increasing_invariant`` applies.

    Example::

        monotone_decreasing_invariant("error_count")
        # → error_count must never go up
    """
    _min = [initial_value]

    def _pred(s: State, m: List[float] = _min) -> bool:
        val = float(s.get(key, initial_value) or initial_value)
        if val <= m[0]:
            m[0] = val
            return True
        return False

    _sug = (
        suggestion
        or f"'{key}' increased above its previous minimum. "
           f"The value must only decrease over time. Revert to the last valid value or abort."
    )
    return Invariant(
        f"monotone_decreasing:{key}",
        _pred,
        description=f"'{key}' must be non-increasing",
        enforcement=enforcement,
        suggestion=_sug,
    )


# Structural integrity

def non_empty_invariant(
    key: str,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` is empty (list, string, dict) or None.

    Example::

        non_empty_invariant("output_files")
        # → output_files must not be an empty list or None
    """
    _sug = (
        suggestion
        or f"'{key}' is empty or None. "
           f"Populate it with at least one value before attempting this action."
    )
    return Invariant(
        f"non_empty:{key}",
        lambda s: bool(s.get(key)),
        description=f"'{key}' must be non-empty and non-None",
        enforcement=enforcement,
        suggestion=_sug,
    )


def list_length_invariant(
    key: str,
    max_length: int,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` (list or string) exceeds ``max_length``.

    Common uses: file lists, output token budgets, queue depths.

    Example::

        list_length_invariant("created_files", max_length=100)
    """
    _sug = (
        suggestion
        or f"'{key}' has reached its maximum of {max_length} items. "
           f"Remove stale entries or process existing items before adding more."
    )
    return Invariant(
        f"list_length:{key}",
        lambda s: len(s.get(key) or []) <= max_length,
        description=f"'{key}' length must not exceed {max_length}",
        enforcement=enforcement,
        suggestion=_sug,
    )


def no_duplicate_ids_invariant(
    key: str,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` (list) contains duplicate values.

    Suitable for action logs, ID registries, and deduplication guarantees.

    Example::

        no_duplicate_ids_invariant("processed_ids")
    """
    _sug = (
        suggestion
        or f"'{key}' contains duplicate values. "
           f"Verify the new item is not already in the list before appending."
    )
    return Invariant(
        f"no_duplicate_ids:{key}",
        lambda s: len(lst := (s.get(key) or [])) == len(set(lst)),
        description=f"'{key}' must contain no duplicate values",
        enforcement=enforcement,
        suggestion=_sug,
    )


def custom_invariant(
    name: str,
    validator: Callable[[Any], bool],
    key: str,
    *,
    description: str = "",
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
    max_eval_ms: Optional[float] = None,
) -> Invariant:
    """
    Wrap an arbitrary single-argument validator into an ``Invariant``.

    The validator receives ``state[key]`` (not the full State).
    Use this when your predicate is simpler to write against the field
    value than against the entire state.

    Example::

        custom_invariant(
            "valid_email",
            lambda v: "@" in str(v or ""),
            key="user_email",
            description="user_email must be a valid email address",
        )
    """
    _desc = description or f"custom validator on '{key}'"
    return Invariant(
        name,
        lambda s: validator(s.get(key)),
        description=_desc,
        enforcement=enforcement,
        suggestion=suggestion,
        max_eval_ms=max_eval_ms,
    )


# Compound and domain-specific patterns

def file_operation_limit_invariant(
    key: str = "files_modified",
    max_ops: int = 50,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` exceeds ``max_ops``.

    Convenience alias of ``rate_limit_invariant`` with defaults and naming
    tuned for file-system operation scenarios.

    Example::

        file_operation_limit_invariant(max_ops=10)
        # → at most 10 file modifications per run
    """
    _sug = (
        suggestion
        or f"File operation limit of {max_ops} reached for '{key}'. "
           f"Review and commit existing changes before modifying more files."
    )
    return rate_limit_invariant(key, max_ops, enforcement=enforcement, suggestion=_sug)


def api_call_limit_invariant(
    key: str = "api_calls",
    max_calls: int = 100,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` exceeds ``max_calls``.

    Convenience alias for API call rate limiting.

    Example::

        api_call_limit_invariant(max_calls=50)
    """
    _sug = (
        suggestion
        or f"API call limit of {max_calls} reached for '{key}'. "
           f"Cache responses, batch remaining requests, or increase the limit."
    )
    return rate_limit_invariant(key, max_calls, enforcement=enforcement, suggestion=_sug)


def email_safety_invariant(
    *, enforcement: str = "blocking") -> Invariant:
    """
    Block any action that sets ``state['emails_deleted']`` above 0.

    Purpose-built for the email-deletion demo and any agent that processes
    email: ensures no emails are permanently deleted without an explicit
    override.  Pair with ``human_approval_gate_invariant`` for a two-factor
    deletion guard.

    Example::

        email_safety_invariant()
        # → blocks any action that would delete emails
    """
    return Invariant(
        "email_safety:no_delete",
        lambda s: (s.get("emails_deleted", 0) or 0) == 0,
        description="No emails may be permanently deleted",
        enforcement=enforcement,
        suggestion=(
            "Email deletion is blocked by default. "
            "Set state['human_approved'] = True after explicit user confirmation "
            "before proposing a delete action."
        ),
    )



def string_length_invariant(
    key: str,
    max_length: int,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where the string representation of ``state[key]``
    exceeds ``max_length`` characters.

    Safe (passes) if ``key`` is absent from the state.  Non-string values
    are coerced with ``str()`` before measuring length.

    Common uses: preventing LLM prompt injection via oversized fields,
    enforcing output-length SLAs, capping user-supplied strings.

    Example::

        string_length_invariant("llm_output", max_length=4096)
        # → blocks any action that sets llm_output to a string > 4096 chars
    """
    _sug = (
        suggestion
        or f"'{key}' exceeds the maximum length of {max_length} characters. "
           f"Truncate, summarise, or split the content before proceeding."
    )
    return Invariant(
        f"string_length:{key}",
        lambda s: len(str(s.get(key, "") or "")) <= max_length,
        description=f"'{key}' must not exceed {max_length} characters",
        enforcement=enforcement,
        suggestion=_sug,
    )


# PII patterns compiled once at module load time for efficiency.
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                          # SSN
    re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), # CC (16-digit)
    re.compile(r"\b[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}\b"),         # Email
    re.compile(r"\b(\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),  # Phone (NA)
]


def pii_guard_invariant(
    *keys: str,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where one of the specified keys contains PII patterns.

    Detects four common PII categories in the string representation of each
    key's value:

    - US Social Security Numbers (``NNN-NN-NNNN``)
    - 16-digit credit/debit card numbers (with optional spaces or dashes)
    - Email addresses
    - North American phone numbers

    All four patterns are compiled at import time.  The invariant is
    fail-safe: if a key is absent, it passes.

    Args:
        *keys: One or more state-variable names to scan for PII.
        enforcement: ``"blocking"`` (default) or ``"monitoring"``.
        suggestion: Optional remediation hint.

    Example::

        pii_guard_invariant("user_message", "agent_output")
        # → blocks any action that would put PII in either field
    """
    key_list = list(keys)
    _sug = (
        suggestion
        or f"PII detected (SSN, credit card number, email address, or phone number) "
           f"in fields: {key_list}. Redact or tokenise sensitive data before writing to these fields."
    )

    def _predicate(s: State, _keys: List[str] = key_list) -> bool:
        for k in _keys:
            val = str(s.get(k, "") or "")
            for pat in _PII_PATTERNS:
                if pat.search(val):
                    return False
        return True

    key_label = ",".join(key_list)
    return Invariant(
        f"pii_guard:{key_label}",
        _predicate,
        description=f"Fields [{key_label}] must not contain PII (SSN, CC, email, phone)",
        enforcement=enforcement,
        suggestion=_sug,
    )


def time_window_rate_invariant(
    key: str,
    max_count: int,
    window_seconds: float,
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` (a list of float timestamps) has
    ``max_count`` or more entries within the last ``window_seconds``.

    Usage pattern — store a list of ``time.time()`` values in state::

        state = state.with_updates({"call_times": existing + [time.time()]})

    The predicate counts how many timestamps in the list are within
    ``window_seconds`` of the current wall-clock time.  Timestamps older
    than the window do not count.

    Safe (passes) if ``state[key]`` is absent or not a list.

    Example::

        time_window_rate_invariant("api_timestamps", max_count=10,
                                   window_seconds=60.0)
        # → at most 10 API calls per minute
    """
    _sug = (
        suggestion
        or f"Rate limit exceeded: more than {max_count} events in '{key}' "
           f"within the last {window_seconds}s. Wait for the window to roll over before retrying."
    )
    return Invariant(
        f"time_window_rate:{key}",
        lambda s: _count_recent(s.get(key), window_seconds) < max_count,
        description=(
            f"At most {max_count} entries in '{key}' within "
            f"{window_seconds}s window"
        ),
        enforcement=enforcement,
        suggestion=_sug,
    )


def _count_recent(timestamps: Any, window_seconds: float) -> int:
    """Count timestamps (float list) within window_seconds of now."""
    if not isinstance(timestamps, (list, tuple)):
        return 0
    now = _time.time()
    return sum(
        1 for t in timestamps
        if isinstance(t, (int, float)) and now - t <= window_seconds
    )


def json_schema_invariant(
    key: str,
    schema: Dict[str, type],
    *,
    enforcement: str = "blocking",
    suggestion: Optional[str] = None,
) -> Invariant:
    """
    Block any state where ``state[key]`` (a dict) has a field whose Python
    type does not match the expected type declared in ``schema``.

    ``schema`` maps field names to expected Python types, for example::

        json_schema_invariant("config", {"timeout": int, "retries": int,
                                         "endpoint": str})

    Rules:

    - Fields present in the dict but absent from ``schema`` are ignored.
    - Fields absent from the dict (but present in ``schema``) are not
      checked — use :func:`required_fields_invariant` for presence checks.
    - If ``state[key]`` is absent or ``None``, the invariant passes (safe).
    - If ``state[key]`` is not a dict, the invariant fails (blocks).

    Example::

        json_schema_invariant("payload", {"amount": float, "currency": str})
        # → blocks actions that set payload["amount"] to a non-float
    """
    _schema_frozen = dict(schema)
    field_names = list(_schema_frozen.keys())
    _sug = (
        suggestion
        or f"'{key}' contains a field with the wrong type. "
           f"Check field types against the expected schema: {_schema_frozen}."
    )

    def _predicate(s: State, _schema: Dict[str, type] = _schema_frozen) -> bool:
        val = s.get(key)
        if val is None:
            return True
        if not isinstance(val, dict):
            return False
        for field_name, expected_type in _schema.items():
            if field_name in val and not isinstance(val[field_name], expected_type):
                return False
        return True

    return Invariant(
        f"json_schema:{key}",
        _predicate,
        description=f"'{key}' dict fields must match schema: {field_names}",
        enforcement=enforcement,
        suggestion=_sug,
    )


__all__ = [
    "allowed_values_invariant",
    "api_call_limit_invariant",
    "custom_invariant",
    "email_safety_invariant",
    # Domain-specific
    "file_operation_limit_invariant",
    # Access control
    "human_approval_gate_invariant",
    # Data protection — PII
    "json_schema_invariant",
    "list_length_invariant",
    "max_retries_invariant",
    "monotone_decreasing_invariant",
    # Monotonicity
    "monotone_increasing_invariant",
    "no_action_after_flag_invariant",
    # Data protection
    "no_delete_invariant",
    "no_duplicate_ids_invariant",
    "no_regex_match_invariant",
    "no_sensitive_substring_invariant",
    # Structural
    "non_empty_invariant",
    # PII guard
    "pii_guard_invariant",
    # Resource limits
    "rate_limit_invariant",
    "read_only_keys_invariant",
    "required_fields_invariant",
    "resource_ceiling_invariant",
    # String length cap
    "string_length_invariant",
    # Time-window rate limit
    "time_window_rate_invariant",
    "value_range_invariant",
]
