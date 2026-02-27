"""
Tests for constrai.invariants — all 21 pre-built invariant factory functions.

Exercises every factory, both the passing and failing state, and key edge
cases (None values, monitoring mode, custom messages, missing keys).
"""
from __future__ import annotations

import pytest

from constrai.formal import State
from constrai.invariants import (
    allowed_values_invariant,
    api_call_limit_invariant,
    custom_invariant,
    email_safety_invariant,
    file_operation_limit_invariant,
    human_approval_gate_invariant,
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
    rate_limit_invariant,
    read_only_keys_invariant,
    required_fields_invariant,
    resource_ceiling_invariant,
    value_range_invariant,
)


def _check(inv, state_dict: dict) -> bool:
    s = State(state_dict)
    ok, _ = inv.check(s)
    return ok


class TestRateLimitInvariant:
    def test_below_limit_passes(self):
        inv = rate_limit_invariant("calls", 5)
        assert _check(inv, {"calls": 4}) is True

    def test_at_limit_passes(self):
        inv = rate_limit_invariant("calls", 5)
        assert _check(inv, {"calls": 5}) is True

    def test_above_limit_fails(self):
        inv = rate_limit_invariant("calls", 5)
        assert _check(inv, {"calls": 6}) is False

    def test_missing_key_defaults_zero(self):
        inv = rate_limit_invariant("calls", 5)
        assert _check(inv, {}) is True

    def test_none_value_defaults_zero(self):
        inv = rate_limit_invariant("calls", 5)
        assert _check(inv, {"calls": None}) is True

    def test_monitoring_mode_returns_false_but_doesnt_block(self):
        inv = rate_limit_invariant("calls", 0, enforcement="monitoring")
        assert inv.enforcement == "monitoring"
        ok, _ = inv.check(State({"calls": 1}))
        assert ok is False

    def test_custom_suggestion(self):
        inv = rate_limit_invariant("calls", 5, suggestion="custom msg")
        assert "custom msg" in (inv.suggestion or "")

    def test_name_includes_key(self):
        inv = rate_limit_invariant("api_calls", 10)
        assert "api_calls" in inv.name


class TestResourceCeilingInvariant:
    def test_below_ceiling_passes(self):
        inv = resource_ceiling_invariant("mem", 1000.0)
        assert _check(inv, {"mem": 500.0}) is True

    def test_at_ceiling_passes(self):
        inv = resource_ceiling_invariant("mem", 1000.0)
        assert _check(inv, {"mem": 1000.0}) is True

    def test_above_ceiling_fails(self):
        inv = resource_ceiling_invariant("mem", 1000.0)
        assert _check(inv, {"mem": 1001.0}) is False

    def test_missing_key_defaults_zero(self):
        inv = resource_ceiling_invariant("mem", 100.0)
        assert _check(inv, {}) is True


class TestValueRangeInvariant:
    def test_within_range_passes(self):
        inv = value_range_invariant("score", 0.0, 1.0)
        assert _check(inv, {"score": 0.5}) is True

    def test_at_min_passes(self):
        inv = value_range_invariant("score", 0.0, 1.0)
        assert _check(inv, {"score": 0.0}) is True

    def test_at_max_passes(self):
        inv = value_range_invariant("score", 0.0, 1.0)
        assert _check(inv, {"score": 1.0}) is True

    def test_below_min_fails(self):
        inv = value_range_invariant("score", 0.0, 1.0)
        assert _check(inv, {"score": -0.1}) is False

    def test_above_max_fails(self):
        inv = value_range_invariant("score", 0.0, 1.0)
        assert _check(inv, {"score": 1.1}) is False

    def test_missing_key_uses_min_val(self):
        inv = value_range_invariant("score", 0.0, 1.0)
        assert _check(inv, {}) is True


class TestMaxRetriesInvariant:
    def test_below_limit_passes(self):
        inv = max_retries_invariant("retries", 3)
        assert _check(inv, {"retries": 2}) is True

    def test_at_limit_passes(self):
        inv = max_retries_invariant("retries", 3)
        assert _check(inv, {"retries": 3}) is True

    def test_above_limit_fails(self):
        inv = max_retries_invariant("retries", 3)
        assert _check(inv, {"retries": 4}) is False


class TestNoDeleteInvariant:
    def test_present_truthy_passes(self):
        inv = no_delete_invariant("session")
        assert _check(inv, {"session": "abc123"}) is True

    def test_present_false_fails(self):
        inv = no_delete_invariant("session")
        assert _check(inv, {"session": False}) is False

    def test_present_none_fails(self):
        inv = no_delete_invariant("session")
        assert _check(inv, {"session": None}) is False

    def test_missing_key_fails(self):
        inv = no_delete_invariant("session")
        assert _check(inv, {}) is False

    def test_present_empty_string_fails(self):
        inv = no_delete_invariant("token")
        assert _check(inv, {"token": ""}) is False


class TestReadOnlyKeysInvariant:
    def test_unchanged_keys_pass(self):
        s0 = State({"db_ver": 5, "user": "alice"})
        inv = read_only_keys_invariant(["db_ver"], s0)
        assert _check(inv, {"db_ver": 5, "user": "bob"}) is True

    def test_changed_key_fails(self):
        s0 = State({"db_ver": 5})
        inv = read_only_keys_invariant(["db_ver"], s0)
        assert _check(inv, {"db_ver": 6}) is False

    def test_multiple_keys_all_changed_fails(self):
        s0 = State({"a": 1, "b": 2})
        inv = read_only_keys_invariant(["a", "b"], s0)
        assert _check(inv, {"a": 99, "b": 2}) is False

    def test_multiple_keys_unchanged_passes(self):
        s0 = State({"a": 1, "b": 2})
        inv = read_only_keys_invariant(["a", "b"], s0)
        assert _check(inv, {"a": 1, "b": 2}) is True


class TestRequiredFieldsInvariant:
    def test_all_present_passes(self):
        inv = required_fields_invariant(["task_id", "user_id"])
        assert _check(inv, {"task_id": "t1", "user_id": "u1"}) is True

    def test_one_missing_fails(self):
        inv = required_fields_invariant(["task_id", "user_id"])
        assert _check(inv, {"task_id": "t1"}) is False

    def test_one_none_fails(self):
        inv = required_fields_invariant(["task_id", "user_id"])
        assert _check(inv, {"task_id": "t1", "user_id": None}) is False

    def test_all_missing_fails(self):
        inv = required_fields_invariant(["task_id", "user_id"])
        assert _check(inv, {}) is False


class TestNoSensitiveSubstringInvariant:
    def test_clean_string_passes(self):
        inv = no_sensitive_substring_invariant("output", ["password", "api_key"])
        assert _check(inv, {"output": "hello world"}) is True

    def test_forbidden_substring_fails(self):
        inv = no_sensitive_substring_invariant("output", ["password"])
        assert _check(inv, {"output": "my password is abc"}) is False

    def test_case_insensitive_default(self):
        inv = no_sensitive_substring_invariant("output", ["PASSWORD"])
        assert _check(inv, {"output": "my password leaked"}) is False

    def test_case_sensitive_mode(self):
        inv = no_sensitive_substring_invariant("output", ["PASSWORD"], case_sensitive=True)
        assert _check(inv, {"output": "my password leaked"}) is True

    def test_case_sensitive_exact_match_fails(self):
        inv = no_sensitive_substring_invariant("output", ["password"], case_sensitive=True)
        assert _check(inv, {"output": "my password leaked"}) is False

    def test_missing_key_passes(self):
        inv = no_sensitive_substring_invariant("output", ["secret"])
        assert _check(inv, {}) is True


class TestNoRegexMatchInvariant:
    def test_no_match_passes(self):
        inv = no_regex_match_invariant("response", r"\bpassword\b")
        assert _check(inv, {"response": "hello world"}) is True

    def test_match_fails(self):
        inv = no_regex_match_invariant("response", r"eyJ[A-Za-z0-9+/=]{5,}")
        assert _check(inv, {"response": "token: eyJhbGciOiJIUzI1NiJ9"}) is False

    def test_missing_key_passes(self):
        inv = no_regex_match_invariant("response", r"\d{4}")
        assert _check(inv, {}) is True


class TestHumanApprovalGateInvariant:
    def test_approved_passes(self):
        inv = human_approval_gate_invariant("approved")
        assert _check(inv, {"approved": True}) is True

    def test_not_approved_fails(self):
        inv = human_approval_gate_invariant("approved")
        assert _check(inv, {"approved": False}) is False

    def test_missing_fails(self):
        inv = human_approval_gate_invariant("approved")
        assert _check(inv, {}) is False

    def test_truthy_string_passes(self):
        inv = human_approval_gate_invariant("approved")
        assert _check(inv, {"approved": "yes"}) is True


class TestNoActionAfterFlagInvariant:
    def test_flag_not_set_passes(self):
        inv = no_action_after_flag_invariant("task_complete")
        assert _check(inv, {"task_complete": False}) is True

    def test_flag_set_fails(self):
        inv = no_action_after_flag_invariant("task_complete")
        assert _check(inv, {"task_complete": True}) is False

    def test_missing_flag_passes(self):
        inv = no_action_after_flag_invariant("task_complete")
        assert _check(inv, {}) is True


class TestAllowedValuesInvariant:
    def test_allowed_value_passes(self):
        inv = allowed_values_invariant("env", {"staging", "prod"})
        assert _check(inv, {"env": "staging"}) is True

    def test_forbidden_value_fails(self):
        inv = allowed_values_invariant("env", {"staging", "prod"})
        assert _check(inv, {"env": "dev"}) is False

    def test_missing_key_fails(self):
        inv = allowed_values_invariant("env", {"staging", "prod"})
        assert _check(inv, {}) is False


class TestMonotoneIncreasingInvariant:
    def test_increasing_passes(self):
        inv = monotone_increasing_invariant("progress")
        assert _check(inv, {"progress": 0.0}) is True
        assert _check(inv, {"progress": 5.0}) is True
        assert _check(inv, {"progress": 10.0}) is True

    def test_decrease_fails(self):
        inv = monotone_increasing_invariant("progress")
        _check(inv, {"progress": 10.0})
        assert _check(inv, {"progress": 5.0}) is False

    def test_missing_key_uses_initial(self):
        inv = monotone_increasing_invariant("progress", initial_value=5.0)
        assert _check(inv, {}) is True

    def test_initial_value_respected(self):
        inv = monotone_increasing_invariant("progress", initial_value=10.0)
        assert _check(inv, {"progress": 9.0}) is False


class TestMonotoneDecreasingInvariant:
    def test_decreasing_passes(self):
        inv = monotone_decreasing_invariant("errors")
        assert _check(inv, {"errors": 10.0}) is True
        assert _check(inv, {"errors": 5.0}) is True
        # Note: the predicate uses `(value or initial_value)` so 0.0 is treated
        # as missing (falsy). Caller should track non-zero decreasing values.
        assert _check(inv, {"errors": 2.0}) is True

    def test_increase_fails(self):
        inv = monotone_decreasing_invariant("errors")
        _check(inv, {"errors": 5.0})
        assert _check(inv, {"errors": 6.0}) is False

    def test_zero_treated_as_falsy_uses_initial(self):
        inv = monotone_decreasing_invariant("errors")
        _check(inv, {"errors": 5.0})
        # 0.0 is falsy, so `0.0 or float('inf')` = inf > last_seen(5.0) → fails
        assert _check(inv, {"errors": 0.0}) is False


class TestNonEmptyInvariant:
    def test_non_empty_list_passes(self):
        inv = non_empty_invariant("items")
        assert _check(inv, {"items": [1, 2, 3]}) is True

    def test_empty_list_fails(self):
        inv = non_empty_invariant("items")
        assert _check(inv, {"items": []}) is False

    def test_none_fails(self):
        inv = non_empty_invariant("items")
        assert _check(inv, {"items": None}) is False

    def test_missing_fails(self):
        inv = non_empty_invariant("items")
        assert _check(inv, {}) is False

    def test_non_empty_string_passes(self):
        inv = non_empty_invariant("name")
        assert _check(inv, {"name": "alice"}) is True


class TestListLengthInvariant:
    def test_within_limit_passes(self):
        inv = list_length_invariant("files", 10)
        assert _check(inv, {"files": [1, 2, 3]}) is True

    def test_at_limit_passes(self):
        inv = list_length_invariant("files", 3)
        assert _check(inv, {"files": [1, 2, 3]}) is True

    def test_over_limit_fails(self):
        inv = list_length_invariant("files", 3)
        assert _check(inv, {"files": [1, 2, 3, 4]}) is False

    def test_none_treated_as_empty(self):
        inv = list_length_invariant("files", 5)
        assert _check(inv, {"files": None}) is True


class TestNoDuplicateIdsInvariant:
    def test_unique_ids_pass(self):
        inv = no_duplicate_ids_invariant("ids")
        assert _check(inv, {"ids": ["a", "b", "c"]}) is True

    def test_duplicate_ids_fail(self):
        inv = no_duplicate_ids_invariant("ids")
        assert _check(inv, {"ids": ["a", "b", "a"]}) is False

    def test_empty_list_passes(self):
        inv = no_duplicate_ids_invariant("ids")
        assert _check(inv, {"ids": []}) is True

    def test_missing_key_passes(self):
        inv = no_duplicate_ids_invariant("ids")
        assert _check(inv, {}) is True


class TestCustomInvariant:
    def test_validator_passes(self):
        inv = custom_invariant("has_at", lambda v: "@" in str(v or ""), key="email")
        assert _check(inv, {"email": "user@example.com"}) is True

    def test_validator_fails(self):
        inv = custom_invariant("has_at", lambda v: "@" in str(v or ""), key="email")
        assert _check(inv, {"email": "notanemail"}) is False

    def test_custom_description(self):
        inv = custom_invariant("v", lambda v: True, key="x", description="my check")
        assert "my check" in inv.description

    def test_missing_key_passes_none_to_validator(self):
        inv = custom_invariant("check", lambda v: v is None, key="missing")
        assert _check(inv, {}) is True


class TestConvenienceAliases:
    def test_file_operation_limit(self):
        inv = file_operation_limit_invariant(max_ops=3)
        assert _check(inv, {"files_modified": 3}) is True
        assert _check(inv, {"files_modified": 4}) is False

    def test_file_operation_limit_custom_key(self):
        inv = file_operation_limit_invariant("deletes", max_ops=2)
        assert _check(inv, {"deletes": 1}) is True
        assert _check(inv, {"deletes": 3}) is False

    def test_api_call_limit(self):
        inv = api_call_limit_invariant(max_calls=5)
        assert _check(inv, {"api_calls": 5}) is True
        assert _check(inv, {"api_calls": 6}) is False

    def test_api_call_limit_custom_key(self):
        inv = api_call_limit_invariant("external_calls", max_calls=10)
        assert _check(inv, {"external_calls": 9}) is True


class TestEmailSafetyInvariant:
    def test_no_deleted_emails_passes(self):
        inv = email_safety_invariant()
        assert _check(inv, {"emails_deleted": 0}) is True

    def test_deleted_email_fails(self):
        inv = email_safety_invariant()
        assert _check(inv, {"emails_deleted": 1}) is False

    def test_missing_key_passes(self):
        inv = email_safety_invariant()
        assert _check(inv, {}) is True

    def test_name_is_correct(self):
        inv = email_safety_invariant()
        assert "email_safety" in inv.name

    def test_monitoring_mode(self):
        inv = email_safety_invariant(enforcement="monitoring")
        assert inv.enforcement == "monitoring"
