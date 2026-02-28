"""
Tests for the four new invariant factory functions:
  string_length_invariant, pii_guard_invariant,
  time_window_rate_invariant, json_schema_invariant.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from clampai.formal import State
from clampai.invariants import (
    json_schema_invariant,
    pii_guard_invariant,
    string_length_invariant,
    time_window_rate_invariant,
)


def _state(**kw) -> State:
    return State(dict(kw))


# ─── string_length_invariant ───────────────────────────────────────────────────


class TestStringLengthInvariant:
    def test_under_limit_passes(self):
        inv = string_length_invariant("msg", 10)
        ok, _ = inv.check(_state(msg="hello"))
        assert ok

    def test_at_limit_passes(self):
        inv = string_length_invariant("msg", 5)
        ok, _ = inv.check(_state(msg="hello"))
        assert ok

    def test_over_limit_blocks(self):
        inv = string_length_invariant("msg", 4)
        ok, _ = inv.check(_state(msg="hello"))
        assert not ok

    def test_absent_key_passes(self):
        inv = string_length_invariant("missing_key", 5)
        ok, _ = inv.check(_state())
        assert ok

    def test_none_value_passes(self):
        inv = string_length_invariant("k", 5)
        ok, _ = inv.check(_state(k=None))
        assert ok

    def test_non_string_coerced(self):
        inv = string_length_invariant("num", 3)
        # str(1234) == "1234" — 4 chars → blocked for limit 3
        ok, _ = inv.check(_state(num=1234))
        assert not ok

    def test_non_string_coerced_passes(self):
        inv = string_length_invariant("num", 5)
        ok, _ = inv.check(_state(num=99))
        assert ok

    def test_empty_string_always_passes(self):
        inv = string_length_invariant("k", 0)
        ok, _ = inv.check(_state(k=""))
        assert ok

    def test_limit_zero_nonempty_blocks(self):
        inv = string_length_invariant("k", 0)
        ok, _ = inv.check(_state(k="a"))
        assert not ok

    def test_large_limit(self):
        inv = string_length_invariant("doc", 100_000)
        ok, _ = inv.check(_state(doc="x" * 99_999))
        assert ok

    def test_default_enforcement_blocking(self):
        inv = string_length_invariant("k", 5)
        assert inv.enforcement == "blocking"

    def test_monitoring_enforcement(self):
        inv = string_length_invariant("k", 5, enforcement="monitoring")
        assert inv.enforcement == "monitoring"

    def test_name_contains_key(self):
        inv = string_length_invariant("my_field", 100)
        assert "my_field" in inv.name

    def test_suggestion_propagated(self):
        inv = string_length_invariant("k", 5, suggestion="Keep it short")
        assert "Keep it short" in (inv.suggestion or "")


# ─── pii_guard_invariant ───────────────────────────────────────────────────────


class TestPiiGuardInvariant:
    def test_clean_state_passes(self):
        inv = pii_guard_invariant("text")
        ok, _ = inv.check(_state(text="Hello, how are you?"))
        assert ok

    def test_ssn_detected(self):
        inv = pii_guard_invariant("text")
        ok, _ = inv.check(_state(text="My SSN is 123-45-6789."))
        assert not ok

    def test_credit_card_detected(self):
        inv = pii_guard_invariant("text")
        ok, _ = inv.check(_state(text="Card: 4111 1111 1111 1111"))
        assert not ok

    def test_credit_card_dashes_detected(self):
        inv = pii_guard_invariant("text")
        ok, _ = inv.check(_state(text="4111-1111-1111-1111"))
        assert not ok

    def test_email_detected(self):
        inv = pii_guard_invariant("text")
        ok, _ = inv.check(_state(text="Contact alice@example.com for help."))
        assert not ok

    def test_phone_number_detected(self):
        inv = pii_guard_invariant("text")
        ok, _ = inv.check(_state(text="Call me at 555-867-5309."))
        assert not ok

    def test_multiple_keys_any_triggers(self):
        inv = pii_guard_invariant("a", "b")
        ok, _ = inv.check(_state(a="safe text", b="My SSN is 123-45-6789"))
        assert not ok

    def test_multiple_keys_all_clean_passes(self):
        inv = pii_guard_invariant("a", "b")
        ok, _ = inv.check(_state(a="hello", b="world"))
        assert ok

    def test_absent_key_passes(self):
        inv = pii_guard_invariant("missing")
        ok, _ = inv.check(_state())
        assert ok

    def test_none_value_passes(self):
        inv = pii_guard_invariant("k")
        ok, _ = inv.check(_state(k=None))
        assert ok

    def test_non_string_coerced(self):
        inv = pii_guard_invariant("k")
        # int value 42 — no PII pattern in "42"
        ok, _ = inv.check(_state(k=42))
        assert ok

    def test_default_enforcement_blocking(self):
        inv = pii_guard_invariant("k")
        assert inv.enforcement == "blocking"

    def test_monitoring_enforcement(self):
        inv = pii_guard_invariant("k", enforcement="monitoring")
        assert inv.enforcement == "monitoring"

    def test_name_contains_key(self):
        inv = pii_guard_invariant("output")
        assert "output" in inv.name

    def test_no_keys_always_passes(self):
        inv = pii_guard_invariant()
        ok, _ = inv.check(_state(text="123-45-6789"))
        assert ok


# ─── time_window_rate_invariant ────────────────────────────────────────────────


class TestTimeWindowRateInvariant:
    def _now_timestamps(self, n: int) -> list:
        now = time.time()
        return [now - i * 0.1 for i in range(n)]

    def _old_timestamps(self, n: int, age_s: float = 3600.0) -> list:
        now = time.time()
        return [now - age_s - i for i in range(n)]

    def test_under_count_passes(self):
        inv = time_window_rate_invariant("ts", max_count=5, window_seconds=60.0)
        ok, _ = inv.check(_state(ts=self._now_timestamps(4)))
        assert ok

    def test_at_count_blocks(self):
        inv = time_window_rate_invariant("ts", max_count=5, window_seconds=60.0)
        ok, _ = inv.check(_state(ts=self._now_timestamps(5)))
        assert not ok

    def test_over_count_blocks(self):
        inv = time_window_rate_invariant("ts", max_count=3, window_seconds=60.0)
        ok, _ = inv.check(_state(ts=self._now_timestamps(10)))
        assert not ok

    def test_old_timestamps_excluded(self):
        inv = time_window_rate_invariant("ts", max_count=3, window_seconds=60.0)
        # 10 old timestamps outside window + 2 recent → only 2 recent count
        ts = self._old_timestamps(10) + self._now_timestamps(2)
        ok, _ = inv.check(_state(ts=ts))
        assert ok

    def test_absent_key_passes(self):
        inv = time_window_rate_invariant("ts", max_count=1, window_seconds=60.0)
        ok, _ = inv.check(_state())
        assert ok

    def test_non_list_value_passes(self):
        inv = time_window_rate_invariant("ts", max_count=1, window_seconds=60.0)
        ok, _ = inv.check(_state(ts="not a list"))
        assert ok

    def test_none_value_passes(self):
        inv = time_window_rate_invariant("ts", max_count=1, window_seconds=60.0)
        ok, _ = inv.check(_state(ts=None))
        assert ok

    def test_empty_list_passes(self):
        inv = time_window_rate_invariant("ts", max_count=1, window_seconds=60.0)
        ok, _ = inv.check(_state(ts=[]))
        assert ok

    def test_non_float_items_ignored(self):
        # max_count=3: 2 valid floats (non-float items discarded) → count=2 < 3 → passes
        inv = time_window_rate_invariant("ts", max_count=3, window_seconds=60.0)
        now = time.time()
        ts = [now, "bad", None, now - 1.0]
        ok, _ = inv.check(_state(ts=ts))
        assert ok  # only 2 valid floats counted → 2 < max_count=3 → passes

    def test_window_exactly_boundaries(self):
        fake_now = 1_000_000.0
        inv = time_window_rate_invariant("ts", max_count=1, window_seconds=10.0)
        ts = [fake_now - 10.0]  # exactly at boundary (now - t == window_seconds → included)
        with patch("clampai.invariants._time.time", return_value=fake_now):
            ok, _ = inv.check(_state(ts=ts))
        assert not ok  # count=1 >= max_count=1

    def test_default_enforcement_blocking(self):
        inv = time_window_rate_invariant("ts", 5, 60.0)
        assert inv.enforcement == "blocking"

    def test_monitoring_enforcement(self):
        inv = time_window_rate_invariant("ts", 5, 60.0, enforcement="monitoring")
        assert inv.enforcement == "monitoring"

    def test_name_contains_key(self):
        inv = time_window_rate_invariant("my_timestamps", 5, 60.0)
        assert "my_timestamps" in inv.name

    def test_uses_time_time(self):
        fake_now = 1_000_000.0
        inv = time_window_rate_invariant("ts", max_count=2, window_seconds=60.0)
        ts = [fake_now - 30.0, fake_now - 45.0]  # both within 60s of fake_now
        with patch("clampai.invariants._time.time", return_value=fake_now):
            ok, _ = inv.check(_state(ts=ts))
        assert not ok  # count=2 >= max_count=2


# ─── json_schema_invariant ─────────────────────────────────────────────────────


class TestJsonSchemaInvariant:
    def test_all_correct_types_passes(self):
        inv = json_schema_invariant("cfg", {"timeout": int, "url": str})
        ok, _ = inv.check(_state(cfg={"timeout": 30, "url": "http://x"}))
        assert ok

    def test_wrong_type_blocks(self):
        inv = json_schema_invariant("cfg", {"timeout": int})
        ok, _ = inv.check(_state(cfg={"timeout": "30"}))  # str not int
        assert not ok

    def test_float_wrong_for_int(self):
        inv = json_schema_invariant("cfg", {"count": int})
        ok, _ = inv.check(_state(cfg={"count": 3.0}))  # float not int
        assert not ok

    def test_missing_field_passes(self):
        inv = json_schema_invariant("cfg", {"optional_key": str})
        ok, _ = inv.check(_state(cfg={"other_key": "x"}))
        assert ok

    def test_extra_fields_ignored(self):
        inv = json_schema_invariant("cfg", {"timeout": int})
        ok, _ = inv.check(_state(cfg={"timeout": 5, "extra": "ok"}))
        assert ok

    def test_absent_key_passes(self):
        inv = json_schema_invariant("missing", {"k": str})
        ok, _ = inv.check(_state())
        assert ok

    def test_none_value_passes(self):
        inv = json_schema_invariant("cfg", {"k": str})
        ok, _ = inv.check(_state(cfg=None))
        assert ok

    def test_not_a_dict_blocks(self):
        inv = json_schema_invariant("cfg", {"k": str})
        ok, _ = inv.check(_state(cfg="not_a_dict"))
        assert not ok

    def test_not_a_dict_list_blocks(self):
        inv = json_schema_invariant("cfg", {"k": str})
        ok, _ = inv.check(_state(cfg=[1, 2, 3]))
        assert not ok

    def test_empty_schema_always_passes_for_dict(self):
        inv = json_schema_invariant("cfg", {})
        ok, _ = inv.check(_state(cfg={"any": "value"}))
        assert ok

    def test_bool_subclass_of_int(self):
        inv = json_schema_invariant("cfg", {"flag": bool})
        ok, _ = inv.check(_state(cfg={"flag": True}))
        assert ok

    def test_float_value_for_float_passes(self):
        inv = json_schema_invariant("cfg", {"price": float})
        ok, _ = inv.check(_state(cfg={"price": 9.99}))
        assert ok

    def test_nested_dict_field_type(self):
        inv = json_schema_invariant("cfg", {"meta": dict})
        ok, _ = inv.check(_state(cfg={"meta": {"k": "v"}}))
        assert ok

    def test_default_enforcement_blocking(self):
        inv = json_schema_invariant("cfg", {})
        assert inv.enforcement == "blocking"

    def test_monitoring_enforcement(self):
        inv = json_schema_invariant("cfg", {}, enforcement="monitoring")
        assert inv.enforcement == "monitoring"

    def test_name_contains_key(self):
        inv = json_schema_invariant("my_config", {})
        assert "my_config" in inv.name
