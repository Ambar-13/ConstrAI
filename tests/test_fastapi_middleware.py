"""Tests for clampai.adapters.fastapi_middleware.

Tests the ClampAIMiddleware budget enforcement, invariant violation, state_fn,
error response format, and reset behaviour without requiring starlette/fastapi
to be installed as a hard dependency.

The middleware dispatch() is tested by constructing the object directly (bypassing
BaseHTTPMiddleware.__init__) and calling dispatch() as a plain coroutine.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from clampai.adapters.fastapi_middleware import ClampAIMiddleware
from clampai.formal import ActionSpec, Invariant, SafetyKernel
from clampai.invariants import (
    no_delete_invariant,
    no_sensitive_substring_invariant,
    rate_limit_invariant,
    value_range_invariant,
)

# Helpers — mock Starlette objects

def _make_request(
    method: str = "GET",
    path: str = "/",
    client_ip: str = "127.0.0.1",
    headers: Optional[Dict[str, str]] = None,
) -> MagicMock:
    req = MagicMock()
    req.method = method
    req.url = MagicMock()
    req.url.path = path
    req.client = MagicMock()
    req.client.host = client_ip
    req.headers = headers or {}
    return req


def _make_request_no_client(path: str = "/") -> MagicMock:
    req = MagicMock()
    req.method = "GET"
    req.url = MagicMock()
    req.url.path = path
    req.client = None
    req.headers = {}
    return req


async def _ok_handler(request: Any) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    return resp


def _run(coro: Any) -> Any:
    """Run a coroutine using asyncio.run (Python 3.7+)."""
    return asyncio.run(coro)


def _make_middleware(
    budget: float = 100.0,
    cost_per_request: float = 1.0,
    invariants: Any = (),
    state_fn: Any = None,
) -> ClampAIMiddleware:
    """
    Construct ClampAIMiddleware directly, bypassing BaseHTTPMiddleware.__init__.

    No starlette install required — dispatch() uses only clampai.formal internally,
    and _rejection_response falls back to a plain response object when starlette
    is absent.
    """
    mw: ClampAIMiddleware = ClampAIMiddleware.__new__(ClampAIMiddleware)
    mw._budget = budget
    mw._cost = cost_per_request
    mw._invariants = list(invariants)
    mw._state_fn = state_fn
    mw._budget_status_code = 429
    mw._invariant_status_code = 422
    mw._kernel = SafetyKernel(budget, list(invariants))
    mw._action = ActionSpec(
        id="http_request",
        name="HTTP Request",
        description="Test request",
        effects=(),
        cost=cost_per_request,
        reversible=False,
    )
    return mw


# TestClampAIMiddlewareInit

class TestClampAIMiddlewareInit:
    def test_require_starlette_raises_without_library(self) -> None:
        with patch(
            "clampai.adapters.fastapi_middleware._STARLETTE_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="clampai\\[fastapi\\]"):
                ClampAIMiddleware(MagicMock(), budget=10.0)

    def test_repr_contains_budget(self) -> None:
        mw = _make_middleware(budget=42.0)
        assert "42" in repr(mw)

    def test_repr_contains_cost(self) -> None:
        mw = _make_middleware(cost_per_request=3.0)
        assert "3.0" in repr(mw)

    def test_repr_contains_invariants_count(self) -> None:
        mw = _make_middleware(invariants=[no_delete_invariant("k")])
        assert "1" in repr(mw)

    def test_budget_remaining_initial(self) -> None:
        mw = _make_middleware(budget=99.0)
        assert mw.budget_remaining == pytest.approx(99.0, abs=0.01)

    def test_requests_processed_initial(self) -> None:
        mw = _make_middleware()
        assert mw.requests_processed == 0


# TestClampAIMiddlewareAllowed

class TestClampAIMiddlewareAllowed:
    def test_passes_through_to_handler(self) -> None:
        mw = _make_middleware(budget=10.0)
        request = _make_request()
        response = _run(mw.dispatch(request, _ok_handler))
        assert response.status_code == 200

    def test_budget_decreases_after_request(self) -> None:
        mw = _make_middleware(budget=10.0, cost_per_request=3.0)
        request = _make_request()
        _run(mw.dispatch(request, _ok_handler))
        assert mw.budget_remaining == pytest.approx(7.0, abs=0.01)

    def test_step_count_increments(self) -> None:
        mw = _make_middleware(budget=10.0)
        _run(mw.dispatch(_make_request(), _ok_handler))
        _run(mw.dispatch(_make_request(), _ok_handler))
        assert mw.requests_processed == 2

    def test_multiple_requests_within_budget(self) -> None:
        mw = _make_middleware(budget=5.0, cost_per_request=1.0)
        for _ in range(5):
            resp = _run(mw.dispatch(_make_request(), _ok_handler))
            assert resp.status_code == 200

    def test_request_with_no_client(self) -> None:
        mw = _make_middleware(budget=10.0)
        request = _make_request_no_client()
        response = _run(mw.dispatch(request, _ok_handler))
        assert response.status_code == 200

    def test_post_request_passes(self) -> None:
        mw = _make_middleware(budget=10.0)
        request = _make_request(method="POST", path="/api/data")
        response = _run(mw.dispatch(request, _ok_handler))
        assert response.status_code == 200


# TestClampAIMiddlewareBudgetExhausted

class TestClampAIMiddlewareBudgetExhausted:
    def _exhaust(self, mw: ClampAIMiddleware, n: int) -> None:
        for _ in range(n):
            _run(mw.dispatch(_make_request(), _ok_handler))

    def test_returns_429_when_exhausted(self) -> None:
        mw = _make_middleware(budget=2.0, cost_per_request=1.0)
        self._exhaust(mw, 2)
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp.status_code == 429

    def test_handler_not_called_when_budget_exhausted(self) -> None:
        mw = _make_middleware(budget=1.0, cost_per_request=1.0)
        self._exhaust(mw, 1)
        called: list[int] = []

        async def tracking_handler(req: Any) -> MagicMock:
            called.append(1)
            return MagicMock(status_code=200)

        _run(mw.dispatch(_make_request(), tracking_handler))
        assert called == []

    def test_rejection_response_is_json(self) -> None:
        mw = _make_middleware(budget=1.0, cost_per_request=1.0)
        self._exhaust(mw, 1)
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        body = json.loads(resp.body)
        assert "error" in body

    def test_custom_budget_status_code(self) -> None:
        mw = _make_middleware(budget=1.0)
        mw._budget_status_code = 503
        self._exhaust(mw, 1)
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp.status_code == 503

    def test_step_count_not_incremented_on_rejection(self) -> None:
        mw = _make_middleware(budget=1.0, cost_per_request=1.0)
        self._exhaust(mw, 1)
        count_before = mw.requests_processed
        _run(mw.dispatch(_make_request(), _ok_handler))
        assert mw.requests_processed == count_before


# TestClampAIMiddlewareInvariantViolation

class TestClampAIMiddlewareInvariantViolation:
    def test_returns_422_on_invariant_violation(self) -> None:
        inv = no_sensitive_substring_invariant("path", ["/forbidden"])
        mw = _make_middleware(budget=100.0, invariants=[inv])
        request = _make_request(path="/forbidden")
        resp = _run(mw.dispatch(request, _ok_handler))
        assert resp.status_code == 422

    def test_passes_on_invariant_satisfaction(self) -> None:
        inv = no_sensitive_substring_invariant("path", ["/forbidden"])
        mw = _make_middleware(budget=100.0, invariants=[inv])
        request = _make_request(path="/allowed")
        resp = _run(mw.dispatch(request, _ok_handler))
        assert resp.status_code == 200

    def test_violation_response_is_json(self) -> None:
        inv = no_sensitive_substring_invariant("path", ["/secret"])
        mw = _make_middleware(budget=100.0, invariants=[inv])
        request = _make_request(path="/secret/data")
        resp = _run(mw.dispatch(request, _ok_handler))
        body = json.loads(resp.body)
        assert "error" in body

    def test_custom_invariant_status_code(self) -> None:
        inv = Invariant("always_fail", lambda s: False, "always fails")
        mw = _make_middleware(budget=100.0, invariants=[inv])
        mw._invariant_status_code = 403
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp.status_code == 403

    def test_monitoring_invariant_does_not_block(self) -> None:
        inv = Invariant(
            "mon_inv",
            lambda s: False,
            "always fails",
            enforcement="monitoring",
        )
        mw = _make_middleware(budget=100.0, invariants=[inv])
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp is not None


# TestClampAIMiddlewareStateFn

class TestClampAIMiddlewareStateFn:
    def test_state_fn_injects_extra_fields(self) -> None:
        def my_state_fn(req: Any) -> Dict[str, Any]:
            return {"user_id": "alice", "user_calls": 0}

        inv = value_range_invariant("user_calls", 0, 100)
        mw = _make_middleware(budget=10.0, state_fn=my_state_fn, invariants=[inv])
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp.status_code == 200

    def test_state_fn_exception_is_silenced(self) -> None:
        def bad_state_fn(req: Any) -> Dict[str, Any]:
            raise ValueError("state_fn exploded")

        mw = _make_middleware(budget=10.0, state_fn=bad_state_fn)
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp.status_code == 200

    def test_state_fn_returning_none_is_safe(self) -> None:
        mw = _make_middleware(budget=10.0, state_fn=lambda req: None)
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp.status_code == 200

    def test_state_fn_returning_non_dict_is_safe(self) -> None:
        mw = _make_middleware(budget=10.0, state_fn=lambda req: "not a dict")
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp.status_code == 200

    def test_no_state_fn_uses_default_method_path_ip(self) -> None:
        inv = no_sensitive_substring_invariant("path", ["/blocked"])
        mw = _make_middleware(budget=10.0, invariants=[inv])
        resp = _run(mw.dispatch(_make_request(path="/ok"), _ok_handler))
        assert resp.status_code == 200


# TestClampAIMiddlewareReset

class TestClampAIMiddlewareReset:
    def _exhaust(self, mw: ClampAIMiddleware, n: int) -> None:
        for _ in range(n):
            _run(mw.dispatch(_make_request(), _ok_handler))

    def test_reset_restores_budget(self) -> None:
        mw = _make_middleware(budget=2.0, cost_per_request=1.0)
        self._exhaust(mw, 2)
        assert mw.budget_remaining == pytest.approx(0.0, abs=0.01)
        mw.reset()
        assert mw.budget_remaining == pytest.approx(2.0, abs=0.01)

    def test_reset_restores_step_count(self) -> None:
        mw = _make_middleware(budget=10.0)
        self._exhaust(mw, 3)
        mw.reset()
        assert mw.requests_processed == 0

    def test_requests_succeed_after_reset(self) -> None:
        mw = _make_middleware(budget=1.0, cost_per_request=1.0)
        self._exhaust(mw, 1)
        mw.reset()
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp.status_code == 200

    def test_invariants_still_enforced_after_reset(self) -> None:
        inv = no_delete_invariant("required")
        mw = _make_middleware(budget=10.0, invariants=[inv])
        mw.reset()
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        assert resp.status_code == 422


# TestClampAIMiddlewareErrorResponseFormat

class TestClampAIMiddlewareErrorResponseFormat:
    def _get_rejection(
        self,
        budget: float = 1.0,
        n_exhaust: int = 1,
        invariants: Any = (),
    ) -> Any:
        mw = _make_middleware(budget=budget, invariants=list(invariants))
        for _ in range(n_exhaust):
            _run(mw.dispatch(_make_request(), _ok_handler))
        return _run(mw.dispatch(_make_request(), _ok_handler))

    def test_budget_error_body_has_error_key(self) -> None:
        resp = self._get_rejection(budget=1.0, n_exhaust=1)
        body = json.loads(resp.body)
        assert "error" in body

    def test_budget_error_body_has_detail_key(self) -> None:
        resp = self._get_rejection(budget=1.0, n_exhaust=1)
        body = json.loads(resp.body)
        assert "detail" in body

    def test_budget_error_type_is_budget_exhausted(self) -> None:
        resp = self._get_rejection(budget=1.0, n_exhaust=1)
        body = json.loads(resp.body)
        assert body["error"] == "budget_exhausted"

    def test_invariant_error_type(self) -> None:
        inv = Invariant("always_fail", lambda s: False, "always fails")
        mw = _make_middleware(budget=100.0, invariants=[inv])
        resp = _run(mw.dispatch(_make_request(), _ok_handler))
        body = json.loads(resp.body)
        assert body["error"] == "invariant_violated"

    def test_response_media_type_is_json(self) -> None:
        resp = self._get_rejection(budget=1.0, n_exhaust=1)
        assert "json" in resp.media_type.lower()
