"""
Tests for clampai.server.

Covers:
  - _state_from_dict: empty dict, int, str, nested, .get() access
  - _action_from_dict: minimal dict, default cost, default reversible, effects
    parsing, missing id defaults, name fallback to id, empty description
  - ClampAIServer init: url property, repr, kernel budget, stop on unstarted,
    start_background + stop round-trip
  - GET /health: 200 + {"status": "ok"}
  - GET /status: 200 + correct fields
  - POST /evaluate: approved=True, budget_remaining present, invariant blocks,
    invalid JSON 400, missing action fields use defaults
  - POST /execute: approved + new_state + step_count + budget_remaining,
    budget decrements, step_count increments, budget exhaustion 429,
    invariant violation 422, invalid JSON 400
  - GET|POST /unknown: 404
  - POST /reset: budget_remaining restored, status="ok", step_count resets
"""

from __future__ import annotations

import http.client
import json
import socket
import time
from typing import Any, Dict, List, Optional

import pytest

from clampai.formal import Invariant
from clampai.server import ClampAIServer, _action_from_dict, _state_from_dict


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(host: str, port: int, path: str, body: Any) -> tuple[int, Dict[str, Any]]:
    payload = json.dumps(body).encode()
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request(
        "POST",
        path,
        body=payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
    )
    resp = conn.getresponse()
    status = resp.status
    data = json.loads(resp.read())
    conn.close()
    return status, data


def _get(host: str, port: int, path: str) -> tuple[int, Dict[str, Any]]:
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    status = resp.status
    data = json.loads(resp.read())
    conn.close()
    return status, data


def _make_action_body(
    *,
    action_id: str = "test_action",
    name: str = "Test Action",
    description: str = "A test action",
    cost: float = 1.0,
    reversible: bool = False,
    effects: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": action_id,
        "name": name,
        "description": description,
        "cost": cost,
        "reversible": reversible,
        "effects": effects or [],
    }


def _blocking_invariant(name: str, pred) -> Invariant:
    return Invariant(name, pred, f"{name} invariant", enforcement="blocking")


class TestStateParsing:
    def test_empty_dict(self):
        state = _state_from_dict({})
        assert state.get("anything") is None

    def test_int_value(self):
        state = _state_from_dict({"count": 5})
        assert state.get("count") == 5

    def test_str_value(self):
        state = _state_from_dict({"msg": "hello"})
        assert state.get("msg") == "hello"

    def test_nested_dict(self):
        state = _state_from_dict({"nested": {"a": 1}})
        val = state.get("nested")
        assert val == {"a": 1}

    def test_get_returns_none_for_missing_key(self):
        state = _state_from_dict({"x": 1})
        assert state.get("y") is None


class TestActionParsing:
    def test_minimal_dict_parses(self):
        action = _action_from_dict({"id": "do_thing"})
        assert action.id == "do_thing"

    def test_default_cost_is_one(self):
        action = _action_from_dict({"id": "do_thing"})
        assert action.cost == pytest.approx(1.0)

    def test_default_reversible_is_false(self):
        action = _action_from_dict({"id": "do_thing"})
        assert action.reversible is False

    def test_effects_list_parsed(self):
        raw = {
            "id": "do_thing",
            "effects": [{"variable": "x", "mode": "set", "value": 10}],
        }
        action = _action_from_dict(raw)
        assert len(action.effects) == 1
        assert action.effects[0].variable == "x"
        assert action.effects[0].value == 10

    def test_empty_effects_list(self):
        action = _action_from_dict({"id": "do_thing", "effects": []})
        assert action.effects == ()

    def test_missing_id_defaults_to_action(self):
        action = _action_from_dict({})
        assert action.id == "action"

    def test_name_falls_back_to_id_when_missing(self):
        action = _action_from_dict({"id": "fallback_name"})
        assert action.name == "fallback_name"

    def test_description_defaults_to_empty_string(self):
        action = _action_from_dict({"id": "do_thing"})
        assert action.description == ""


class TestClampAIServerInit:
    def test_url_property(self):
        server = ClampAIServer(100.0, host="127.0.0.1", port=19001)
        assert server.url == "http://127.0.0.1:19001"

    def test_repr_contains_host_port_budget(self):
        server = ClampAIServer(250.0, host="127.0.0.1", port=19002)
        r = repr(server)
        assert "127.0.0.1" in r
        assert "19002" in r
        assert "250.0" in r

    def test_kernel_budget_matches(self):
        server = ClampAIServer(500.0, port=19003)
        assert server.kernel.budget.budget == pytest.approx(500.0)

    def test_stop_on_unstarted_server_is_safe(self):
        server = ClampAIServer(100.0, port=19004)
        server.stop()

    def test_start_background_and_stop_round_trip(self):
        port = _free_port()
        server = ClampAIServer(100.0, port=port)
        server.start_background()
        time.sleep(0.05)
        status, data = _get("127.0.0.1", port, "/health")
        server.stop()
        assert status == 200
        assert data.get("status") == "ok"


class TestHealthEndpoint:
    @pytest.fixture(autouse=True)
    def server(self):
        port = _free_port()
        s = ClampAIServer(100.0, port=port)
        s.start_background()
        time.sleep(0.05)
        self._host = "127.0.0.1"
        self._port = port
        yield s
        s.stop()

    def test_health_returns_200(self):
        status, _ = _get(self._host, self._port, "/health")
        assert status == 200

    def test_health_body_has_status_ok(self):
        _, data = _get(self._host, self._port, "/health")
        assert data == {"status": "ok"}


class TestStatusEndpoint:
    @pytest.fixture(autouse=True)
    def server(self):
        port = _free_port()
        s = ClampAIServer(200.0, port=port)
        s.start_background()
        time.sleep(0.05)
        self._host = "127.0.0.1"
        self._port = port
        yield s
        s.stop()

    def test_status_returns_200(self):
        status, _ = _get(self._host, self._port, "/status")
        assert status == 200

    def test_status_has_budget_total(self):
        _, data = _get(self._host, self._port, "/status")
        assert data["budget_total"] == pytest.approx(200.0)

    def test_status_has_budget_remaining(self):
        _, data = _get(self._host, self._port, "/status")
        assert data["budget_remaining"] == pytest.approx(200.0)

    def test_status_has_budget_spent(self):
        _, data = _get(self._host, self._port, "/status")
        assert data["budget_spent"] == pytest.approx(0.0)

    def test_status_has_step_count_and_trace_length(self):
        _, data = _get(self._host, self._port, "/status")
        assert "step_count" in data
        assert "trace_length" in data
        assert data["step_count"] == 0
        assert data["trace_length"] == 0


class TestEvaluateEndpoint:
    @pytest.fixture(autouse=True)
    def server(self):
        port = _free_port()
        never_pass = _blocking_invariant("never_pass_eval", lambda s: s.get("allow") is True)
        s = ClampAIServer(100.0, invariants=[never_pass], port=port)
        s.start_background()
        time.sleep(0.05)
        self._host = "127.0.0.1"
        self._port = port
        yield s
        s.stop()

    def test_valid_action_with_passing_invariant_returns_approved_true(self):
        body = {
            "state": {"allow": True},
            "action": _make_action_body(),
        }
        status, data = _post(self._host, self._port, "/evaluate", body)
        assert status == 200
        assert data["approved"] is True

    def test_response_contains_budget_remaining(self):
        body = {
            "state": {"allow": True},
            "action": _make_action_body(cost=5.0),
        }
        _, data = _post(self._host, self._port, "/evaluate", body)
        assert "budget_remaining" in data
        assert data["budget_remaining"] == pytest.approx(100.0)

    def test_invariant_blocks_returns_approved_false(self):
        body = {
            "state": {"allow": False},
            "action": _make_action_body(),
        }
        status, data = _post(self._host, self._port, "/evaluate", body)
        assert status == 200
        assert data["approved"] is False

    def test_invalid_json_returns_400(self):
        conn = http.client.HTTPConnection(self._host, self._port, timeout=5)
        payload = b"this is not json {"
        conn.request(
            "POST",
            "/evaluate",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()

    def test_missing_action_fields_use_defaults(self):
        body = {
            "state": {"allow": True},
            "action": {},
        }
        status, data = _post(self._host, self._port, "/evaluate", body)
        assert status == 200
        assert "approved" in data


class TestExecuteEndpoint:
    @pytest.fixture(autouse=True)
    def server(self):
        port = _free_port()
        s = ClampAIServer(10.0, port=port)
        s.start_background()
        time.sleep(0.05)
        self._host = "127.0.0.1"
        self._port = port
        yield s
        s.stop()

    def test_valid_action_returns_approved_true(self):
        body = {"state": {}, "action": _make_action_body(cost=1.0)}
        status, data = _post(self._host, self._port, "/execute", body)
        assert status == 200
        assert data["approved"] is True

    def test_response_contains_new_state(self):
        body = {"state": {"x": 42}, "action": _make_action_body(cost=1.0)}
        _, data = _post(self._host, self._port, "/execute", body)
        assert "new_state" in data

    def test_response_contains_step_count(self):
        body = {"state": {}, "action": _make_action_body(cost=1.0)}
        _, data = _post(self._host, self._port, "/execute", body)
        assert "step_count" in data
        assert data["step_count"] == 1

    def test_response_contains_budget_remaining(self):
        body = {"state": {}, "action": _make_action_body(cost=3.0)}
        _, data = _post(self._host, self._port, "/execute", body)
        assert "budget_remaining" in data
        assert data["budget_remaining"] == pytest.approx(7.0)

    def test_budget_decrements(self):
        body = {"state": {}, "action": _make_action_body(cost=2.0)}
        _post(self._host, self._port, "/execute", body)
        _post(self._host, self._port, "/execute", body)
        _, status_data = _get(self._host, self._port, "/status")
        assert status_data["budget_remaining"] == pytest.approx(6.0)

    def test_step_count_increments(self):
        body = {"state": {}, "action": _make_action_body(cost=1.0)}
        _post(self._host, self._port, "/execute", body)
        _post(self._host, self._port, "/execute", body)
        _, status_data = _get(self._host, self._port, "/status")
        assert status_data["step_count"] == 2

    def test_budget_exhaustion_returns_429(self):
        body = {"state": {}, "action": _make_action_body(cost=4.0)}
        _post(self._host, self._port, "/execute", body)
        _post(self._host, self._port, "/execute", body)
        status, data = _post(self._host, self._port, "/execute", body)
        assert status == 429
        assert data.get("error") == "budget_exhausted"

    def test_invariant_violation_returns_422(self):
        port = _free_port()
        inv = _blocking_invariant("no_dangerous", lambda s: s.get("dangerous") is not True)
        s = ClampAIServer(100.0, invariants=[inv], port=port)
        s.start_background()
        time.sleep(0.05)
        try:
            body = {"state": {"dangerous": True}, "action": _make_action_body(cost=1.0)}
            status, data = _post("127.0.0.1", port, "/execute", body)
            assert status == 422
            assert data.get("error") == "invariant_violated"
        finally:
            s.stop()

    def test_invalid_json_returns_400(self):
        conn = http.client.HTTPConnection(self._host, self._port, timeout=5)
        payload = b"not valid json }{{"
        conn.request(
            "POST",
            "/execute",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()


class TestResetEndpoint:
    @pytest.fixture(autouse=True)
    def server(self):
        port = _free_port()
        s = ClampAIServer(10.0, port=port)
        s.start_background()
        time.sleep(0.05)
        self._host = "127.0.0.1"
        self._port = port
        yield s
        s.stop()

    def test_reset_restores_budget_remaining(self):
        body = {"state": {}, "action": _make_action_body(cost=5.0)}
        _post(self._host, self._port, "/execute", body)
        _, before = _get(self._host, self._port, "/status")
        assert before["budget_remaining"] == pytest.approx(5.0)

        _post(self._host, self._port, "/reset", {})
        _, after = _get(self._host, self._port, "/status")
        assert after["budget_remaining"] == pytest.approx(10.0)

    def test_reset_returns_status_ok(self):
        _, data = _post(self._host, self._port, "/reset", {})
        assert data.get("status") == "ok"

    def test_reset_restores_step_count(self):
        body = {"state": {}, "action": _make_action_body(cost=1.0)}
        _post(self._host, self._port, "/execute", body)
        _post(self._host, self._port, "/execute", body)
        _, before = _get(self._host, self._port, "/status")
        assert before["step_count"] == 2

        _post(self._host, self._port, "/reset", {})
        _, after = _get(self._host, self._port, "/status")
        assert after["step_count"] == 0


class TestUnknownRoutes:
    @pytest.fixture(autouse=True)
    def server(self):
        port = _free_port()
        s = ClampAIServer(100.0, port=port)
        s.start_background()
        time.sleep(0.05)
        self._host = "127.0.0.1"
        self._port = port
        yield s
        s.stop()

    def test_get_unknown_returns_404(self):
        status, data = _get(self._host, self._port, "/unknown")
        assert status == 404
        assert data.get("error") == "not_found"

    def test_post_unknown_returns_404(self):
        status, data = _post(self._host, self._port, "/unknown", {})
        assert status == 404
        assert data.get("error") == "not_found"
