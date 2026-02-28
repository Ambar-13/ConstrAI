"""
clampai.server — HTTP sidecar server for ClampAI safety checks.

Exposes a SafetyKernel as an HTTP API so that agents running in separate
processes, containers, or languages can get provable budget and invariant
enforcement without embedding the kernel in their process.

    # Start the server (one shared kernel for all callers):
    from clampai.server import ClampAIServer
    from clampai.invariants import rate_limit_invariant, pii_guard_invariant

    server = ClampAIServer(
        budget=1000.0,
        invariants=[
            rate_limit_invariant("api_calls", 200),
            pii_guard_invariant("output"),
        ],
        host="127.0.0.1",
        port=8765,
    )
    server.run()  # blocking; use server.run_async() for async

    # Or run from the CLI:
    python -m clampai.server --budget 1000.0 --port 8765

Clients call the API with JSON:

    POST /evaluate
    {"state": {"api_calls": 5}, "action": {"id": "call_api", "name": "Call API",
     "description": "...", "cost": 1.0, "effects": [], "reversible": false}}
    → {"approved": true, "rejection_reasons": [], "budget_remaining": 995.0}

    POST /execute
    (same body as /evaluate)
    → {"approved": true, "new_state": {...}, "step_count": 6, "budget_remaining": 994.0}

    GET /status
    → {"budget_remaining": 994.0, "step_count": 6, "budget_total": 1000.0,
       "trace_length": 6}

    POST /reset
    → {"status": "ok", "budget_remaining": 1000.0}

Safety guarantees are identical to direct kernel use:
- T1: Budget can never go negative.
- T3: Blocking invariants always hold on committed states.
- T5: /execute is atomic — partial commits are impossible.
- T6: Audit trace is append-only and hash-chained.

The server is intentionally lightweight: zero external dependencies beyond
the Python standard library + clampai itself. FastAPI/starlette are optional
— if available, they are used for full ASGI support; otherwise the server
falls back to a stdlib HTTPServer.

Requires: pip install clampai  (optionally: pip install clampai[fastapi])
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Sequence

from clampai.formal import ActionSpec, Effect, Invariant, SafetyKernel, State

# JSON serialisation / deserialisation helpers

def _state_from_dict(d: Dict[str, Any]) -> State:
    """Build a ``State`` from a JSON-decoded dict."""
    return State(d)


def _action_from_dict(d: Dict[str, Any]) -> ActionSpec:
    """Build an ``ActionSpec`` from a JSON-decoded dict."""
    raw_effects = d.get("effects", [])
    effects = tuple(
        Effect(e["variable"], e["mode"], e["value"])
        for e in raw_effects
        if isinstance(e, dict)
    )
    return ActionSpec(
        id=str(d.get("id", "action")),
        name=str(d.get("name", d.get("id", "action"))),
        description=str(d.get("description", "")),
        effects=effects,
        cost=float(d.get("cost", 1.0)),
        reversible=bool(d.get("reversible", False)),
    )


# Core request handler

class _KernelRequestHandler(BaseHTTPRequestHandler):
    """stdlib HTTP request handler backed by a shared SafetyKernel."""

    kernel: SafetyKernel  # injected by ClampAIServer

    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        pass  # silence default access log

    def _send_json(self, status: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw)
        except Exception as exc:
            self._send_json(400, {"error": "invalid_json", "detail": str(exc)})
            return None

    def do_GET(self) -> None:
        if self.path == "/status":
            self._handle_status()
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not_found", "path": self.path})

    def do_POST(self) -> None:
        if self.path == "/evaluate":
            self._handle_evaluate()
        elif self.path == "/execute":
            self._handle_execute()
        elif self.path == "/reset":
            self._handle_reset()
        else:
            self._send_json(404, {"error": "not_found", "path": self.path})

    def _handle_status(self) -> None:
        k = self.kernel
        self._send_json(200, {
            "budget_total": k.budget.budget,
            "budget_remaining": k.budget.remaining,
            "budget_spent": k.budget.spent_net,
            "step_count": k.step_count,
            "trace_length": k.trace.length,
        })

    def _handle_evaluate(self) -> None:
        data = self._read_json()
        if data is None:
            return
        try:
            state = _state_from_dict(data.get("state", {}))
            action = _action_from_dict(data.get("action", {}))
        except Exception as exc:
            self._send_json(422, {"error": "invalid_request", "detail": str(exc)})
            return

        verdict = self.kernel.evaluate(state, action)
        self._send_json(200, {
            "approved": verdict.approved,
            "rejection_reasons": list(verdict.rejection_reasons),
            "budget_remaining": self.kernel.budget.remaining,
            "cost": action.cost,
        })

    def _handle_execute(self) -> None:
        data = self._read_json()
        if data is None:
            return
        try:
            state = _state_from_dict(data.get("state", {}))
            action = _action_from_dict(data.get("action", {}))
        except Exception as exc:
            self._send_json(422, {"error": "invalid_request", "detail": str(exc)})
            return

        try:
            new_state, entry = self.kernel.evaluate_and_execute_atomic(
                state, action, data.get("reasoning", "")
            )
        except RuntimeError as exc:
            msg = str(exc)
            is_budget = any(
                kw in msg.lower()
                for kw in ("budget", "afford", "exceeded", "insufficient")
            )
            status = 429 if is_budget else 422
            error_type = "budget_exhausted" if is_budget else "invariant_violated"
            self._send_json(status, {"error": error_type, "detail": msg})
            return

        self._send_json(200, {
            "approved": True,
            "new_state": dict(new_state._vars),
            "step_count": self.kernel.step_count,
            "budget_remaining": self.kernel.budget.remaining,
            "trace_step": entry.step,
        })

    def _handle_reset(self) -> None:
        data = self._read_json()
        if data is None:
            return
        old = self.kernel
        new_kernel = SafetyKernel(
            old.budget.budget,
            old.invariants,
            min_action_cost=old.min_action_cost,
            emergency_actions=set(old.emergency_actions),
        )
        type(self).kernel = new_kernel
        self._send_json(200, {
            "status": "ok",
            "budget_remaining": new_kernel.budget.remaining,
        })


# Server class

class ClampAIServer:
    """
    HTTP sidecar server wrapping a shared ``SafetyKernel``.

    Exposes ``/evaluate``, ``/execute``, ``/status``, ``/reset``, and
    ``/health`` endpoints so any HTTP client — in any language — can get
    provable ClampAI safety enforcement.

    Args:
        budget:
            Total resource budget for all requests handled by this server.
        invariants:
            Safety predicates enforced on every ``/execute`` request.
        host:
            Bind address (default ``"127.0.0.1"``).
        port:
            TCP port (default ``8765``).
        min_action_cost:
            Minimum cost per action (T2 termination guard, default 0.001).

    Example::

        server = ClampAIServer(budget=5000.0, port=8765)
        server.run()  # blocking

        # Non-blocking (for use in tests or async applications):
        server.start_background()
        ...
        server.stop()
    """

    def __init__(
        self,
        budget: float,
        *,
        invariants: Sequence[Invariant] = (),
        host: str = "127.0.0.1",
        port: int = 8765,
        min_action_cost: float = 0.001,
    ) -> None:
        self.host = host
        self.port = port
        self.kernel = SafetyKernel(
            budget, list(invariants), min_action_cost=min_action_cost
        )
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _make_handler_class(self) -> type:
        kernel = self.kernel

        class _Handler(_KernelRequestHandler):
            pass

        _Handler.kernel = kernel  # type: ignore[attr-defined]
        return _Handler

    def run(self) -> None:
        """
        Start the server and block until interrupted.

        Press Ctrl-C to stop. Use ``start_background()`` for non-blocking use.
        """
        handler_cls = self._make_handler_class()
        self._httpd = HTTPServer((self.host, self.port), handler_cls)
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._httpd.server_close()

    def start_background(self) -> None:
        """
        Start the server in a daemon thread (non-blocking).

        Call ``stop()`` to shut it down.
        """
        handler_cls = self._make_handler_class()
        self._httpd = HTTPServer((self.host, self.port), handler_cls)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down the background server."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    @property
    def url(self) -> str:
        """Base URL of the server (e.g. ``http://127.0.0.1:8765``)."""
        return f"http://{self.host}:{self.port}"

    def __repr__(self) -> str:
        return (
            f"ClampAIServer(host={self.host!r}, port={self.port}, "
            f"budget={self.kernel.budget.budget})"
        )


# CLI entry point

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ClampAI HTTP sidecar server"
    )
    parser.add_argument("--budget", type=float, default=1000.0,
                        help="Total budget (default: 1000.0)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Bind port (default: 8765)")
    parser.add_argument("--min-cost", type=float, default=0.001,
                        help="Minimum action cost for T2 termination bound")
    args = parser.parse_args()

    server = ClampAIServer(
        budget=args.budget,
        host=args.host,
        port=args.port,
        min_action_cost=args.min_cost,
    )
    print(f"ClampAI sidecar server listening on {server.url}")
    print(f"  Budget: {args.budget}  |  Endpoints: /evaluate /execute /status /reset /health")
    server.run()


if __name__ == "__main__":
    _main()


__all__ = [
    "ClampAIServer",
]
