"""
clampai.adapters.fastapi_middleware — Starlette / FastAPI middleware.

Wraps a Starlette or FastAPI application with ClampAI budget and invariant
enforcement on every incoming HTTP request.  Each request is treated as an
"action" with a configurable cost.  If the budget is exhausted the middleware
returns **429 Too Many Requests**.  If a blocking invariant is violated it
returns **422 Unprocessable Entity**.  Both responses are JSON and the endpoint
handler is never called.

    from fastapi import FastAPI
    from clampai.adapters import ClampAIMiddleware
    from clampai.invariants import rate_limit_invariant, no_sensitive_substring_invariant

    app = FastAPI()
    app.add_middleware(
        ClampAIMiddleware,
        budget=1000.0,
        cost_per_request=1.0,
        invariants=[
            rate_limit_invariant("endpoint_calls", 100),
            no_sensitive_substring_invariant("path", ["/admin", "/internal"]),
        ],
    )

The state dict passed to invariants is built from request metadata:

    {
        "method":    "GET",           # HTTP verb
        "path":      "/api/items",    # URL path
        "client_ip": "127.0.0.1",     # client host (None if unavailable)
    }

Additional fields can be injected by supplying a ``state_fn``:

    app.add_middleware(
        ClampAIMiddleware,
        budget=500.0,
        state_fn=lambda req: {"user_id": req.headers.get("X-User-Id")},
    )

Safety guarantees:

- T1 (Budget Safety): budget is charged atomically; requests beyond the cap
  always receive 429 — the endpoint is never called.
- T3 (Invariant Safety): blocking invariants are checked before every request.
- T5 (Atomicity): charge + trace-append are all-or-nothing.

Requires: pip install 'clampai[fastapi]'  (starlette>=0.27 or fastapi>=0.100)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

# Starlette imports are deferred so that importing this module does not fail
# when starlette / fastapi is not installed.
try:
    from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore[import]
    _STARLETTE_AVAILABLE = True
except ImportError:
    _STARLETTE_AVAILABLE = False

    class BaseHTTPMiddleware:  # type: ignore[no-redef]
        """Stub used when starlette is not installed."""

        def __init__(self, app: Any, **kwargs: Any) -> None:
            pass


from clampai.formal import ActionSpec, Invariant, SafetyKernel, State


def _require_starlette() -> None:
    if not _STARLETTE_AVAILABLE:
        raise ImportError(
            "Starlette is required for ClampAIMiddleware. "
            "Install it with: pip install 'clampai[fastapi]'"
        )


class ClampAIMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """
    Starlette / FastAPI middleware with ClampAI budget and invariant enforcement.

    Every HTTP request is evaluated by a ``SafetyKernel`` before the endpoint
    handler is called.  Requests that exceed the budget or violate a blocking
    invariant are rejected with a JSON error response.

    Args:
        app:
            The ASGI application to wrap (passed automatically by Starlette).
        budget:
            Total budget for all requests processed by this middleware instance.
        cost_per_request:
            Budget charged per request (default 1.0).
        invariants:
            ``Invariant`` objects checked against every request's state dict.
        state_fn:
            Optional callable ``(request) -> dict`` to inject extra fields into
            the state dict checked by invariants.  Called with the Starlette
            ``Request`` object.
        budget_status_code:
            HTTP status returned when the budget is exhausted (default 429).
        invariant_status_code:
            HTTP status returned on invariant violation (default 422).

    Example::

        app.add_middleware(
            ClampAIMiddleware,
            budget=5000.0,
            cost_per_request=1.0,
            invariants=[no_delete_invariant("active_session")],
        )
    """

    def __init__(
        self,
        app: Any,
        *,
        budget: float,
        cost_per_request: float = 1.0,
        invariants: Sequence[Invariant] = (),
        state_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        budget_status_code: int = 429,
        invariant_status_code: int = 422,
    ) -> None:
        _require_starlette()
        super().__init__(app)
        self._budget = budget
        self._cost = cost_per_request
        self._invariants: List[Invariant] = list(invariants)
        self._state_fn = state_fn
        self._budget_status_code = budget_status_code
        self._invariant_status_code = invariant_status_code

        self._kernel = SafetyKernel(budget, self._invariants)
        self._action = ActionSpec(
            id="http_request",
            name="HTTP Request",
            description="Incoming HTTP request handled by ClampAIMiddleware",
            effects=(),
            cost=cost_per_request,
            reversible=False,
        )

    async def dispatch(self, request: Any, call_next: Callable) -> Any:
        """
        Evaluate the request against the safety kernel before dispatching.

        Builds a state dict from request metadata (and ``state_fn`` if provided),
        runs ``SafetyKernel.execute_atomic``, and either forwards to ``call_next``
        or returns a rejection response.
        """
        state_data: Dict[str, Any] = {
            "method": request.method,
            "path": str(request.url.path),
            "client_ip": (
                request.client.host if getattr(request, "client", None) else None
            ),
        }

        if self._state_fn is not None:
            try:
                extra = self._state_fn(request)
                if extra and isinstance(extra, dict):
                    state_data.update(extra)
            except Exception:
                pass

        clampai_state = State(state_data)

        try:
            self._kernel.evaluate_and_execute_atomic(clampai_state, self._action)
        except Exception as exc:
            return self._rejection_response(exc)

        return await call_next(request)

    def _rejection_response(self, exc: Exception) -> Any:
        """Build an appropriate JSON rejection response."""
        msg = str(exc)
        is_budget = any(
            kw in msg.lower() for kw in ("budget", "afford", "exceeded", "insufficient")
        )
        status_code = (
            self._budget_status_code if is_budget else self._invariant_status_code
        )
        error_type = "budget_exhausted" if is_budget else "invariant_violated"

        body = json.dumps({"error": error_type, "detail": msg}).encode()

        if _STARLETTE_AVAILABLE:
            from starlette.responses import Response as _StarletteResponse
            return _StarletteResponse(
                content=body,
                status_code=status_code,
                media_type="application/json",
            )

        # Minimal response object for testing without starlette installed
        class _PlainResponse:
            def __init__(
                self, body: bytes, status_code: int, media_type: str
            ) -> None:
                self.body = body
                self.status_code = status_code
                self.media_type = media_type

        return _PlainResponse(body, status_code, "application/json")

    @property
    def budget_remaining(self) -> float:
        """Budget remaining across all processed requests."""
        return self._kernel.budget.remaining

    @property
    def requests_processed(self) -> int:
        """Number of requests that have passed the safety check."""
        return self._kernel.step_count

    def reset(self) -> None:
        """
        Recreate the kernel with the original budget and invariants.

        Useful between test runs or application restarts where the budget
        should start fresh.
        """
        self._kernel = SafetyKernel(self._budget, self._invariants)

    def __repr__(self) -> str:
        return (
            f"ClampAIMiddleware(budget={self._budget}, "
            f"cost_per_request={self._cost}, "
            f"invariants={len(self._invariants)})"
        )


__all__ = [
    "ClampAIMiddleware",
]
