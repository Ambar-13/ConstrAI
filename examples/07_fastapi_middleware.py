"""
examples/07_fastapi_middleware.py — ClampAI FastAPI/Starlette middleware.

Demonstrates ClampAIMiddleware: per-request budget enforcement and invariant
checking on a FastAPI application. Requests that exhaust the budget receive
HTTP 429; invariant violations receive HTTP 422.

Run with:
    pip install clampai[fastapi] uvicorn
    uvicorn examples.07_fastapi_middleware:app --reload

Or run the built-in demo (no uvicorn needed):
    python examples/07_fastapi_middleware.py

Endpoints available:
  GET /search?q=<query>     — lightweight search (costs 1.0)
  GET /llm?prompt=<text>    — expensive LLM call (costs 5.0)
  GET /status               — current budget and request count
  POST /reset               — reset the middleware budget (for testing)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

from clampai.adapters.fastapi_middleware import ClampAIMiddleware
from clampai.invariants import no_sensitive_substring_invariant, rate_limit_invariant

# Try to import FastAPI; fall back to a minimal demo runner if not installed.

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


def build_app() -> Any:
    """Build and return a configured FastAPI application."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for this example. "
            "Install it with: pip install clampai[fastapi]"
        )

    app = FastAPI(title="ClampAI FastAPI Example", version="1.0.1")

    # Add ClampAI middleware: 50 budget units shared across all requests.
    # /admin and /internal paths are blocked by invariant.
    app.add_middleware(
        ClampAIMiddleware,
        budget=50.0,
        cost_per_request=1.0,
        invariants=[
            rate_limit_invariant("requests_served", 40),
            no_sensitive_substring_invariant("path", ["/admin", "/internal", "/debug"]),
        ],
        state_fn=lambda req: {"requests_served": 0},  # demo: real app queries DB
    )

    @app.get("/search")
    async def search(q: str = "default") -> dict:
        """Search endpoint — costs 1 budget unit."""
        return {"results": [f"Result for: {q}"], "query": q}

    @app.get("/llm")
    async def llm_call(prompt: str = "hello") -> dict:
        """Simulate an LLM call — in production this costs more per request."""
        return {"response": f"LLM response to: {prompt}", "tokens": 150}

    @app.get("/status")
    async def status(request: Request) -> dict:
        """Show current budget status."""
        mw = _get_middleware(app)
        if mw:
            return {
                "budget_remaining": mw.budget_remaining,
                "requests_processed": mw.requests_processed,
            }
        return {"error": "middleware not found"}

    @app.post("/reset")
    async def reset(request: Request) -> dict:
        """Reset the budget (useful between test runs)."""
        mw = _get_middleware(app)
        if mw:
            mw.reset()
            return {"status": "ok", "budget_remaining": mw.budget_remaining}
        return {"error": "middleware not found"}

    return app


def _get_middleware(app: Any) -> Any:
    """Extract the ClampAIMiddleware instance from a FastAPI app."""
    stack = app.middleware_stack
    while stack is not None:
        if isinstance(stack, ClampAIMiddleware):
            return stack
        stack = getattr(stack, "app", None)
    return None


# Module-level app for uvicorn
if _FASTAPI_AVAILABLE:
    app = build_app()


# Standalone demo (no uvicorn required)

def _make_mock_request(method: str = "GET", path: str = "/") -> Any:
    """Create a minimal mock Starlette request for the demo."""
    req = MagicMock()
    req.method = method
    req.url = MagicMock()
    req.url.path = path
    req.client = MagicMock()
    req.client.host = "127.0.0.1"
    req.headers = {}
    return req


async def _ok_handler(request: Any) -> Any:
    resp = MagicMock()
    resp.status_code = 200
    return resp


async def run_demo() -> None:
    """Simulate middleware behaviour without a running server."""
    print("ClampAI FastAPI Middleware — standalone demo")
    print("=" * 50)

    from clampai.formal import ActionSpec, SafetyKernel

    # Manually construct middleware (bypassing BaseHTTPMiddleware.__init__)
    mw: ClampAIMiddleware = ClampAIMiddleware.__new__(ClampAIMiddleware)
    mw._budget = 5.0
    mw._cost = 1.0
    mw._invariants = [
        no_sensitive_substring_invariant("path", ["/admin", "/internal"]),
    ]
    mw._state_fn = None
    mw._budget_status_code = 429
    mw._invariant_status_code = 422
    mw._kernel = SafetyKernel(5.0, mw._invariants)
    mw._action = ActionSpec(
        id="http_request",
        name="HTTP Request",
        description="Demo request",
        effects=(),
        cost=1.0,
        reversible=False,
    )

    print("\n1. Normal requests within budget:")
    for i in range(1, 6):
        req = _make_mock_request(path=f"/search?q={i}")
        resp = await mw.dispatch(req, _ok_handler)
        status = resp.status_code
        remaining = mw.budget_remaining
        print(f"   Request {i}: HTTP {status}  |  budget_remaining={remaining:.1f}")

    print("\n2. Request after budget exhausted:")
    req = _make_mock_request(path="/search?q=overflow")
    resp = await mw.dispatch(req, _ok_handler)
    body = json.loads(resp.body)
    print(f"   HTTP {resp.status_code}: {body}")

    print("\n3. Invariant violation (blocked path):")
    mw.reset()
    req = _make_mock_request(path="/admin/users")
    resp = await mw.dispatch(req, _ok_handler)
    body = json.loads(resp.body)
    print(f"   HTTP {resp.status_code}: {body}")

    print("\n4. After reset, budget is restored:")
    mw.reset()
    req = _make_mock_request(path="/search?q=after_reset")
    resp = await mw.dispatch(req, _ok_handler)
    print(f"   HTTP {resp.status_code}  |  budget_remaining={mw.budget_remaining:.1f}")

    print("\n\nTo run as a real FastAPI server:")
    print("  pip install clampai[fastapi] uvicorn")
    print("  uvicorn examples.07_fastapi_middleware:app --reload")
    print("  curl http://127.0.0.1:8000/search?q=hello")
    print("  curl http://127.0.0.1:8000/status")


if __name__ == "__main__":
    asyncio.run(run_demo())
