"""
clampai.adapters — Pre-built LLM adapters and observability backends.

Each LLM adapter implements the ``complete()`` / ``acomplete()`` interface
defined by ``clampai.reasoning.LLMAdapter``.

    from clampai.adapters import AnthropicAdapter
    adapter = AnthropicAdapter(anthropic.Anthropic())
    engine = Orchestrator(task, llm=adapter)

    from clampai.adapters import OpenAIAdapter
    adapter = OpenAIAdapter(openai.OpenAI())

    # Wrap any ClampAI Orchestrator as a LangChain BaseTool:
    from clampai.adapters import ClampAISafeTool
    tool = ClampAISafeTool(orchestrator=engine, name="safe_agent")

    # Wrap an MCP tool server with shared budget enforcement:
    from clampai.adapters import SafeMCPServer
    server = SafeMCPServer("my-agent", budget=500.0)

    @server.tool()
    def send_email(to: str, subject: str) -> str: ...

    # LangGraph safety nodes — enforce budget and invariants in a graph:
    from clampai.adapters import SafetyNode, clampai_node, budget_guard, invariant_guard
    from clampai.invariants import no_delete_invariant, rate_limit_invariant

    @clampai_node(budget=100.0, cost_per_step=1.0)
    def research_node(state: dict) -> dict:
        return {"result": "..."}

    graph.add_node("budget_gate", budget_guard(budget=100.0))
    graph.add_node("inv_gate", invariant_guard([
        no_delete_invariant("audit_log"),
        rate_limit_invariant("api_calls", 50),
    ]))

    # FastAPI / Starlette middleware — per-request budget and invariant enforcement:
    from clampai.adapters import ClampAIMiddleware

    app.add_middleware(
        ClampAIMiddleware,
        budget=1000.0,
        cost_per_request=1.0,
    )

    # LangChain callback — wraps ANY LangChain agent in 2 lines (no agent changes):
    from clampai.adapters import ClampAICallbackHandler
    handler = ClampAICallbackHandler(budget=50.0, cost_per_action=2.0)
    result = agent_executor.invoke({"input": "..."}, config={"callbacks": [handler]})
    print(handler.budget_remaining)

    # CrewAI — safe tool and step callback:
    from clampai.adapters import ClampAISafeCrewTool, ClampAICrewCallback, safe_crew_tool

    @safe_crew_tool(budget=100.0, cost=2.0)
    def search(query: str) -> str: ...

    callback = ClampAICrewCallback(budget=200.0)
    crew = Crew(agents=[...], tasks=[...], step_callback=callback.step_callback)

    # AutoGen — safe reply function:
    from clampai.adapters import ClampAISafeAutoGenAgent, autogen_reply_fn

    @autogen_reply_fn(budget=100.0, cost_per_reply=2.0)
    def my_reply(messages: list) -> str: ...

    # OTel trace export — wire audit log to distributed tracing:
    from clampai.adapters import OTelTraceExporter
    exporter = OTelTraceExporter(tracer)
    kernel = SafetyKernel(..., reconcile_fn=exporter.make_reconcile_fn())

    # OpenClaw personal AI assistant (CLI-based, no extra pip install):
    #
    # Direct adapter:
    from clampai.adapters import AsyncOpenClawAdapter
    adapter = AsyncOpenClawAdapter(thinking="medium")
    response = await adapter.acomplete("Choose the next action.")

    # Session-scoped adapter (Gateway maintains conversation history):
    from clampai.adapters import openclaw_session, async_openclaw_session

    with openclaw_session(thinking="medium") as adapter:
        r1 = adapter.complete("Step 1.")
        r2 = adapter.complete("Step 2, building on the above.")

    async with async_openclaw_session(thinking="high") as adapter:
        r = await adapter.acomplete("Summarise my inbox.")

    # Gateway client (health, models, sessions, RPC, memory):
    from clampai.adapters import OpenClawGateway

    gw = OpenClawGateway()
    health = gw.health()
    models = gw.list_models()
    sessions = gw.list_sessions(all_agents=True)
    result = gw.call("health")
    hits = gw.search_memory("unpaid invoices")

Installation:
    pip install clampai[anthropic]      # Anthropic SDK
    pip install clampai[openai]         # OpenAI SDK
    pip install clampai[langchain]      # LangChain >= 0.3
    pip install clampai[langgraph]      # LangGraph >= 0.2
    pip install clampai[fastapi]        # FastAPI / Starlette
    pip install clampai[mcp]            # MCP (Model Context Protocol)
    pip install clampai[prometheus]     # Prometheus metrics
    pip install clampai[opentelemetry]  # OpenTelemetry metrics
    # OpenClaw: npm install -g openclaw@latest  (Node.js CLI, no pip package)
"""

from .anthropic_adapter import AnthropicAdapter, AsyncAnthropicAdapter
from .autogen_adapter import (
    ClampAIAutoGenBudgetError,
    ClampAIAutoGenError,
    ClampAIAutoGenInvariantError,
    ClampAISafeAutoGenAgent,
    autogen_reply_fn,
)
from .crewai_adapter import (
    ClampAICrewBudgetError,
    ClampAICrewCallback,
    ClampAICrewError,
    ClampAICrewInvariantError,
    ClampAISafeCrewTool,
    safe_crew_tool,
)
from .fastapi_middleware import ClampAIMiddleware
from .langchain_callback import ClampAICallbackError, ClampAICallbackHandler
from .langchain_tool import ClampAISafeTool
from .langgraph_adapter import (
    ClampAIBudgetError,
    ClampAIInvariantError,
    ClampAISafetyError,
    SafetyNode,
    budget_guard,
    clampai_node,
    invariant_guard,
)
from .mcp_server import SafeMCPServer
from .metrics import OTelMetrics, OTelTraceExporter, PrometheusMetrics
from .openai_adapter import AsyncOpenAIAdapter, OpenAIAdapter
from .openclaw_adapter import (
    AsyncOpenClawAdapter,
    GatewayHealth,
    OpenClawAdapter,
    OpenClawGateway,
    OpenClawResponse,
    async_openclaw_session,
    openclaw_session,
)

__all__ = [
    "AnthropicAdapter",
    "AsyncAnthropicAdapter",
    "AsyncOpenAIAdapter",
    "AsyncOpenClawAdapter",
    "ClampAIAutoGenBudgetError",
    "ClampAIAutoGenError",
    "ClampAIAutoGenInvariantError",
    "ClampAIBudgetError",
    "ClampAICallbackError",
    "ClampAICallbackHandler",
    "ClampAICrewBudgetError",
    "ClampAICrewCallback",
    "ClampAICrewError",
    "ClampAICrewInvariantError",
    "ClampAIInvariantError",
    "ClampAIMiddleware",
    "ClampAISafeAutoGenAgent",
    "ClampAISafeCrewTool",
    "ClampAISafeTool",
    "ClampAISafetyError",
    "GatewayHealth",
    "OTelMetrics",
    "OTelTraceExporter",
    "OpenAIAdapter",
    "OpenClawAdapter",
    "OpenClawGateway",
    "OpenClawResponse",
    "PrometheusMetrics",
    "SafeMCPServer",
    "SafetyNode",
    "async_openclaw_session",
    "autogen_reply_fn",
    "budget_guard",
    "clampai_node",
    "invariant_guard",
    "openclaw_session",
    "safe_crew_tool",
]
