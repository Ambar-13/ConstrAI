"""
constrai.adapters — Pre-built LLM adapters and observability backends.

Each LLM adapter implements the ``complete()`` / ``acomplete()`` interface
defined by ``constrai.reasoning.LLMAdapter``.

    from constrai.adapters import AnthropicAdapter
    adapter = AnthropicAdapter(anthropic.Anthropic())
    engine = Orchestrator(task, llm=adapter)

    from constrai.adapters import OpenAIAdapter
    adapter = OpenAIAdapter(openai.OpenAI())

    # Wrap any ConstrAI Orchestrator as a LangChain BaseTool:
    from constrai.adapters import ConstrAISafeTool
    tool = ConstrAISafeTool(orchestrator=engine, name="safe_agent")

    # Wrap an MCP tool server with shared budget enforcement:
    from constrai.adapters import SafeMCPServer
    server = SafeMCPServer("my-agent", budget=500.0)

    @server.tool()
    def send_email(to: str, subject: str) -> str: ...

    # OpenClaw personal AI assistant (CLI-based, no extra pip install):
    #
    # Direct adapter:
    from constrai.adapters import AsyncOpenClawAdapter
    adapter = AsyncOpenClawAdapter(thinking="medium")
    response = await adapter.acomplete("Choose the next action.")

    # Session-scoped adapter (Gateway maintains conversation history):
    from constrai.adapters import openclaw_session, async_openclaw_session

    with openclaw_session(thinking="medium") as adapter:
        r1 = adapter.complete("Step 1.")
        r2 = adapter.complete("Step 2, building on the above.")

    async with async_openclaw_session(thinking="high") as adapter:
        r = await adapter.acomplete("Summarise my inbox.")

    # Gateway client (health, models, sessions, RPC, memory):
    from constrai.adapters import OpenClawGateway

    gw = OpenClawGateway()
    health = gw.health()
    models = gw.list_models()
    sessions = gw.list_sessions(all_agents=True)
    result = gw.call("health")
    hits = gw.search_memory("unpaid invoices")

Installation:
    pip install constrai[anthropic]      # Anthropic SDK
    pip install constrai[openai]         # OpenAI SDK
    pip install constrai[langchain]      # LangChain >= 0.3
    pip install constrai[mcp]            # MCP (Model Context Protocol)
    pip install constrai[prometheus]     # Prometheus metrics
    pip install constrai[opentelemetry]  # OpenTelemetry metrics
    # OpenClaw: npm install -g openclaw@latest  (Node.js CLI, no pip package)
"""

from .anthropic_adapter import AnthropicAdapter, AsyncAnthropicAdapter
from .langchain_tool import ConstrAISafeTool
from .mcp_server import SafeMCPServer
from .metrics import OTelMetrics, PrometheusMetrics
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
    "ConstrAISafeTool",
    "GatewayHealth",
    "OTelMetrics",
    "OpenAIAdapter",
    "OpenClawAdapter",
    "OpenClawGateway",
    "OpenClawResponse",
    "PrometheusMetrics",
    "SafeMCPServer",
    "async_openclaw_session",
    "openclaw_session",
]
