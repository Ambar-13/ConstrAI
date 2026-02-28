"""
examples/claude_integration.py — End-to-end demo with a real Claude model.

Plugs AnthropicAdapter into an Orchestrator and runs a task with full
safety guarantees enforced at the kernel level.

    export ANTHROPIC_API_KEY="your-key-here"
    python examples/claude_integration.py
"""
import json
import os
from typing import Optional

from clampai import (
    ActionSpec,
    ClampAI_SYSTEM_PROMPT,
    Effect,
    Invariant,
    LLMAdapter,
    Orchestrator,
    State,
    TaskDefinition,
)

# Claude adapter — plugs Anthropic API into ClampAI

class ClaudeAdapter:
    """
    Anthropic Claude adapter for ClampAI.

    Implements the LLMAdapter protocol by delegating to the Anthropic
    messages API. ClampAI handles all structured parsing, validation,
    and safety checking; Claude just reasons.
    """
    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.total_tokens = 0

    def complete(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or ClampAI_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        return response.content[0].text


# Define a task

def build_web_app_task() -> TaskDefinition:
    """Example: Build and deploy a web application."""
    return TaskDefinition(
        goal="Build, test, and deploy a React web application",

        initial_state=State({
            "project_initialized": False,
            "components_built": 0,
            "tests_written": False,
            "tests_passing": False,
            "built": False,
            "deployed": False,
            "domain_configured": False,
        }),

        available_actions=[
            ActionSpec(
                id="init_project",
                name="Initialize Project",
                description="Create React project with TypeScript template",
                effects=(Effect("project_initialized", "set", True),),
                cost=2.0, category="setup", risk_level="low",
                postconditions_text="Project scaffold exists",
            ),
            ActionSpec(
                id="build_component",
                name="Build Component",
                description="Create a React component (header, main, footer)",
                effects=(Effect("components_built", "increment", 1),),
                cost=3.0, category="dev", risk_level="low",
                preconditions_text="Project initialized",
                postconditions_text="One more component exists",
            ),
            ActionSpec(
                id="write_tests",
                name="Write Tests",
                description="Write unit tests for all components",
                effects=(Effect("tests_written", "set", True),),
                cost=4.0, category="test", risk_level="low",
                preconditions_text="At least 3 components built",
            ),
            ActionSpec(
                id="run_tests",
                name="Run Tests",
                description="Execute test suite",
                effects=(Effect("tests_passing", "set", True),),
                cost=2.0, category="test", risk_level="medium",
                preconditions_text="Tests written",
            ),
            ActionSpec(
                id="build_prod",
                name="Production Build",
                description="Create optimized production build",
                effects=(Effect("built", "set", True),),
                cost=3.0, category="build", risk_level="medium",
                preconditions_text="Tests passing",
            ),
            ActionSpec(
                id="deploy",
                name="Deploy to Production",
                description="Deploy built app to hosting provider",
                effects=(Effect("deployed", "set", True),),
                cost=5.0, category="deploy", risk_level="high",
                preconditions_text="Production build exists",
                reversible=True,
            ),
            ActionSpec(
                id="configure_domain",
                name="Configure Domain",
                description="Set up custom domain and SSL",
                effects=(Effect("domain_configured", "set", True),),
                cost=2.0, category="deploy", risk_level="medium",
                preconditions_text="App deployed",
            ),
        ],

        invariants=[
            Invariant(
                "max_components",
                lambda s: s.get("components_built", 0) <= 20,
                "No more than 20 components",
            ),
            Invariant(
                "no_deploy_without_tests",
                lambda s: not s.get("deployed", False) or s.get("tests_passing", False),
                "Cannot deploy without passing tests",
                severity="critical",
            ),
        ],

        budget=50.0,

        goal_predicate=lambda s: (
            s.get("deployed", False) and s.get("domain_configured", False)
        ),

        goal_progress_fn=lambda s: min(1.0, sum([
            0.1 if s.get("project_initialized", False) else 0,
            min(s.get("components_built", 0), 3) / 3 * 0.3,
            0.1 if s.get("tests_written", False) else 0,
            0.1 if s.get("tests_passing", False) else 0,
            0.1 if s.get("built", False) else 0,
            0.2 if s.get("deployed", False) else 0,
            0.1 if s.get("domain_configured", False) else 0,
        ])),

        dependencies={
            "init_project": [],
            "build_component": [("init_project", "Need project first")],
            "write_tests": [("build_component", "Need components")],
            "run_tests": [("write_tests", "Need tests")],
            "build_prod": [("run_tests", "Need passing tests")],
            "deploy": [("build_prod", "Need build")],
            "configure_domain": [("deploy", "Need deployment")],
        },

        priors={
            "action:init_project:succeeds": (9.0, 1.0),      # Very likely
            "action:build_component:succeeds": (8.0, 2.0),    # Likely
            "action:write_tests:succeeds": (7.0, 3.0),        # Usually works
            "action:run_tests:succeeds": (6.0, 4.0),          # Can fail
            "action:build_prod:succeeds": (7.0, 3.0),         # Usually works
            "action:deploy:succeeds": (5.0, 5.0),             # Uncertain
            "action:configure_domain:succeeds": (7.0, 3.0),   # Usually works
        },

        system_prompt=ClampAI_SYSTEM_PROMPT + """
DOMAIN CONTEXT:
You are building a React web application. Follow standard practices:
- Initialize before building components
- Build at least 3 components before testing
- Test before building for production
- Build before deploying
- Deploy before configuring domain
""",
    )


# Run it

if __name__ == "__main__":
    task = build_web_app_task()

    # Use Claude if API key is available, otherwise MockLLM
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print("🧠 Using Claude for reasoning")
        llm = ClaudeAdapter(api_key=api_key)
    else:
        print("Using MockLLM (set ANTHROPIC_API_KEY for Claude)")
        llm = None  # Will use MockLLMAdapter

    engine = Orchestrator(task, llm=llm)
    result = engine.run()

    print(result.summary())

    # Verify safety guarantees
    print("\nSafety Verification:")
    print(f"  T1 Budget:     {'ok' if result.total_cost <= task.budget else 'FAIL'} "
          f"(${result.total_cost:.2f} ≤ ${task.budget:.2f})")
    print("  T2 Terminated: ok")
    print(f"  T3 Invariants: {'ok' if all(i.violation_count == 0 for i in task.invariants) else 'FAIL'}")
    ok, msg = engine.kernel.trace.verify_integrity()
    print(f"  T6 Trace:      {'ok' if ok else 'FAIL'} {msg}")

    if api_key:
        print(f"\nToken usage: {llm.total_tokens}")
