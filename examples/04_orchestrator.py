"""
examples/04_orchestrator.py — Full TaskDefinition + Orchestrator workflow.

The Orchestrator is the high-level entry point for agentic tasks. It wraps a
SafetyKernel with an LLM reasoning layer: the LLM proposes the next action,
the kernel accepts or rejects it, and the loop continues until the goal
predicate is satisfied or resources are exhausted.

This example uses MockLLMAdapter (no API key needed) so you can run it
immediately. Swap in AnthropicAdapter or OpenAIAdapter to use a real model.

Run:
    python examples/04_orchestrator.py
"""

from constrai import (
    ActionSpec,
    ConstrAI_SYSTEM_PROMPT,
    Effect,
    Invariant,
    Orchestrator,
    State,
    TaskDefinition,
    rate_limit_invariant,
    resource_ceiling_invariant,
)


def build_data_pipeline_task() -> TaskDefinition:
    """Define a data ingestion + transformation + reporting pipeline."""
    return TaskDefinition(
        goal="Ingest raw data, transform it, validate quality, and generate a report",

        initial_state=State({
            "data_ingested": False,
            "records_loaded": 0,
            "transformed": False,
            "quality_score": 0.0,
            "report_generated": False,
        }),

        available_actions=[
            ActionSpec(
                id="ingest_data",
                name="Ingest Data",
                description="Pull raw records from the data source",
                effects=(
                    Effect("data_ingested", "set", True),
                    Effect("records_loaded", "set", 1000),
                ),
                cost=5.0,
                category="ingest",
                risk_level="low",
                postconditions_text="Raw data is available",
            ),
            ActionSpec(
                id="transform_data",
                name="Transform Data",
                description="Normalize, deduplicate, and enrich records",
                effects=(
                    Effect("transformed", "set", True),
                    Effect("records_loaded", "multiply", 0.9),  # ~10% deduplication
                ),
                cost=8.0,
                category="transform",
                risk_level="low",
                preconditions_text="Data must be ingested first",
            ),
            ActionSpec(
                id="validate_quality",
                name="Validate Quality",
                description="Compute data quality score (0–100)",
                effects=(Effect("quality_score", "set", 92.5),),
                cost=3.0,
                category="validate",
                risk_level="low",
                preconditions_text="Data must be transformed",
            ),
            ActionSpec(
                id="generate_report",
                name="Generate Report",
                description="Create executive summary report",
                effects=(Effect("report_generated", "set", True),),
                cost=6.0,
                category="report",
                risk_level="medium",
                preconditions_text="Quality score must be computed",
            ),
        ],

        invariants=[
            Invariant(
                "no_report_without_quality_check",
                lambda s: not s.get("report_generated", False)
                          or s.get("quality_score", 0.0) > 0,
                "Cannot generate report without a quality score",
                severity="critical",
            ),
            Invariant(
                "no_transform_without_ingest",
                lambda s: not s.get("transformed", False) or s.get("data_ingested", False),
                "Cannot transform data that hasn't been ingested",
                severity="critical",
            ),
            resource_ceiling_invariant("records_loaded", ceiling=10_000),
        ],

        budget=50.0,

        goal_predicate=lambda s: (
            s.get("data_ingested", False)
            and s.get("transformed", False)
            and s.get("report_generated", False)
        ),

        goal_progress_fn=lambda s: sum([
            0.20 if s.get("data_ingested", False) else 0.0,
            0.30 if s.get("transformed", False) else 0.0,
            0.20 if s.get("quality_score", 0.0) > 0 else 0.0,
            0.30 if s.get("report_generated", False) else 0.0,
        ]),

        dependencies={
            "ingest_data": [],
            "transform_data": [("ingest_data", "Requires ingested data")],
            "validate_quality": [("transform_data", "Requires transformed data")],
            "generate_report": [("validate_quality", "Requires quality score")],
        },

        priors={
            "action:ingest_data:succeeds": (9.0, 1.0),
            "action:transform_data:succeeds": (8.0, 2.0),
            "action:validate_quality:succeeds": (9.0, 1.0),
            "action:generate_report:succeeds": (7.0, 3.0),
        },

        system_prompt=ConstrAI_SYSTEM_PROMPT + """
DOMAIN CONTEXT:
You are running a data pipeline. The correct order is:
  1. ingest_data
  2. transform_data
  3. validate_quality
  4. generate_report

Follow this sequence. Do not skip steps.
""",
    )


def main() -> None:
    task = build_data_pipeline_task()
    engine = Orchestrator(task)  # uses MockLLMAdapter by default
    result = engine.run()

    print(result.summary())

    print("\nSafety verification:")
    print(f"  T1 Budget:     {'PASS' if result.total_cost <= task.budget else 'FAIL'} "
          f"(${result.total_cost:.2f} of ${task.budget:.2f})")
    print("  T2 Terminated: PASS")
    print(f"  T3 Invariants: {'PASS' if all(i.violation_count == 0 for i in task.invariants) else 'FAIL'}")
    ok, msg = engine.kernel.trace.verify_integrity()
    print(f"  T6 Trace:      {'PASS' if ok else 'FAIL'} {msg}")

    if result.final_state:
        fs = result.final_state
        print("\nFinal pipeline state:")
        print(f"  data_ingested:    {fs.get('data_ingested')}")
        print(f"  transformed:      {fs.get('transformed')}")
        print(f"  quality_score:    {fs.get('quality_score')}")
        print(f"  report_generated: {fs.get('report_generated')}")


if __name__ == "__main__":
    main()
