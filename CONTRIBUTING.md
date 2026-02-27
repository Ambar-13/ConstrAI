# Contributing to ConstrAI

Thank you for your interest in contributing.  ConstrAI is a formal safety
framework, so correctness matters more than velocity.  Read this guide before
opening a pull request.

**Maintainer:** Ambar — ambar13@u.nus.edu  
**Repository:** https://github.com/Ambar-13/ConstrAI

---

## Table of contents

1. [Development setup](#development-setup)
2. [Running the tests](#running-the-tests)
3. [API conventions — keyword-only parameters](#api-conventions--keyword-only-parameters)
4. [Code style](#code-style)
5. [Adding invariants to the pre-built library](#adding-invariants-to-the-pre-built-library)
6. [How to submit a pull request](#how-to-submit-a-pull-request)
7. [Good first issues](#good-first-issues)
8. [Rules](#rules)

---

## Development setup

```bash
# 1. Fork then clone your fork.
git clone https://github.com/YOUR_USERNAME/ConstrAI.git
cd ConstrAI

# 2. Create a virtual environment (optional but recommended).
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package in editable mode with dev dependencies.
pip install -e ".[dev]"

# 4. Verify everything works.
pytest tests/ -v
```

You do not need any LLM API keys to run the core tests — the built-in
`MockLLMAdapter` drives execution.

---

## Running the tests

```bash
# Full test suite (required before every PR):
pytest tests/ -v --tb=short

# Individual suites:
pytest tests/test_constrai.py            # T1-T8 unit tests
pytest tests/test_monte_carlo.py         # 1,000-run probabilistic validation
pytest tests/test_composition.py         # Operadic composition verification
pytest tests/test_integration.py         # End-to-end orchestrator scenarios
pytest tests/test_soft_gaps_fixed.py     # Inverse algebra, QP repair, monitors
pytest tests/test_boundary_enforcement.py

# Adversarial scenarios (run as a script, not via pytest):
python tests/chaos_fuzzer.py

# With coverage:
pytest tests/ -v --cov=constrai --cov-report=term-missing
```

All tests must pass before a PR will be merged.  No exceptions.

---

## API conventions — keyword-only parameters

### Mandatory rule during the 0.x phase

**All optional parameters in public API methods MUST be keyword-only.**

This is enforced by placing a bare `*` before the first optional parameter in
every public function and method signature.  It is not a stylistic preference —
it is a hard requirement for the entire 0.x series.

**Why this matters:**  
During the 0.x phase, we may need to reorder parameters or insert new ones as
the API evolves.  Keyword-only parameters let us do that without silently
breaking callers who pass arguments positionally.  Once 1.0.0 is released,
positional argument order will be frozen; until then, keyword-only is the safe
default.

**Example — correct:**

```python
class SafetyKernel:
    def evaluate(
        self,
        action: ActionSpec,
        state: State,
        *,                          # <-- everything after here is keyword-only
        dry_run: bool = False,
        timeout_ms: int = 500,
    ) -> SafetyVerdict:
        ...
```

**Example — wrong (do not do this):**

```python
class SafetyKernel:
    def evaluate(
        self,
        action: ActionSpec,
        state: State,
        dry_run: bool = False,      # positional optional -- breaks on reorder
        timeout_ms: int = 500,
    ) -> SafetyVerdict:
        ...
```

Required (non-default) parameters may remain positional.  Only parameters that
have a default value must be keyword-only.

If you are adding a new parameter to an existing method, it must go after the
`*` and must have a default value so that existing callers are unaffected.

---

## Code style

`ruff check` runs in CI and must pass before a PR will be merged.  Follow the
existing patterns in the file you are editing:

- **Indentation:** 4 spaces.  No tabs.
- **Line length:** aim for 88 characters; the ruff limit is 100.
- **Type hints:** required on all new public functions and methods.  Optional
  but appreciated on private helpers.
- **Docstrings:** required on all new public classes and methods.  Use plain
  prose — no specific docstring format is enforced.
- **Guarantee levels:** every new feature must declare its epistemic status
  (`PROVEN`, `CONDITIONAL`, `EMPIRICAL`, or `HEURISTIC`) in a docstring or
  comment.  Do not claim `PROVEN` unless the guarantee holds by construction
  and induction.
- **No `shell=True`:** ever.  Not in tests, not in examples, not anywhere.
- **No floating-point for budget/cost arithmetic:**  budget and cost values
  are stored as Python `float` for the public API surface, but accumulated
  using integer millicents internally to avoid drift.  Do not introduce
  new raw float accumulation for safety-critical counters.

---

## Adding invariants to the pre-built library

ConstrAI ships with several ready-to-use `Invariant` constructors (see
`constrai/invariants.py`).  To add a new one:

1. **Write the predicate as a pure function** — no I/O, no randomness, no
   global state.  The predicate receives a `dict`-like `State` and must return
   a `bool`.  Slow predicates block the kernel; keep them O(1) or O(n) with
   a small, bounded n.

2. **Add a constructor function** in `constrai/invariants.py`:

   ```python
   def max_retries_invariant(
       key: str,
       limit: int,
       *,
       enforcement: str = "blocking",
       suggestion: Optional[str] = None,
   ) -> Invariant:
       """
       Blocks any action that would push ``state[key]`` above ``limit``.

       Guarantee level: PROVEN (by construction -- predicate is pure and
       the kernel simulates before committing).
       """
       _sug = suggestion or f"Reduce '{key}' below {limit} before proceeding."
       return Invariant(
           name=f"max_retries:{key}",
           predicate=lambda s: s.get(key, 0) <= limit,
           description=f"'{key}' must not exceed {limit}",
           enforcement=enforcement,
           suggestion=_sug,
       )
   ```

3. **Export it** from `constrai/invariants.py` — add the function name to `__all__` at the top of that file.

4. **Add a unit test** in `tests/test_constrai.py` covering:
   - The invariant passes on valid states.
   - The invariant blocks actions that would violate it (T3 check).
   - The state is unchanged after rejection (T5 check).

5. **Update the docs** — add a row to the invariant table in `docs/API.md`.

---

## How to submit a pull request

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b my-feature
   ```

2. **Make your changes.**  Keep each PR focused — one fix or feature per PR.
   Mixed concerns are harder to review and slower to merge.

3. **Run the full test suite** and confirm everything passes:
   ```bash
   pytest tests/ -v
   python tests/chaos_fuzzer.py
   ```

4. **Open the pull request** against the `main` branch.  In the PR description:
   - Explain *what* changed and *why*.
   - Reference any related issue numbers.
   - State the guarantee level of any new safety claims.
   - If you changed a public API signature, confirm that keyword-only
     conventions are followed and that no existing callers are silently broken.

5. **Address review comments.**  Maintainer review is the final gate before
   merge.

---

## Good first issues

If you are new to the codebase, these areas are well-scoped and do not require
deep knowledge of the formal kernel:

| Area | Description |
|------|-------------|
| Examples | Add domain examples to `examples/` — file operations, REST API orchestration, CI/CD pipelines, database migration tasks. |
| Docs | Fix typos, clarify prose, or expand explanations in `docs/`. |
| Docstrings | Many private helpers lack docstrings.  Adding them is always welcome. |
| Chaos fuzzer | Add new adversarial attack vectors to `tests/chaos_fuzzer.py`. |
| Attestors | Implement new `Attestor` subclasses — Docker health checks, HTTP endpoint verification, database query attestors, Kubernetes pod readiness. |
| Readiness probes | Add new `ReadinessProbe` patterns for common infrastructure (S3 bucket reachability, Redis ping, etc.). |
| Benchmarks | Write performance benchmarks for large state spaces and many invariants. |

**Medium difficulty:**

- New `ReadinessProbe` types for cloud infrastructure.
- Additional `guards.py` invariants (rate limits, SLA windows, quota checks).

**Advanced:**

- Multi-agent coordination layer.
- Lean4 or Coq formalisation of T1–T7.
- Real LLM adapter testing and benchmarks (Claude, GPT-4, local models via Ollama).
- Dynamic cost / weighted allocation (expensive LLM calls get proportionally more budget).

---

## Rules

**Do:**

- Add tests for every new feature.  If it lives in `constrai/`, it needs a test.
- Keep the zero-dependency constraint for core.  External dependencies belong
  in optional extras (`extras_require` / `[project.optional-dependencies]`).
- Be honest about guarantee levels.  `EMPIRICAL` is fine; calling it `PROVEN`
  when it is not is not.
- Use keyword-only parameters for all optional arguments in public APIs (see
  above).

**Do not:**

- Break existing tests.  If a test needs updating, explain why in the PR.
- Add `shell=True` anywhere.  This is a hard rule — no exceptions.
- Use floating-point accumulation for budget/cost tracking.
- Remove or weaken existing safety checks.
- Claim a theorem holds if you have not verified it holds by construction
  or under clearly stated assumptions.

---

## Questions?

Open an issue on GitHub.  No question is too basic.
