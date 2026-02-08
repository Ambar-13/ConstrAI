# Contributing to ConstrAI

Thanks for wanting to contribute. Here's how to get started.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/constrai.git
cd constrai
pip install -e ".[dev]"
```

## Running tests

Before submitting anything, make sure all three suites pass:

```bash
python tests/test_constrai.py       # Core theorems (69 tests)
python tests/chaos_fuzzer.py       # Adversarial attacks (45 tests)
python tests/test_v2_hardening.py  # Hardening layer (41 tests)
```

All 155 tests must pass. No exceptions.

## What we'd love help with

**Good first issues:**
- Add more domain examples (file operations, API orchestration, CI/CD pipelines) in `examples/`
- Improve docstrings or fix typos in `docs/`
- Make the docs more comprehensive
- Write additional chaos fuzzer attack vectors in `tests/chaos_fuzzer.py`

**Medium:**
- New `Attestor` implementations (Docker health checks, HTTP endpoint verification, database query attestors)
- New `ReadinessProbe` patterns for common infrastructure
- Performance benchmarks on large state spaces

**Advanced:**
- Multi-agent coordination layer
- Formal proof document (Lean4 or Coq formalization of T1–T7)
- Real LLM adapter testing and benchmarks (Claude, GPT-4, local models)
- Add Dynamic cost / weighted allocation (where expensive LLM calls get more tokens/budget proportionally)
- Distributed locking for concurrent agent access

## How to submit

1. Fork the repo
2. Create a branch (`git checkout -b my-fix`)
3. Make your changes
4. Run all three test suites
5. Open a PR with a short description of what and why

Keep PRs focused. One fix or feature per PR is easier to review than a big batch.

## Rules

**Do:**
- Add tests for new features. If it's in `constrai/`, it needs a test.
- Keep the zero-dependency constraint for core (`constrai/`). External deps go in optional extras.
- Be honest about guarantee levels. If your feature is EMPIRICAL, don't call it PROVEN.

**Don't:**
- Break existing tests. If a test needs updating, explain why in the PR.
- Add `shell=True` anywhere. Ever.
- Use floating-point arithmetic for budget/cost tracking (we use integer millicents internally for a reason).
- Remove or weaken existing safety checks.

## Code style

- No linter is enforced. Just be readable.
- Type hints are appreciated but not mandatory.
- Docstrings on public classes and methods.

## Questions?

Open an issue. There's no question too basic. Feel free to discuss!
