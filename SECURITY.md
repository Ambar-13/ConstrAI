# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

Only the latest minor release receives security patches during the 0.x phase.

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately using
[GitHub Security Advisories](https://github.com/Ambar-13/ClampAI/security/advisories/new).
This keeps details confidential until a fix is available and coordinated disclosure
has been agreed upon.

Please include:

- A clear description of the vulnerability and its impact.
- Step-by-step reproduction instructions (minimal reproducer preferred).
- The ClampAI version(s) affected.
- Whether you believe a CVE should be requested.

If you need to contact the maintainer directly before using the advisory form,
email **ambar13@u.nus.edu** with `[SECURITY]` in the subject line.

---

## Response SLA

| Severity (CVSS v3.1 base score) | Acknowledgement | Patch target |
|---------------------------------|-----------------|--------------|
| Critical (>= 9.0)               | 48 hours        | 14 days      |
| High (7.0 – 8.9)                | 48 hours        | 30 days      |
| Medium (4.0 – 6.9)              | 48 hours        | 90 days      |
| Low (< 4.0)                     | 48 hours        | 90 days      |

"Acknowledgement" means a maintainer reply confirming the report was received and
is being evaluated.  "Patch target" is a best-effort commitment, not a guarantee;
complex vulnerabilities in the formal kernel may require additional time.

---

## CVE Policy

A CVE will be requested for any confirmed vulnerability that:

- Enables a **kernel bypass** — i.e., allows an action to execute while
  violating T1 (budget), T3 (invariant), or T5 (atomicity) under normal
  usage conditions (no deliberate process-level memory corruption).
- Defeats the **budget invariant** — causes `spent(t) > budget` without
  the kernel raising a rejection.
- **Tampers with the execution trace** in a way that is undetected by
  `ExecutionTrace.verify_integrity()`, breaking T6 (hash-chain integrity).

CVEs will not be requested for theoretical weaknesses that require
exploiting SHA-256 preimage resistance or other cryptographic assumptions
that are not realistic attack vectors.

---

## Scope

### In scope

| Category | Description |
|----------|-------------|
| Kernel bypass | Any code path that causes T1, T3, or T5 to be violated under ordinary Python usage |
| Budget invariant defeat | `spent(t) > budget` reachable without kernel rejection |
| Trace tampering | Modifications to `ExecutionTrace` that pass `verify_integrity()` unchanged |
| Invariant circumvention | A correctly written blocking invariant that the kernel silently skips |
| Atomicity violation | A rejected action that nonetheless mutates `State` or charges budget |

### Out of scope

| Category | Reason |
|----------|--------|
| Theoretical SHA-256 weaknesses | Not a realistic attack vector; no fix possible at the library level |
| Bugs in third-party dependencies (anthropic, openai, langchain) | Report directly to those projects |
| User-written invariant predicate bugs | The kernel enforces what you declare; incorrect predicates are user error |
| Slow invariant predicates (performance degradation) | Not a safety bypass; use `Invariant(..., max_eval_ms=N)` to set a timeout — slow predicates are treated as violations, not exploitable gaps. See `clampai/formal.py:Invariant._check_with_timeout`. |
| Vulnerabilities requiring `ctypes`, `gc`, or other deliberate CPython internals abuse | Partially mitigated; full memory safety is outside scope (see `docs/VULNERABILITIES.md`) |
| Issues requiring physical or OS-level access to the host | Out of threat model |

---

## Coordinated Disclosure

The maintainer will work with reporters to agree on a disclosure timeline before
any public statement is made.  The default target is **90 days** from initial
report, or sooner once a patch is released — whichever comes first.

---

## Reporter Credit

Security reporters who identify valid in-scope vulnerabilities will, with their
consent, be:

- Named in **`CHANGELOG.md`** in the release that contains the fix.
- Listed permanently in **`SECURITY_HALL_OF_FAME.md`** (to be created on the
  first confirmed report).

Anonymous reports are welcome; credit will be listed as "anonymous reporter"
unless you request otherwise.
