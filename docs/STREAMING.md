# ClampAI and Streaming LLMs

**Status:** Design decision, documented to preempt questions.
**Date:** 2026-02-27

---

## Why ClampAI Buffers LLM Responses

ClampAI's orchestrator calls `LLMAdapter.complete()` and waits for the full response before doing anything with it. This is **intentional and correct**. Here is why.

### The core clampaint: T5 atomicity requires a complete action proposal

ClampAI's safety kernel evaluates `ActionSpec` objects — structured data with a declared `id`, `cost`, and `effects`. Before any action executes, the kernel:

1. Simulates the action's effects on a copy of the current state (not the real state).
2. Checks all blocking invariants against the simulated result.
3. Checks the budget.
4. Only if all checks pass: commits atomically (updates state, charges budget, appends trace).

**This evaluation requires the complete ActionSpec.** You cannot run a budget check on a partial cost. You cannot simulate an incomplete effect list. You cannot check an invariant against a half-formed state update.

If ClampAI evaluated a streaming response mid-token, it would be evaluating an incomplete action proposal — the equivalent of a bank authorising a transaction before seeing the full amount. This is not a limitation to be worked around; it is a necessary consequence of the safety model.

### The latency argument: streaming adds nothing observable

The ClampAI safety evaluation adds **0.061 ms** average latency per check (see `BENCHMARKS.md`). A real LLM API call takes 1,000–5,000 ms of network latency plus token generation time. The safety check is **three to five orders of magnitude faster** than the LLM response.

Waiting for the complete response and then evaluating it takes the same wall-clock time as evaluating a streaming response, because the safety evaluation completes before the next token would have been delivered anyway.

```
Timeline (not to scale, illustrative):

  LLM starts responding
  │
  ├─ token 1 arrives ──────────────────────────────────── 0 ms
  │   (incomplete JSON: `{"chosen_act`)
  │
  ├─ token N arrives (response complete) ──────────────── 2,100 ms
  │   (full JSON: `{"chosen_action_id": "archive_email", ...}`)
  │
  ├─ SafetyKernel.evaluate() ──────────────────────────── 2,100.06 ms
  │   (checks budget, invariants, simulates effects)
  │
  └─ action executes or is rejected ───────────────────── 2,100.06 ms
                                                            ↑
                              0.06 ms overhead: unmeasurable in practice
```

The "streaming delay" that users notice in chat interfaces (words appearing one by one) is **display latency**, not evaluation latency. ClampAI is not a chat interface. It is an execution framework. The question is not "how quickly does text appear?" but "is this action approved?". Those are different questions.

---

## Where Streaming Is a Legitimate Concern

The above argument is about safety evaluation. There is one scenario where streaming matters for ClampAI: **UX in applications that display the LLM's reasoning while the agent runs**.

If you are building an application that shows the user the agent's reasoning tokens in real time (like a streaming chat UI), ClampAI's buffering means the reasoning text only appears after the complete response is received — not as tokens arrive.

This is a **real UX gap**. The fix is the `stream_tokens` callback.

### The `stream_tokens` callback

`LLMAdapter.complete()` accepts an optional `stream_tokens: Callable[[str], None]` parameter. When provided, the adapter calls this function with each token chunk as it arrives. The full response is still returned at the end for safety evaluation.

```python
# Display tokens in real time for UX; safety kernel still gets full response
def on_token(chunk: str):
    print(chunk, end="", flush=True)

class MyAdapter:
    def complete(self, prompt, system_prompt="", temperature=0.3,
                 max_tokens=2000, stream_tokens=None) -> str:
        full_text = ""
        with client.messages.stream(...) as stream:
            for chunk in stream.text_stream:
                if stream_tokens:
                    stream_tokens(chunk)     # ← fires for UX
                full_text += chunk
        return full_text                     # ← returned for safety evaluation

# Usage:
adapter = MyAdapter()
engine = Orchestrator(task, llm=adapter)
engine._llm_stream_callback = on_token    # application wires this up
result = engine.run()
```

The `stream_tokens` callback fires for **display only**. The safety kernel never sees individual chunks — it evaluates the complete response returned by `complete()`. The safety path is unchanged.

ClampAI's built-in `AnthropicAdapter` and `OpenAIAdapter` both implement `stream_tokens` support.

---

## MCP Context

MCP (Model Context Protocol) tool calls are **always fully-formed objects** within a streaming response. When a Claude model uses a tool, the `tool_use` block (which contains the tool name and input parameters) is buffered and delivered as a complete unit — even within a streaming response. The surrounding text tokens stream; the tool call itself does not.

This means the streaming concern for MCP is less acute than for the general case:

- ClampAI evaluates the `tool_use` block (complete, always).
- The surrounding reasoning text streams through the `stream_tokens` callback.
- The safety kernel never sees partial tool inputs.

An MCP adapter for ClampAI does not need to implement partial-response handling. It receives complete tool call objects and evaluates them normally.

---

## Summary

| Question | Answer |
|----------|--------|
| Does ClampAI support streaming tokens? | Yes, via the `stream_tokens` callback on `complete()`. |
| Does the safety kernel evaluate partial responses? | No, by design — partial actions cannot be formally evaluated. |
| Does buffering add observable latency? | No — safety evaluation (0.061 ms) is unmeasurable against LLM latency (1,000–5,000 ms). |
| Why doesn't ClampAI support incremental safety evaluation? | Evaluating half of an action violates T5 atomicity — a partial commit is not atomic. |
| What about streaming reasoning UX? | Use the `stream_tokens` callback to display tokens as they arrive; the kernel still evaluates the full response. |
| What about MCP? | MCP tool calls are always complete objects; the streaming concern does not apply. |

---

## References

- `clampai/reasoning.py:LLMAdapter` — Protocol definition with `stream_tokens` and `acomplete()` documentation
- `clampai/adapters/anthropic_adapter.py` — Streaming implementation for Anthropic SDK
- `clampai/adapters/openai_adapter.py` — Streaming implementation for OpenAI SDK
- `clampai/formal.py:SafetyKernel.evaluate()` — Why complete ActionSpec is required
- `BENCHMARKS.md` — 0.061 ms per check benchmark methodology
- `MATHEMATICAL_COMPLIANCE.md §T5` — Atomicity proof (evaluate on copy, commit only if passing)
