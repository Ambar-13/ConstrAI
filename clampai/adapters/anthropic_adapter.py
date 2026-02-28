"""
clampai.adapters.anthropic_adapter — Claude API adapters (sync and async).

Sync adapter (``AnthropicAdapter``):
    import anthropic
    from clampai.adapters import AnthropicAdapter
    from clampai import Orchestrator

    client = anthropic.Anthropic()
    adapter = AnthropicAdapter(client)
    engine = Orchestrator(task, llm=adapter)

Native-async adapter (``AsyncAnthropicAdapter``):
    import anthropic
    from clampai.adapters import AsyncAnthropicAdapter
    from clampai import AsyncSafetyKernel

    client = anthropic.AsyncAnthropic()
    adapter = AsyncAnthropicAdapter(client)
    response = await adapter.acomplete("Summarise this task.")

Pass a ``stream_tokens`` callable to ``complete()`` or ``acomplete()`` to
receive token chunks as they arrive. This is for UX only — the safety
kernel waits for the full response before evaluating any action.

Default model is ``claude-haiku-4-5-20251001``. For higher-quality reasoning,
pass ``model="claude-sonnet-4-6"`` or ``model="claude-opus-4-6"``.

Requires: pip install clampai[anthropic]
"""

from __future__ import annotations

from typing import Callable, Optional


class AnthropicAdapter:
    """
    ClampAI LLM adapter for the Anthropic Python SDK.

    Args:
    client:
        An ``anthropic.Anthropic`` instance (or compatible).  The caller
        is responsible for providing API credentials.
    model:
        Anthropic model ID.  Default: ``claude-haiku-4-5-20251001``.
    default_system_prompt:
        System prompt used when the caller does not provide one.  Override
        when you want a consistent persona across all orchestrator calls.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        client,
        model: str = DEFAULT_MODEL,
        default_system_prompt: str = "You are a helpful AI assistant.",
    ) -> None:
        self._client = client
        self._model = model
        self._default_system_prompt = default_system_prompt


    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream_tokens: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Call the Anthropic Messages API and return the assistant's text.

        Args:
        prompt:
            User-turn content sent to the model.
        system_prompt:
            System prompt.  Falls back to ``default_system_prompt`` if empty.
        temperature:
            Sampling temperature (0.0–1.0).
        max_tokens:
            Maximum tokens in the response.
        stream_tokens:
            If provided, called with each text chunk as it arrives.
            The full text is still returned after streaming completes.
            Safety evaluation always uses the full text — never a partial chunk.

        Returns:
        str
            The complete assistant response text.
        """
        system = system_prompt or self._default_system_prompt

        if stream_tokens is not None:
            return self._complete_streaming(
                prompt, system, temperature, max_tokens, stream_tokens
            )
        return self._complete_blocking(prompt, system, temperature, max_tokens)


    def _complete_blocking(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _complete_streaming(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        stream_tokens: Callable[[str], None],
    ) -> str:
        full_text = ""
        with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for chunk in stream.text_stream:
                stream_tokens(chunk)
                full_text += chunk
        return full_text

    async def acomplete(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream_tokens: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Async variant using ``asyncio.to_thread``.

        Runs the synchronous ``complete()`` in a thread pool.  For
        truly non-blocking async I/O, use ``AsyncAnthropicAdapter``
        with an ``anthropic.AsyncAnthropic`` client instead.

        Guarantee: HEURISTIC (thread pool; not native async I/O)
        """
        import asyncio
        return await asyncio.to_thread(
            self.complete, prompt, system_prompt, temperature, max_tokens, stream_tokens
        )

    def __repr__(self) -> str:
        return f"AnthropicAdapter(model={self._model!r})"


class AsyncAnthropicAdapter:
    """
    Native-async Anthropic adapter using ``anthropic.AsyncAnthropic``.

    Uses the Anthropic SDK's native async client — network I/O does not
    block OS threads or occupy a thread pool slot.  This is the recommended
    adapter for use with ``AsyncSafetyKernel`` or any coroutine-based
    orchestration.

    Args:
        client: An ``anthropic.AsyncAnthropic`` instance.
        model: Anthropic model ID.  Default: ``claude-haiku-4-5-20251001``.
        default_system_prompt: Used when the caller does not provide one.

    Usage:
        import anthropic
        from clampai.adapters import AsyncAnthropicAdapter

        client = anthropic.AsyncAnthropic()   # reads ANTHROPIC_API_KEY
        adapter = AsyncAnthropicAdapter(client)

        response = await adapter.acomplete("Choose the next action.")

    Guarantee: CONDITIONAL (correct if the Anthropic SDK async client
    behaves correctly; network errors propagate as exceptions).
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        client,
        model: str = DEFAULT_MODEL,
        default_system_prompt: str = "You are a helpful AI assistant.",
    ) -> None:
        self._client = client
        self._model = model
        self._default_system_prompt = default_system_prompt

    async def acomplete(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream_tokens: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Call the Anthropic Messages API asynchronously.

        Uses the native async SDK client.  No thread pool is allocated.

        Args:
            prompt: User-turn content sent to the model.
            system_prompt: System prompt.  Falls back to
                ``default_system_prompt`` if empty.
            temperature: Sampling temperature (0.0–1.0).
            max_tokens: Maximum tokens in the response.
            stream_tokens: If provided, called with each text chunk as
                it arrives.  The full text is still returned after
                streaming completes.  Safety evaluation always uses the
                full text — never a partial chunk.

        Returns:
            The complete assistant response text.
        """
        system = system_prompt or self._default_system_prompt
        if stream_tokens is not None:
            return await self._acomplete_streaming(
                prompt, system, temperature, max_tokens, stream_tokens
            )
        return await self._acomplete_blocking(prompt, system, temperature, max_tokens)

    async def _acomplete_blocking(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    async def _acomplete_streaming(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        stream_tokens: Callable[[str], None],
    ) -> str:
        full_text = ""
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                stream_tokens(text)
                full_text += text
        return full_text

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream_tokens: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Not implemented on the async adapter.

        Use ``acomplete()`` from an async context, or use
        ``AnthropicAdapter`` (the synchronous adapter) for non-async use.
        """
        raise NotImplementedError(
            "AsyncAnthropicAdapter is async-only. "
            "Use acomplete() from an async context, or "
            "AnthropicAdapter for synchronous use."
        )

    def __repr__(self) -> str:
        return f"AsyncAnthropicAdapter(model={self._model!r})"
