"""
clampai.adapters.openai_adapter — OpenAI Chat Completions adapters (sync and async).

Sync adapter (``OpenAIAdapter``):
    import openai
    from clampai.adapters import OpenAIAdapter
    from clampai import Orchestrator

    client = openai.OpenAI()
    adapter = OpenAIAdapter(client)
    engine = Orchestrator(task, llm=adapter)

Native-async adapter (``AsyncOpenAIAdapter``):
    import openai
    from clampai.adapters import AsyncOpenAIAdapter
    from clampai import AsyncSafetyKernel

    client = openai.AsyncOpenAI()
    adapter = AsyncOpenAIAdapter(client)
    response = await adapter.acomplete("Choose the next action.")

For Azure OpenAI, pass an ``openai.AzureOpenAI`` or ``openai.AsyncAzureOpenAI``
client and set ``model`` to your deployment name.

Pass a ``stream_tokens`` callable to ``complete()`` or ``acomplete()`` for
live token delivery.  Safety evaluation always waits for the full response.

Requires: pip install clampai[openai]
"""

from __future__ import annotations

from typing import Callable, Optional


class OpenAIAdapter:
    """
    ClampAI LLM adapter for the OpenAI Python SDK (v1.x).

    Args:
    client:
        An ``openai.OpenAI`` or ``openai.AzureOpenAI`` instance.
    model:
        Chat model ID.  Default: ``gpt-4o-mini``.
    default_system_prompt:
        System message used when the caller does not provide one.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

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
        Call the OpenAI Chat Completions API and return the assistant's text.

        Args:
        prompt:
            User-turn content sent to the model.
        system_prompt:
            System message.  Falls back to ``default_system_prompt`` if empty.
        temperature:
            Sampling temperature (0.0–2.0 for OpenAI).
        max_tokens:
            Maximum tokens in the response.
        stream_tokens:
            If provided, called with each text delta as it arrives.
            The complete text is still returned after streaming completes.

        Returns:
        str
            The complete assistant response text.
        """
        system = system_prompt or self._default_system_prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        if stream_tokens is not None:
            return self._complete_streaming(
                messages, temperature, max_tokens, stream_tokens
            )
        return self._complete_blocking(messages, temperature, max_tokens)


    def _complete_blocking(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _complete_streaming(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        stream_tokens: Callable[[str], None],
    ) -> str:
        full_text = ""
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                stream_tokens(delta)
                full_text += delta
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

        EXPERIMENTAL: The OpenAI SDK's ``AsyncOpenAI`` client is preferred
        for true async I/O.  This wrapper runs the synchronous ``complete()``
        in a thread pool.

        For production async use, instantiate with ``AsyncOpenAI`` and call
        ``chat.completions.create`` natively.  An ``AsyncOpenAIAdapter`` is
        planned for v0.6.

        Guarantee: HEURISTIC (thread pool; not native async)
        """
        import asyncio
        return await asyncio.to_thread(
            self.complete, prompt, system_prompt, temperature, max_tokens, stream_tokens
        )

    def __repr__(self) -> str:
        return f"OpenAIAdapter(model={self._model!r})"


class AsyncOpenAIAdapter:
    """
    Native-async OpenAI adapter using ``openai.AsyncOpenAI``.

    Uses the OpenAI SDK's native async client — network I/O does not
    block OS threads or occupy a thread pool slot.  This is the recommended
    adapter for use with ``AsyncSafetyKernel`` or any coroutine-based
    orchestration.

    Args:
        client: An ``openai.AsyncOpenAI`` or ``openai.AsyncAzureOpenAI``
            instance.
        model: Chat model ID.  Default: ``gpt-4o-mini``.
        default_system_prompt: Used when the caller does not provide one.

    Usage:
        import openai
        from clampai.adapters import AsyncOpenAIAdapter

        client = openai.AsyncOpenAI()   # reads OPENAI_API_KEY
        adapter = AsyncOpenAIAdapter(client)

        response = await adapter.acomplete("Choose the next action.")

    Guarantee: CONDITIONAL (correct if the OpenAI SDK async client
    behaves correctly; network errors propagate as exceptions).
    """

    DEFAULT_MODEL = "gpt-4o-mini"

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
        Call the OpenAI Chat Completions API asynchronously.

        Uses the native async SDK client.  No thread pool is allocated.

        Args:
            prompt: User-turn content sent to the model.
            system_prompt: System message.  Falls back to
                ``default_system_prompt`` if empty.
            temperature: Sampling temperature (0.0–2.0 for OpenAI).
            max_tokens: Maximum tokens in the response.
            stream_tokens: If provided, called with each text delta as
                it arrives.  The full text is still returned after
                streaming completes.  Safety evaluation always uses the
                full text — never a partial delta.

        Returns:
            The complete assistant response text.
        """
        system = system_prompt or self._default_system_prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if stream_tokens is not None:
            return await self._acomplete_streaming(
                messages, temperature, max_tokens, stream_tokens
            )
        return await self._acomplete_blocking(messages, temperature, max_tokens)

    async def _acomplete_blocking(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def _acomplete_streaming(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        stream_tokens: Callable[[str], None],
    ) -> str:
        full_text = ""
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                stream_tokens(delta)
                full_text += delta
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
        ``OpenAIAdapter`` (the synchronous adapter) for non-async use.
        """
        raise NotImplementedError(
            "AsyncOpenAIAdapter is async-only. "
            "Use acomplete() from an async context, or "
            "OpenAIAdapter for synchronous use."
        )

    def __repr__(self) -> str:
        return f"AsyncOpenAIAdapter(model={self._model!r})"
