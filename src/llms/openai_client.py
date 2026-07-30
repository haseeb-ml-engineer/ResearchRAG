"""
OpenAI LLM client for ResearchRAG.

This module implements `OpenAIClient`, a concrete subclass of
`BaseLLM` that integrates with the OpenAI API. It mirrors the Groq
client behavior so the rest of the application can use the shared LLM
factory without special cases.
"""

import time
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.llms.base_llm import (
    BaseLLM,
    LLMConfigurationError,
    LLMGenerationError,
    LLMResponse,
)

logger = get_logger(__name__)


class OpenAIClient(BaseLLM):
    """Concrete LLM provider for OpenAI."""

    def __init__(self) -> None:
        self._client: Optional[OpenAI] = None
        self._model: str = settings.llm.model
        self._default_temperature: float = settings.llm.temperature
        self._default_max_tokens: int = settings.llm.max_tokens

        self.validate_configuration()
        self.initialize()

    def validate_configuration(self) -> None:
        api_key = settings.api_keys.openai_api_key
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                "OpenAI API key is missing. Please set OPENAI_API_KEY in the environment or .env file."
            )

    def initialize(self) -> None:
        logger.info("Initializing OpenAI client (model=%s)", self._model)
        try:
            self._client = OpenAI(api_key=settings.api_keys.openai_api_key)
        except Exception as error:
            raise LLMConfigurationError(f"Failed to initialize OpenAI client: {error}") from error

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._client is None:
            self.initialize()

        assert self._client is not None

        messages = self._build_messages(prompt, system_prompt, history)
        request_kwargs = self._build_request_kwargs(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
            **kwargs,
        )

        logger.info(
            "Sending generation request to OpenAI (model=%s, temp=%.2f)",
            self._model,
            request_kwargs["temperature"],
        )
        start_time = time.perf_counter()

        response = None

        try:
            response = self._client.chat.completions.create(**request_kwargs)
        except Exception as error:
            self._handle_api_error(error)

        assert response is not None
        latency = time.perf_counter() - start_time
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            metadata={
                "finish_reason": choice.finish_reason,
                "model": response.model,
                "latency_seconds": round(latency, 4),
            },
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self._client is None:
            self.initialize()

        assert self._client is not None

        messages = self._build_messages(prompt, system_prompt, history)
        request_kwargs = self._build_request_kwargs(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            **kwargs,
        )

        try:
            stream = self._client.chat.completions.create(**request_kwargs)
            for chunk in stream:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk is not None:
                    yield content_chunk
        except Exception as error:
            self._handle_api_error(error)

    def health_check(self) -> bool:
        try:
            self.validate_configuration()
            if self._client is None:
                self.initialize()

            assert self._client is not None
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception as error:
            logger.warning("OpenAI health check failed: %s", error)
            return False

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_request_kwargs(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        request_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._default_max_tokens,
            "stream": stream,
        }
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        request_kwargs.update(kwargs)
        return request_kwargs

    @staticmethod
    def _handle_api_error(error: Exception) -> None:
        raise LLMGenerationError(f"Unexpected error during OpenAI generation: {error}") from error
