"""
Groq LLM client for ResearchRAG.

This module implements `GroqClient`, a concrete subclass of `BaseLLM`
that integrates with the Groq API for lightning-fast inference. It
encapsulates all Groq SDK interactions, configuration validation, and
error translation, ensuring the core application remains agnostic to
Groq's specific API design.
"""

import time
from typing import Any, Dict, Iterator, List, Optional

import groq
from groq import Groq

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.llms.base_llm import (
    BaseLLM,
    LLMConfigurationError,
    LLMGenerationError,
    LLMResponse,
)

logger = get_logger(__name__)


class GroqClient(BaseLLM):
    """
    Concrete LLM provider for Groq.

    This client wraps the official Groq Python SDK, mapping the generic
    `BaseLLM` interface to Groq's chat completion endpoints. It handles
    SDK initialization, parameter resolution, streaming, and maps Groq
    API exceptions into standard project-level exceptions.
    """

    def __init__(self) -> None:
        """
        Initialize the Groq client configuration.

        Validates the configuration immediately to fail fast if API
        keys are missing, then initializes the underlying SDK.
        """
        self._client: Optional[Groq] = None
        self._model: str = settings.llm.model
        self._default_temperature: float = settings.llm.temperature
        self._default_max_tokens: int = settings.llm.max_tokens

        self.validate_configuration()
        self.initialize()

    def validate_configuration(self) -> None:
        """
        Validate that the Groq API key is present in settings.

        Raises:
            LLMConfigurationError: If the API key is missing or empty.
        """
        api_key = settings.api_keys.groq_api_key
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                "Groq API key is missing. Please set GROQ_API_KEY in the "
                "environment or .env file."
            )

    def initialize(self) -> None:
        """
        Initialize the official Groq SDK client.

        Raises:
            LLMConfigurationError: If SDK initialization fails.
        """
        logger.info("Initializing Groq client (model=%s)", self._model)
        try:
            self._client = Groq(api_key=settings.api_keys.groq_api_key)
        except Exception as error:
            raise LLMConfigurationError(
                f"Failed to initialize Groq SDK client: {error}"
            ) from error

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
        """
        Generate a complete synchronous answer using Groq.

        Args:
            prompt: The user query or main instruction.
            system_prompt: Optional system-level instructions.
            history: Optional list of previous conversation turns.
            temperature: Override for sampling temperature.
            max_tokens: Override for maximum tokens.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional Groq-specific parameters (e.g., stop sequences).

        Returns:
            An `LLMResponse` containing the generated text and usage stats.

        Raises:
            LLMGenerationError: If the Groq API request fails.
        """
        if self._client is None:
            self.initialize()

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
            "Sending generation request to Groq (model=%s, temp=%.2f)",
            self._model,
            request_kwargs["temperature"],
        )
        start_time = time.perf_counter()

        assert self._client is not None
        response = None

        try:
            response = self._client.chat.completions.create(**request_kwargs)
        except Exception as error:
            self._handle_api_error(error)

        assert response is not None

        latency = time.perf_counter() - start_time
        
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason
        usage = response.usage

        logger.info(
            "Groq generation completed in %.3fs (finish_reason=%s)",
            latency,
            finish_reason,
        )

        return LLMResponse(
            content=content,
            metadata={
                "finish_reason": finish_reason,
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
        """
        Generate an answer incrementally using Groq's streaming API.

        Args:
            prompt: The user query or main instruction.
            system_prompt: Optional system-level instructions.
            history: Optional list of previous conversation turns.
            temperature: Override for sampling temperature.
            max_tokens: Override for maximum tokens.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional Groq-specific parameters.

        Yields:
            Chunks of generated text as strings.

        Raises:
            LLMGenerationError: If the Groq streaming request fails.
        """
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

        logger.info("Starting Groq streaming generation (model=%s)", self._model)

        try:
            stream = self._client.chat.completions.create(**request_kwargs)
            for chunk in stream:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk is not None:
                    yield content_chunk
        except Exception as error:
            self._handle_api_error(error)

    def health_check(self) -> bool:
        """
        Verify that the Groq API is reachable and authenticated.

        Performs a minimal generation request to ensure the API
        is fully operational.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            self.validate_configuration()
            if self._client is None:
                self.initialize()

            assert self._client is not None
            # Perform a minimal, low-latency request
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception as error:
            logger.warning("Groq health check failed: %s", error)
            return False

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """
        Construct the message history payload required by Groq.

        Args:
            prompt: The latest user query.
            system_prompt: Optional system boundary prompt.
            history: Optional list of prior conversation turns.

        Returns:
            A list of formatted message dictionaries.
        """
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
        """
        Resolve parameters into a final kwargs dictionary for the API.

        Combines runtime overrides with settings defaults.

        Args:
            messages: The formatted message list.
            temperature: Runtime temperature override.
            max_tokens: Runtime max tokens override.
            top_p: Optional top_p parameter.
            stream: Whether to request a streaming response.
            **kwargs: Extra parameters like stop sequences.

        Returns:
            A dictionary ready to unpack into the Groq SDK call.
        """
        resolved_temp = temperature if temperature is not None else self._default_temperature
        resolved_tokens = max_tokens if max_tokens is not None else self._default_max_tokens

        request_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": resolved_temp,
            "max_tokens": resolved_tokens,
            "stream": stream,
        }
        if top_p is not None:
            request_kwargs["top_p"] = top_p

        request_kwargs.update(kwargs)
        return request_kwargs

    @staticmethod
    def _handle_api_error(error: Exception) -> None:
        """
        Map Groq SDK exceptions to standard project exceptions.

        Args:
            error: The exception caught during an API call.

        Raises:
            LLMGenerationError: Always raises this standardized error.
        """
        if isinstance(error, groq.APIConnectionError):
            raise LLMGenerationError(
                f"Network error connecting to Groq API: {error}"
            ) from error
        elif isinstance(error, groq.RateLimitError):
            raise LLMGenerationError(
                f"Groq API rate limit exceeded: {error}"
            ) from error
        elif isinstance(error, groq.APIStatusError):
            raise LLMGenerationError(
                f"Groq API returned an error status ({error.status_code}): {error.response}"
            ) from error
        else:
            raise LLMGenerationError(
                f"Unexpected error during Groq generation: {error}"
            ) from error
