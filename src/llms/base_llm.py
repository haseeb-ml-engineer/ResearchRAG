"""
Base LLM interface for ResearchRAG.

This module defines `BaseLLM`, the abstract base class that establishes
the mandatory contract for all language model providers in the system.
It defines how the pipeline asks for text generation and how providers
return results, ensuring the rest of the application remains completely
ignorant of provider-specific SDKs, HTTP clients, or data formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    """Base exception for errors raised by any LLM provider."""


class LLMConfigurationError(LLMError):
    """Raised when an LLM provider is incorrectly configured."""


class LLMGenerationError(LLMError):
    """Raised when text generation fails (e.g., API errors, rate limits)."""


@dataclass(frozen=True)
class LLMResponse:
    """
    Provider-independent structured response from an LLM.

    This object ensures that the orchestration layer never receives
    provider-specific objects (like OpenAI's `ChatCompletion` or Groq's
    equivalent).

    Attributes:
        content: The generated text answer.
        metadata: Provider-specific metadata (e.g., finish reason, model used).
        prompt_tokens: The number of tokens consumed by the prompt.
        completion_tokens: The number of tokens generated in the response.
        total_tokens: Total tokens consumed (prompt + completion).
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class BaseLLM(ABC):
    """
    Abstract interface for all language model providers.

    Any supported provider (OpenAI, Groq, Gemini, Anthropic) must
    inherit from this class and implement its abstract methods.
    The orchestration layers interact exclusively with this interface.
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the provider client.

        This method should be used to establish network sessions,
        validate API keys, or load local models into memory.

        Raises:
            LLMConfigurationError: If initialization fails due to
                missing keys or inaccessible resources.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_configuration(self) -> None:
        """
        Validate that the provider is fully and correctly configured.

        This method checks for the presence of required environment
        variables, valid model names, and acceptable generation
        parameters before any generation attempts are made.

        Raises:
            LLMConfigurationError: If the configuration is invalid.
        """
        raise NotImplementedError

    @abstractmethod
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
        Generate a complete synchronous answer given a prompt.

        Args:
            prompt: The user query or main instruction.
            system_prompt: Optional system-level instructions to guide
                model behavior and set boundaries.
            history: Optional list of previous conversation turns,
                typically formatted as dicts with 'role' and 'content'.
            temperature: Sampling temperature controlling randomness.
            max_tokens: Maximum tokens to generate in the response.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An `LLMResponse` containing the generated text and usage
            statistics.

        Raises:
            LLMGenerationError: If the generation fails due to API
                errors, rate limits, or context length violations.
        """
        raise NotImplementedError

    @abstractmethod
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
        Generate an answer incrementally using a stream.

        This method yields text chunks as they become available from
        the provider, enabling real-time UI updates.

        Args:
            prompt: The user query or main instruction.
            system_prompt: Optional system-level instructions.
            history: Optional list of previous conversation turns.
            temperature: Sampling temperature controlling randomness.
            max_tokens: Maximum tokens to generate in the response.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Chunks of generated text as strings.

        Raises:
            LLMGenerationError: If the stream fails to initialize or
                interrupts unexpectedly.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify that the LLM provider is reachable and operational.

        Returns:
            True if the provider is healthy and ready to generate,
            False otherwise.
        """
        raise NotImplementedError
