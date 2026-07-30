"""
LLM Factory for ResearchRAG.

This module defines `LLMFactory`, responsible for instantiating the
configured language model client. It completely isolates LLM provider
specifics (e.g., OpenAI, Groq) from the rest of the application.
The pipeline and other components interact only with this factory and
the provider-independent `BaseLLM` interface it returns.
"""

from typing import Dict, Optional, Type

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.llms.base_llm import BaseLLM

logger = get_logger(__name__)


class LLMFactoryError(Exception):
    """Base exception for errors raised by `LLMFactory`."""


class UnsupportedProviderError(LLMFactoryError):
    """Raised when the configured or requested LLM provider is not supported."""


class LLMConfigurationError(LLMFactoryError):
    """Raised when a provider fails to initialize due to invalid configuration."""


class LLMFactory:
    """
    Manages LLM provider selection and instantiates the active client.

    `LLMFactory` is the only component in ResearchRAG that should know
    about concrete LLM provider classes. The orchestration layer
    (`RAGPipeline`) depends solely on this factory to obtain a
    `BaseLLM` instance. Switching providers (e.g., from OpenAI to
    Groq) requires only a configuration change, not a code change in
    the pipeline.
    """

    # Registry mapping provider names to their implementation class.
    # Future providers can be added dynamically via `register_provider`
    # without modifying this file.
    _PROVIDER_REGISTRY: Dict[str, Type[BaseLLM]] = {}

    def __init__(self, provider_name: Optional[str] = None) -> None:
        """
        Initialize the LLM factory with the configured provider.

        Args:
            provider_name: Name of the LLM provider to use. Defaults
                to the provider configured in application settings.

        Raises:
            UnsupportedProviderError: If the requested provider is
                not supported.
            LLMConfigurationError: If the selected provider fails to
                initialize due to configuration issues.
        """
        self._provider_name: str = provider_name or settings.llm.provider
        self._client: BaseLLM = self._create_client(self._provider_name)

        logger.info("LLMFactory initialized with provider: %s", self._provider_name)

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLM]) -> None:
        """
        Register a new LLM provider implementation.

        This allows new providers (e.g., Anthropic Claude, Gemini,
        Ollama) to be added to the system without modifying existing
        logic in `LLMFactory`.

        Args:
            name: Identifier used to select this provider via
                configuration.
            provider_class: A concrete `BaseLLM` subclass
                implementing the provider.

        Raises:
            TypeError: If `provider_class` does not subclass
                `BaseLLM`.
        """
        if not issubclass(provider_class, BaseLLM):
            raise TypeError(
                f"provider_class must be a subclass of BaseLLM, got {provider_class}"
            )
        cls._PROVIDER_REGISTRY[name] = provider_class
        logger.info("Registered LLM provider: %s", name)

    def get_client(self) -> BaseLLM:
        """
        Retrieve the instantiated LLM client.

        Returns:
            The configured `BaseLLM` instance.
        """
        return self._client

    def _create_client(self, provider_name: str) -> BaseLLM:
        """
        Instantiate the LLM client corresponding to the given name.

        Args:
            provider_name: Name of the LLM provider to instantiate.

        Returns:
            An initialized instance of a `BaseLLM` subclass.

        Raises:
            UnsupportedProviderError: If `provider_name` is not
                recognized.
            LLMConfigurationError: If the client class fails to
                initialize.
        """
        provider_class = self._get_provider_class(provider_name)

        logger.info("Creating LLM client instance for provider: %s", provider_name)
        try:
            return provider_class()
        except Exception as error:
            raise LLMConfigurationError(
                f"Failed to initialize LLM provider '{provider_name}': {error}"
            ) from error

    def _get_provider_class(self, provider_name: str) -> Type[BaseLLM]:
        """
        Resolve the provider name to a concrete `BaseLLM` class.

        Local imports are used for built-in providers to prevent
        top-level ImportErrors if a module is currently empty or
        missing external dependencies, adhering to lazy-loading.

        Args:
            provider_name: Name of the LLM provider.

        Returns:
            The `BaseLLM` subclass for the provider.

        Raises:
            UnsupportedProviderError: If the provider is not supported.
            LLMConfigurationError: If the provider's module cannot
                be imported.
        """
        # 1. Check dynamic registry first (allows overriding built-ins)
        if provider_name in self._PROVIDER_REGISTRY:
            return self._PROVIDER_REGISTRY[provider_name]

        # 2. Fall back to built-in providers using local imports.
        if provider_name == "openai":
            try:
                from src.llms.openai_client import OpenAIClient
                return OpenAIClient
            except ImportError as error:
                raise LLMConfigurationError(
                    f"The 'openai' provider module is missing or incomplete: {error}"
                ) from error
        elif provider_name == "groq":
            try:
                from src.llms.groq_client import GroqClient
                return GroqClient
            except ImportError as error:
                raise LLMConfigurationError(
                    f"The 'groq' provider module is missing or incomplete: {error}"
                ) from error
        else:
            supported = sorted(list(self._PROVIDER_REGISTRY.keys()) + ["groq", "openai"])
            raise UnsupportedProviderError(
                f"Unsupported LLM provider '{provider_name}'. "
                f"Supported providers: {supported}"
            )
