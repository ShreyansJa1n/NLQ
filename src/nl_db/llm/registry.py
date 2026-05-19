from __future__ import annotations

from ..config import Settings, require_api_key
from .anthropic_provider import AnthropicProvider
from .openai_compatible import OpenAICompatibleProvider
from .openai_provider import OpenAIProvider
from .provider import LLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    """Construct the configured LLMProvider. Pulls API key as needed."""
    name = settings.provider.name
    model = settings.provider.model

    if name == "anthropic":
        return AnthropicProvider(model=model, api_key=require_api_key(settings))
    if name == "openai":
        return OpenAIProvider(model=model, api_key=require_api_key(settings))
    if name == "openai_compatible":
        assert settings.provider.base_url is not None  # validated by ProviderConfig
        api_key = (
            settings.provider.api_key.get_secret_value()
            if settings.provider.api_key
            else None
        )
        return OpenAICompatibleProvider(
            model=model,
            base_url=settings.provider.base_url,
            api_key=api_key,
        )
    raise ValueError(f"Unknown provider: {name}")
