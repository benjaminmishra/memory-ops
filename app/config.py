"""Configuration utilities.

This module centralises all environment variables used by the service.  The
defaults defined here are sensible for local development.  In production
environments you should override these values via environment variables.
"""
from functools import lru_cache
import os
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # API keys allowed to access this service.  Comma‑separated list.
    api_keys: str = Field("dev-key", env="API_KEYS")
    # Upstream provider base URL (e.g. https://api.openai.com)
    upstream_base: str = Field("https://api.openai.com", env="UPSTREAM_BASE")
    # Default model name used when none is provided by the caller
    upstream_model: str = Field("gpt-3.5-turbo", env="UPSTREAM_MODEL")
    # API key used when forwarding to the upstream provider if caller
    # does not provide one in Authorization header
    upstream_api_key: str = Field("", env="UPSTREAM_API_KEY")
    # Number of tokens to keep after compression (QR‑HEAD top‑K)
    top_k: int = Field(64, env="TOP_K")
    # Maximum number of API requests per minute allowed per identity
    requests_per_minute: int = Field(60, env="REQUESTS_PER_MINUTE")
    # Maximum number of prompt+completion tokens that may be processed per minute per identity.
    # When this limit is hit, the API will return HTTP 429 until the quota resets.
    tokens_per_minute: int = Field(20000, env="TOKENS_PER_MINUTE")
    # Backwards‑compatible alias for tokens_per_minute.  This environment variable
    # is still respected if provided.
    rate_limit_tpm: int = Field(20000, env="RATE_LIMIT_TPM")
    # Path or HuggingFace model name of the base model to use
    model_name: str = Field("mistralai/Mistral-7B-v0.2", env="MODEL_NAME")
    # Path or HuggingFace identifier of the LoRA weights; if empty,
    # compression will fall back to a naive summariser
    lora_id: str = Field("", env="LORA_ID")
    # Database file path for storing conversation memory
    database_url: str = Field("sqlite:///./memory.db", env="DATABASE_URL")

    @property
    def parsed_api_keys(self) -> set[str]:
        """Return the configured API keys as a set of stripped strings."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    """Return a singleton instance of Settings."""
    return Settings()