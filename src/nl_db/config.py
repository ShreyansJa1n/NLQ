from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ProviderName = Literal["anthropic", "openai", "openai_compatible"]


class LimitsConfig(BaseModel):
    max_rows: int = Field(default=1000, ge=1, le=1_000_000)
    timeout_s: float = Field(default=10.0, gt=0)
    max_prompt_tokens: int = Field(default=8000, ge=512)


class DatabaseConfig(BaseModel):
    path: Path | None = None
    dialect: Literal["sqlite"] = "sqlite"


class ProviderConfig(BaseModel):
    name: ProviderName = "anthropic"
    model: str = "claude-sonnet-4-6"
    base_url: str | None = None
    api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _check_compatible_needs_base_url(self) -> ProviderConfig:
        if self.name == "openai_compatible" and not self.base_url:
            raise ValueError(
                "provider 'openai_compatible' requires base_url "
                "(e.g. http://localhost:8080/v1)"
            )
        return self


class GenerationConfig(BaseModel):
    """Tuning knobs exposed by the Streamlit playground."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=32, le=8192)
    paraphrase: bool = True
    paraphrase_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    paraphrase_max_output_tokens: int = Field(default=128, ge=32, le=2048)
    auto_limit: bool = True
    num_few_shot: int = Field(default=-1, ge=-1, le=20)  # -1 = use builder default


class _TomlSource(PydanticBaseSettingsSource):
    """Pydantic settings source that reads a TOML file at construction time."""

    def __init__(
        self, settings_cls: type[BaseSettings], path: Path | None
    ) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        if path is not None and path.exists():
            with path.open("rb") as fh:
                self._data = tomllib.load(fh)

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:  # pragma: no cover - pydantic plumbing
        value = self._data.get(field_name)
        return value, field_name, value is not None

    def __call__(self) -> dict[str, Any]:
        return self._data


def _make_settings_cls(toml_path: Path | None) -> type[Settings]:
    """Build a Settings subclass that knows where to look for the TOML file."""

    class _ConfiguredSettings(Settings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                _TomlSource(settings_cls, toml_path),
                file_secret_settings,
            )

    return _ConfiguredSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NL_DB_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    log_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "nl-db" / "logs"
    )


_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
}


CONFIG_FILENAME = "nl-db.toml"


def default_config_path() -> Path:
    """Resolve where the config file lives.

    Precedence:
    1. `NL_DB_CONFIG_FILE` env var (absolute path)
    2. `./nl-db.toml` in the current working directory (project-local)
    """
    override = os.environ.get("NL_DB_CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    return Path.cwd() / CONFIG_FILENAME


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings with precedence (highest → lowest):

    1. Environment variables (NL_DB_*)
    2. `.env` in CWD
    3. Config TOML file (default: ./nl-db.toml, override: NL_DB_CONFIG_FILE)
    4. Built-in defaults

    Provider API keys are pulled from the provider-specific env var
    (ANTHROPIC_API_KEY, etc.) when not set explicitly in TOML/env.
    """
    cls = _make_settings_cls(config_path or default_config_path())
    settings = cls()

    if settings.provider.api_key is None:
        env_var = _API_KEY_ENV[settings.provider.name]
        env_value = os.environ.get(env_var)
        if env_value:
            settings.provider.api_key = SecretStr(env_value)

    return settings


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    """Persist non-secret settings to a TOML file.

    NEVER writes the API key — that always stays in env / .env.
    Returns the path that was written.
    """
    target = path or default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    provider_block: dict[str, Any] = {
        "name": settings.provider.name,
        "model": settings.provider.model,
    }
    if settings.provider.base_url:
        provider_block["base_url"] = settings.provider.base_url

    db_block: dict[str, Any] = {"dialect": settings.db.dialect}
    if settings.db.path is not None:
        db_block["path"] = str(settings.db.path)

    data: dict[str, Any] = {
        "provider": provider_block,
        "db": db_block,
        "limits": settings.limits.model_dump(),
        "generation": settings.generation.model_dump(),
    }

    with target.open("wb") as fh:
        tomli_w.dump(data, fh)
    return target


def require_api_key(settings: Settings) -> str:
    """Return the configured API key as a plain string, or raise."""
    if settings.provider.api_key is None:
        env_var = _API_KEY_ENV[settings.provider.name]
        raise RuntimeError(
            f"No API key configured for provider '{settings.provider.name}'. "
            f"Set {env_var} in your environment or .env file."
        )
    return settings.provider.api_key.get_secret_value()
