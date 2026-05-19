from __future__ import annotations

from pathlib import Path

import pytest

from nl_db.config import load_settings, require_api_key


def test_defaults_when_no_config_and_no_env(tmp_path: Path) -> None:
    s = load_settings(config_path=tmp_path / "missing.toml")
    assert s.provider.name == "anthropic"
    assert s.provider.api_key is None
    assert s.limits.max_rows == 1000
    assert s.limits.timeout_s == 10.0
    assert s.db.dialect == "sqlite"


def test_env_overrides_provider_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NL_DB_PROVIDER__NAME", "openai")
    monkeypatch.setenv("NL_DB_PROVIDER__MODEL", "gpt-4o")
    s = load_settings(config_path=tmp_path / "missing.toml")
    assert s.provider.name == "openai"
    assert s.provider.model == "gpt-4o"


def test_api_key_pulled_from_provider_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NL_DB_PROVIDER__NAME", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    s = load_settings(config_path=tmp_path / "missing.toml")
    assert require_api_key(s) == "sk-fake"


def test_missing_api_key_raises(tmp_path: Path) -> None:
    s = load_settings(config_path=tmp_path / "missing.toml")
    with pytest.raises(RuntimeError, match="No API key configured"):
        require_api_key(s)


def test_openai_compatible_requires_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NL_DB_PROVIDER__NAME", "openai_compatible")
    with pytest.raises(ValueError, match="requires base_url"):
        load_settings(config_path=tmp_path / "missing.toml")


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[provider]
name = "openai_compatible"
model = "local-llama"
base_url = "http://localhost:8080/v1"

[limits]
max_rows = 50
timeout_s = 2.5
""".strip()
    )
    s = load_settings(config_path=cfg)
    assert s.provider.name == "openai_compatible"
    assert s.provider.base_url == "http://localhost:8080/v1"
    assert s.limits.max_rows == 50
    assert s.limits.timeout_s == 2.5


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[provider]
name = "anthropic"
model = "claude-sonnet-4-6"
""".strip()
    )
    monkeypatch.setenv("NL_DB_PROVIDER__MODEL", "claude-opus-4-7")
    s = load_settings(config_path=cfg)
    assert s.provider.model == "claude-opus-4-7"
