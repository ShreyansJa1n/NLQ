from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from nl_db.config import (
    GenerationConfig,
    LimitsConfig,
    ProviderConfig,
    Settings,
    default_config_path,
    load_settings,
    save_settings,
)


def test_default_config_path_is_cwd_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NL_DB_CONFIG_FILE", raising=False)
    assert default_config_path() == tmp_path / "nl-db.toml"


def test_default_config_path_honors_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NL_DB_CONFIG_FILE", str(tmp_path / "elsewhere.toml"))
    assert default_config_path() == tmp_path / "elsewhere.toml"


def test_save_then_load_roundtrips(tmp_path: Path) -> None:
    cfg = tmp_path / "nl-db.toml"
    src = Settings(
        provider=ProviderConfig(
            name="openai_compatible",
            model="apple.local",
            base_url="http://localhost:8080/v1",
        ),
        limits=LimitsConfig(max_rows=250, timeout_s=4.0),
        generation=GenerationConfig(
            temperature=0.4,
            paraphrase=False,
            num_few_shot=2,
        ),
    )
    written = save_settings(src, cfg)
    assert written == cfg
    assert cfg.exists()

    loaded = load_settings(config_path=cfg)
    assert loaded.provider.name == "openai_compatible"
    assert loaded.provider.model == "apple.local"
    assert loaded.provider.base_url == "http://localhost:8080/v1"
    assert loaded.limits.max_rows == 250
    assert loaded.limits.timeout_s == 4.0
    assert loaded.generation.temperature == 0.4
    assert loaded.generation.paraphrase is False
    assert loaded.generation.num_few_shot == 2


def test_save_never_writes_api_key(tmp_path: Path) -> None:
    from pydantic import SecretStr

    cfg = tmp_path / "nl-db.toml"
    src = Settings(
        provider=ProviderConfig(
            name="anthropic",
            model="claude-haiku-4-5-20251001",
            api_key=SecretStr("sk-ant-supersecret"),
        ),
    )
    save_settings(src, cfg)
    raw = cfg.read_text()
    assert "supersecret" not in raw
    assert "api_key" not in raw

    parsed = tomllib.loads(raw)
    assert "api_key" not in parsed["provider"]


def test_save_includes_db_path_when_set(tmp_path: Path) -> None:
    cfg = tmp_path / "nl-db.toml"
    db = tmp_path / "data.db"
    src = Settings()
    src.db.path = db
    save_settings(src, cfg)

    parsed = tomllib.loads(cfg.read_text())
    assert parsed["db"]["path"] == str(db)


def test_save_omits_base_url_when_unset(tmp_path: Path) -> None:
    cfg = tmp_path / "nl-db.toml"
    src = Settings()  # default provider = anthropic, no base_url
    save_settings(src, cfg)
    parsed = tomllib.loads(cfg.read_text())
    assert "base_url" not in parsed["provider"]


def test_generation_block_loaded_from_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "nl-db.toml"
    cfg.write_text(
        """
[provider]
name = "anthropic"
model = "claude-haiku-4-5-20251001"

[generation]
temperature = 0.7
max_output_tokens = 1024
paraphrase = false
auto_limit = false
num_few_shot = 1
""".strip()
    )
    s = load_settings(config_path=cfg)
    assert s.generation.temperature == 0.7
    assert s.generation.max_output_tokens == 1024
    assert s.generation.paraphrase is False
    assert s.generation.auto_limit is False
    assert s.generation.num_few_shot == 1
