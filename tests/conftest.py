from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fixtures.build_sample_db import SAMPLE_DB, build as build_sample_db


@pytest.fixture(scope="session")
def sample_db_path() -> Path:
    """Build the canonical sample.db on demand. Cached per test session."""
    if not SAMPLE_DB.exists():
        build_sample_db(SAMPLE_DB, overwrite=False)
    return SAMPLE_DB


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip provider env vars by default so tests start from a clean slate."""
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "OPENAI_COMPATIBLE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    for key in list(os.environ):
        if key.startswith("NL_DB_"):
            monkeypatch.delenv(key, raising=False)
    yield
