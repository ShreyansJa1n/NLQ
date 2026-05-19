from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


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
