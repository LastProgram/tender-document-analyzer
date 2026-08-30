import importlib
import sys

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app


def test_create_app_returns_fastapi_application() -> None:
    application = create_app()

    assert isinstance(application, FastAPI)
    assert application.title == "Tender Summarizer"


def test_importing_app_does_not_require_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("app.main", None)
    monkeypatch.delitem(sys.modules, "ollama", raising=False)

    module = importlib.import_module("app.main")

    assert isinstance(module.app, FastAPI)
    assert "ollama" not in sys.modules


def test_default_settings_are_valid() -> None:
    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "qwen3:8b"
    assert settings.ollama_timeout_seconds == 120.0
    assert settings.max_upload_size_mb == 20
    assert settings.max_pdf_pages == 300
    assert settings.llm_chunk_max_chars == 30_000
    assert settings.chunk_overlap_pages == 1


def test_environment_overrides_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model:latest")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "10")
    monkeypatch.setenv("MAX_PDF_PAGES", "150")
    monkeypatch.setenv("LLM_CHUNK_MAX_CHARS", "12000")
    monkeypatch.setenv("CHUNK_OVERLAP_PAGES", "2")

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://ollama.internal:11434"
    assert settings.ollama_model == "custom-model:latest"
    assert settings.ollama_timeout_seconds == 45.5
    assert settings.max_upload_size_mb == 10
    assert settings.max_pdf_pages == 150
    assert settings.llm_chunk_max_chars == 12_000
    assert settings.chunk_overlap_pages == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_upload_size_mb", 0),
        ("ollama_timeout_seconds", -1),
        ("chunk_overlap_pages", -1),
    ],
)
def test_invalid_numeric_settings_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
