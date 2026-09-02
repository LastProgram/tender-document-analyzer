from collections.abc import Sequence
from http import HTTPStatus
from time import perf_counter
from typing import Any, Protocol

from httpx import RequestError, TimeoutException
from ollama import AsyncClient, ChatResponse, ResponseError
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import (
    InvalidLLMResponseError,
    LLMConfigurationError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.core.logging import elapsed_ms, log_event
from app.prompts import consolidation_messages, extraction_messages
from app.schemas.document import DocumentChunk
from app.schemas.tender import TenderSummary

# Ошибки конфигурации отделены от временных сбоев, потому что
# повторный запрос не исправит неверные настройки.
_TIMEOUT_STATUS_CODES = frozenset(
    {HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.GATEWAY_TIMEOUT}
)
_CONFIGURATION_STATUS_CODES = frozenset(
    {
        HTTPStatus.BAD_REQUEST,
        HTTPStatus.UNAUTHORIZED,
        HTTPStatus.FORBIDDEN,
        HTTPStatus.NOT_FOUND,
        HTTPStatus.UNPROCESSABLE_ENTITY,
    }
)


class _OllamaClient(Protocol):
    async def chat(self, **kwargs: Any) -> ChatResponse: ...


class OllamaTenderExtractor:
    """Преобразует ответы Ollama в проверенную структуру тендерной выжимки."""

    def __init__(self, settings: Settings, client: _OllamaClient | None = None) -> None:
        self._model = settings.ollama_model
        self._client = (
            client
            if client is not None
            else AsyncClient(
                host=settings.ollama_base_url,
                timeout=settings.ollama_timeout_seconds,
            )
        )

    async def extract(self, chunk: DocumentChunk) -> TenderSummary:
        started_at = perf_counter()
        summary = await self._request(extraction_messages(chunk))
        log_event(
            "llm_extraction_completed",
            duration_ms=elapsed_ms(started_at),
            model=self._model,
        )
        return summary

    async def consolidate(
        self,
        summaries: Sequence[TenderSummary],
    ) -> TenderSummary:
        started_at = perf_counter()
        summary = await self._request(consolidation_messages(summaries))
        log_event(
            "llm_consolidation_completed",
            duration_ms=elapsed_ms(started_at),
            model=self._model,
        )
        return summary

    async def _request(self, messages: list[dict[str, str]]) -> TenderSummary:
        try:
            response = await self._client.chat(
                model=self._model,
                messages=messages,
                format=TenderSummary.model_json_schema(),
                # Нулевая температура уменьшает разброс ответов
                # при извлечении по строгой схеме.
                options={"temperature": 0},
            )
        except TimeoutException as exc:
            raise LLMTimeoutError from exc
        except RequestError as exc:
            raise LLMUnavailableError from exc
        except ConnectionError as exc:
            raise LLMUnavailableError from exc
        except ResponseError as exc:
            if exc.status_code in _TIMEOUT_STATUS_CODES:
                raise LLMTimeoutError from exc
            if exc.status_code in _CONFIGURATION_STATUS_CODES:
                raise LLMConfigurationError from exc
            raise LLMUnavailableError from exc

        content = response.message.content
        if not content or not content.strip():
            raise InvalidLLMResponseError

        # JSON Schema задаёт модели ожидаемый формат ответа.
        # Соответствие публичному контракту подтверждает проверка Pydantic.
        try:
            return TenderSummary.model_validate_json(content)
        except ValidationError as exc:
            raise InvalidLLMResponseError from exc
