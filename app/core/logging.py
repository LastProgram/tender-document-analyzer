import json
import logging
import re
from contextvars import ContextVar
from time import perf_counter
from typing import Any
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_MAX_LOG_FILENAME_LENGTH = 255

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

logger = logging.getLogger("app.processing")


class JsonLogFormatter(logging.Formatter):
    """Форматирует безопасные события приложения в строки JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "event": getattr(record, "event", record.getMessage()),
        }
        payload.update(getattr(record, "log_fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class RequestTracingMiddleware:
    """Связывает события одного HTTP запроса проверенным идентификатором."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = resolve_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        started_at = perf_counter()
        scope.setdefault("state", {})["request_id"] = request_id
        scope["state"]["request_started_at"] = started_at
        token = _request_id.set(request_id)

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self._app(scope, receive, send_with_request_id)
        finally:
            _request_id.reset(token)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logger.setLevel(logging.INFO)


def resolve_request_id(candidate: str | None) -> str:
    # Ограниченный набор символов не позволяет клиентскому значению
    # создавать ложные границы между записями журнала.
    if candidate is not None and _REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid4().hex


def sanitize_filename(filename: str | None) -> str | None:
    if filename is None:
        return None
    # Замена управляющих символов не позволяет имени файла
    # создавать поддельные записи журнала.
    sanitized = "".join(
        character if character.isprintable() else "_" for character in filename
    )
    return sanitized[:_MAX_LOG_FILENAME_LENGTH]


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    request_id: str | None = None,
    include_exception: bool = False,
    **fields: object,
) -> None:
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "request_id": request_id or _request_id.get(),
            "log_fields": fields,
        },
        exc_info=include_exception,
    )
