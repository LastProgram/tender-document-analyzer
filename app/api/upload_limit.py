from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import application_error_response
from app.core.exceptions import PdfTooLargeError
from app.core.logging import elapsed_ms, log_event


class _RequestBodyTooLarge(Exception):
    pass


class UploadSizeLimitMiddleware:
    """Останавливает приём тела запроса до разбора multipart."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        content_length = self._content_length(scope)
        if content_length is not None and content_length > self._max_bytes:
            await self._reject(scope, receive, send)
            return

        received_bytes = 0

        async def receive_limited() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self._max_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self._app(scope, receive_limited, send)
        except _RequestBodyTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        value = Headers(scope=scope).get("content-length")
        if value is None:
            return None
        try:
            content_length = int(value)
        except ValueError:
            return None
        return content_length if content_length >= 0 else None

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        state = scope.get("state", {})
        request_id = state.get("request_id", "-")
        started_at = state.get("request_started_at")
        fields: dict[str, object] = {
            "error_code": "PDF_TOO_LARGE",
            "http_status": 413,
        }
        if isinstance(started_at, float):
            fields["duration_ms"] = elapsed_ms(started_at)
        log_event(
            "tender_processing_failed",
            request_id=request_id,
            **fields,
        )
        response = application_error_response(PdfTooLargeError, request_id)
        await response(scope, receive, send)
