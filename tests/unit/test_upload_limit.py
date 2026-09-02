import json
from time import perf_counter

import pytest
from starlette.types import Message, Receive, Scope, Send

from app.api.upload_limit import UploadSizeLimitMiddleware


@pytest.mark.asyncio
async def test_streamed_body_is_stopped_when_received_bytes_exceed_limit() -> None:
    downstream_completed = False

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal downstream_completed
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        downstream_completed = True

    incoming: list[Message] = [
        {"type": "http.request", "body": b"123", "more_body": True},
        {"type": "http.request", "body": b"456", "more_body": False},
    ]
    incoming_iterator = iter(incoming)
    sent: list[Message] = []

    async def receive() -> Message:
        return next(incoming_iterator)

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = UploadSizeLimitMiddleware(downstream, max_bytes=5)
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/tenders/summarize",
        "headers": [],
        "state": {
            "request_id": "streamed-request",
            "request_started_at": perf_counter(),
        },
    }

    await middleware(scope, receive, send)

    assert downstream_completed is False
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"]["code"] == "PDF_TOO_LARGE"
