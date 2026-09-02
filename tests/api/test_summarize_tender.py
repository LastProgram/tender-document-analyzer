from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_settings, get_summary_service
from app.core.config import Settings
from app.core.exceptions import (
    ApplicationError,
    EncryptedPdfError,
    InvalidLLMResponseError,
    InvalidPdfError,
    LLMConfigurationError,
    LLMTimeoutError,
    LLMUnavailableError,
    PdfPageLimitExceededError,
    PdfTextNotFoundError,
    PdfTooLargeError,
    UnsupportedPdfTypeError,
)
from app.main import create_app
from app.schemas.document import UploadedDocument
from app.schemas.tender import TenderSummary

SUMMARIZE_URL = "/api/v1/tenders/summarize"


class FakeSummaryService:
    def __init__(
        self,
        result: TenderSummary | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.documents: list[UploadedDocument] = []

    async def summarize(self, document: UploadedDocument) -> TenderSummary:
        self.documents.append(document)
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise AssertionError("Summary result was not configured")
        return self._result


def make_summary() -> TenderSummary:
    return TenderSummary(
        contract_amount="1 000 000 рублей",
        deadlines=["до 30 ноября 2026 года"],
        contractor_requirements=["наличие действующей лицензии"],
        penalties=["пеня за каждый день просрочки"],
    )


def create_test_app(
    service: FakeSummaryService,
    settings: Settings | None = None,
) -> FastAPI:
    application = create_app(settings)
    application.dependency_overrides[get_summary_service] = lambda: service
    if settings is not None:
        application.dependency_overrides[get_settings] = lambda: settings
    return application


@asynccontextmanager
async def request_client(
    service: FakeSummaryService,
    settings: Settings | None = None,
    *,
    raise_app_exceptions: bool = True,
) -> AsyncIterator[AsyncClient]:
    application = create_test_app(service, settings)
    transport = ASGITransport(
        app=application,
        raise_app_exceptions=raise_app_exceptions,
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_multipart_upload_returns_summary_without_envelope() -> None:
    summary = make_summary()
    service = FakeSummaryService(result=summary)

    async with request_client(service) as client:
        response = await client.post(
            SUMMARIZE_URL,
            files={"file": ("tender.pdf", b"%PDF-content", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == summary.model_dump(mode="json")
    assert set(response.json()) == {
        "contract_amount",
        "deadlines",
        "contractor_requirements",
        "penalties",
    }
    assert service.documents == [
        UploadedDocument(
            filename="tender.pdf",
            content_type="application/pdf",
            content=b"%PDF-content",
        )
    ]


@pytest.mark.asyncio
async def test_missing_file_uses_fastapi_validation() -> None:
    service = FakeSummaryService(result=make_summary())

    async with request_client(service) as client:
        response = await client.post(SUMMARIZE_URL)

    assert response.status_code == 422
    assert service.documents == []


@pytest.mark.asyncio
async def test_unsupported_content_type_returns_415() -> None:
    service = FakeSummaryService(error=UnsupportedPdfTypeError())

    async with request_client(service) as client:
        response = await client.post(
            SUMMARIZE_URL,
            files={"file": ("tender.txt", b"text", "text/plain")},
        )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "UNSUPPORTED_PDF_TYPE"


@pytest.mark.asyncio
async def test_actual_upload_size_over_limit_returns_413() -> None:
    service = FakeSummaryService(result=make_summary())
    settings = Settings(_env_file=None, max_upload_size_mb=1)
    content = b"x" * (1024 * 1024 + 1)

    async with request_client(service, settings) as client:
        response = await client.post(
            SUMMARIZE_URL,
            files={"file": ("tender.pdf", content, "application/pdf")},
        )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "PDF_TOO_LARGE"
    assert service.documents == []


@pytest.mark.asyncio
async def test_request_body_limit_runs_before_multipart_parsing() -> None:
    service = FakeSummaryService(result=make_summary())
    settings = Settings(_env_file=None, max_upload_size_mb=1)

    async with request_client(service, settings) as client:
        response = await client.post(
            SUMMARIZE_URL,
            headers={
                "Content-Type": "multipart/form-data; boundary=invalid",
                "X-Request-ID": "oversized-request",
            },
            content=b"x" * (1024 * 1024 + 1),
        )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "PDF_TOO_LARGE"
    assert response.headers["X-Request-ID"] == "oversized-request"
    assert service.documents == []


@pytest.mark.parametrize(
    ("error", "status_code", "code", "message"),
    [
        (
            UnsupportedPdfTypeError(),
            415,
            "UNSUPPORTED_PDF_TYPE",
            "Only PDF files are supported.",
        ),
        (
            PdfTooLargeError(),
            413,
            "PDF_TOO_LARGE",
            "PDF exceeds the upload size limit.",
        ),
        (
            PdfPageLimitExceededError(),
            413,
            "PDF_PAGE_LIMIT_EXCEEDED",
            "PDF exceeds the page limit.",
        ),
        (InvalidPdfError(), 422, "INVALID_PDF", "PDF is invalid."),
        (
            EncryptedPdfError(),
            422,
            "ENCRYPTED_PDF",
            "Encrypted PDF files are not supported.",
        ),
        (
            PdfTextNotFoundError(),
            422,
            "PDF_TEXT_NOT_FOUND",
            "PDF does not contain extractable text.",
        ),
        (
            LLMUnavailableError(),
            503,
            "LLM_UNAVAILABLE",
            "LLM service is unavailable.",
        ),
        (
            LLMTimeoutError(),
            504,
            "LLM_TIMEOUT",
            "LLM request timed out.",
        ),
        (
            LLMConfigurationError(),
            503,
            "LLM_CONFIGURATION_ERROR",
            "LLM service is not configured correctly.",
        ),
        (
            InvalidLLMResponseError(),
            502,
            "INVALID_LLM_RESPONSE",
            "LLM returned an invalid response.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_application_error_has_public_status_and_code(
    error: ApplicationError,
    status_code: int,
    code: str,
    message: str,
) -> None:
    service = FakeSummaryService(error=error)

    async with request_client(service) as client:
        response = await client.post(
            SUMMARIZE_URL,
            files={"file": ("tender.pdf", b"%PDF-content", "application/pdf")},
        )

    assert response.status_code == status_code
    assert response.json() == {"detail": {"code": code, "message": message}}


@pytest.mark.asyncio
async def test_unknown_error_returns_safe_500() -> None:
    service = FakeSummaryService(error=RuntimeError("private failure details"))

    async with request_client(service, raise_app_exceptions=False) as client:
        response = await client.post(
            SUMMARIZE_URL,
            files={"file": ("tender.pdf", b"%PDF-content", "application/pdf")},
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "code": "INTERNAL_ERROR",
            "message": "Internal server error.",
        }
    }
    assert "private failure details" not in response.text
