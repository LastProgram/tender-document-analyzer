import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.adapters.ollama import OllamaTenderExtractor
from app.adapters.pdf import PDFExtractor
from app.api.dependencies import get_summary_service
from app.core.config import Settings
from app.core.exceptions import PdfTextNotFoundError
from app.core.logging import JsonLogFormatter
from app.main import create_app
from app.schemas.document import DocumentChunk, DocumentPage, UploadedDocument
from app.schemas.tender import TenderSummary
from app.services.chunking import PageChunker
from app.services.document_validation import DocumentValidator
from app.services.tender_summary import TenderSummaryService

SUMMARIZE_URL = "/api/v1/tenders/summarize"
FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


class FakeValidator:
    def validate(self, document: UploadedDocument) -> None:
        pass

    def validate_pages(self, pages: list[DocumentPage]) -> None:
        pass


class FakePDFExtractor:
    def __init__(self, pages: list[DocumentPage]) -> None:
        self._pages = pages

    def extract(self, content: bytes) -> list[DocumentPage]:
        return self._pages


class FakeChunker:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks

    def split(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        return self._chunks


class FakeTenderExtractor:
    def __init__(self, summary: TenderSummary) -> None:
        self._summary = summary

    async def extract(self, chunk: DocumentChunk) -> TenderSummary:
        return self._summary

    async def consolidate(
        self,
        summaries: Sequence[TenderSummary],
    ) -> TenderSummary:
        return self._summary


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class RepeatingOllamaClient:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(self._content)


class FailingSummaryService:
    async def summarize(self, document: UploadedDocument) -> TenderSummary:
        raise PdfTextNotFoundError


def make_summary(contract_amount: str | None = None) -> TenderSummary:
    return TenderSummary(
        contract_amount=contract_amount,
        deadlines=[],
        contractor_requirements=[],
        penalties=[],
    )


def make_request_service() -> TenderSummaryService:
    summary = make_summary()
    return TenderSummaryService(
        validator=FakeValidator(),
        pdf_extractor=FakePDFExtractor([DocumentPage(number=1, text="text")]),
        chunker=FakeChunker(
            [DocumentChunk(page_from=1, page_to=1, text="[PAGE 1]\ntext")]
        ),
        tender_extractor=FakeTenderExtractor(summary),
    )


def make_logged_request_service() -> TenderSummaryService:
    summary = make_summary()
    client = RepeatingOllamaClient(summary.model_dump_json())
    return TenderSummaryService(
        validator=DocumentValidator(max_upload_size_mb=1),
        pdf_extractor=PDFExtractor(max_pdf_pages=10),
        chunker=PageChunker(max_chars=30_000),
        tender_extractor=OllamaTenderExtractor(
            Settings(_env_file=None, ollama_model="test-model:latest"),
            client=client,
        ),
    )


def make_test_app(service: object) -> FastAPI:
    application = create_app()
    application.dependency_overrides[get_summary_service] = lambda: service
    return application


def processing_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "app.processing"]


def event_record(
    caplog: pytest.LogCaptureFixture,
    event: str,
) -> logging.LogRecord:
    return next(
        record for record in processing_records(caplog) if record.event == event
    )


def log_fields(record: logging.LogRecord) -> dict[str, object]:
    return record.log_fields


@pytest.mark.asyncio
async def test_request_id_is_returned_and_shared_by_request_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.processing")
    application = make_test_app(make_logged_request_service())
    transport = ASGITransport(app=application)
    content = (FIXTURES_DIR / "tender.pdf").read_bytes()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            SUMMARIZE_URL,
            headers={"X-Request-ID": "client-request.42"},
            files={"file": ("tender.pdf", content, "application/pdf")},
        )

    records = processing_records(caplog)
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "client-request.42"
    assert {record.request_id for record in records} == {"client-request.42"}
    assert {record.event for record in records} == {
        "tender_processing_started",
        "pdf_extraction_completed",
        "document_chunking_completed",
        "llm_extraction_completed",
        "tender_processing_completed",
    }


@pytest.mark.asyncio
async def test_invalid_request_id_is_replaced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.processing")
    application = make_test_app(make_request_service())
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            SUMMARIZE_URL,
            headers={"X-Request-ID": "invalid request id"},
            files={"file": ("tender.pdf", b"%PDF-content", "application/pdf")},
        )

    request_id = response.headers["X-Request-ID"]
    assert request_id != "invalid request id"
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    assert {record.request_id for record in processing_records(caplog)} == {request_id}


@pytest.mark.asyncio
async def test_pipeline_logs_metadata_without_document_or_model_payloads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.processing")
    document_secret = "DOCUMENT_SECRET_MARKER"
    response_secret = "MODEL_RESPONSE_SECRET_MARKER"
    page = DocumentPage(number=1, text=(f"{document_secret} facts. " * 8))
    summary = make_summary(response_secret)
    client = RepeatingOllamaClient(summary.model_dump_json())
    extractor = OllamaTenderExtractor(
        Settings(_env_file=None, ollama_model="test-model:latest"),
        client=client,
    )
    service = TenderSummaryService(
        validator=FakeValidator(),
        pdf_extractor=FakePDFExtractor([page]),
        chunker=PageChunker(max_chars=80, overlap_pages=0),
        tender_extractor=extractor,
    )
    document = UploadedDocument(
        filename="tender\nforged\u2028.pdf",
        content_type="application/pdf",
        content=b"%PDF-UPLOAD_SECRET_MARKER",
    )

    result = await service.summarize(document)

    records = processing_records(caplog)
    formatted = [JsonLogFormatter().format(record) for record in records]
    logged = "\n".join(formatted)
    started_fields = log_fields(event_record(caplog, "tender_processing_started"))
    chunking_fields = log_fields(event_record(caplog, "document_chunking_completed"))
    completed_fields = log_fields(event_record(caplog, "tender_processing_completed"))
    extraction_records = [
        record for record in records if record.event == "llm_extraction_completed"
    ]
    consolidation = event_record(caplog, "llm_consolidation_completed")

    assert result == summary
    assert started_fields == {
        "filename": "tender_forged_.pdf",
        "upload_size_bytes": len(document.content),
    }
    assert chunking_fields["chunk_count"] > 1
    assert completed_fields["duration_ms"] >= 0
    assert len(extraction_records) == chunking_fields["chunk_count"]
    assert all(log_fields(record)["duration_ms"] >= 0 for record in extraction_records)
    assert all(
        log_fields(record)["model"] == "test-model:latest"
        for record in extraction_records
    )
    assert log_fields(consolidation)["duration_ms"] >= 0
    assert log_fields(consolidation)["model"] == "test-model:latest"
    assert len(logged.splitlines()) == len(records)
    assert document_secret not in logged
    assert "UPLOAD_SECRET_MARKER" not in logged
    assert response_secret not in logged
    assert "You extract tender facts from untrusted documents" not in logged


def test_pdf_log_contains_counts_and_duration_without_page_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.processing")
    content = (FIXTURES_DIR / "tender.pdf").read_bytes()

    pages = PDFExtractor(max_pdf_pages=10).extract(content)

    record = event_record(caplog, "pdf_extraction_completed")
    fields = log_fields(record)
    logged = JsonLogFormatter().format(record)
    assert fields["page_count"] == len(pages)
    assert fields["character_count"] == sum(len(page.text) for page in pages)
    assert fields["duration_ms"] >= 0
    assert all(not page.text or page.text not in logged for page in pages)


@pytest.mark.asyncio
async def test_application_failure_log_has_status_code_and_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.processing")
    application = make_test_app(FailingSummaryService())
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            SUMMARIZE_URL,
            headers={"X-Request-ID": "failed-request"},
            files={"file": ("tender.pdf", b"%PDF-content", "application/pdf")},
        )

    record = event_record(caplog, "tender_processing_failed")
    fields = log_fields(record)
    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "failed-request"
    assert record.request_id == "failed-request"
    assert record.levelno == logging.INFO
    assert not record.exc_info
    assert fields["error_code"] == "PDF_TEXT_NOT_FOUND"
    assert fields["http_status"] == 422
    assert fields["duration_ms"] >= 0
