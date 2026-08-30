from collections.abc import Sequence
from threading import get_ident

import pytest

from app.core.exceptions import (
    InvalidLLMResponseError,
    InvalidPdfError,
    PdfTextNotFoundError,
)
from app.schemas.document import DocumentChunk, DocumentPage, UploadedDocument
from app.schemas.tender import TenderSummary
from app.services.tender_summary import TenderSummaryService


class FakeValidator:
    def __init__(
        self,
        calls: list[str],
        upload_error: Exception | None = None,
        pages_error: Exception | None = None,
    ) -> None:
        self._calls = calls
        self._upload_error = upload_error
        self._pages_error = pages_error

    def validate(self, document: UploadedDocument) -> None:
        self._calls.append("validate_upload")
        if self._upload_error is not None:
            raise self._upload_error

    def validate_pages(self, pages: list[DocumentPage]) -> None:
        self._calls.append("validate_pages")
        if self._pages_error is not None:
            raise self._pages_error


class FakePDFExtractor:
    def __init__(self, calls: list[str], pages: list[DocumentPage]) -> None:
        self._calls = calls
        self._pages = pages
        self.thread_id: int | None = None

    def extract(self, content: bytes) -> list[DocumentPage]:
        self._calls.append("parse_pdf")
        self.thread_id = get_ident()
        return self._pages


class FakeChunker:
    def __init__(self, calls: list[str], chunks: list[DocumentChunk]) -> None:
        self._calls = calls
        self._chunks = chunks
        self.received_pages: list[DocumentPage] | None = None

    def split(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        self._calls.append("split_pages")
        self.received_pages = pages
        return self._chunks


class FakeTenderExtractor:
    def __init__(
        self,
        calls: list[str],
        results: Sequence[TenderSummary],
        extract_error_at: int | None = None,
        extract_error: Exception | None = None,
        consolidation_error: Exception | None = None,
        consolidated_result: TenderSummary | None = None,
    ) -> None:
        self._calls = calls
        self._results = iter(results)
        self._extract_error_at = extract_error_at
        self._extract_error = extract_error
        self._consolidation_error = consolidation_error
        self._consolidated_result = consolidated_result
        self.extracted_chunks: list[DocumentChunk] = []
        self.consolidated_summaries: list[TenderSummary] | None = None

    async def extract(self, chunk: DocumentChunk) -> TenderSummary:
        self._calls.append(f"extract_{chunk.page_from}")
        self.extracted_chunks.append(chunk)
        if len(self.extracted_chunks) == self._extract_error_at:
            if self._extract_error is None:
                raise AssertionError("Extraction error was not configured")
            raise self._extract_error
        return next(self._results)

    async def consolidate(
        self,
        summaries: Sequence[TenderSummary],
    ) -> TenderSummary:
        self._calls.append("consolidate")
        self.consolidated_summaries = list(summaries)
        if self._consolidation_error is not None:
            raise self._consolidation_error
        if self._consolidated_result is None:
            raise AssertionError("Consolidated result was not configured")
        return self._consolidated_result


def make_document() -> UploadedDocument:
    return UploadedDocument(
        filename="tender.pdf",
        content_type="application/pdf",
        content=b"%PDF-test",
    )


def make_pages() -> list[DocumentPage]:
    return [
        DocumentPage(number=1, text="first"),
        DocumentPage(number=2, text="second"),
    ]


def make_chunk(number: int) -> DocumentChunk:
    return DocumentChunk(
        page_from=number,
        page_to=number,
        text=f"[PAGE {number}]\ntext",
    )


def make_summary(amount: str | None = None) -> TenderSummary:
    return TenderSummary(
        contract_amount=amount,
        deadlines=[],
        contractor_requirements=[],
        penalties=[],
    )


@pytest.mark.asyncio
async def test_short_document_uses_single_extraction() -> None:
    calls: list[str] = []
    pages = make_pages()
    chunk = make_chunk(1)
    summary = make_summary("1 000 000 рублей")
    validator = FakeValidator(calls)
    pdf_extractor = FakePDFExtractor(calls, pages)
    chunker = FakeChunker(calls, [chunk])
    tender_extractor = FakeTenderExtractor(calls, [summary])
    service = TenderSummaryService(validator, pdf_extractor, chunker, tender_extractor)

    result = await service.summarize(make_document())

    assert result == summary
    assert tender_extractor.extracted_chunks == [chunk]
    assert tender_extractor.consolidated_summaries is None
    assert calls == [
        "validate_upload",
        "parse_pdf",
        "validate_pages",
        "split_pages",
        "extract_1",
    ]


@pytest.mark.asyncio
async def test_long_document_consolidates_results() -> None:
    calls: list[str] = []
    chunks = [make_chunk(1), make_chunk(2)]
    partials = [make_summary("1 000 000 рублей"), make_summary()]
    consolidated = make_summary("1 000 000 рублей")
    validator = FakeValidator(calls)
    pdf_extractor = FakePDFExtractor(calls, make_pages())
    chunker = FakeChunker(calls, chunks)
    tender_extractor = FakeTenderExtractor(
        calls,
        partials,
        consolidated_result=consolidated,
    )
    service = TenderSummaryService(validator, pdf_extractor, chunker, tender_extractor)

    result = await service.summarize(make_document())

    assert result == consolidated
    assert tender_extractor.consolidated_summaries == partials
    assert calls[-3:] == ["extract_1", "extract_2", "consolidate"]


@pytest.mark.asyncio
async def test_invalid_upload_does_not_parse_pdf() -> None:
    calls: list[str] = []
    validator = FakeValidator(calls, upload_error=InvalidPdfError())
    pdf_extractor = FakePDFExtractor(calls, make_pages())
    chunker = FakeChunker(calls, [make_chunk(1)])
    tender_extractor = FakeTenderExtractor(calls, [make_summary()])
    service = TenderSummaryService(validator, pdf_extractor, chunker, tender_extractor)

    with pytest.raises(InvalidPdfError):
        await service.summarize(make_document())

    assert calls == ["validate_upload"]
    assert tender_extractor.extracted_chunks == []


@pytest.mark.asyncio
async def test_textless_pdf_does_not_call_llm() -> None:
    calls: list[str] = []
    validator = FakeValidator(calls, pages_error=PdfTextNotFoundError())
    pdf_extractor = FakePDFExtractor(calls, make_pages())
    chunker = FakeChunker(calls, [make_chunk(1)])
    tender_extractor = FakeTenderExtractor(calls, [make_summary()])
    service = TenderSummaryService(validator, pdf_extractor, chunker, tender_extractor)

    with pytest.raises(PdfTextNotFoundError):
        await service.summarize(make_document())

    assert calls == ["validate_upload", "parse_pdf", "validate_pages"]
    assert tender_extractor.extracted_chunks == []


@pytest.mark.asyncio
async def test_parser_runs_outside_event_loop() -> None:
    calls: list[str] = []
    event_loop_thread = get_ident()
    validator = FakeValidator(calls)
    pdf_extractor = FakePDFExtractor(calls, make_pages())
    chunker = FakeChunker(calls, [make_chunk(1)])
    tender_extractor = FakeTenderExtractor(calls, [make_summary()])
    service = TenderSummaryService(validator, pdf_extractor, chunker, tender_extractor)

    await service.summarize(make_document())

    assert pdf_extractor.thread_id is not None
    assert pdf_extractor.thread_id != event_loop_thread


@pytest.mark.asyncio
async def test_chunk_failure_aborts_pipeline() -> None:
    calls: list[str] = []
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    error = InvalidLLMResponseError()
    validator = FakeValidator(calls)
    pdf_extractor = FakePDFExtractor(calls, make_pages())
    chunker = FakeChunker(calls, chunks)
    tender_extractor = FakeTenderExtractor(
        calls,
        [make_summary(), make_summary()],
        extract_error_at=2,
        extract_error=error,
    )
    service = TenderSummaryService(validator, pdf_extractor, chunker, tender_extractor)

    with pytest.raises(InvalidLLMResponseError) as caught:
        await service.summarize(make_document())

    assert caught.value is error
    assert tender_extractor.extracted_chunks == chunks[:2]
    assert tender_extractor.consolidated_summaries is None


@pytest.mark.asyncio
async def test_consolidation_failure_is_propagated() -> None:
    calls: list[str] = []
    chunks = [make_chunk(1), make_chunk(2)]
    error = InvalidLLMResponseError()
    validator = FakeValidator(calls)
    pdf_extractor = FakePDFExtractor(calls, make_pages())
    chunker = FakeChunker(calls, chunks)
    tender_extractor = FakeTenderExtractor(
        calls,
        [make_summary(), make_summary()],
        consolidation_error=error,
    )
    service = TenderSummaryService(validator, pdf_extractor, chunker, tender_extractor)

    with pytest.raises(InvalidLLMResponseError) as caught:
        await service.summarize(make_document())

    assert caught.value is error
    assert tender_extractor.consolidated_summaries is not None


@pytest.mark.asyncio
async def test_chunks_are_processed_in_order() -> None:
    calls: list[str] = []
    chunks = [make_chunk(3), make_chunk(1), make_chunk(2)]
    partials = [make_summary(), make_summary(), make_summary()]
    validator = FakeValidator(calls)
    pdf_extractor = FakePDFExtractor(calls, make_pages())
    chunker = FakeChunker(calls, chunks)
    tender_extractor = FakeTenderExtractor(
        calls,
        partials,
        consolidated_result=make_summary(),
    )
    service = TenderSummaryService(validator, pdf_extractor, chunker, tender_extractor)

    await service.summarize(make_document())

    assert tender_extractor.extracted_chunks == chunks
    assert calls[-4:] == ["extract_3", "extract_1", "extract_2", "consolidate"]
