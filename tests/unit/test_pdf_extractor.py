from pathlib import Path

import pytest
from pypdf import PageObject
from pypdf.errors import PdfReadError

from app.adapters.pdf import PDFExtractor
from app.core.exceptions import (
    EncryptedPdfError,
    InvalidPdfError,
    PdfPageLimitExceededError,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def read_fixture(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


def test_extracts_pages_in_physical_order() -> None:
    pages = PDFExtractor(max_pdf_pages=10).extract(read_fixture("tender.pdf"))

    assert [page.number for page in pages] == [1, 2, 3]
    assert "First tender page" in pages[0].text
    assert pages[1].text == ""
    assert "Third tender page" in pages[2].text


def test_empty_page_does_not_shift_following_page_number() -> None:
    pages = PDFExtractor(max_pdf_pages=10).extract(read_fixture("tender.pdf"))

    assert pages[2].number == 3


def test_rejects_broken_pdf_without_exposing_parser_message() -> None:
    with pytest.raises(InvalidPdfError) as error:
        PDFExtractor(max_pdf_pages=10).extract(read_fixture("broken.pdf"))

    assert str(error.value) == "PDF document is invalid."
    assert isinstance(error.value.__cause__, PdfReadError)


def test_rejects_encrypted_pdf() -> None:
    with pytest.raises(EncryptedPdfError):
        PDFExtractor(max_pdf_pages=10).extract(read_fixture("encrypted.pdf"))


def test_checks_page_limit_before_extracting_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(page: PageObject) -> str:
        raise AssertionError("extract_text must not be called")

    monkeypatch.setattr(PageObject, "extract_text", fail_if_called)

    with pytest.raises(PdfPageLimitExceededError):
        PDFExtractor(max_pdf_pages=2).extract(read_fixture("tender.pdf"))


def test_rejects_document_when_page_extraction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_extraction(page: PageObject) -> str:
        raise PdfReadError("sensitive parser detail")

    monkeypatch.setattr(PageObject, "extract_text", fail_extraction)

    with pytest.raises(InvalidPdfError) as error:
        PDFExtractor(max_pdf_pages=10).extract(read_fixture("tender.pdf"))

    assert str(error.value) == "PDF document is invalid."
    assert isinstance(error.value.__cause__, PdfReadError)
    assert "sensitive parser detail" not in str(error.value)


def test_empty_pdf_has_no_extractable_text() -> None:
    pages = PDFExtractor(max_pdf_pages=10).extract(read_fixture("empty.pdf"))

    assert len(pages) == 1
    assert pages[0].text == ""
