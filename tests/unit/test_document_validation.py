import pytest

from app.core.exceptions import (
    InvalidPdfError,
    PdfTextNotFoundError,
    PdfTooLargeError,
    UnsupportedPdfTypeError,
)
from app.schemas.document import DocumentPage, UploadedDocument
from app.services.document_validation import DocumentValidator


def make_document(
    content: bytes = b"%PDF-1.7\n",
    filename: str | None = "document.pdf",
    content_type: str | None = "application/pdf",
) -> UploadedDocument:
    return UploadedDocument(
        filename=filename,
        content_type=content_type,
        content=content,
    )


def test_rejects_empty_upload() -> None:
    validator = DocumentValidator(max_upload_size_mb=1)

    with pytest.raises(InvalidPdfError):
        validator.validate(make_document(content=b""))


def test_rejects_upload_over_size_limit() -> None:
    validator = DocumentValidator(max_upload_size_mb=1)
    content = b"%PDF-" + b"x" * (1024 * 1024)

    with pytest.raises(PdfTooLargeError):
        validator.validate(make_document(content=content))


def test_rejects_unsupported_content_type() -> None:
    validator = DocumentValidator(max_upload_size_mb=1)

    with pytest.raises(UnsupportedPdfTypeError):
        validator.validate(make_document(content_type="text/plain"))


def test_rejects_pdf_filename_without_pdf_signature() -> None:
    validator = DocumentValidator(max_upload_size_mb=1)

    with pytest.raises(UnsupportedPdfTypeError):
        validator.validate(make_document(content=b"not a pdf"))


def test_accepts_pdf_signature_within_first_1024_bytes() -> None:
    validator = DocumentValidator(max_upload_size_mb=1)
    document = make_document(
        content=b"x" * 100 + b"%PDF-1.7\n",
        filename="document.bin",
    )

    validator.validate(document)


def test_rejects_pdf_signature_after_first_1024_bytes() -> None:
    validator = DocumentValidator(max_upload_size_mb=1)
    document = make_document(content=b"x" * 1024 + b"%PDF-1.7\n")

    with pytest.raises(UnsupportedPdfTypeError):
        validator.validate(document)


def test_rejects_document_without_extractable_text() -> None:
    validator = DocumentValidator(max_upload_size_mb=1)
    pages = [DocumentPage(number=1, text=""), DocumentPage(number=2, text="   ")]

    with pytest.raises(PdfTextNotFoundError):
        validator.validate_pages(pages)


def test_accepts_document_with_text_on_any_page() -> None:
    validator = DocumentValidator(max_upload_size_mb=1)
    pages = [DocumentPage(number=1, text=""), DocumentPage(number=2, text="Tender")]

    validator.validate_pages(pages)
