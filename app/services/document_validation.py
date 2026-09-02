from app.core.exceptions import (
    InvalidPdfError,
    PdfTextNotFoundError,
    PdfTooLargeError,
    UnsupportedPdfTypeError,
)
from app.schemas.document import DocumentPage, UploadedDocument

PDF_SIGNATURE = b"%PDF-"
PDF_SIGNATURE_SEARCH_BYTES = 1024
SUPPORTED_PDF_CONTENT_TYPES = frozenset({"application/pdf"})


class DocumentValidator:
    def __init__(self, max_upload_size_mb: int) -> None:
        self._max_upload_size_bytes = max_upload_size_mb * 1024 * 1024

    def validate(self, document: UploadedDocument) -> None:
        if not document.content:
            raise InvalidPdfError("PDF document is empty.")

        if len(document.content) > self._max_upload_size_bytes:
            raise PdfTooLargeError("PDF document exceeds the size limit.")

        if document.content_type not in SUPPORTED_PDF_CONTENT_TYPES:
            raise UnsupportedPdfTypeError("Unsupported PDF content type.")

        header = document.content[:PDF_SIGNATURE_SEARCH_BYTES]
        if PDF_SIGNATURE not in header:
            raise UnsupportedPdfTypeError("PDF signature was not found.")

    def validate_pages(self, pages: list[DocumentPage]) -> None:
        if not any(page.text.strip() for page in pages):
            raise PdfTextNotFoundError("PDF document contains no extractable text.")
