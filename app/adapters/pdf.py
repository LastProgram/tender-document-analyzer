from io import BytesIO
from time import perf_counter

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PyPdfError

from app.core.exceptions import (
    EncryptedPdfError,
    InvalidPdfError,
    PdfPageLimitExceededError,
)
from app.core.logging import elapsed_ms, log_event
from app.schemas.document import DocumentPage


class PDFExtractor:
    def __init__(self, max_pdf_pages: int) -> None:
        self._max_pdf_pages = max_pdf_pages

    def extract(self, content: bytes) -> list[DocumentPage]:
        started_at = perf_counter()
        try:
            reader = PdfReader(BytesIO(content))
        except FileNotDecryptedError as exc:
            raise EncryptedPdfError(
                "Encrypted PDF documents are not supported."
            ) from exc
        except PyPdfError as exc:
            raise InvalidPdfError("PDF document is invalid.") from exc

        if reader.is_encrypted:
            raise EncryptedPdfError("Encrypted PDF documents are not supported.")

        try:
            page_count = len(reader.pages)
            if page_count > self._max_pdf_pages:
                raise PdfPageLimitExceededError("PDF document exceeds the page limit.")

            pages: list[DocumentPage] = []
            for page_number, page in enumerate(reader.pages, start=1):
                pages.append(
                    DocumentPage(
                        number=page_number,
                        text=page.extract_text() or "",
                    )
                )
        except PyPdfError as exc:
            raise InvalidPdfError("PDF document is invalid.") from exc

        log_event(
            "pdf_extraction_completed",
            duration_ms=elapsed_ms(started_at),
            page_count=len(pages),
            character_count=sum(len(page.text) for page in pages),
        )
        return pages
