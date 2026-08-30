from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PyPdfError

from app.core.exceptions import (
    EncryptedPdfError,
    InvalidPdfError,
    PdfPageLimitExceededError,
)
from app.schemas.document import DocumentPage


class PDFExtractor:
    def __init__(self, max_pdf_pages: int) -> None:
        self._max_pdf_pages = max_pdf_pages

    def extract(self, content: bytes) -> list[DocumentPage]:
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

        return pages
