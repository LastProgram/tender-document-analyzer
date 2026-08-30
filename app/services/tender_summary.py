import anyio

from app.adapters.pdf import PDFExtractor
from app.schemas.document import UploadedDocument
from app.schemas.tender import TenderSummary
from app.services.chunking import PageChunker
from app.services.document_validation import DocumentValidator
from app.services.ports import TenderExtractor


class TenderSummaryService:
    """Объединяет проверку документа, разбиение и извлечение в единый сценарий."""

    def __init__(
        self,
        validator: DocumentValidator,
        pdf_extractor: PDFExtractor,
        chunker: PageChunker,
        tender_extractor: TenderExtractor,
    ) -> None:
        self._validator = validator
        self._pdf_extractor = pdf_extractor
        self._chunker = chunker
        self._tender_extractor = tender_extractor

    async def summarize(self, document: UploadedDocument) -> TenderSummary:
        self._validator.validate(document)
        pages = await anyio.to_thread.run_sync(
            self._pdf_extractor.extract,
            document.content,
        )
        self._validator.validate_pages(pages)
        chunks = self._chunker.split(pages)

        # Последовательные вызовы не конкурируют за память
        # и процессор локальной модели.
        partials: list[TenderSummary] = []
        for chunk in chunks:
            partials.append(await self._tender_extractor.extract(chunk))

        # Результат единственного блока уже является полной выжимкой
        # и не требует отдельного вызова консолидации.
        if len(partials) == 1:
            return partials[0]
        return await self._tender_extractor.consolidate(partials)
