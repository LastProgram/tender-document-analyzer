from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.adapters.pdf import PDFExtractor
from app.core.config import Settings
from app.services.chunking import PageChunker
from app.services.document_validation import DocumentValidator
from app.services.tender_summary import TenderSummaryService


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_summary_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> TenderSummaryService:
    # Отложенный импорт позволяет создать приложение FastAPI
    # без загрузки клиента локальной модели.
    from app.adapters.ollama import OllamaTenderExtractor

    return TenderSummaryService(
        validator=DocumentValidator(settings.max_upload_size_mb),
        pdf_extractor=PDFExtractor(settings.max_pdf_pages),
        chunker=PageChunker(
            max_chars=settings.llm_chunk_max_chars,
            overlap_pages=settings.chunk_overlap_pages,
        ),
        tender_extractor=OllamaTenderExtractor(settings),
    )


SettingsDependency = Annotated[Settings, Depends(get_settings)]
SummaryServiceDependency = Annotated[
    TenderSummaryService,
    Depends(get_summary_service),
]
