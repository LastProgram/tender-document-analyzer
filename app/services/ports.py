from collections.abc import Sequence
from typing import Protocol

from app.schemas.document import DocumentChunk
from app.schemas.tender import TenderSummary


class TenderExtractor(Protocol):
    """Определяет границу извлечения тендерных данных для прикладного слоя."""

    async def extract(self, chunk: DocumentChunk) -> TenderSummary: ...

    async def consolidate(
        self,
        summaries: Sequence[TenderSummary],
    ) -> TenderSummary: ...
