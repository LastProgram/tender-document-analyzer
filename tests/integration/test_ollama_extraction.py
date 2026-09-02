import pytest

from app.adapters.ollama import OllamaTenderExtractor
from app.core.config import Settings
from app.schemas.document import DocumentChunk
from app.schemas.tender import TenderSummary


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_real_ollama_returns_valid_tender_summary() -> None:
    extractor = OllamaTenderExtractor(Settings())
    chunk = DocumentChunk(
        page_from=1,
        page_to=1,
        text=(
            "[PAGE 1]\n"
            "Начальная (максимальная) цена контракта составляет 1 500 000 рублей. "
            "Работы должны быть завершены не позднее 10 сентября 2026 года. "
            "Подрядчик обязан иметь действующую лицензию на выполнение "
            "электромонтажных работ. За каждый день просрочки начисляется пеня "
            "в размере 0,1 процента от стоимости просроченного обязательства."
        ),
    )

    summary = await extractor.extract(chunk)

    assert isinstance(summary, TenderSummary)
    assert set(summary.model_dump()) == {
        "contract_amount",
        "deadlines",
        "contractor_requirements",
        "penalties",
    }
