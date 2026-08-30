from pydantic import BaseModel, ConfigDict


class UploadedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str | None
    content_type: str | None
    content: bytes


class DocumentPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: int
    text: str


class DocumentChunk(BaseModel):
    """Представляет текстовый блок с диапазоном исходных страниц."""

    model_config = ConfigDict(frozen=True)

    page_from: int
    page_to: int
    text: str
