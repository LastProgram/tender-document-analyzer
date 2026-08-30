from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.api.dependencies import SettingsDependency, SummaryServiceDependency
from app.core.exceptions import PdfTooLargeError
from app.schemas.document import UploadedDocument
from app.schemas.tender import TenderSummary

router = APIRouter(prefix="/api/v1/tenders")

_UPLOAD_READ_CHUNK_SIZE = 1024 * 1024


async def read_bounded_upload(
    file: UploadFile,
    max_bytes: int,
) -> UploadedDocument:
    parts: list[bytes] = []
    total = 0

    while chunk := await file.read(_UPLOAD_READ_CHUNK_SIZE):
        total += len(chunk)
        if total > max_bytes:
            raise PdfTooLargeError
        parts.append(chunk)

    return UploadedDocument(
        filename=file.filename,
        content_type=file.content_type,
        content=b"".join(parts),
    )


@router.post("/summarize", response_model=TenderSummary)
async def summarize_tender(
    file: Annotated[UploadFile, File()],
    service: SummaryServiceDependency,
    settings: SettingsDependency,
) -> TenderSummary:
    document = await read_bounded_upload(
        file,
        settings.max_upload_size_mb * 1024 * 1024,
    )
    return await service.summarize(document)
