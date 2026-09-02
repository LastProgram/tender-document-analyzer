from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.routes import router
from app.api.upload_limit import UploadSizeLimitMiddleware
from app.core.config import Settings
from app.core.logging import RequestTracingMiddleware, configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    runtime_settings = settings or Settings()
    application = FastAPI(title="Tender Summarizer")
    application.add_middleware(
        UploadSizeLimitMiddleware,
        max_bytes=runtime_settings.max_upload_size_mb * 1024 * 1024,
    )
    application.add_middleware(RequestTracingMiddleware)
    application.include_router(router)
    register_error_handlers(application)
    return application


app = create_app()
