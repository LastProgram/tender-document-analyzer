from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.routes import router
from app.core.logging import RequestTracingMiddleware, configure_logging


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title="Tender Summarizer")
    application.add_middleware(RequestTracingMiddleware)
    application.include_router(router)
    register_error_handlers(application)
    return application


app = create_app()
