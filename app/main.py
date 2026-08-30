from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.routes import router


def create_app() -> FastAPI:
    application = FastAPI(title="Tender Summarizer")
    application.include_router(router)
    register_error_handlers(application)
    return application


app = create_app()
