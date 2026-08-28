from fastapi import FastAPI


def create_app() -> FastAPI:
    return FastAPI(title="Tender Summarizer")


app = create_app()
