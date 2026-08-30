from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    ApplicationError,
    EncryptedPdfError,
    InvalidLLMResponseError,
    InvalidPdfError,
    LLMConfigurationError,
    LLMTimeoutError,
    LLMUnavailableError,
    PdfPageLimitExceededError,
    PdfTextNotFoundError,
    PdfTooLargeError,
    UnsupportedPdfTypeError,
)

ErrorDetails = tuple[HTTPStatus, str, str]

ERROR_RESPONSES: dict[type[ApplicationError], ErrorDetails] = {
    UnsupportedPdfTypeError: (
        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        "UNSUPPORTED_PDF_TYPE",
        "Only PDF files are supported.",
    ),
    PdfTooLargeError: (
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        "PDF_TOO_LARGE",
        "PDF exceeds the upload size limit.",
    ),
    PdfPageLimitExceededError: (
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        "PDF_PAGE_LIMIT_EXCEEDED",
        "PDF exceeds the page limit.",
    ),
    InvalidPdfError: (
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "INVALID_PDF",
        "PDF is invalid.",
    ),
    EncryptedPdfError: (
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "ENCRYPTED_PDF",
        "Encrypted PDF files are not supported.",
    ),
    PdfTextNotFoundError: (
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "PDF_TEXT_NOT_FOUND",
        "PDF does not contain extractable text.",
    ),
    LLMUnavailableError: (
        HTTPStatus.SERVICE_UNAVAILABLE,
        "LLM_UNAVAILABLE",
        "LLM service is unavailable.",
    ),
    LLMTimeoutError: (
        HTTPStatus.GATEWAY_TIMEOUT,
        "LLM_TIMEOUT",
        "LLM request timed out.",
    ),
    LLMConfigurationError: (
        HTTPStatus.SERVICE_UNAVAILABLE,
        "LLM_CONFIGURATION_ERROR",
        "LLM service is not configured correctly.",
    ),
    InvalidLLMResponseError: (
        HTTPStatus.BAD_GATEWAY,
        "INVALID_LLM_RESPONSE",
        "LLM returned an invalid response.",
    ),
}

_INTERNAL_ERROR: ErrorDetails = (
    HTTPStatus.INTERNAL_SERVER_ERROR,
    "INTERNAL_ERROR",
    "Internal server error.",
)


def _error_response(details: ErrorDetails) -> JSONResponse:
    status_code, code, message = details
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": code, "message": message}},
    )


async def _application_error_handler(
    _request: Request,
    exc: ApplicationError,
) -> JSONResponse:
    return _error_response(ERROR_RESPONSES.get(type(exc), _INTERNAL_ERROR))


async def _unexpected_error_handler(
    _request: Request,
    _exc: Exception,
) -> JSONResponse:
    return _error_response(_INTERNAL_ERROR)


def register_error_handlers(application: FastAPI) -> None:
    application.add_exception_handler(ApplicationError, _application_error_handler)
    application.add_exception_handler(Exception, _unexpected_error_handler)
