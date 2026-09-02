import logging
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
from app.core.logging import REQUEST_ID_HEADER, elapsed_ms, log_event

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


def _error_response(details: ErrorDetails, request_id: str) -> JSONResponse:
    status_code, code, message = details
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": code, "message": message}},
        headers={REQUEST_ID_HEADER: request_id},
    )


def application_error_response(
    error_type: type[ApplicationError],
    request_id: str,
) -> JSONResponse:
    return _error_response(ERROR_RESPONSES.get(error_type, _INTERNAL_ERROR), request_id)


async def _application_error_handler(
    request: Request,
    exc: ApplicationError,
) -> JSONResponse:
    details = ERROR_RESPONSES.get(type(exc), _INTERNAL_ERROR)
    status_code, code, _message = details
    request_id = request.state.request_id
    log_event(
        "tender_processing_failed",
        request_id=request_id,
        duration_ms=elapsed_ms(request.state.request_started_at),
        error_code=code,
        http_status=status_code,
    )
    return application_error_response(type(exc), request_id)


async def _unexpected_error_handler(
    request: Request,
    _exc: Exception,
) -> JSONResponse:
    status_code, code, _message = _INTERNAL_ERROR
    request_id = request.state.request_id
    log_event(
        "tender_processing_failed",
        level=logging.ERROR,
        request_id=request_id,
        include_exception=True,
        duration_ms=elapsed_ms(request.state.request_started_at),
        error_code=code,
        http_status=status_code,
    )
    return _error_response(_INTERNAL_ERROR, request_id)


def register_error_handlers(application: FastAPI) -> None:
    application.add_exception_handler(ApplicationError, _application_error_handler)
    application.add_exception_handler(Exception, _unexpected_error_handler)
