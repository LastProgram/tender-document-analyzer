class ApplicationError(Exception):
    pass


class InvalidPdfError(ApplicationError):
    pass


class UnsupportedPdfTypeError(ApplicationError):
    pass


class PdfTooLargeError(ApplicationError):
    pass


class EncryptedPdfError(ApplicationError):
    pass


class PdfPageLimitExceededError(ApplicationError):
    pass


class PdfTextNotFoundError(ApplicationError):
    pass
