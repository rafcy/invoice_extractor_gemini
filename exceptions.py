"""
Custom exceptions for the Invoice Processing API
"""


class DocumentProcessingError(Exception):
    """Base exception for document processing errors"""
    pass


class PDFNotSupportedError(DocumentProcessingError):
    """Raised when PDF processing is attempted without required dependencies"""
    pass


class PDFConversionError(DocumentProcessingError):
    """Raised when PDF to image conversion fails"""
    pass


class JSONParseError(DocumentProcessingError):
    """Raised when JSON parsing fails"""
    pass


class DocumentDownloadError(DocumentProcessingError):
    """Raised when document download from URL fails"""
    pass


class InvalidDocumentError(DocumentProcessingError):
    """Raised when document format is invalid"""
    pass
