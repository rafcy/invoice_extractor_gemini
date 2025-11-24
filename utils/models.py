"""
Pydantic models for request/response validation and type safety
"""

from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional, List, Union, Literal
from datetime import datetime
from enum import Enum


class DocumentSource(str, Enum):
    """Source type for document processing"""
    FILE = "file"
    URL = "url"
    BASE64 = "base64"


class MimeType(str, Enum):
    """Supported MIME types"""
    JPEG = "image/jpeg"
    PNG = "image/png"
    PDF = "application/pdf"
    WEBP = "image/webp"
    TIFF = "image/tiff"


class VendorInfo(BaseModel):
    """Vendor information extracted from invoice"""
    name: Optional[str] = None
    address: Optional[str] = None
    tax_id: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class CustomerInfo(BaseModel):
    """Customer information extracted from invoice"""
    name: Optional[str] = None
    address: Optional[str] = None
    tax_id: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class LineItem(BaseModel):
    """Individual line item from invoice"""
    description: str
    quantity: float
    unit_price: float
    tax_rate: float = 0.0
    amount: float


class Totals(BaseModel):
    """Invoice totals and currency information"""
    subtotal: float
    tax_amount: float
    discount: float = 0.0
    total: float
    currency: Optional[str] = "USD"


class InvoiceMetadata(BaseModel):
    """Metadata about the parsing process"""
    source_file: Optional[str] = None
    source_url: Optional[str] = None
    parsed_at: datetime
    model: str
    processing_time_ms: Optional[float] = None


class InvoiceData(BaseModel):
    """Complete invoice data structure"""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    vendor: VendorInfo
    customer: CustomerInfo
    line_items: List[LineItem] = []
    totals: Totals
    payment_terms: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[InvoiceMetadata] = None


class DocumentRequest(BaseModel):
    """Request model for document processing"""
    url: Optional[HttpUrl] = Field(None, description="URL to image or PDF")
    base64_data: Optional[str] = Field(None, description="Base64 encoded image/PDF data")
    mime_type: Optional[MimeType] = Field(None, description="MIME type of the document")
    additional_instructions: Optional[str] = Field(None, description="Additional extraction instructions")

    @field_validator('base64_data')
    @classmethod
    def validate_base64(cls, v):
        if v:
            import base64
            try:
                base64.b64decode(v)
            except Exception:
                raise ValueError("Invalid base64 data")
        return v

    def model_post_init(self, __context):
        """Validate that either URL or base64_data is provided"""
        if not self.url and not self.base64_data:
            raise ValueError("Either 'url' or 'base64_data' must be provided")
        if self.base64_data and not self.mime_type:
            raise ValueError("'mime_type' is required when using 'base64_data'")


class BatchDocumentRequest(BaseModel):
    """Request model for batch document processing"""
    documents: List[DocumentRequest] = Field(..., min_length=1, max_length=100)
    additional_instructions: Optional[str] = None


class ProcessingStatus(str, Enum):
    """Status of document processing"""
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


class DocumentResponse(BaseModel):
    """Response model for single document processing"""
    status: ProcessingStatus
    data: Optional[InvoiceData] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None


class BatchDocumentResponse(BaseModel):
    """Response model for batch document processing"""
    total_documents: int
    successful: int
    failed: int
    results: List[DocumentResponse]
    total_processing_time_ms: float
    rate_limit_info: Optional[dict] = None


class RateLimitInfo(BaseModel):
    """Rate limit status information"""
    requests_in_current_window: int
    max_requests_per_window: int
    window_reset_at: datetime
    requests_remaining: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "unhealthy"]
    timestamp: datetime
    gemini_api_configured: bool
    rate_limit_info: Optional[RateLimitInfo] = None
