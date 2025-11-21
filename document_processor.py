"""
Document processor that handles invoice extraction using Gemini API
Refactored with proper OOP structure, reusing logic from existing parsers
"""

import os
import json
import io
import httpx
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Union
from PIL import Image
import google.generativeai as genai

# Try to import pdf2image for PDF support
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from models import (
    InvoiceData, VendorInfo, CustomerInfo, LineItem,
    Totals, InvoiceMetadata, DocumentRequest, MimeType
)
from exceptions import (
    PDFNotSupportedError, PDFConversionError,
    JSONParseError, DocumentProcessingError
)


# Structured prompt for consistent JSON output (reused from existing parsers)
STRUCTURED_PROMPT = """
You are an expert invoice data extraction system. Extract ALL information from this invoice and return it as a valid JSON object with the following exact structure:

{
    "invoice_number": "string or null",
    "invoice_date": "YYYY-MM-DD or null",
    "due_date": "YYYY-MM-DD or null",
    "vendor": {
        "name": "string or null",
        "address": "string or null",
        "tax_id": "string or null",
        "phone": "string or null",
        "email": "string or null"
    },
    "customer": {
        "name": "string or null",
        "address": "string or null",
        "tax_id": "string or null",
        "phone": "string or null",
        "email": "string or null"
    },
    "line_items": [
        {
            "description": "string",
            "quantity": number,
            "unit_price": number,
            "tax_rate": number,
            "amount": number
        }
    ],
    "totals": {
        "subtotal": number,
        "tax_amount": number,
        "discount": number,
        "total": number,
        "currency": "string (e.g., USD, EUR, GBP)"
    },
    "payment_terms": "string or null",
    "notes": "string or null"
}

CRITICAL RULES:
1. Return ONLY valid JSON, no markdown, no explanations, no code blocks
2. If a field is not found, use null for strings or 0 for numbers
3. Convert all dates to YYYY-MM-DD format
4. Extract all line items as an array
5. Calculate totals if not explicitly stated
6. Be precise with numbers (use decimals)
"""


class DocumentProcessor:
    """
    Document processor for invoice extraction
    Handles images, PDFs, and URLs with Gemini API
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize document processor

        Args:
            model_name: Gemini model to use
            api_key: Google API key (if not provided, uses env variable)
        """
        self.model_name = model_name
        api_key = api_key or os.getenv('GOOGLE_API_KEY')

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    async def download_from_url(self, url: str) -> tuple[bytes, str]:
        """
        Download document from URL

        Args:
            url: URL to download from

        Returns:
            Tuple of (file_bytes, mime_type)
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Get MIME type from response headers
            mime_type = response.headers.get('content-type', 'application/octet-stream')

            return response.content, mime_type

    def prepare_image_from_bytes(
        self,
        file_bytes: bytes,
        mime_type: str
    ) -> Dict[str, Union[str, bytes]]:
        """
        Prepare image data for Gemini API from raw bytes
        Handles PDF conversion if needed

        Args:
            file_bytes: Raw file bytes
            mime_type: MIME type of the file

        Returns:
            Image data dict for Gemini API
        """
        # Handle PDF files
        if mime_type == "application/pdf":
            if not PDF_SUPPORT:
                raise PDFNotSupportedError(
                    "PDF support not available. Install pdf2image and poppler: "
                    "pip install pdf2image && brew install poppler (macOS)"
                )

            # Convert PDF first page to image
            images = convert_from_bytes(file_bytes, first_page=1, last_page=1)
            if not images:
                raise PDFConversionError("Could not extract image from PDF")

            # Convert PIL image to PNG bytes
            img_byte_arr = io.BytesIO()
            images[0].save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            image_bytes = img_byte_arr.read()

            return {
                "mime_type": "image/png",
                "data": image_bytes
            }
        else:
            # Handle regular image files
            return {
                "mime_type": mime_type,
                "data": file_bytes
            }

    def clean_json_response(self, response_text: str) -> str:
        """
        Clean JSON response from Gemini (remove markdown code blocks)

        Args:
            response_text: Raw response from Gemini

        Returns:
            Cleaned JSON string
        """
        cleaned = response_text.strip()

        # Remove markdown code blocks if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    async def process_document(
        self,
        request: DocumentRequest,
        source_identifier: Optional[str] = None
    ) -> InvoiceData:
        """
        Process a document and extract invoice data

        Args:
            request: Document request with URL or base64 data
            source_identifier: Optional identifier for the source

        Returns:
            Extracted invoice data

        Raises:
            Exception: If processing fails
        """
        start_time = datetime.now()

        try:
            # Get document bytes
            if request.url:
                file_bytes, mime_type = await self.download_from_url(str(request.url))
                source = str(request.url)
            elif request.base64_data:
                file_bytes = base64.b64decode(request.base64_data)
                mime_type = request.mime_type.value if request.mime_type else "image/jpeg"
                source = source_identifier or "base64_upload"
            else:
                raise ValueError("Either url or base64_data must be provided")

            # Prepare image for Gemini
            image_data = self.prepare_image_from_bytes(file_bytes, mime_type)

            # Build prompt
            full_prompt = STRUCTURED_PROMPT
            if request.additional_instructions:
                full_prompt += f"\n\nAdditional instructions: {request.additional_instructions}"

            # Call Gemini API
            response = self.model.generate_content([full_prompt, image_data])

            # Clean and parse JSON
            cleaned_json = self.clean_json_response(response.text)
            parsed_data = json.loads(cleaned_json)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Build structured response
            invoice_data = InvoiceData(
                invoice_number=parsed_data.get('invoice_number'),
                invoice_date=parsed_data.get('invoice_date'),
                due_date=parsed_data.get('due_date'),
                vendor=VendorInfo(**parsed_data.get('vendor', {})),
                customer=CustomerInfo(**parsed_data.get('customer', {})),
                line_items=[LineItem(**item) for item in parsed_data.get('line_items', [])],
                totals=Totals(**parsed_data.get('totals', {})),
                payment_terms=parsed_data.get('payment_terms'),
                notes=parsed_data.get('notes'),
                metadata=InvoiceMetadata(
                    source_file=source if not request.url else None,
                    source_url=str(request.url) if request.url else None,
                    parsed_at=datetime.now(),
                    model=self.model_name,
                    processing_time_ms=processing_time
                )
            )

            return invoice_data

        except json.JSONDecodeError as e:
            raise JSONParseError(f"Failed to parse JSON response: {str(e)}")
        except (PDFNotSupportedError, PDFConversionError, JSONParseError):
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")

    async def process_batch(
        self,
        requests: List[DocumentRequest],
        additional_instructions: Optional[str] = None
    ) -> List[Optional[InvoiceData]]:
        """
        Process multiple documents in batch
        Note: For now, processes sequentially. Can be optimized with parallel processing.

        Args:
            requests: List of document requests
            additional_instructions: Optional global instructions for all documents

        Returns:
            List of invoice data (None for failed documents)
        """
        results = []

        for idx, req in enumerate(requests):
            try:
                # Apply global instructions if provided
                if additional_instructions and not req.additional_instructions:
                    req.additional_instructions = additional_instructions

                result = await self.process_document(req, source_identifier=f"batch_item_{idx}")
                results.append(result)
            except Exception as e:
                print(f"Error processing document {idx}: {str(e)}")
                results.append(None)

        return results


class DocumentCache:
    """Simple in-memory cache for processed documents"""

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, InvoiceData] = {}
        self.max_size = max_size
        self.access_order: List[str] = []

    def _generate_key(self, request: DocumentRequest) -> str:
        """Generate cache key from request"""
        if request.url:
            return f"url:{request.url}"
        elif request.base64_data:
            # Use hash of base64 data
            import hashlib
            return f"b64:{hashlib.md5(request.base64_data.encode()).hexdigest()}"
        return ""

    def get(self, request: DocumentRequest) -> Optional[InvoiceData]:
        """Get cached result if available"""
        key = self._generate_key(request)
        if key in self.cache:
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, request: DocumentRequest, data: InvoiceData) -> None:
        """Cache a result"""
        key = self._generate_key(request)

        # Evict oldest if at max size
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = data
        if key not in self.access_order:
            self.access_order.append(key)

    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.access_order.clear()
