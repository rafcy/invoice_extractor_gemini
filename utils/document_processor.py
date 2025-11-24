"""
Document processor that handles invoice extraction using Gemini API
Refactored with proper OOP structure, reusing logic from existing parsers
"""

import os
import json
import io
import httpx
import base64
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Union
from PIL import Image
import imagehash
from google import genai
from google.genai import types

# Configure logging
logger = logging.getLogger(__name__)

# Try to import pdf2image for PDF support
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from utils.models import (
    InvoiceData, VendorInfo, CustomerInfo, LineItem,
    Totals, InvoiceMetadata, DocumentRequest, MimeType
)
from utils.exceptions import (
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
3. Convert all dates to DD-MM-YYYY format
4. Extract all line items as an array
5. Calculate totals if not explicitly stated
6. Be precise with numbers (use decimals)
"""


class DocumentProcessor:
    """
    Document processor for invoice extraction
    Handles images, PDFs, and URLs with Gemini API
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        """
        Initialize document processor with new Google GenAI SDK

        Args:
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
            api_key: Google API key (if not provided, uses env variable)
        """
        self.model_name = model_name
        api_key = api_key or os.getenv('GOOGLE_API_KEY')

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # New SDK: Create centralized client
        self.client = genai.Client(api_key=api_key)
        logger.info(f"✅ Initialized Google GenAI Client with model: {model_name}")

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

    async def get_file_bytes(
        self,
        request: DocumentRequest
    ) -> tuple[bytes, str, str]:
        """
        Get file bytes from request (URL or base64)

        Args:
            request: Document request with URL or base64 data

        Returns:
            Tuple of (file_bytes, mime_type, source_identifier)
        """
        if request.url:
            file_bytes, mime_type = await self.download_from_url(str(request.url))
            source = str(request.url)
        elif request.base64_data:
            file_bytes = base64.b64decode(request.base64_data)
            mime_type = request.mime_type.value if request.mime_type else "image/jpeg"
            source = "base64_upload"
        else:
            raise ValueError("Either url or base64_data must be provided")

        return file_bytes, mime_type, source

    async def process_document(
        self,
        request: DocumentRequest,
        source_identifier: Optional[str] = None,
        cache: Optional['DocumentCache'] = None
    ) -> InvoiceData:
        """
        Process a document and extract invoice data

        Args:
            request: Document request with URL or base64 data
            source_identifier: Optional identifier for the source
            cache: Optional DocumentCache instance for caching

        Returns:
            Extracted invoice data

        Raises:
            Exception: If processing fails
        """
        start_time = datetime.now()
        logger.info("🚀 Starting document processing")
        logger.info("="*60)

        try:
            # Get document bytes
            step_start = datetime.now()
            file_bytes, mime_type, source = await self.get_file_bytes(request)
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            logger.info(f"⏱️  [1/5] Get file bytes: {step_duration:.2f}ms")
            logger.info(f"     File size: {len(file_bytes)} bytes, mime_type: {mime_type}")

            if source_identifier:
                source = source_identifier

            # Check cache first (content-based with ImageHash for images)
            if cache:
                step_start = datetime.now()
                cached_result = cache.get_by_content(
                    file_bytes,
                    mime_type,
                    request.additional_instructions
                )
                step_duration = (datetime.now() - step_start).total_seconds() * 1000
                logger.info(f"⏱️  [2/5] Cache check: {step_duration:.2f}ms")

                if cached_result:
                    total_duration = (datetime.now() - start_time).total_seconds() * 1000
                    logger.info("✅ CACHE HIT - Returning cached result")
                    logger.info(f"⏱️  Total time: {total_duration:.2f}ms")
                    logger.info("="*60)
                    return cached_result

                logger.info("     Cache miss - continuing to process")
            else:
                logger.info("⏱️  [2/5] Cache: disabled")

            # Prepare image for Gemini
            step_start = datetime.now()
            image_data = self.prepare_image_from_bytes(file_bytes, mime_type)
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            logger.info(f"⏱️  [3/5] Prepare image for Gemini: {step_duration:.2f}ms")

            # Build prompt
            step_start = datetime.now()
            full_prompt = STRUCTURED_PROMPT
            if request.additional_instructions:
                full_prompt += f"\n\nAdditional instructions: {request.additional_instructions}"
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            logger.info(f"⏱️  [4/5] Build prompt: {step_duration:.2f}ms")

            # Call Gemini API (async for parallel processing) - New SDK
            logger.info("     Calling Gemini API (async with new SDK)...")
            step_start = datetime.now()

            # New SDK: Use client.aio.models.generate_content
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_text(text=full_prompt),
                    types.Part.from_bytes(data=image_data['data'], mime_type=image_data['mime_type'])
                ]
            )

            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            logger.info(f"⏱️  [4/5] Gemini API call: {step_duration:.2f}ms ⚡")
            logger.info(f"     Response length: {len(response.text)} chars")

            # Clean and parse JSON
            step_start = datetime.now()
            cleaned_json = self.clean_json_response(response.text)
            parsed_data = json.loads(cleaned_json)
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            logger.info(f"⏱️  [5/5] Parse JSON response: {step_duration:.2f}ms")

            # Build structured response
            logger.info("     Building structured response...")
            step_start = datetime.now()

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
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            )

            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            logger.info(f"⏱️  [6/7] Build InvoiceData object: {step_duration:.2f}ms")

            # Cache the result (content-based with ImageHash for images)
            if cache:
                step_start = datetime.now()
                cache.set_by_content(
                    file_bytes,
                    mime_type,
                    invoice_data,
                    request.additional_instructions
                )
                step_duration = (datetime.now() - step_start).total_seconds() * 1000
                logger.info(f"⏱️  [7/7] Cache result: {step_duration:.2f}ms")

            # Calculate total time
            total_duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.info("="*60)
            logger.info(f"✅ Processing complete!")
            logger.info(f"⏱️  TOTAL TIME: {total_duration:.2f}ms")
            logger.info(f"     Invoice: {invoice_data.invoice_number}")
            logger.info("="*60)

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
        additional_instructions: Optional[str] = None,
        cache: Optional['DocumentCache'] = None
    ) -> List[Optional[InvoiceData]]:
        """
        Process multiple documents in batch with content-based caching

        Checks cache for each individual file. Only processes uncached files.
        This allows partial cache hits when some files in the batch were previously processed.

        Args:
            requests: List of document requests
            additional_instructions: Optional global instructions for all documents
            cache: Optional DocumentCache instance for caching

        Returns:
            List of invoice data (None for failed documents)
        """
        results = []

        for idx, req in enumerate(requests):
            try:
                # Apply global instructions if provided
                if additional_instructions and not req.additional_instructions:
                    req.additional_instructions = additional_instructions

                result = await self.process_document(
                    req,
                    source_identifier=f"batch_item_{idx}",
                    cache=cache
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing document {idx}: {str(e)}")
                results.append(None)

        return results


class DocumentCache:
    """
    Content-based cache for processed documents using perceptual image hashing

    Uses ImageHash for images (perceptual hashing) and SHA-256 for PDFs.
    This is much more efficient than hashing entire file bytes:
    - ImageHash: ~16 bytes per image vs MBs of image data
    - Detects similar/duplicate images even with minor differences
    - Fast computation and comparison

    Cache key = image_hash + instructions_hash
    """

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, InvoiceData] = {}
        self.max_size = max_size
        self.access_order: List[str] = []

    def _generate_content_hash(
        self,
        file_bytes: bytes,
        mime_type: str,
        additional_instructions: Optional[str] = None
    ) -> str:
        """
        Generate cache key from file content + instructions

        Uses ImageHash for images (perceptual hash) and SHA-256 for PDFs/other files.

        Args:
            file_bytes: Raw file content bytes
            mime_type: MIME type of the file
            additional_instructions: Optional processing instructions that affect output

        Returns:
            Cache key combining content hash + instructions hash
        """
        logger.info(f"🔍 Generating cache hash for mime_type: {mime_type}, file_size: {len(file_bytes)} bytes")

        # Generate content hash based on file type
        if mime_type.startswith('image/'):
            # Use perceptual image hash (much more efficient)
            try:
                logger.info("  → Using ImageHash for image file")
                image = Image.open(io.BytesIO(file_bytes))
                logger.info(f"  → Image opened: size={image.size}, mode={image.mode}")

                # Using average hash (aHash) - fast and works well for duplicates
                # Alternative: phash (more robust), dhash (good for transformations)
                content_hash = str(imagehash.average_hash(image, hash_size=16))
                logger.info(f"  ✓ ImageHash generated: {content_hash}")
            except Exception as e:
                # Fallback to SHA-256 if image hash fails
                logger.warning(f"  ⚠️  ImageHash failed: {str(e)}, falling back to SHA-256")
                content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
                logger.info(f"  ✓ SHA-256 fallback hash: {content_hash}")
        else:
            # For PDFs and other files, use SHA-256 (still better than full bytes)
            logger.info("  → Using SHA-256 for non-image file")
            content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
            logger.info(f"  ✓ SHA-256 hash generated: {content_hash}")

        # Hash additional instructions separately
        if additional_instructions:
            inst_hash = hashlib.sha256(additional_instructions.encode('utf-8')).hexdigest()[:8]
            final_hash = f"{content_hash}:{inst_hash}"
            logger.info(f"  → Instructions provided, appending hash: {inst_hash}")
            logger.info(f"  ✓ Final cache key: {final_hash}")
            return final_hash

        logger.info(f"  ✓ Final cache key: {content_hash}")
        return content_hash

    def get_by_content(
        self,
        file_bytes: bytes,
        mime_type: str,
        additional_instructions: Optional[str] = None
    ) -> Optional[InvoiceData]:
        """
        Get cached result by file content hash

        Args:
            file_bytes: Raw file content bytes
            mime_type: MIME type of the file
            additional_instructions: Optional processing instructions

        Returns:
            Cached InvoiceData if available, None otherwise
        """
        logger.info("="*60)
        logger.info("📦 CACHE GET REQUEST")
        key = self._generate_content_hash(file_bytes, mime_type, additional_instructions)

        if key in self.cache:
            # Update access order (LRU)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            logger.info(f"✅ CACHE HIT! Key found in cache: {key}")
            logger.info(f"   Returning cached invoice: {self.cache[key].invoice_number}")
            logger.info("="*60)
            return self.cache[key]

        logger.info(f"❌ CACHE MISS! Key not found: {key}")
        logger.info(f"   Current cache size: {len(self.cache)}/{self.max_size}")
        logger.info(f"   Cached keys: {list(self.cache.keys())[:5]}..." if len(self.cache) > 5 else f"   Cached keys: {list(self.cache.keys())}")
        logger.info("="*60)
        return None

    def set_by_content(
        self,
        file_bytes: bytes,
        mime_type: str,
        data: InvoiceData,
        additional_instructions: Optional[str] = None
    ) -> None:
        """
        Cache a result by file content hash

        Args:
            file_bytes: Raw file content bytes
            mime_type: MIME type of the file
            data: Processed invoice data to cache
            additional_instructions: Optional processing instructions
        """
        logger.info("="*60)
        logger.info("💾 CACHE SET REQUEST")
        key = self._generate_content_hash(file_bytes, mime_type, additional_instructions)

        # Evict least recently used if at max size
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            logger.warning(f"⚠️  Cache full! Evicting oldest key: {oldest}")
            del self.cache[oldest]

        self.cache[key] = data
        if key not in self.access_order:
            self.access_order.append(key)

        logger.info(f"✅ CACHED! Stored invoice: {data.invoice_number}")
        logger.info(f"   Cache key: {key}")
        logger.info(f"   Cache size: {len(self.cache)}/{self.max_size}")
        logger.info("="*60)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "available_slots": self.max_size - len(self.cache)
        }

    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.access_order.clear()
