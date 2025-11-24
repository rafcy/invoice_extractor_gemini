"""
FastAPI application for invoice document processing with Gemini API
Supports images, PDFs, and URLs with rate limiting
"""

import os
import logging
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from utils.models import (
    DocumentRequest, DocumentResponse, BatchDocumentRequest,
    BatchDocumentResponse, HealthResponse, ProcessingStatus,
    RateLimitInfo, MimeType
)
from utils.document_processor import DocumentProcessor, DocumentCache
from utils.rate_limiter import RateLimiter, BatchRateLimiter

# Load environment variables
load_dotenv()

# Configuration
RATE_LIMIT_MAX_PER_MINUTE = int(os.getenv("RATE_LIMIT_MAX_PER_MINUTE", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "100"))


# Global instances
rate_limiter: Optional[RateLimiter] = None
batch_rate_limiter: Optional[BatchRateLimiter] = None
document_processor: Optional[DocumentProcessor] = None
document_cache: Optional[DocumentCache] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI application"""
    # Startup
    global rate_limiter, batch_rate_limiter, document_processor, document_cache

    print("🚀 Starting Invoice Processing API...")
    print(f"   Model: {GEMINI_MODEL}")
    print(f"   Rate Limit: {RATE_LIMIT_MAX_PER_MINUTE} requests per {RATE_LIMIT_WINDOW_SECONDS}s")
    print(f"   Cache: {'Enabled' if ENABLE_CACHE else 'Disabled'}")

    # Initialize components
    rate_limiter = RateLimiter(
        max_requests=RATE_LIMIT_MAX_PER_MINUTE,
        window_seconds=RATE_LIMIT_WINDOW_SECONDS
    )

    batch_rate_limiter = BatchRateLimiter(
        max_per_batch=RATE_LIMIT_MAX_PER_MINUTE,
        batch_delay_seconds=RATE_LIMIT_WINDOW_SECONDS
    )

    document_processor = DocumentProcessor(model_name=GEMINI_MODEL)

    if ENABLE_CACHE:
        document_cache = DocumentCache(max_size=MAX_CACHE_SIZE)

    print("✅ API ready!\n")

    yield

    # Shutdown
    print("\n🛑 Shutting down Invoice Processing API...")


# Create FastAPI app
app = FastAPI(
    title="Invoice Processing API",
    description="Extract structured data from invoices using Gemini API with rate limiting",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    gemini_configured = bool(os.getenv('GOOGLE_API_KEY'))

    rate_limit_status = rate_limiter.get_current_usage() if rate_limiter else None

    return HealthResponse(
        status="healthy" if gemini_configured else "unhealthy",
        timestamp=datetime.now(),
        gemini_api_configured=gemini_configured,
        rate_limit_info=RateLimitInfo(**rate_limit_status) if rate_limit_status else None
    )


# @app.get("/api/v1/rate-limit", tags=["Rate Limiting"])
# async def get_rate_limit_status():
#     """Get current rate limit status"""
#     if not rate_limiter:
#         raise HTTPException(status_code=500, detail="Rate limiter not initialized")

#     usage = rate_limiter.get_current_usage()
#     return {
#         "rate_limit": usage,
#         "configuration": {
#             "max_requests_per_window": RATE_LIMIT_MAX_PER_MINUTE,
#             "window_seconds": RATE_LIMIT_WINDOW_SECONDS
#         }
#     }


@app.post("/api/v1/process", response_model=DocumentResponse, tags=["Processing"])
async def process_document(request: DocumentRequest):
    """
    Process a single document (image or PDF) from URL or base64

    - **url**: URL to image or PDF file
    - **base64_data**: Base64 encoded image/PDF data
    - **mime_type**: MIME type (required for base64)
    - **additional_instructions**: Optional extraction instructions
    """
    request_start = datetime.now()
    logger.info("="*80)
    logger.info("📥 NEW REQUEST: /api/v1/process")
    logger.info("="*80)

    if not document_processor or not rate_limiter:
        raise HTTPException(status_code=500, detail="Service not initialized")

    # Check rate limit
    step_start = datetime.now()
    if not rate_limiter.can_make_request():
        wait_time = rate_limiter.wait_time_until_available()
        logger.warning(f"⚠️  Rate limit exceeded. Wait: {wait_time:.1f}s")
        return DocumentResponse(
            status=ProcessingStatus.RATE_LIMITED,
            error=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds",
            processing_time_ms=0.0
        )

    # Acquire rate limit slot
    await rate_limiter.acquire(1)
    step_duration = (datetime.now() - step_start).total_seconds() * 1000
    logger.info(f"⏱️  Rate limit check: {step_duration:.2f}ms")

    # Process document (caching is handled inside process_document now)
    start_time = datetime.now()
    try:
        cache = document_cache if ENABLE_CACHE else None
        logger.info(f"     Cache enabled: {ENABLE_CACHE}")

        invoice_data = await document_processor.process_document(
            request,
            cache=cache
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        total_time = (datetime.now() - request_start).total_seconds() * 1000

        logger.info("="*80)
        logger.info("✅ REQUEST COMPLETE")
        logger.info(f"⏱️  Processing time: {processing_time:.2f}ms")
        logger.info(f"⏱️  Total request time: {total_time:.2f}ms")
        logger.info("="*80)

        return DocumentResponse(
            status=ProcessingStatus.SUCCESS,
            data=invoice_data,
            processing_time_ms=processing_time
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        total_time = (datetime.now() - request_start).total_seconds() * 1000

        logger.error("="*80)
        logger.error("❌ REQUEST FAILED")
        logger.error(f"⏱️  Time before error: {processing_time:.2f}ms")
        logger.error(f"⏱️  Total request time: {total_time:.2f}ms")
        logger.error(f"     Error: {str(e)}")
        logger.error("="*80)

        return DocumentResponse(
            status=ProcessingStatus.FAILED,
            error=str(e),
            processing_time_ms=processing_time
        )


@app.post("/api/v1/process/batch", response_model=BatchDocumentResponse, tags=["Processing"])
async def process_batch(request: BatchDocumentRequest):
    """
    Process multiple documents in batch with rate limiting

    If more than rate limit allows, processes in chunks with delays.
    Returns all results together after processing completes.

    - **documents**: List of document requests (max 100)
    - **additional_instructions**: Optional global instructions
    """
    if not document_processor or not batch_rate_limiter:
        raise HTTPException(status_code=500, detail="Service not initialized")

    total_docs = len(request.documents)

    if total_docs > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 documents per batch request"
        )

    start_time = datetime.now()
    all_results: List[DocumentResponse] = []

    # Get batch info
    batch_info = batch_rate_limiter.get_batch_info(total_docs)

    # Merge global instructions with individual requests
    processed_requests = []
    for doc_req in request.documents:
        # If document doesn't have instructions, use global ones
        if not doc_req.additional_instructions and request.additional_instructions:
            doc_req.additional_instructions = request.additional_instructions
        processed_requests.append(doc_req)

    # Process function for a single batch - NOW WITH PARALLEL PROCESSING!
    async def process_single_batch(batch_requests: List[DocumentRequest]) -> List[DocumentResponse]:
        cache = document_cache if ENABLE_CACHE else None

        logger.info(f"🚀 Processing batch of {len(batch_requests)} documents in PARALLEL")
        batch_start = datetime.now()

        # Process each document as a separate async task
        async def process_single_doc(req: DocumentRequest, index: int) -> DocumentResponse:
            try:
                doc_start = datetime.now()
                logger.info(f"  [{index+1}/{len(batch_requests)}] Starting document processing...")

                invoice_data = await document_processor.process_document(
                    req,
                    cache=cache
                )

                processing_time = (datetime.now() - doc_start).total_seconds() * 1000
                logger.info(f"  [{index+1}/{len(batch_requests)}] ✅ Completed in {processing_time:.2f}ms")

                return DocumentResponse(
                    status=ProcessingStatus.SUCCESS,
                    data=invoice_data,
                    processing_time_ms=processing_time
                )

            except Exception as e:
                logger.error(f"  [{index+1}/{len(batch_requests)}] ❌ Failed: {str(e)}")
                return DocumentResponse(
                    status=ProcessingStatus.FAILED,
                    error=str(e),
                    processing_time_ms=0.0
                )

        # Use asyncio.gather to process all documents concurrently
        import asyncio
        tasks = [process_single_doc(req, i) for i, req in enumerate(batch_requests)]
        batch_results = await asyncio.gather(*tasks)

        batch_duration = (datetime.now() - batch_start).total_seconds() * 1000
        logger.info(f"✅ Batch complete! Processed {len(batch_requests)} documents in {batch_duration:.2f}ms")
        logger.info(f"   Average: {batch_duration/len(batch_requests):.2f}ms per document")

        return list(batch_results)

    # Process in batches with rate limiting
    all_results = await batch_rate_limiter.process_batches(
        processed_requests,
        process_single_batch
    )

    total_processing_time = (datetime.now() - start_time).total_seconds() * 1000

    # Count successful and failed
    successful = sum(1 for r in all_results if r.status == ProcessingStatus.SUCCESS)
    failed = sum(1 for r in all_results if r.status == ProcessingStatus.FAILED)

    return BatchDocumentResponse(
        total_documents=total_docs,
        successful=successful,
        failed=failed,
        results=all_results,
        total_processing_time_ms=total_processing_time,
        rate_limit_info=batch_info
    )


@app.get("/api/v1/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """Get cache statistics"""
    if not ENABLE_CACHE or not document_cache:
        raise HTTPException(status_code=400, detail="Cache is not enabled")

    return document_cache.get_stats()


@app.delete("/api/v1/cache", tags=["Cache"])
async def clear_cache():
    """Clear the document cache"""
    if not ENABLE_CACHE or not document_cache:
        raise HTTPException(status_code=400, detail="Cache is not enabled")

    document_cache.clear()
    return {"message": "Cache cleared successfully"}


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
