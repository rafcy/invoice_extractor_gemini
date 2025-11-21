# Invoice Processing API

A production-ready FastAPI application for extracting structured data from invoices using Google Gemini API. Supports images, PDFs, and URLs with intelligent rate limiting and caching.

## Features

- **Multiple Input Methods**: Process documents via URL, base64 encoding, or direct file upload
- **Format Support**: Images (JPEG, PNG, WebP) and PDF files
- **Intelligent Rate Limiting**: Configurable rate limits with automatic batching and delays
- **Batch Processing**: Process multiple documents with automatic chunking
- **Caching**: Optional caching for faster repeated requests
- **Type Safety**: Full Pydantic validation for requests and responses
- **OOP Architecture**: Clean, maintainable code structure with proper separation of concerns

## Project Structure

```
API/
├── main.py                 # FastAPI application and endpoints
├── models.py               # Pydantic models for request/response validation
├── document_processor.py   # Document processing logic with Gemini API
├── rate_limiter.py        # Rate limiting implementation
├── exceptions.py          # Custom exception classes
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.9+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- poppler (for PDF support)

### macOS

```bash
# Install poppler for PDF support
brew install poppler

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Linux

```bash
# Install poppler for PDF support
sudo apt-get install poppler-utils

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your Google API key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
RATE_LIMIT_MAX_PER_MINUTE=10
GEMINI_MODEL=gemini-2.5-flash
```

## Usage

### Starting the Server

```bash
# Development mode with auto-reload
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

Check API status and rate limit information.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00",
  "gemini_api_configured": true,
  "rate_limit_info": {
    "requests_in_current_window": 3,
    "max_requests_per_window": 10,
    "window_reset_at": "2024-01-20T10:31:00",
    "requests_remaining": 7
  }
}
```

### 2. Process Document from URL

Process an invoice from a public URL.

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/invoice.pdf",
    "additional_instructions": "Focus on extracting line items"
  }'
```

### 3. Process Document from Base64

Process a base64-encoded document.

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -d '{
    "base64_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB...",
    "mime_type": "image/png"
  }'
```

### 4. Upload File

Upload and process a file directly.

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@invoice.pdf" \
  -F "additional_instructions=Extract all tax details"
```

### 5. Batch Processing

Process multiple documents in a single request.

```bash
curl -X POST "http://localhost:8000/api/v1/process/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "url": "https://example.com/invoice1.pdf"
      },
      {
        "url": "https://example.com/invoice2.pdf"
      }
    ],
    "additional_instructions": "Extract vendor information"
  }'
```

**Response:**
```json
{
  "total_documents": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "status": "success",
      "data": {
        "invoice_number": "INV-001",
        "invoice_date": "2024-01-15",
        "vendor": {
          "name": "ABC Company",
          "tax_id": "123456789"
        },
        "totals": {
          "total": 1250.50,
          "currency": "USD"
        }
      },
      "processing_time_ms": 1523.45
    }
  ],
  "total_processing_time_ms": 3250.12,
  "rate_limit_info": {
    "total_items": 2,
    "items_per_batch": 10,
    "number_of_batches": 1,
    "delay_between_batches_seconds": 60,
    "estimated_total_time_seconds": 0
  }
}
```

### 6. Rate Limit Status

Check current rate limit status.

```bash
curl http://localhost:8000/api/v1/rate-limit
```

### 7. Clear Cache

Clear the document cache (if enabled).

```bash
curl -X DELETE http://localhost:8000/api/v1/cache
```

## Response Format

All successful document processing returns a structured JSON object:

```json
{
  "status": "success",
  "data": {
    "invoice_number": "INV-12345",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
    "vendor": {
      "name": "Acme Corporation",
      "address": "123 Main St, City, State 12345",
      "tax_id": "12-3456789",
      "phone": "+1-555-0123",
      "email": "billing@acme.com"
    },
    "customer": {
      "name": "Customer Inc",
      "address": "456 Oak Ave, Town, State 67890",
      "tax_id": "98-7654321",
      "phone": "+1-555-0456",
      "email": "ap@customer.com"
    },
    "line_items": [
      {
        "description": "Product A",
        "quantity": 10,
        "unit_price": 50.00,
        "tax_rate": 0.08,
        "amount": 500.00
      }
    ],
    "totals": {
      "subtotal": 500.00,
      "tax_amount": 40.00,
      "discount": 0.00,
      "total": 540.00,
      "currency": "USD"
    },
    "payment_terms": "Net 30",
    "notes": "Thank you for your business",
    "metadata": {
      "source_url": "https://example.com/invoice.pdf",
      "parsed_at": "2024-01-20T10:30:00",
      "model": "gemini-2.5-flash",
      "processing_time_ms": 1234.56
    }
  },
  "processing_time_ms": 1234.56
}
```

## Rate Limiting

The API implements intelligent rate limiting to comply with Gemini API quotas:

- **Default**: 10 requests per minute (configurable via `RATE_LIMIT_MAX_PER_MINUTE`)
- **Batch Processing**: Automatically splits large batches and adds delays
- **Behavior**:
  - If limit is reached, single requests return `rate_limited` status
  - Batch requests automatically wait and process in chunks
  - All results are returned together after processing completes

### Example: Processing 25 Documents

If you send 25 documents with a 10/minute rate limit:

1. **Batch 1**: Processes documents 1-10 immediately
2. **Delay**: Waits 60 seconds
3. **Batch 2**: Processes documents 11-20
4. **Delay**: Waits 60 seconds
5. **Batch 3**: Processes documents 21-25
6. **Response**: Returns all 25 results together

## Configuration

All configuration is done via environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Your Google Gemini API key | Required |
| `GEMINI_MODEL` | Model to use | `gemini-2.5-flash` |
| `RATE_LIMIT_MAX_PER_MINUTE` | Max requests per minute | `10` |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window in seconds | `60` |
| `ENABLE_CACHE` | Enable document caching | `true` |
| `MAX_CACHE_SIZE` | Maximum cached documents | `100` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

### Model Options

- `gemini-2.5-flash-lite`: Fastest, most cost-effective
- `gemini-2.5-flash`: Balanced speed and accuracy (recommended)
- `gemini-2.5-pro`: Highest accuracy

## Python Client Examples

### Example 1: Process Single Invoice

```python
import requests

url = "http://localhost:8000/api/v1/process"
payload = {
    "url": "https://example.com/invoice.pdf"
}

response = requests.post(url, json=payload)
data = response.json()

if data["status"] == "success":
    invoice = data["data"]
    print(f"Invoice: {invoice['invoice_number']}")
    print(f"Total: {invoice['totals']['currency']} {invoice['totals']['total']}")
```

### Example 2: Upload Local File

```python
import requests

url = "http://localhost:8000/api/v1/upload"

with open("invoice.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

data = response.json()
print(data)
```

### Example 3: Batch Processing

```python
import requests

url = "http://localhost:8000/api/v1/process/batch"
payload = {
    "documents": [
        {"url": "https://example.com/invoice1.pdf"},
        {"url": "https://example.com/invoice2.pdf"},
        {"url": "https://example.com/invoice3.pdf"}
    ]
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Processed: {data['successful']}/{data['total_documents']}")
for result in data['results']:
    if result['status'] == 'success':
        print(f"  - {result['data']['invoice_number']}: ${result['data']['totals']['total']}")
```

### Example 4: With Base64 Encoding

```python
import requests
import base64

# Read and encode file
with open("invoice.png", "rb") as f:
    base64_data = base64.b64encode(f.read()).decode('utf-8')

url = "http://localhost:8000/api/v1/process"
payload = {
    "base64_data": base64_data,
    "mime_type": "image/png",
    "additional_instructions": "Extract all line items with tax details"
}

response = requests.post(url, json=payload)
data = response.json()
print(data)
```

## Error Handling

The API returns structured error responses:

```json
{
  "status": "failed",
  "error": "Document processing failed: PDF support not available",
  "processing_time_ms": 12.34
}
```

Common error scenarios:
- **Rate Limited**: Wait for the time indicated in the error message
- **Invalid Format**: Check that your file is a supported format
- **PDF Support**: Install `pdf2image` and `poppler`
- **Invalid API Key**: Check your `GOOGLE_API_KEY` in `.env`

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests (when available)
pytest
```

### Code Formatting

```bash
# Install formatters
pip install black ruff

# Format code
black .
ruff check --fix .
```

## Architecture

### OOP Structure

The application follows clean OOP principles:

- **`DocumentProcessor`**: Handles all Gemini API interactions and document processing
- **`RateLimiter`**: Manages per-request rate limiting
- **`BatchRateLimiter`**: Handles batch processing with delays
- **`DocumentCache`**: Simple LRU cache for processed documents
- **Pydantic Models**: Type-safe request/response validation

### Reused Components

This API reuses and refactors logic from existing parsers:
- Invoice extraction prompt from `invoice_parser.py`
- Batch processing logic from `batch_invoice_parser.py`
- PDF handling and image preparation utilities

## Production Deployment

### Docker (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t invoice-api .
docker run -p 8000:8000 --env-file .env invoice-api
```

### Environment Variables in Production

- Store `GOOGLE_API_KEY` securely (e.g., AWS Secrets Manager, Azure Key Vault)
- Set `HOST=0.0.0.0` for container environments
- Configure `RATE_LIMIT_MAX_PER_MINUTE` based on your Gemini API quota
- Consider disabling cache (`ENABLE_CACHE=false`) in serverless environments

## License

This project is part of the invoice processing suite. See parent directory for license information.

## Support

For issues or questions:
1. Check the API documentation at `/docs`
2. Review error messages and logs
3. Ensure all dependencies are installed
4. Verify your Google API key is valid

## Changelog

### Version 1.0.0
- Initial release
- Support for images, PDFs, and URLs
- Rate limiting with batch processing
- Caching support
- Comprehensive API documentation
