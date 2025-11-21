"""
Example usage of the Invoice Processing API
Demonstrates different ways to interact with the API
"""

import requests
import base64
import json
from pathlib import Path


# API base URL
BASE_URL = "http://localhost:8000"


def check_health():
    """Check if the API is healthy"""
    print("=" * 60)
    print("Checking API Health")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    data = response.json()

    print(f"Status: {data['status']}")
    print(f"Gemini Configured: {data['gemini_api_configured']}")

    if data.get('rate_limit_info'):
        rl = data['rate_limit_info']
        print(f"Rate Limit: {rl['requests_in_current_window']}/{rl['max_requests_per_window']}")
        print(f"Remaining: {rl['requests_remaining']}")

    print()


def process_from_url():
    """Process an invoice from a URL"""
    print("=" * 60)
    print("Example 1: Process from URL")
    print("=" * 60)

    payload = {
        "url": "https://example.com/sample-invoice.pdf",
        "additional_instructions": "Extract all vendor and customer details"
    }

    response = requests.post(f"{BASE_URL}/api/v1/process", json=payload)
    data = response.json()

    if data['status'] == 'success':
        invoice = data['data']
        print(f"✅ Invoice Number: {invoice['invoice_number']}")
        print(f"   Date: {invoice['invoice_date']}")
        print(f"   Vendor: {invoice['vendor']['name']}")
        print(f"   Total: {invoice['totals']['currency']} {invoice['totals']['total']}")
        print(f"   Processing Time: {data['processing_time_ms']:.2f}ms")
    else:
        print(f"❌ Error: {data.get('error', 'Unknown error')}")

    print()


def upload_local_file(file_path: str):
    """Upload and process a local file"""
    print("=" * 60)
    print(f"Example 2: Upload Local File - {file_path}")
    print("=" * 60)

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        print()
        return

    with open(file_path, "rb") as f:
        files = {"file": (Path(file_path).name, f)}
        data = {"additional_instructions": "Focus on line items"}

        response = requests.post(
            f"{BASE_URL}/api/v1/upload",
            files=files,
            data=data
        )

    result = response.json()

    if result['status'] == 'success':
        invoice = result['data']
        print(f"✅ Successfully processed: {invoice['invoice_number']}")
        print(f"   Line Items: {len(invoice['line_items'])}")
        print(f"   Total: {invoice['totals']['total']}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

    print()


def process_base64(file_path: str):
    """Process a file using base64 encoding"""
    print("=" * 60)
    print(f"Example 3: Process Base64 - {file_path}")
    print("=" * 60)

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        print()
        return

    # Read and encode file
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        base64_data = base64.b64encode(file_bytes).decode('utf-8')

    # Determine MIME type
    suffix = Path(file_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.pdf': 'application/pdf'
    }
    mime_type = mime_types.get(suffix, 'image/jpeg')

    payload = {
        "base64_data": base64_data,
        "mime_type": mime_type
    }

    response = requests.post(f"{BASE_URL}/api/v1/process", json=payload)
    data = response.json()

    if data['status'] == 'success':
        invoice = data['data']
        print(f"✅ Invoice: {invoice['invoice_number']}")
        print(f"   Vendor: {invoice['vendor']['name']}")
    else:
        print(f"❌ Error: {data.get('error', 'Unknown error')}")

    print()


def batch_process(urls: list):
    """Process multiple invoices in batch"""
    print("=" * 60)
    print(f"Example 4: Batch Process - {len(urls)} documents")
    print("=" * 60)

    payload = {
        "documents": [{"url": url} for url in urls],
        "additional_instructions": "Extract vendor and total amount"
    }

    response = requests.post(f"{BASE_URL}/api/v1/process/batch", json=payload)
    data = response.json()

    print(f"Total Documents: {data['total_documents']}")
    print(f"Successful: {data['successful']}")
    print(f"Failed: {data['failed']}")
    print(f"Total Processing Time: {data['total_processing_time_ms']:.2f}ms")
    print()

    if data.get('rate_limit_info'):
        rl = data['rate_limit_info']
        print(f"Batch Info:")
        print(f"  - Batches: {rl['number_of_batches']}")
        print(f"  - Items per batch: {rl['items_per_batch']}")
        print(f"  - Estimated time: {rl['estimated_total_time_seconds']}s")
        print()

    # Show first few results
    for i, result in enumerate(data['results'][:3], 1):
        if result['status'] == 'success':
            inv = result['data']
            print(f"{i}. ✅ {inv['invoice_number']} - {inv['totals']['currency']} {inv['totals']['total']}")
        else:
            print(f"{i}. ❌ Failed: {result.get('error', 'Unknown')}")

    print()


def check_rate_limit():
    """Check current rate limit status"""
    print("=" * 60)
    print("Rate Limit Status")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/api/v1/rate-limit")
    data = response.json()

    rl = data['rate_limit']
    print(f"Current Window: {rl['requests_in_current_window']}/{rl['max_requests_per_window']}")
    print(f"Remaining: {rl['requests_remaining']}")
    print(f"Window Reset: {rl['window_reset_at']}")

    config = data['configuration']
    print(f"\nConfiguration:")
    print(f"  Max per window: {config['max_requests_per_window']}")
    print(f"  Window seconds: {config['window_seconds']}")

    print()


def save_results_to_file(invoice_data: dict, output_file: str = "extracted_invoice.json"):
    """Save extracted invoice data to a file"""
    print("=" * 60)
    print(f"Saving to {output_file}")
    print("=" * 60)

    with open(output_file, 'w') as f:
        json.dump(invoice_data, f, indent=2)

    print(f"✅ Saved to {output_file}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Invoice Processing API - Example Usage")
    print("=" * 60 + "\n")

    # 1. Check health
    check_health()

    # 2. Check rate limit
    check_rate_limit()

    # Note: The following examples require actual files/URLs
    # Uncomment and modify as needed:

    # # Process from URL
    # process_from_url()

    # # Upload local file
    # upload_local_file("path/to/invoice.pdf")

    # # Process with base64
    # process_base64("path/to/invoice.png")

    # # Batch process
    # batch_process([
    #     "https://example.com/invoice1.pdf",
    #     "https://example.com/invoice2.pdf",
    #     "https://example.com/invoice3.pdf"
    # ])

    print("=" * 60)
    print("💡 Tip: Uncomment the examples above and provide actual file paths/URLs")
    print("=" * 60)
