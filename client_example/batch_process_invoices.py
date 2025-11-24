#!/usr/bin/env python3
"""
Batch Invoice Request Generator

This script reads images from the 'invoices' folder and generates
batch request JSON files with base64-encoded images.

Usage:
    python batch_process_invoices.py

Requirements:
    None (uses only Python standard library)
"""

import os
import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Any


# Configuration
BATCH_SIZE = 10  # Number of images per batch request
INVOICES_FOLDER = Path(__file__).parent / "invoices"
OUTPUT_FOLDER = Path(__file__).parent / "batch_requests"

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".pdf"}


def get_mime_type(file_path: Path) -> str:
    """Get MIME type based on file extension."""
    extension = file_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".pdf": "application/pdf"
    }
    return mime_types.get(extension, "application/octet-stream")


def encode_file_to_base64(file_path: Path) -> str:
    """Read file and encode to base64 string."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return base64.b64encode(file_bytes).decode('utf-8')


def get_invoice_files() -> List[Path]:
    """Get all supported invoice files from the invoices folder."""
    if not INVOICES_FOLDER.exists():
        print(f"Error: Invoices folder not found at {INVOICES_FOLDER}")
        print(f"Please create the folder and add invoice images: mkdir -p {INVOICES_FOLDER}")
        return []

    files = []
    for file_path in INVOICES_FOLDER.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            files.append(file_path)

    return sorted(files)


def create_batch_request(files: List[Path], additional_instructions: str = None) -> Dict[str, Any]:
    """Create a batch request payload for the API."""
    batch_requests = []

    for file_path in files:
        try:
            base64_data = encode_file_to_base64(file_path)
            mime_type = get_mime_type(file_path)

            request_data = {
                "base64_data": base64_data,
                "mime_type": mime_type
            }

            if additional_instructions:
                request_data["additional_instructions"] = additional_instructions

            batch_requests.append(request_data)
            print(f"  ✓ Encoded: {file_path.name} ({mime_type})")

        except Exception as e:
            print(f"  ✗ Failed to encode {file_path.name}: {e}")

    return {"documents": batch_requests}


def save_batch_request(payload: Dict[str, Any], batch_num: int, file_names: List[str]) -> str:
    """Save batch request payload to a JSON file."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_FOLDER / f"batch_request_{batch_num}_{timestamp}.json"

    # Add metadata to the request
    request_with_metadata = {
        "metadata": {
            "batch_number": batch_num,
            "total_documents": len(payload.get("documents", [])),
            "file_names": file_names,
            "generated_at": timestamp
        },
        "request": payload
    }

    with open(output_file, "w") as f:
        json.dump(request_with_metadata, f, indent=2)

    # Calculate file size
    file_size = output_file.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    print(f"💾 Request saved to: {output_file}")
    print(f"   File size: {file_size_mb:.2f} MB")

    return str(output_file)


def print_batch_summary(payload: Dict[str, Any], file_names: List[str]):
    """Print a summary of the batch request."""
    documents = payload.get("documents", [])
    total = len(documents)

    print("\n📊 Batch Request Summary:")
    print(f"   Total documents: {total}")

    # Calculate total base64 size
    total_base64_size = sum(len(doc.get("base64_data", "")) for doc in documents)
    total_size_mb = total_base64_size / (1024 * 1024)
    print(f"   Total base64 size: {total_size_mb:.2f} MB")

    # Show mime types
    mime_types = {}
    for doc in documents:
        mime = doc.get("mime_type", "unknown")
        mime_types[mime] = mime_types.get(mime, 0) + 1

    print("\n📋 Document Types:")
    for mime, count in mime_types.items():
        print(f"   {mime}: {count} file(s)")

    print("\n📄 Files Included:")
    for i, name in enumerate(file_names, 1):
        print(f"   {i}. {name}")




def main():
    """Main function to generate batch request JSON files."""
    print("=" * 60)
    print("🧾 Batch Invoice Request Generator")
    print("=" * 60)

    # Get all invoice files
    print(f"\n📁 Scanning folder: {INVOICES_FOLDER}")
    invoice_files = get_invoice_files()

    if not invoice_files:
        print("\n⚠️  No invoice files found!")
        print(f"   Add invoice images to: {INVOICES_FOLDER}")
        print(f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return

    print(f"✓ Found {len(invoice_files)} invoice file(s)")

    # Optional: Add additional instructions
    additional_instructions = None
    # additional_instructions = "Extract all line items with descriptions"

    # Process files in batches
    total_batches = (len(invoice_files) + BATCH_SIZE - 1) // BATCH_SIZE
    generated_files = []

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(invoice_files))
        batch_files = invoice_files[start_idx:end_idx]

        print(f"\n" + "=" * 60)
        print(f"📦 Generating Batch {batch_num + 1}/{total_batches} ({len(batch_files)} files)")
        print("=" * 60)

        # Create batch request
        print(f"\n🔧 Encoding files to base64...")
        payload = create_batch_request(batch_files, additional_instructions)

        # Get file names for metadata
        file_names = [f.name for f in batch_files]

        # Print summary
        print_batch_summary(payload, file_names)

        # Save batch request to JSON file
        print("\n💾 Saving batch request...")
        output_file = save_batch_request(payload, batch_num + 1, file_names)
        generated_files.append(output_file)

    print("\n" + "=" * 60)
    print("✨ All batch requests generated!")
    print("=" * 60)
    print(f"\n📂 Generated {len(generated_files)} batch request file(s):")
    for file in generated_files:
        print(f"   • {file}")

    print("\n💡 Next steps:")
    print(f"   1. Review the generated JSON files in: {OUTPUT_FOLDER}")
    print("   2. Use these files to send requests to your API")
    print(f"   3. Example: curl -X POST http://localhost:8000/api/v1/batch \\")
    print("              -H 'Content-Type: application/json' \\")
    print(f"              -d @{OUTPUT_FOLDER}/batch_request_1_*.json")


if __name__ == "__main__":
    main()
