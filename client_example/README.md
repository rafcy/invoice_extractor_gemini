# Batch Invoice Processor Client

A Python client script for processing invoice images in batches using the Invoice Processing API.

## Features

- Reads invoice images from a local folder
- Encodes images to base64 automatically
- Sends batch requests to the API
- Handles multiple image formats (JPG, PNG, GIF, BMP, WebP, PDF)
- Saves results to JSON files
- Shows detailed processing summary
- Supports custom batch sizes
- Handles errors gracefully

## Setup

### 1. Install Dependencies

```bash
pip install requests
```

### 2. Add Invoice Images

Place your invoice images in the `invoices` folder:

```bash
# The folder is already created, just add your images
cp /path/to/your/invoice1.jpg ./invoices/
cp /path/to/your/invoice2.png ./invoices/
```

### 3. Configure API URL (Optional)

By default, the script connects to `http://localhost:8000`. To change this:

```bash
export API_BASE_URL="http://your-api-server:8000"
```

Or edit the `API_BASE_URL` variable in the script.

## Usage

### Basic Usage

```bash
python batch_process_invoices.py
```

### Custom Batch Size

Edit the `BATCH_SIZE` variable in the script:

```python
BATCH_SIZE = 5  # Process 5 images per batch
```

### Additional Instructions

To add custom instructions for invoice processing, uncomment and modify:

```python
additional_instructions = "Extract all line items with descriptions and tax breakdown"
```

## Output

### Console Output

The script provides detailed progress information:

```
============================================================
🧾 Batch Invoice Processor Client
============================================================

📁 Scanning folder: /path/to/invoices
✓ Found 15 invoice file(s)

============================================================
📦 Processing Batch 1/2 (10 files)
============================================================

🔧 Encoding files to base64...
  ✓ Encoded: invoice_001.jpg (image/jpeg)
  ✓ Encoded: invoice_002.png (image/png)
  ...

🚀 Sending batch request to http://localhost:8000/api/v1/batch...
✅ Batch completed in 3.45s

📊 Batch Processing Summary:
   Total invoices: 10
   ✅ Successful: 10
   ❌ Failed: 0
   ⏱️  Avg processing time: 234.56ms

📋 Individual Results:
   ✅ Invoice 1: success (45.23ms)
      Invoice #: INV-2024-001, Total: $1,234.56
   ✅ Invoice 2: success (23.12ms)
      Invoice #: INV-2024-002, Total: $567.89
   ...

💾 Results saved to: results/batch_1_20250124_143022.json
```

### JSON Output

Results are saved to the `results` folder with timestamps:

```json
{
  "results": [
    {
      "status": "success",
      "data": {
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-01-15",
        "total_amount": "$1,234.56",
        "vendor_name": "Acme Corp",
        "line_items": [...]
      },
      "processing_time_ms": 45.23
    },
    ...
  ]
}
```

## Folder Structure

```
client_example/
├── batch_process_invoices.py    # Main script
├── README.md                      # This file
├── invoices/                      # Place your invoice images here
│   ├── invoice_001.jpg
│   ├── invoice_002.png
│   └── ...
└── results/                       # Processing results (auto-created)
    ├── batch_1_20250124_143022.json
    └── batch_2_20250124_143025.json
```

## Supported File Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)
- PDF (.pdf)

## Error Handling

The script handles various error scenarios:

- **Connection errors**: Verifies API is running
- **Timeout errors**: 5-minute timeout for large batches
- **HTTP errors**: Shows detailed error messages
- **File encoding errors**: Skips problematic files and continues

## Configuration Options

Edit these variables in the script:

```python
# API endpoint
API_BASE_URL = "http://localhost:8000"

# Number of images per batch request
BATCH_SIZE = 10

# Folders
INVOICES_FOLDER = Path(__file__).parent / "invoices"
OUTPUT_FOLDER = Path(__file__).parent / "results"

# Supported formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".pdf"}
```

## Performance Tips

1. **Batch Size**: Larger batches are faster but may timeout. Start with 10 and adjust.
2. **Network**: Keep the client close to the API server for best performance.
3. **Cache**: The API caches results - reprocessing the same invoices will be instant!
4. **Rate Limits**: The script automatically handles API rate limits.

## Examples

### Example 1: Process 100 Invoices

```bash
# Add 100 invoices to the folder
cp /path/to/invoices/*.jpg ./invoices/

# Process in batches of 10
python batch_process_invoices.py
```

### Example 2: Custom Instructions

Edit the script to add instructions:

```python
additional_instructions = "Focus on extracting tax breakdown and payment terms"
```

Then run:

```bash
python batch_process_invoices.py
```

### Example 3: Different API Server

```bash
export API_BASE_URL="http://production-server:8000"
python batch_process_invoices.py
```

## Troubleshooting

### No Files Found

```
⚠️  No invoice files found!
```

**Solution**: Add invoice images to the `invoices` folder.

### Connection Error

```
❌ Connection error. Is the API running at http://localhost:8000?
```

**Solution**:
1. Start the API server: `cd ../API && uvicorn main:app --reload`
2. Or update `API_BASE_URL` to point to your server

### Timeout Error

```
❌ Request timed out after 5 minutes
```

**Solution**: Reduce `BATCH_SIZE` to process fewer images per request.

## API Endpoints Used

- `POST /api/v1/batch` - Batch processing endpoint

## License

This client example is provided as-is for demonstration purposes.
