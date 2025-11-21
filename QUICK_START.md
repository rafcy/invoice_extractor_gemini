# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
cd API
pip install -r requirements.txt

# macOS
brew install poppler

# Linux
sudo apt-get install poppler-utils
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Start Server

```bash
# Option 1: Use the start script
./start.sh

# Option 2: Run directly
python main.py

# Option 3: Use uvicorn
uvicorn main:app --reload
```

### 4. Test API

Open browser: http://localhost:8000/docs

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Process a document
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@/path/to/invoice.pdf"
```

## Docker Quick Start

```bash
# Build
docker build -t invoice-api .

# Run
docker run -p 8000:8000 --env-file .env invoice-api
```

## Common Commands

```bash
# Check rate limit
curl http://localhost:8000/api/v1/rate-limit

# Process from URL
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/invoice.pdf"}'

# Clear cache
curl -X DELETE http://localhost:8000/api/v1/cache
```

## Configuration

Edit `.env` to adjust:
- `RATE_LIMIT_MAX_PER_MINUTE`: Default 10
- `GEMINI_MODEL`: flash-lite, flash, or pro
- `ENABLE_CACHE`: true/false

## Troubleshooting

**PDF not working?**
```bash
# macOS
brew install poppler

# Linux
sudo apt-get install poppler-utils
```

**Rate limit errors?**
- Increase `RATE_LIMIT_MAX_PER_MINUTE` in `.env`
- Use batch endpoint for multiple documents

**API key errors?**
- Check `GOOGLE_API_KEY` in `.env`
- Get key from: https://makersuite.google.com/app/apikey

## Next Steps

1. Read full [README.md](README.md) for detailed documentation
2. Try [example_usage.py](example_usage.py) for Python examples
3. Explore interactive docs at http://localhost:8000/docs
