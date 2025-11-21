#!/bin/bash

# Quick start script for Invoice Processing API

echo "🚀 Starting Invoice Processing API..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "   Creating from .env.example..."
    cp .env.example .env
    echo ""
    echo "❗ IMPORTANT: Edit .env and add your GOOGLE_API_KEY"
    echo "   Get your API key from: https://makersuite.google.com/app/apikey"
    echo ""
    read -p "Press Enter after adding your API key to .env..."
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.12 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check for poppler (PDF support)
if ! command -v pdfinfo &> /dev/null; then
    echo ""
    echo "⚠️  Warning: poppler not found! PDF support will be disabled."
    echo "   Install with:"
    echo "     macOS: brew install poppler"
    echo "     Linux: sudo apt-get install poppler-utils"
    echo ""
fi

# Start the server
echo ""
echo "✅ Starting server..."
echo "   API Docs: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/health"
echo ""

python main.py
