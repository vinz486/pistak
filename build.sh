#!/bin/bash

# Exit on error
set -e

echo "=== Pistak Build Script ==="

echo "Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing project requirements..."
pip install -r requirements.txt

echo "Installing PyInstaller..."
pip install pyinstaller

echo "🧹 Cleaning up previous builds..."
rm -rf build/ dist/

echo "📦 Packaging Pistak application..."
pyinstaller \
    --name "pistak" \
    --clean \
    --onefile \
            --collect-all "textual" \
            --collect-all "openvino" \
            --collect-all "openvino_genai" \
            --hidden-import "server" \
            --hidden-import "uvicorn" \
            --hidden-import "fastapi" \
            --hidden-import "openvino_genai" \
            --hidden-import "openvino" \
            --hidden-import "psutil" \
            --hidden-import "pydantic" \
            --hidden-import "huggingface_hub" \
            main.py

echo "==================================="
echo "Build complete! 🎉"
echo "Your executable is ready at: ./dist/pistak"
echo "You can run it simply with: ./dist/pistak"
echo "==================================="
