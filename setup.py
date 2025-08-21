#!/usr/bin/env python3
# Enterprise RAG System - Setup Script

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    print("=" * 60)
    print("🚀 ENTERPRISE RAG SYSTEM SETUP")
    print("   Perfect for Techolution AI Intern Application")
    print("=" * 60)
    print()

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def run_command(command, check=True):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False

    if result.stdout:
        print(result.stdout)
    return True

def install_dependencies():
    print("\n📦 Installing Python dependencies...")

    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        print("⚠️ Warning: Could not upgrade pip")

    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
        print("❌ Failed to install dependencies")
        return False

    print("✅ Dependencies installed successfully")
    return True

def setup_directories():
    directories = [
        "data/documents",
        "data/vectorstore/chroma", 
        "data/sample_docs",
        "logs",
        "models/fine_tuned"
    ]

    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   📂 {directory}")

    print("✅ Directories created")

def create_env_file():
    env_content = '''# Enterprise RAG System Configuration

# Application Settings
APP_NAME=Enterprise RAG System
VERSION=1.0.0
DEBUG=true
HOST=localhost
PORT=8000

# LLM Configuration
DEFAULT_MODEL=microsoft/DialoGPT-small
HUGGINGFACE_TOKEN=your-huggingface-token-here
MAX_TOKENS=512
TEMPERATURE=0.7

# Vector Database
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=./data/vectorstore/chroma

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE=52428800

# Security
SECRET_KEY=enterprise-rag-secret-key-change-in-production
'''

    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("⚠️ .env file already exists, skipping...")

def test_installation():
    print("\n🧪 Testing installation...")

    try:
        # Test core imports
        import torch
        print("✅ PyTorch available")

        import transformers
        print("✅ Transformers available")

        import chromadb
        print("✅ ChromaDB available")

        import fastapi
        print("✅ FastAPI available")

        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            print("⚠️ CUDA not available, using CPU")

        print("✅ All core dependencies working")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def show_next_steps():
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("📋 NEXT STEPS:")
    print("1. Start the system:")
    print("   python src/api/main.py")
    print()
    print("2. Open your browser:")
    print("   http://localhost:8000")
    print()
    print("3. Or use Docker:")
    print("   docker-compose up -d")
    print()
    print("🎯 FOR TECHOLUTION INTERVIEW:")
    print("✅ LLM Fine-tuning: POST to /api/v1/finetune")
    print("✅ Vector Databases: Chroma, Pinecone, Faiss support")
    print("✅ Document Processing: Upload files to /api/v1/ingest/files")
    print("✅ RAG Queries: POST to /api/v1/query")
    print("✅ API Documentation: http://localhost:8000/docs")
    print()
    print("🚀 Ready for your AI Intern application demo!")
    print("=" * 60)

def main():
    print_header()

    # Check Python version
    check_python_version()

    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)

    # Setup directories
    setup_directories()

    # Create environment file
    create_env_file()

    # Test installation
    if not test_installation():
        print("❌ Setup completed but some tests failed")
        print("   You may still be able to run the system")

    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
