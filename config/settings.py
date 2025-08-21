# Enterprise RAG System Configuration

import os
from typing import Optional, List
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "Enterprise RAG System"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "localhost"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite:///./data/enterprise_rag.db"

    # LLM Configuration  
    DEFAULT_MODEL: str = "microsoft/DialoGPT-medium"
    HUGGINGFACE_TOKEN: Optional[str] = None
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7

    # Vector Database
    VECTOR_DB_TYPE: str = "chroma"  # chroma, pinecone, faiss
    CHROMA_PERSIST_DIR: str = "./data/vectorstore/chroma"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX: str = "enterprise-rag"

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FORMATS: List[str] = [".txt", ".pdf", ".docx", ".md", ".html", ".csv"]

    # Security
    SECRET_KEY: str = "enterprise-rag-secret-key-change-in-production"

    class Config:
        env_file = ".env"

settings = Settings()
