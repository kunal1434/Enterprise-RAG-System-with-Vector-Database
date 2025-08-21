# Document Processing Module

import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import mimetypes

# Document processing libraries
import PyPDF2
from docx import Document as DocxDocument  
import pandas as pd
from bs4 import BeautifulSoup
import markdown

# LangChain for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Logging
import logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process various document formats for RAG system"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        # File processors mapping
        self.processors = {
            '.txt': self._process_txt,
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.md': self._process_markdown,
            '.html': self._process_html,
            '.csv': self._process_csv
        }

    def _process_txt(self, file_path: str) -> str:
        """Process plain text files"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _process_pdf(self, file_path: str) -> str:
        """Process PDF files"""
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
        return text

    def _process_docx(self, file_path: str) -> str:
        """Process Word documents"""
        try:
            doc = DocxDocument(file_path)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs]
            return "\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            raise

    def _process_markdown(self, file_path: str) -> str:
        """Process Markdown files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()

    def _process_html(self, file_path: str) -> str:
        """Process HTML files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()

    def _process_csv(self, file_path: str) -> str:
        """Process CSV files"""
        df = pd.read_csv(file_path)
        return df.to_string()

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and return chunks with metadata"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()
        if file_ext not in self.processors:
            raise ValueError(f"Unsupported file format: {file_ext}")

        logger.info(f"Processing file: {file_path}")

        # Extract text content
        text_content = self.processors[file_ext](str(file_path))

        # Split into chunks
        documents = self.text_splitter.create_documents([text_content])

        # Create chunks with metadata
        chunks = []
        for i, doc in enumerate(documents):
            chunk_metadata = {
                'source': str(file_path),
                'chunk_id': f"{file_path.stem}_{i}",
                'chunk_index': i,
                'file_type': file_ext,
                'chunk_length': len(doc.page_content)
            }

            chunks.append({
                'content': doc.page_content,
                'metadata': chunk_metadata
            })

        return {
            'file_path': str(file_path),
            'file_type': file_ext,
            'chunks_count': len(chunks),
            'chunks': chunks,
            'total_length': len(text_content)
        }

    async def process_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """Process all supported files in a directory"""
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Find supported files
        files_to_process = []
        pattern = "**/*" if recursive else "*"

        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.processors:
                files_to_process.append(file_path)

        logger.info(f"Found {len(files_to_process)} files to process")

        # Process files
        results = []
        for file_path in files_to_process:
            try:
                result = self.process_file(file_path)
                results.append(result)
                logger.info(f"✅ Processed: {file_path}")
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
                continue

        return results

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.processors.keys())
