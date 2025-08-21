# Enterprise RAG Pipeline - Core System Integration

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

# Core components
from .document_processor import DocumentProcessor
from .vector_db import VectorDBManager
from .llm_manager import LLMManager

# Configuration
from config.settings import settings

import logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Enterprise RAG Pipeline integrating all components"""

    def __init__(self):
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        self.vector_db = VectorDBManager(
            db_type=settings.VECTOR_DB_TYPE,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )

        self.llm_manager = LLMManager(model_name=settings.DEFAULT_MODEL)

        # Load model on initialization
        self.llm_manager.load_model()

        # Query history
        self.query_history = []

        logger.info("✅ RAG Pipeline initialized successfully")

    async def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest multiple documents into the system"""
        ingestion_results = {
            'processed_files': [],
            'failed_files': [],
            'total_chunks': 0,
            'processing_time': 0
        }

        start_time = datetime.now()

        for file_path in file_paths:
            try:
                logger.info(f"Processing: {file_path}")

                # Process document
                result = self.document_processor.process_file(file_path)

                # Add to vector database
                chunks = result['chunks']
                success = self.vector_db.add_documents(chunks)

                if success:
                    ingestion_results['processed_files'].append({
                        'path': file_path,
                        'chunks': len(chunks),
                        'size': result.get('total_length', 0)
                    })
                    ingestion_results['total_chunks'] += len(chunks)
                    logger.info(f"✅ Successfully processed: {file_path}")
                else:
                    ingestion_results['failed_files'].append({
                        'path': file_path,
                        'error': 'Failed to add to vector database'
                    })

            except Exception as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                logger.error(error_msg)
                ingestion_results['failed_files'].append({
                    'path': file_path,
                    'error': str(e)
                })

        end_time = datetime.now()
        ingestion_results['processing_time'] = (end_time - start_time).total_seconds()

        logger.info(f"Ingestion completed: {len(ingestion_results['processed_files'])} success, "
                   f"{len(ingestion_results['failed_files'])} failed")

        return ingestion_results

    async def ingest_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Ingest all supported documents from a directory"""
        try:
            results = await self.document_processor.process_directory(directory_path, recursive)

            ingestion_results = {
                'processed_files': [],
                'failed_files': [],
                'total_chunks': 0,
                'processing_time': 0
            }

            start_time = datetime.now()

            for result in results:
                try:
                    # Add chunks to vector database
                    chunks = result['chunks']
                    success = self.vector_db.add_documents(chunks)

                    if success:
                        ingestion_results['processed_files'].append({
                            'path': result['file_path'],
                            'chunks': len(chunks),
                            'size': result['total_length']
                        })
                        ingestion_results['total_chunks'] += len(chunks)
                    else:
                        ingestion_results['failed_files'].append({
                            'path': result['file_path'],
                            'error': 'Failed to add to vector database'
                        })

                except Exception as e:
                    ingestion_results['failed_files'].append({
                        'path': result['file_path'],
                        'error': str(e)
                    })

            end_time = datetime.now()
            ingestion_results['processing_time'] = (end_time - start_time).total_seconds()

            return ingestion_results

        except Exception as e:
            logger.error(f"Directory ingestion failed: {e}")
            return {'error': str(e)}

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            results = self.vector_db.search(query, k)
            logger.info(f"Found {len(results)} relevant documents for query")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM with retrieved context"""
        try:
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(context_docs[:3]):  # Use top 3 documents
                source = doc['metadata'].get('source', f'Document {i+1}')
                content = doc['content'][:500]  # Truncate for context window
                context_parts.append(f"Source: {source}\n{content}")

            context = "\n\n".join(context_parts)

            # Create prompt for RAG
            prompt = f"""Answer the following question based on the provided context. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

            # Generate response
            answer = self.llm_manager.generate_response(
                prompt,
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )

            return answer

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"

    async def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Complete RAG query: retrieve + generate"""
        start_time = datetime.now()

        try:
            # Step 1: Retrieve relevant documents
            relevant_docs = self.search_documents(question, k)

            if not relevant_docs:
                result = {
                    'question': question,
                    'answer': "I couldn't find any relevant documents to answer your question. Please make sure you have uploaded relevant documents to the system.",
                    'sources': [],
                    'processing_time': 0
                }
            else:
                # Step 2: Generate answer
                answer = self.generate_answer(question, relevant_docs)

                # Step 3: Prepare sources
                sources = []
                for doc in relevant_docs:
                    sources.append({
                        'source': doc['metadata'].get('source', 'Unknown'),
                        'chunk_id': doc['metadata'].get('chunk_id', ''),
                        'score': doc.get('score', 0.0),
                        'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                    })

                result = {
                    'question': question,
                    'answer': answer,
                    'sources': sources,
                    'num_sources': len(sources)
                }

            end_time = datetime.now()
            result['processing_time'] = (end_time - start_time).total_seconds()

            # Add to query history
            self.query_history.append({
                'timestamp': start_time.isoformat(),
                'question': question,
                'num_sources': len(relevant_docs),
                'processing_time': result['processing_time']
            })

            logger.info(f"Query processed in {result['processing_time']:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'question': question,
                'answer': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    async def chat(self, message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Chat interface with conversation context"""
        try:
            # For now, treat as a regular query
            # TODO: Implement proper conversation handling with memory
            result = await self.query(message)

            result.update({
                'type': 'chat',
                'conversation_context': len(conversation_history) if conversation_history else 0
            })

            return result

        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return {
                'question': message,
                'answer': f"Chat error: {str(e)}",
                'sources': [],
                'error': str(e),
                'type': 'chat'
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and metrics"""
        try:
            vector_db_info = self.vector_db.get_info()
            llm_info = self.llm_manager.get_model_info()

            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'document_processor': {
                        'supported_formats': self.document_processor.get_supported_formats(),
                        'chunk_size': self.document_processor.chunk_size,
                        'chunk_overlap': self.document_processor.chunk_overlap
                    },
                    'vector_database': vector_db_info,
                    'llm': llm_info
                },
                'query_history': {
                    'total_queries': len(self.query_history),
                    'recent_queries': self.query_history[-5:] if self.query_history else []
                },
                'settings': {
                    'max_tokens': settings.MAX_TOKENS,
                    'temperature': settings.TEMPERATURE,
                    'vector_db_type': settings.VECTOR_DB_TYPE,
                    'chunk_size': settings.CHUNK_SIZE
                }
            }

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def fine_tune_llm(
        self,
        training_data: List[Dict[str, str]],
        model_name: str,
        epochs: int = 3
    ) -> Dict[str, Any]:
        """Fine-tune the LLM with domain-specific data"""
        try:
            output_dir = f"./models/fine_tuned/{model_name}"
            os.makedirs(output_dir, exist_ok=True)

            result = self.llm_manager.fine_tune_model(
                training_data=training_data,
                output_dir=output_dir,
                num_epochs=epochs
            )

            return result

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return {
                'status': 'error',
                'message': f'Fine-tuning failed: {str(e)}'
            }

    def get_analytics(self) -> Dict[str, Any]:
        """Get system analytics and metrics"""
        try:
            vector_db_info = self.vector_db.get_info()

            return {
                'documents': {
                    'total_count': vector_db_info.get('count', 0),
                    'vector_db_type': vector_db_info.get('type', 'unknown')
                },
                'queries': {
                    'total_count': len(self.query_history),
                    'average_processing_time': sum(q.get('processing_time', 0) for q in self.query_history) / len(self.query_history) if self.query_history else 0,
                    'recent_activity': self.query_history[-10:] if self.query_history else []
                },
                'system': {
                    'uptime': 'Running',
                    'status': 'healthy',
                    'last_updated': datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {'error': str(e)}
