# Vector Database Manager

import os
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

# Vector databases
import chromadb
from chromadb.config import Settings as ChromaSettings

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Embeddings
from sentence_transformers import SentenceTransformer

import logging
logger = logging.getLogger(__name__)

class VectorDatabase(ABC):
    """Abstract base class for vector databases"""

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        pass

class ChromaVectorDB(VectorDatabase):
    """ChromaDB implementation"""

    def __init__(self, persist_directory: str, collection_name: str = "enterprise_docs"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(allow_reset=True)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Enterprise document collection"}
            )
            logger.info(f"Initialized ChromaDB at {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to ChromaDB"""
        try:
            ids = [doc['metadata']['chunk_id'] for doc in documents]
            contents = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]

            self.collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': 1.0 - results['distances'][0][i] if 'distances' in results else 1.0
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'type': 'ChromaDB',
                'status': 'healthy'
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {'error': str(e), 'type': 'ChromaDB', 'status': 'error'}

class SimpleVectorDB(VectorDatabase):
    """Simple in-memory vector database for fallback"""

    def __init__(self):
        self.documents = {}
        self.embeddings = {}
        self.embedding_model = None

        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized SimpleVectorDB with SentenceTransformer")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to simple vector DB"""
        try:
            if not self.embedding_model:
                logger.error("No embedding model available")
                return False

            for doc in documents:
                doc_id = doc['metadata']['chunk_id']
                self.documents[doc_id] = doc

                # Generate embedding
                embedding = self.embedding_model.encode(doc['content'])
                self.embeddings[doc_id] = embedding

            logger.info(f"Added {len(documents)} documents to SimpleVectorDB")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using cosine similarity"""
        try:
            if not self.embedding_model or not self.embeddings:
                return []

            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)

            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((doc_id, similarity))

            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:k]

            # Format results
            formatted_results = []
            for doc_id, score in top_results:
                doc = self.documents[doc_id]
                formatted_results.append({
                    'id': doc_id,
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(score)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            for doc_id in document_ids:
                self.documents.pop(doc_id, None)
                self.embeddings.pop(doc_id, None)
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        return {
            'name': 'simple_vector_db',
            'count': len(self.documents),
            'type': 'SimpleVectorDB',
            'status': 'healthy'
        }

class VectorDBManager:
    """Manager for different vector database implementations"""

    def __init__(self, db_type: str = "chroma", **kwargs):
        self.db_type = db_type
        self.db = self._initialize_db(db_type, **kwargs)

    def _initialize_db(self, db_type: str, **kwargs) -> VectorDatabase:
        """Initialize the specified vector database"""
        try:
            if db_type == "chroma":
                return ChromaVectorDB(
                    persist_directory=kwargs.get('persist_directory', './data/vectorstore/chroma'),
                    collection_name=kwargs.get('collection_name', 'enterprise_docs')
                )
            elif db_type == "simple":
                return SimpleVectorDB()
            else:
                logger.warning(f"Unsupported vector database type: {db_type}, falling back to SimpleVectorDB")
                return SimpleVectorDB()
        except Exception as e:
            logger.error(f"Failed to initialize {db_type}: {e}, falling back to SimpleVectorDB")
            return SimpleVectorDB()

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to vector database"""
        return self.db.add_documents(documents)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        return self.db.search(query, k)

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents"""
        return self.db.delete_documents(document_ids)

    def get_info(self) -> Dict[str, Any]:
        """Get database information"""
        return self.db.get_collection_info()
