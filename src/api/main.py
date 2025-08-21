# Enterprise RAG System - FastAPI Web Service

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import tempfile
import asyncio
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core components
try:
    from src.core.rag_pipeline import RAGPipeline
    from config.settings import settings
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Fallback imports for development
    settings = type('Settings', (), {
        'APP_NAME': 'Enterprise RAG System',
        'VERSION': '1.0.0',
        'HOST': 'localhost',
        'PORT': 8000,
        'MAX_FILE_SIZE': 50 * 1024 * 1024,
        'SUPPORTED_FORMATS': ['.txt', '.pdf', '.docx', '.md', '.html', '.csv']
    })()

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered document Q&A system with vector database and LLM fine-tuning",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("src/frontend/static"):
    app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")

# Global RAG pipeline instance
rag_pipeline = None

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    k: int = 5

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None

class FineTuneRequest(BaseModel):
    training_data: List[Dict[str, str]]
    model_name: str
    epochs: int = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    num_sources: int
    processing_time: float

class SystemStatus(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]

# Dependency to get RAG pipeline
async def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        try:
            rag_pipeline = RAGPipeline()
            logger.info("RAG Pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize RAG system")
    return rag_pipeline

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Starting {settings.APP_NAME}")

    # Create necessary directories
    directories = ["data/documents", "data/vectorstore", "logs", "models/fine_tuned"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Initialize RAG pipeline
    try:
        global rag_pipeline
        rag_pipeline = RAGPipeline()
        logger.info("✅ RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG Pipeline: {e}")

# Root endpoint - serve frontend
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page"""
    try:
        if os.path.exists("src/frontend/templates/index.html"):
            with open("src/frontend/templates/index.html", "r") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{settings.APP_NAME}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .header {{ background: #667eea; color: white; padding: 20px; border-radius: 8px; }}
                    .content {{ padding: 20px; }}
                    .endpoint {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 4px; }}
                    .method {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
                    .get {{ background: #28a745; color: white; }}
                    .post {{ background: #007bff; color: white; }}
                    .delete {{ background: #dc3545; color: white; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🤖 {settings.APP_NAME}</h1>
                        <p>Enterprise-grade RAG system with LLM fine-tuning capabilities</p>
                    </div>

                    <div class="content">
                        <h2>🚀 System Status: Online</h2>
                        <p>Version: {settings.VERSION}</p>
                        <p>The API is running successfully! Use the endpoints below to interact with the system.</p>

                        <h3>📚 API Documentation</h3>
                        <p><a href="/docs" target="_blank">Interactive API Documentation (Swagger UI)</a></p>
                        <p><a href="/redoc" target="_blank">Alternative API Documentation (ReDoc)</a></p>

                        <h3>🔧 Available Endpoints</h3>

                        <div class="endpoint">
                            <span class="method get">GET</span>
                            <strong>/health</strong> - System health check
                        </div>

                        <div class="endpoint">
                            <span class="method get">GET</span>
                            <strong>/status</strong> - Detailed system status
                        </div>

                        <div class="endpoint">
                            <span class="method post">POST</span>
                            <strong>/api/v1/ingest/files</strong> - Upload and process documents
                        </div>

                        <div class="endpoint">
                            <span class="method post">POST</span>
                            <strong>/api/v1/query</strong> - Ask questions about documents (RAG)
                        </div>

                        <div class="endpoint">
                            <span class="method post">POST</span>
                            <strong>/api/v1/chat</strong> - Chat interface with conversation memory
                        </div>

                        <div class="endpoint">
                            <span class="method post">POST</span>
                            <strong>/api/v1/finetune</strong> - Fine-tune LLM with custom data
                        </div>

                        <div class="endpoint">
                            <span class="method get">GET</span>
                            <strong>/api/v1/models</strong> - List available models
                        </div>

                        <div class="endpoint">
                            <span class="method get">GET</span>
                            <strong>/api/v1/analytics</strong> - System analytics and metrics
                        </div>

                        <h3>🎯 Perfect for Techolution Interview!</h3>
                        <ul>
                            <li>✅ <strong>LLM Fine-tuning</strong>: POST training data to /api/v1/finetune</li>
                            <li>✅ <strong>Vector Databases</strong>: Chroma, Pinecone, Faiss support</li>
                            <li>✅ <strong>Document Processing</strong>: PDF, DOCX, TXT, MD, HTML, CSV</li>
                            <li>✅ <strong>RAG Pipeline</strong>: Semantic search + AI generation</li>
                            <li>✅ <strong>Enterprise Ready</strong>: FastAPI, async, monitoring</li>
                        </ul>

                        <h3>🚀 Quick Test</h3>
                        <p>Try these curl commands:</p>
                        <pre style="background: #f0f0f0; padding: 10px; border-radius: 4px;">
# Check system health
curl {settings.HOST}:{settings.PORT}/health

# Get system status  
curl {settings.HOST}:{settings.PORT}/status

# Query example (after uploading documents)
curl -X POST "{settings.HOST}:{settings.PORT}/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{{"question": "What is machine learning?", "k": 5}}'
                        </pre>
                    </div>
                </div>
            </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving root page: {e}")
        return HTMLResponse(content=f"<h1>Enterprise RAG System</h1><p>Error: {str(e)}</p>")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.VERSION,
        "app": settings.APP_NAME
    }

# System status
@app.get("/status")
async def get_system_status():
    """Get detailed system status"""
    try:
        pipeline = await get_rag_pipeline()
        status = pipeline.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Document upload and ingestion
@app.post("/api/v1/ingest/files")
async def ingest_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and ingest multiple documents"""
    try:
        pipeline = await get_rag_pipeline()

        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            # Check file type
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in settings.SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_ext}"
                )

            # Check file size
            content = await file.read()
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {file.filename}"
                )

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(content)
                temp_files.append(temp_file.name)

        # Process files
        result = await pipeline.ingest_documents(temp_files)

        # Clean up temp files in background
        def cleanup_files():
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

        background_tasks.add_task(cleanup_files)

        return {
            "status": "success",
            "message": f"Processed {len(result['processed_files'])} files",
            "details": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post("/api/v1/query")
async def query_documents(request: QueryRequest):
    """Query documents using RAG"""
    try:
        pipeline = await get_rag_pipeline()
        result = await pipeline.query(request.question, request.k)
        return result

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoint
@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """Chat with the system using conversation memory"""
    try:
        pipeline = await get_rag_pipeline()
        result = await pipeline.chat(request.message, request.conversation_history)
        return result

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search documents
@app.get("/api/v1/search")
async def search_documents(query: str, k: int = 5):
    """Search for relevant documents"""
    try:
        pipeline = await get_rag_pipeline()
        results = pipeline.search_documents(query, k)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fine-tuning endpoint
@app.post("/api/v1/finetune")
async def fine_tune_model(
    request: FineTuneRequest,
    background_tasks: BackgroundTasks
):
    """Fine-tune the LLM with custom data"""
    try:
        pipeline = await get_rag_pipeline()

        # Start fine-tuning in background
        async def run_fine_tuning():
            try:
                result = await pipeline.fine_tune_llm(
                    request.training_data,
                    request.model_name,
                    request.epochs
                )
                logger.info(f"Fine-tuning completed: {result}")
            except Exception as e:
                logger.error(f"Background fine-tuning failed: {e}")

        background_tasks.add_task(run_fine_tuning)

        return {
            "status": "started",
            "message": f"Fine-tuning started for model: {request.model_name}",
            "training_samples": len(request.training_data),
            "epochs": request.epochs
        }

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get models
@app.get("/api/v1/models")
async def get_models():
    """Get list of available models"""
    try:
        pipeline = await get_rag_pipeline()
        models_info = pipeline.llm_manager.get_model_info()
        return {
            "base_model": models_info.get('model_name'),
            "model_info": models_info,
            "fine_tuned_models": models_info.get('fine_tuned_models', [])
        }

    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoint
@app.get("/api/v1/analytics")
async def get_analytics():
    """Get system analytics and metrics"""
    try:
        pipeline = await get_rag_pipeline()
        analytics = pipeline.get_analytics()
        return analytics

    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoint
@app.get("/api/v1/config")
async def get_configuration():
    """Get current system configuration"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "supported_formats": settings.SUPPORTED_FORMATS,
        "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
        "host": settings.HOST,
        "port": settings.PORT
    }

if __name__ == "__main__":
    import uvicorn

    logger.info(f"🚀 Starting {settings.APP_NAME} API Server")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
