"""
FastAPI application for RAG-based chatbot with CRAG (Corrective Retrieval Augmented Generation).
"""
# Load environment variables FIRST, before any other imports
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse, UploadResponse, HealthResponse
from vectorstore_manager import vector_store_manager
from file_processor import process_file, get_file_metadata
from crag_service import CRAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global CRAG service instance
crag_service: CRAGService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app startup and shutdown.
    """
    # Startup
    logger.info("Starting up RAG Chatbot API...")
    
    # Verify OpenAI API key is set
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables!")
        raise RuntimeError("OPENAI_API_KEY not found. Please set it in .env file.")
    
    logger.info("Environment variables loaded successfully")
    
    # Initialize CRAG service
    global crag_service
    crag_service = CRAGService(
        model="gpt-4o-mini",
        temperature=0,
        lower_threshold=0.3,
        upper_threshold=0.7
    )
    logger.info("CRAG service initialized")
    
    logger.info("RAG Chatbot API is ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Chatbot API...")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API with CRAG",
    description="A chatbot API using Retrieval Augmented Generation with Corrective RAG technique",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    """
    Serve the frontend HTML page.
    """
    return FileResponse("static/index.html")


@app.get("/api", response_class=HTMLResponse)
async def serve_frontend_api():
    """
    Serve the frontend HTML page at /api path for Azure Functions.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Detailed health check endpoint.
    """
    doc_count = vector_store_manager.get_document_count()
    sources = vector_store_manager.get_sources_summary()
    
    return HealthResponse(
        message=f"API is healthy. {doc_count} document chunks loaded from {len(sources)} files.",
        status="healthy",
        documents_loaded=doc_count
    )


@app.post("/upload_files", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents (PDF, TXT, CSV) to the knowledge base.
    
    Args:
        files: List of files to upload
        
    Returns:
        UploadResponse with processing results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    logger.info(f"Received {len(files)} files for upload")
    
    processed_files = []
    errors = []
    
    for file in files:
        logger.info(f"Processing file: {file.filename}")
        
        # Process the file
        vectorstore, error = process_file(file)
        
        if error:
            errors.append(f"{file.filename}: {error}")
            continue
        
        # Get metadata
        metadata = get_file_metadata(file.filename)
        
        try:
            # Merge into global vectorstore
            vector_store_manager.merge_vectorstores(vectorstore, metadata)
            processed_files.append(file.filename)
            logger.info(f"Successfully processed: {file.filename}")
        except Exception as e:
            error_msg = f"Error merging {file.filename}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Prepare response
    if not processed_files and errors:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process any files. Errors: {'; '.join(errors)}"
        )
    
    message = f"Successfully processed {len(processed_files)} file(s)"
    if errors:
        message += f". {len(errors)} file(s) failed."
    
    return UploadResponse(
        message=message,
        number_of_files_processed=len(processed_files),
        filenames=processed_files,
        errors=errors
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question to the chatbot using RAG with CRAG technique.
    
    Args:
        request: ChatRequest containing the question and options
        
    Returns:
        ChatResponse with the answer and sources
    """
    logger.info(f"Chat request received: '{request.question}' (use_web_search={request.use_web_search})")
    
    # Check if documents have been uploaded
    if vector_store_manager.vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload documents using /upload_files endpoint first."
        )
    
    try:
        # Retrieve similar documents from vectorstore
        docs = vector_store_manager.similarity_search(request.question, k=request.k)
        retrieved_docs = [doc.page_content for doc in docs]
        
        # Extract source filenames from metadata
        doc_sources = []
        for doc in docs:
            source_file = doc.metadata.get('source_file', 'Unknown document')
            if source_file not in doc_sources:
                doc_sources.append(source_file)
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents from: {doc_sources}")
        
        # Process query using CRAG service
        answer, sources, action_taken = crag_service.process_query(
            query=request.question,
            retrieved_docs=retrieved_docs,
            use_web_search=request.use_web_search
        )
        
        # If using documents, prepend the specific source files
        if action_taken in ['correct', 'ambiguous'] and doc_sources:
            # Add document sources at the beginning
            doc_source_labels = [f"📄 {source}" for source in doc_sources]
            sources = doc_source_labels + sources
        
        logger.info(f"Generated answer using CRAG action: {action_taken}")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieval_action=action_taken
        )
    
    except ValueError as e:
        logger.error(f"ValueError in chat endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your request: {str(e)}"
        )


@app.delete("/reset")
async def reset_vectorstore():
    """
    Reset the vectorstore (admin endpoint for testing).
    WARNING: This will delete all uploaded documents from memory.
    """
    logger.warning("Vectorstore reset requested")
    vector_store_manager.reset()
    return JSONResponse(
        content={"message": "Vectorstore has been reset. All documents removed from memory."}
    )


@app.get("/sources")
async def list_sources():
    """
    Get a list of all loaded document sources.
    """
    sources = vector_store_manager.get_sources_summary()
    doc_count = vector_store_manager.get_document_count()
    
    return {
        "total_document_chunks": doc_count,
        "number_of_files": len(sources),
        "sources": sources
    }


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
