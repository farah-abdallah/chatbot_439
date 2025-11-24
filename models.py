"""
Pydantic models for request and response schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., description="The question to ask the chatbot")
    use_web_search: bool = Field(default=False, description="Whether to use web search for additional context")
    k: int = Field(default=4, ge=1, le=10, description="Number of similar documents to retrieve")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(default_factory=list, description="List of sources used for the answer")
    retrieval_action: Optional[str] = Field(None, description="CRAG action taken: 'correct', 'incorrect', or 'ambiguous'")


class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    message: str = Field(..., description="Status message")
    number_of_files_processed: int = Field(..., description="Number of files successfully processed")
    filenames: List[str] = Field(default_factory=list, description="List of uploaded filenames")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered during upload")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    message: str
    status: str
    documents_loaded: int
