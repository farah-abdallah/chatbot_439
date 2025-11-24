"""
File processor module for handling different file types.
"""
import os
import pandas as pd
from typing import Tuple, Optional
from langchain.vectorstores import FAISS
from fastapi import UploadFile
import logging

from helper_functions import encode_pdf, encode_from_string

logger = logging.getLogger(__name__)

# Directory for uploaded files
UPLOAD_DIR = "uploaded_files"


def ensure_upload_directory() -> None:
    """Ensure the upload directory exists."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        logger.info(f"Created upload directory: {UPLOAD_DIR}")


def save_uploaded_file(file: UploadFile) -> str:
    """
    Save an uploaded file to disk.
    
    Args:
        file: The FastAPI UploadFile object
        
    Returns:
        The path where the file was saved
    """
    ensure_upload_directory()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            content = file.file.read()
            f.write(content)
        logger.info(f"Saved file: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {str(e)}")
        raise


def process_pdf(file_path: str, filename: str) -> FAISS:
    """
    Process a PDF file and return a FAISS vectorstore.
    
    Args:
        file_path: Path to the PDF file
        filename: Original filename for metadata
        
    Returns:
        FAISS vectorstore containing the PDF content
    """
    try:
        logger.info(f"Processing PDF: {file_path}")
        vectorstore = encode_pdf(file_path, chunk_size=1000, chunk_overlap=200)
        
        # Add source filename to all documents
        for doc_id, doc in vectorstore.docstore._dict.items():
            doc.metadata['source_file'] = filename
        
        logger.info(f"Successfully encoded PDF: {file_path}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise


def process_txt(file_path: str, filename: str) -> FAISS:
    """
    Process a TXT file and return a FAISS vectorstore.
    
    Args:
        file_path: Path to the TXT file
        filename: Original filename for metadata
        
    Returns:
        FAISS vectorstore containing the TXT content
    """
    try:
        logger.info(f"Processing TXT: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            raise ValueError(f"Empty file: {file_path}")
        
        vectorstore = encode_from_string(content, chunk_size=1000, chunk_overlap=200)
        
        # Add source filename to all documents
        for doc_id, doc in vectorstore.docstore._dict.items():
            doc.metadata['source_file'] = filename
        
        logger.info(f"Successfully encoded TXT: {file_path}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error processing TXT {file_path}: {str(e)}")
        raise


def process_csv(file_path: str, filename: str) -> FAISS:
    """
    Process a CSV file and return a FAISS vectorstore.
    
    Args:
        file_path: Path to the CSV file
        filename: Original filename for metadata
        
    Returns:
        FAISS vectorstore containing the CSV content
    """
    try:
        logger.info(f"Processing CSV: {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to a formatted string
        # Include headers and all rows
        content_parts = []
        content_parts.append("Column names: " + ", ".join(df.columns.tolist()))
        content_parts.append("\nData:\n")
        
        for idx, row in df.iterrows():
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
            content_parts.append(row_str)
        
        content = "\n".join(content_parts)
        
        if not content.strip():
            raise ValueError(f"Empty CSV file: {file_path}")
        
        vectorstore = encode_from_string(content, chunk_size=1000, chunk_overlap=200)
        
        # Add source filename to all documents
        for doc_id, doc in vectorstore.docstore._dict.items():
            doc.metadata['source_file'] = filename
        
        logger.info(f"Successfully encoded CSV: {file_path}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error processing CSV {file_path}: {str(e)}")
        raise


def process_file(file: UploadFile) -> Tuple[Optional[FAISS], Optional[str]]:
    """
    Process an uploaded file based on its extension.
    
    Args:
        file: The FastAPI UploadFile object
        
    Returns:
        Tuple of (vectorstore, error_message)
        If successful, returns (vectorstore, None)
        If failed, returns (None, error_message)
    """
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validate file extension
    if file_extension not in ['.pdf', '.txt', '.csv']:
        error_msg = f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt, .csv"
        logger.warning(error_msg)
        return None, error_msg
    
    try:
        # Save the file
        file_path = save_uploaded_file(file)
        
        # Process based on extension
        if file_extension == '.pdf':
            vectorstore = process_pdf(file_path, file.filename)
        elif file_extension == '.txt':
            vectorstore = process_txt(file_path, file.filename)
        elif file_extension == '.csv':
            vectorstore = process_csv(file_path, file.filename)
        else:
            return None, f"Unsupported file type: {file_extension}"
        
        return vectorstore, None
    
    except Exception as e:
        error_msg = f"Error processing {file.filename}: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def get_file_metadata(filename: str) -> dict:
    """
    Extract metadata from filename.
    
    Args:
        filename: The name of the file
        
    Returns:
        Dictionary containing filename and file_type
    """
    file_extension = os.path.splitext(filename)[1].lower()
    return {
        'filename': filename,
        'file_type': file_extension.lstrip('.')
    }
