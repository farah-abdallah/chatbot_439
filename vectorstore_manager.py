"""
Vector store manager for handling FAISS vectorstore operations.
"""
from typing import Optional, List, Dict
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the global FAISS vectorstore and document metadata."""
    
    def __init__(self):
        """Initialize the vector store manager."""
        self.vectorstore: Optional[FAISS] = None
        self.documents_metadata: List[Dict[str, str]] = []
        self.embeddings = OpenAIEmbeddings()
        logger.info("VectorStoreManager initialized")
    
    def merge_vectorstores(self, new_store: FAISS, metadata: Dict[str, str]) -> None:
        """
        Merge a new vectorstore into the existing global vectorstore.
        
        Args:
            new_store: The new FAISS vectorstore to merge
            metadata: Metadata about the source (filename, file_type)
        """
        if self.vectorstore is None:
            # First vectorstore - just assign it
            self.vectorstore = new_store
            logger.info("Created initial vectorstore")
        else:
            # Merge by extracting documents from new store and adding to existing
            try:
                # Get all documents from the new store
                new_docs = new_store.docstore._dict.values()
                texts = [doc.page_content for doc in new_docs]
                metadatas = [doc.metadata for doc in new_docs]
                
                # Update metadata with source information
                for meta in metadatas:
                    meta.update(metadata)
                
                # Add to existing vectorstore
                self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
                logger.info(f"Merged {len(texts)} documents into vectorstore")
            except Exception as e:
                logger.error(f"Error merging vectorstores: {str(e)}")
                raise
        
        # Store metadata
        self.documents_metadata.append(metadata)
    
    def similarity_search(self, query: str, k: int = 4) -> List:
        """
        Perform similarity search on the vectorstore.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of documents with their content and metadata
        """
        if self.vectorstore is None:
            raise ValueError("No documents have been uploaded yet. Please upload documents first.")
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get the total number of document chunks in the vectorstore.
        
        Returns:
            Number of document chunks
        """
        if self.vectorstore is None:
            return 0
        return len(self.vectorstore.docstore._dict)
    
    def get_sources_summary(self) -> List[str]:
        """
        Get a summary of all loaded document sources.
        
        Returns:
            List of source descriptions
        """
        sources = []
        for meta in self.documents_metadata:
            filename = meta.get('filename', 'unknown')
            file_type = meta.get('file_type', 'unknown')
            sources.append(f"{filename} ({file_type})")
        return sources
    
    def reset(self) -> None:
        """Reset the vectorstore and metadata (for testing or admin purposes)."""
        self.vectorstore = None
        self.documents_metadata = []
        logger.info("VectorStore reset")


# Global instance
vector_store_manager = VectorStoreManager()
