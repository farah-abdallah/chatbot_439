# RAG Chatbot API with CRAG

A modular FastAPI-based chatbot that uses **Retrieval Augmented Generation (RAG)** with the **CRAG (Corrective Retrieval Augmented Generation)** technique. This system intelligently determines when to use retrieved documents, when to perform web searches, or when to combine both approaches based on relevance scores.

## Features

- 📄 **Multi-format Document Support**: Upload and process PDF, TXT, and CSV files
- 🧠 **CRAG Intelligence**: Automatically evaluates document relevance and chooses the best retrieval strategy
- 🔍 **Web Search Integration**: Falls back to DuckDuckGo search when local documents aren't sufficient
- 🎯 **Smart Retrieval**: Uses FAISS vector store for efficient similarity search
- 🔄 **Modular Architecture**: Clean separation of concerns across multiple modules
- 📊 **Document Management**: Tracks and manages multiple uploaded documents

## Architecture

### Modular Structure

```
.
├── main.py                   # FastAPI application with endpoints
├── models.py                 # Pydantic models for requests/responses
├── vectorstore_manager.py    # FAISS vectorstore management
├── file_processor.py         # Document processing for PDF/TXT/CSV
├── crag_service.py          # CRAG logic and LLM interactions
├── helper_functions.py      # Utility functions for encoding
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables (create from .env.example)
```

### CRAG Workflow

1. **Query Received**: User asks a question
2. **Document Retrieval**: Fetch top-k similar documents from vectorstore
3. **Relevance Evaluation**: Score each document's relevance (0-1 scale)
4. **Action Decision**:
   - **Score > 0.7 (Correct)**: Use retrieved documents
   - **Score < 0.3 (Incorrect)**: Perform web search
   - **0.3 ≤ Score ≤ 0.7 (Ambiguous)**: Combine both sources
5. **Knowledge Refinement**: Extract key points from sources
6. **Answer Generation**: Generate final answer with citations

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key

### Setup Steps

1. **Clone or download the project**

2. **Create a virtual environment** (recommended)
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```powershell
   # Copy the example file
   copy .env.example .env
   
   # Edit .env and add your OpenAI API key
   notepad .env
   ```

   Add your API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

## Usage

### Starting the Server

```powershell
python main.py
```

Or using uvicorn directly:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Health Check

```http
GET /
GET /health
```

**Response:**
```json
{
  "message": "RAG Chatbot with CRAG is running...",
  "status": "healthy",
  "documents_loaded": 42
}
```

#### 2. Upload Documents

```http
POST /upload_files
Content-Type: multipart/form-data
```

**Parameters:**
- `files`: List of files (PDF, TXT, or CSV)

**Example using cURL:**
```powershell
curl -X POST "http://localhost:8000/upload_files" `
  -F "files=@document1.pdf" `
  -F "files=@document2.txt" `
  -F "files=@data.csv"
```

**Response:**
```json
{
  "message": "Successfully processed 3 file(s)",
  "number_of_files_processed": 3,
  "filenames": ["document1.pdf", "document2.txt", "data.csv"],
  "errors": []
}
```

#### 3. Chat / Ask Questions

```http
POST /chat
Content-Type: application/json
```

**Request Body:**
```json
{
  "question": "What are the main causes of climate change?",
  "use_web_search": false,
  "k": 4
}
```

**Parameters:**
- `question` (required): Your question
- `use_web_search` (optional, default: false): Force web search
- `k` (optional, default: 4): Number of documents to retrieve

**Response:**
```json
{
  "answer": "Based on the documents, the main causes of climate change include...",
  "sources": [
    "Retrieved document from uploaded files",
    "Climate Science Report (https://example.com)"
  ],
  "retrieval_action": "ambiguous"
}
```

**Retrieval Actions:**
- `correct`: Used high-quality retrieved documents
- `incorrect`: Retrieved documents were not relevant, used web search
- `ambiguous`: Combined retrieved documents with web search
- `forced_web_search`: User explicitly requested web search

#### 4. List Sources

```http
GET /sources
```

**Response:**
```json
{
  "total_document_chunks": 127,
  "number_of_files": 5,
  "sources": [
    "climate_report.pdf (pdf)",
    "notes.txt (txt)",
    "data.csv (csv)"
  ]
}
```

#### 5. Reset Vectorstore (Admin)

```http
DELETE /reset
```

⚠️ **Warning**: This removes all uploaded documents from memory.

## Python Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Upload documents
files = [
    ('files', open('document.pdf', 'rb')),
    ('files', open('notes.txt', 'rb'))
]
response = requests.post(f"{BASE_URL}/upload_files", files=files)
print(response.json())

# 2. Ask a question
question_data = {
    "question": "What is the main topic of the document?",
    "use_web_search": False,
    "k": 4
}
response = requests.post(f"{BASE_URL}/chat", json=question_data)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Action: {result['retrieval_action']}")
```

## Configuration

### CRAG Thresholds

The CRAG service uses two thresholds to determine retrieval actions:

- **Upper Threshold (0.7)**: Documents scoring above this are considered highly relevant
- **Lower Threshold (0.3)**: Documents scoring below this trigger web search

You can customize these in `crag_service.py` or `main.py`:

```python
crag_service = CRAGService(
    model="gpt-4o-mini",
    temperature=0,
    lower_threshold=0.3,  # Adjust as needed
    upper_threshold=0.7   # Adjust as needed
)
```

### Supported File Types

| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| PDF    | `.pdf`    | PyPDF extraction → chunking |
| Text   | `.txt`    | Direct read → chunking |
| CSV    | `.csv`    | Pandas → formatted text → chunking |

## How It Works

### Document Processing Pipeline

1. **Upload**: Files are saved to `uploaded_files/` directory
2. **Extraction**: Content extracted based on file type
3. **Chunking**: Text split into overlapping chunks (1000 chars, 200 overlap)
4. **Embedding**: Chunks converted to vectors using OpenAI embeddings
5. **Storage**: Vectors stored in FAISS index with metadata

### Query Processing Pipeline

1. **Retrieval**: Top-k similar document chunks retrieved
2. **Evaluation**: Each chunk scored for relevance (0-1)
3. **Strategy Selection**:
   - High scores → Use documents
   - Low scores → Web search
   - Medium scores → Combine both
4. **Knowledge Refinement**: Key points extracted
5. **Answer Generation**: LLM generates answer with sources

## Troubleshooting

### Common Issues

**Issue**: `OPENAI_API_KEY not found`
- **Solution**: Create `.env` file with your API key

**Issue**: `No documents have been uploaded yet`
- **Solution**: Upload documents using `/upload_files` endpoint first

**Issue**: `Error processing PDF`
- **Solution**: Ensure PDF is not encrypted or corrupted

**Issue**: Module import errors
- **Solution**: Reinstall dependencies: `pip install -r requirements.txt`

### Logs

The application uses Python's logging module. Check console output for detailed information:
- INFO: Normal operations
- WARNING: Potential issues
- ERROR: Failures with stack traces

## Advanced Usage

### Custom Document Processing

Add custom file processors in `file_processor.py`:

```python
def process_docx(file_path: str) -> FAISS:
    """Process Word documents."""
    # Your implementation
    pass
```

### Extend CRAG Logic

Modify `crag_service.py` to add custom evaluation logic:

```python
def custom_evaluator(self, query: str, doc: str) -> float:
    """Custom relevance scoring."""
    # Your implementation
    pass
```

### MCP Integration (Future)

The system includes placeholders for Model Context Protocol (MCP) integration:

```python
# TODO: Integrate MCP server tools here as additional context providers.
# This could allow for additional data sources or specialized tools
```

## Project Dependencies

Key libraries:
- **FastAPI**: Web framework
- **LangChain**: LLM orchestration
- **FAISS**: Vector similarity search
- **OpenAI**: Embeddings and chat models
- **DuckDuckGo Search**: Web search functionality
- **PyPDF/PyMuPDF**: PDF processing
- **Pandas**: CSV processing

See `requirements.txt` for complete list.

## License

This project is provided as-is for educational and research purposes.

## Contributing

This is a course project. For improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check logs for error details

## Acknowledgments

- Built with LangChain and FastAPI
- Uses OpenAI's GPT models
- CRAG technique implementation based on academic research
- Helper functions adapted from course materials

---

**Happy Chatting! 🤖💬**
