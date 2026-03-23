# YouTube RAG (Retrieval-Augmented Generation) System

A Flask-based web application that downloads YouTube videos, transcribes them to text, stores the transcriptions in a vector database, and provides a RAG (Retrieval-Augmented Generation) interface for querying the content using semantic search and LLM responses.

## Overview

This system enables users to:
1. Download audio from YouTube videos
2. Transcribe audio to text using a Whisper API
3. Store transcriptions in a PostgreSQL database with vector embeddings
4. Query the stored content using semantic search
5. Generate AI-powered responses using Ollama LLM

## Architecture

The system consists of three main components:

1. **Web Application (`app.py`)**: Main Flask server that handles YouTube downloads, transcription, and provides all API endpoints
2. **Ingestion Pipeline (`inges.py`)**: Standalone script that processes text files and creates vector embeddings in PostgreSQL
3. **Query Module (`query.py`)**: Python module that provides RAG functions (semantic search and response generation) imported by `app.py`

**Note**: Only `app.py` needs to be running. `query.py` is imported as a module and doesn't run as a separate service.

## Project Structure

```
youtube-rag/
├── app.py                 # Main Flask application
├── inges.py              # Document ingestion and embedding pipeline
├── query.py              # RAG query and response generation
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker container configuration
├── build.sh             # Docker build and push script
├── test.sh              # Docker test/run script
├── templates/
│   └── index.html       # Web UI for YouTube URL submission
└── docs/
    ├── mp3/             # Directory for downloaded audio files
    └── txt/             # Directory for transcribed text files
```

## Configuration

All configuration is managed through environment variables (with defaults in `config.py`):

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COLLECTION_NAME` | `youtube_finance_docs` | PostgreSQL table name for embeddings |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `EMBEDDING_DIM` | `384` | Embedding vector dimension (must match model output) |
| `OPENAI_API_KEY` | `alpha5cloud` | API key for embedding service |
| `OPENAI_BASE` | `http://embeddings.alpha5.finance:8001/v1` | Base URL for embedding API |
| `MODEL` | `gemma2:27b` | Ollama LLM model name |
| `OLLAMA_HOST` | `http://ollama.alpha5.finance:11434` | Ollama server URL |
| `DOWNLOAD_FOLDER` | `/app/docs/mp3` | Directory for downloaded MP3 files |
| `TEXT_FOLDER` | `/app/docs/txt` | Directory for transcribed text files |
| `API_URL` | `http://whisper-api.alpha5.finance:5005/transcribe` | Whisper transcription API URL |
| `PORT` | `5004` | Flask application port |
| `DEBUG` | `false` | Enable Flask debug mode |

### Database Configuration

PostgreSQL connection string (hardcoded in `config.py`):
```
postgresql://markets:p0w3rb4r@postgres.alpha5.finance:5432/knowledgebase
```

## Components

### 1. app.py - Web Application

Main Flask application that provides:

#### Endpoints

- **`GET /`**: Web interface for submitting YouTube URLs
- **`POST /`**: Processes YouTube URL, downloads audio, and transcribes
- **`GET /success`**: Success page after transcription
- **`GET /health`**: Health check endpoint
- **`POST /run-ingest`**: Triggers the ingestion pipeline (`inges.py`)
- **`POST /query`**: Query the RAG system

#### Features

- YouTube video download using `yt-dlp`
- Audio extraction to MP3 format (192kbps)
- Rate limiting with semaphore (max 3 concurrent downloads)
- Random user-agent rotation to avoid detection
- Automatic transcription via Whisper API
- Text file storage in `docs/txt/` directory

#### Key Functions

- `download_youtube_audio()`: Downloads and converts YouTube video to MP3
- `transcribe_audio()`: Sends audio file to Whisper API for transcription
- `is_valid_youtube_url()`: Validates YouTube URLs

### 2. inges.py - Ingestion Pipeline

Processes text files and creates vector embeddings:

#### Workflow

1. **Load Documents**: Reads all `.txt` files from `/app/docs/txt/`
2. **Split Text**: Chunks text using `RecursiveCharacterTextSplitter`:
   - Chunk size: 2000 characters
   - Chunk overlap: 20 characters
3. **Generate Embeddings**: Creates embeddings using OpenAI-compatible API
4. **Store in Database**: Saves embeddings to PostgreSQL with pgvector extension

#### Database Schema

```python
class TextEmbedding:
    id: Integer (primary key)
    content: String (text chunk)
    embedding: Vector(384) (pgvector)  # 384 dimensions for all-MiniLM-L6-v2
    doc_metadata: JSON (document metadata)
```

**Note**: The embedding dimension (384) matches the `all-MiniLM-L6-v2` model. If you change the embedding model, you must update `N_DIM` in both `inges.py` and `query.py`.

#### Key Functions

- `split_text_into_chunks()`: Splits text into overlapping chunks
- `generate_embeddings()`: Creates embeddings for text chunks
- `insert_embeddings()`: Stores embeddings in PostgreSQL (skips duplicates)

### 3. query.py - RAG Query Module

Python module (not a standalone app) that provides semantic search and AI-powered response functions. This module is imported by `app.py` and provides the following functions:

#### Workflow

1. **Query Embedding**: Converts user query to embedding vector
2. **Semantic Search**: Finds similar chunks using cosine distance in pgvector
3. **Context Extraction**: Retrieves top-k similar chunks (default: 5)
4. **RAG Response**: Generates response using Ollama LLM with retrieved context

#### Key Functions (imported by `app.py`)

- `Extract_context()`: Performs semantic search and retrieves relevant context
- `find_similar_embeddings()`: PostgreSQL query using pgvector cosine distance
- `generate_rag_response()`: Generates AI response using Ollama
- `save_chat_history()`: Placeholder for chat history storage

**Note**: This file does NOT run as a standalone Flask application. It's a module that provides functions to `app.py`.

#### RAG Prompt Engineering

The system uses a specialized prompt for hedge fund manager persona:
- Focuses on maximizing profits
- Analyzes financial data and market trends
- Provides thorough explanations without summarizing
- Relies solely on provided context

## Dependencies

Key Python packages (see `requirements.txt`):

- **Flask**: Web framework
- **yt-dlp**: YouTube video downloader
- **moviepy**: Video/audio processing
- **langchain**: Document processing and text splitting
- **openai**: Embedding generation
- **sqlalchemy**: Database ORM
- **pgvector**: PostgreSQL vector extension support
- **ollama**: LLM client
- **numpy**: Numerical operations

## Setup and Installation

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables** (or modify `config.py`):
```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE=http://your-embedding-service/v1
export OLLAMA_HOST=http://your-ollama-host:11434
```

3. **Create directories**:
```bash
mkdir -p docs/mp3 docs/txt
```

4. **Run the application**:
```bash
python app.py
```

The application will start on `http://localhost:5004`

### Docker Deployment

#### Build Image

```bash
./build.sh
```

Or manually:
```bash
docker build -t registry.alpha5.finance/rags/youtube-qa-agent:prod .
docker push registry.alpha5.finance/rags/youtube-qa-agent:prod
```

#### Run Container

```bash
./test.sh
```

Or manually:
```bash
docker run --rm -it --name youtube-qa-agent \
  -p 5014:5004 \
  -e OPENAI_API_KEY=alpha5cloud \
  -e OPENAI_BASE=http://embeddings.alpha5.finance:8001/v1 \
  -e API_URL=http://whisper-api.alpha5.finance:5005/transcribe \
  -e EMBEDDING_MODEL=all-MiniLM-L6-v2 \
  -e COLLECTION_NAME=youtube_docs \
  -e MODEL=gemma2:27b \
  -e OLLAMA_HOST=http://ollama.alpha5.finance:11434 \
  -v /data/ai-docs/youtube-rag/docs:/app/docs \
  registry.alpha5.finance/rags/youtube-qa-agent:prod
```

## Usage

### 1. Download and Transcribe YouTube Video

**Web Interface**:
- Navigate to `http://localhost:5004`
- Enter YouTube URL
- Click "Download and Transcribe"
- Audio is downloaded to `docs/mp3/`
- Transcription is saved to `docs/txt/`

**API**:
```bash
curl -X POST http://localhost:5004/ \
  -d "youtube_url=https://www.youtube.com/watch?v=VIDEO_ID"
```

### 2. Ingest Documents into Vector Database

**API**:
```bash
curl -X POST http://localhost:5004/run-ingest
```

This processes all `.txt` files in `docs/txt/` and creates embeddings.

### 3. Query the RAG System

**API**:
```bash
curl -X POST http://localhost:5004/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main investment strategies discussed?",
    "user_id": "user123"
  }'
```

**Response**:
```json
{
  "response": "Based on the context, the main investment strategies include..."
}
```

## API Reference

### POST /query

Query the RAG system with a natural language question.

**Request Body**:
```json
{
  "query": "Your question here",
  "user_id": "optional_user_id"
}
```

**Response**:
```json
{
  "response": "AI-generated response based on retrieved context"
}
```

### POST /run-ingest

Trigger the ingestion pipeline to process all text files.

**Response**:
```json
{
  "stdout": "Script output",
  "stderr": "Error output",
  "returncode": 0
}
```

### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "service": "youtube-rag"
}
```

## Database Setup

The system requires PostgreSQL with the pgvector extension. The application automatically creates the extension if it doesn't exist, but you can also create it manually:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The `inges.py` and `query.py` scripts automatically:
1. Create the pgvector extension if missing
2. Create the table if it doesn't exist

**Important**: Ensure your PostgreSQL server has the pgvector extension installed. On Ubuntu/Debian:
```bash
sudo apt-get install postgresql-14-pgvector  # Adjust version as needed
```

## Limitations and Notes

1. **Rate Limiting**: Maximum 3 concurrent downloads (configurable via semaphore)
2. **Duplicate Prevention**: Ingestion skips chunks that already exist in database
3. **Embedding Dimensions**: Fixed at 384 dimensions for `all-MiniLM-L6-v2` model (must match embedding model output)
4. **Similarity Threshold**: Cosine distance threshold of 0.7 for semantic search
5. **Chunk Size**: Text is split into 2000-character chunks with 20-character overlap
6. **pgvector Extension**: The application automatically creates the `vector` extension in PostgreSQL if it doesn't exist

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg system package
2. **Database connection errors**: Verify PostgreSQL connection string and pgvector extension
3. **Embedding API errors**: Check `OPENAI_BASE` and `OPENAI_API_KEY` environment variables
4. **Ollama connection errors**: Verify `OLLAMA_HOST` and ensure model is available

### Logging

The application uses Python's logging module. Set log level via:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] Chat history persistence
- [ ] User authentication
- [ ] Batch processing for multiple videos
- [ ] Support for other video platforms
- [ ] WebSocket support for real-time transcription
- [ ] Advanced filtering and search options
- [ ] Export functionality for transcriptions

## License

[Add license information here]

## Contributing

[Add contributing guidelines here]
