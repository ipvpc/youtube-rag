import os

# You can rename these to whatever naming convention you prefer,
# but for clarity we'll keep them close to your existing variables:
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "youtube_finance_docs")
# Default to a model that the Alpha5 embeddings service exposes.
# (Your error shows available models include 'BAAI/bge-small-en-v1.5'.)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# Embedding model dimension (for other services / legacy tooling if needed).
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
# Set via environment; use empty string if your embedding server does not require auth.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE = os.getenv("OPENAI_BASE", "http://embeddings.alpha5.finance:8001/v1")
MODEL = os.getenv("MODEL", "gemma2:27b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama.alpha5.finance:11434")
DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER", "/app/docs/mp3")
TEXT_FOLDER = os.getenv("TEXT_FOLDER", "/app/docs/txt")
API_URL = os.getenv("API_URL", "http://whisper-api.alpha5.finance:5005/transcribe")

# PostgreSQL
POSTGRES_CONNECTION_URI = os.getenv(
    "POSTGRES_CONNECTION_URI",
    "postgresql://postgres@127.0.0.1:5432/youtube_rag",
)

# Remote LightRAG HTTP API (lightrag_server / FastAPI)
LIGHTRAG_SERVICE_URL = os.getenv("LIGHTRAG_SERVICE_URL", "http://127.0.0.1:9621").strip().rstrip("/")
LIGHTRAG_API_KEY = os.getenv("LIGHTRAG_API_KEY", "").strip()
LIGHTRAG_BEARER_TOKEN = os.getenv("LIGHTRAG_BEARER_TOKEN", "").strip()
# Deprecated: set LIGHTRAG_QUERY_TIMEOUT / LIGHTRAG_INGEST_TIMEOUT instead.
LIGHTRAG_REQUEST_TIMEOUT = float(os.getenv("LIGHTRAG_REQUEST_TIMEOUT", "300"))
LIGHTRAG_CONNECT_TIMEOUT = float(os.getenv("LIGHTRAG_CONNECT_TIMEOUT", "30"))
# Hybrid/global LightRAG queries often exceed 300s on large knowledge bases.
LIGHTRAG_QUERY_TIMEOUT = float(os.getenv("LIGHTRAG_QUERY_TIMEOUT", "900"))
LIGHTRAG_INGEST_TIMEOUT = float(os.getenv("LIGHTRAG_INGEST_TIMEOUT", "600"))
# Query modes: naive, local, global, hybrid, mix, bypass (see LightRAG QueryRequest)
LIGHTRAG_QUERY_MODE = os.getenv("LIGHTRAG_QUERY_MODE", "hybrid").strip().lower()
# Optional: forwarded as `user_prompt` on POST /query (remote prompt tuning)
LIGHTRAG_USER_PROMPT = os.getenv("LIGHTRAG_USER_PROMPT", "").strip()
