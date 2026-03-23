import os

# You can rename these to whatever naming convention you prefer,
# but for clarity we'll keep them close to your existing variables:
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "youtube_finance_docs")
# Default to a model that the Alpha5 embeddings service exposes.
# (Your error shows available models include 'BAAI/bge-small-en-v1.5'.)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# IMPORTANT: pgvector columns are created with a fixed dimension.
# Keep this aligned with the embedding model output dimension.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "alpha5cloud")
OPENAI_BASE = os.getenv("OPENAI_BASE", "http://embeddings.alpha5.finance:8001/v1")
MODEL = os.getenv("MODEL", "gemma2:27b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama.alpha5.finance:11434")
DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER", "/app/docs/mp3")
TEXT_FOLDER = os.getenv("TEXT_FOLDER", "/app/docs/txt")
API_URL = os.getenv("API_URL", "http://whisper-api.alpha5.finance:5005/transcribe")

# PostgreSQL
POSTGRES_CONNECTION_URI = os.getenv(
    "POSTGRES_CONNECTION_URI",
    "postgresql://markets:p0w3rb4r@postgres.alpha5.finance:5432/markets_prod"
)
