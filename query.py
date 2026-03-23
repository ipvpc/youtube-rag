import os
import logging
import re
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import numpy as np
from sqlalchemy.dialects.postgresql import JSON
import config
from ollama import Client
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama.alpha5.finance:11434")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledgebase")
MODEL=os.getenv("MODEL", "deepseek-r1:1.5b")

Base = declarative_base()
# pgvector columns are created with a fixed dimension; keep in sync with config.
N_DIM = config.EMBEDDING_DIM

class TextEmbedding(Base):
    __tablename__ = config.COLLECTION_NAME
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(String)
    embedding = Column(Vector(N_DIM))
    doc_metadata = Column(JSON)

# Connect to PostgreSQL
engine = create_engine(config.POSTGRES_CONNECTION_URI)

# Ensure pgvector extension exists
with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

Base.metadata.create_all(engine)

def _ensure_embedding_schema():
    """
    Ensures the target table has an `embedding vector(N_DIM)` column.
    If the column exists but has a different dimension, raise a clear error.
    """
    with engine.begin() as conn:
        # Create column if missing
        conn.execute(
            text(f'ALTER TABLE "{config.COLLECTION_NAME}" ADD COLUMN IF NOT EXISTS embedding vector({N_DIM})')
        )

        # Validate dimension if present
        row = conn.execute(
            text(
                """
                SELECT
                  format_type(a.atttypid, a.atttypmod) AS type,
                  NULLIF(substring(format_type(a.atttypid, a.atttypmod) from '\\((\\d+)\\)'), '')::int AS dim
                FROM pg_attribute a
                WHERE a.attrelid = to_regclass(:tbl)
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                LIMIT 1
                """
            ),
            {"tbl": config.COLLECTION_NAME},
        ).mappings().first()

        if not row:
            raise RuntimeError(
                f'Could not find embedding column on table "{config.COLLECTION_NAME}". '
                "Check that the table is in the current search_path/schema."
            )

        dim = row.get("dim")
        typ = row.get("type")
        logger.debug("Embedding column check table=%s type=%s dim=%s expected=%s", config.COLLECTION_NAME, typ, dim, N_DIM)
        if dim is not None and int(dim) != int(N_DIM):
            raise RuntimeError(
                f'Postgres column "{config.COLLECTION_NAME}.embedding" is {typ} (dim={dim}), '
                f"but the app is configured for EMBEDDING_DIM={N_DIM}. "
                "Fix by pointing COLLECTION_NAME to a fresh table, or migrate the column type."
            )

_ensure_embedding_schema()

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Initialize OpenAI client for generating query embeddings
openai_client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE)

def find_similar_embeddings(query_embedding, limit=5):
    """
    Find similar embeddings using pgvector cosine distance.

    Note: Some versions of the `pgvector` Python package do not expose
    `.cosine_distance()` comparator helpers on the SQLAlchemy column.
    Using the native pgvector cosine-distance operator (<=>) is compatible.
    """
    k = int(limit)
    similarity_threshold = 0.7

    # Some driver/type combos still fail to bind Python lists into pgvector
    # (e.g., "expected ndim to be 1"). A reliable approach is to pass the
    # query vector as a pgvector text literal and cast it in SQL.
    if len(query_embedding) != N_DIM:
        raise ValueError(f"query_embedding must be length {N_DIM}, got {len(query_embedding)}")

    def _safe_table_name(name: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
            raise ValueError(f"Unsafe COLLECTION_NAME/table name: {name!r}")
        return name

    table = _safe_table_name(config.COLLECTION_NAME)
    q_vec_literal = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    logger.debug("Vector search table=%s k=%s threshold=%s", table, k, similarity_threshold)

    sql = text(
        f"""
        SELECT
          content,
          (embedding <=> (:q)::vector({N_DIM})) AS distance
        FROM "{table}"
        WHERE (embedding <=> (:q)::vector({N_DIM})) < :threshold
        ORDER BY distance
        LIMIT :k
        """
    )

    rows = session.execute(
        sql,
        {"q": q_vec_literal, "threshold": similarity_threshold, "k": k},
    ).mappings().all()
    if rows:
        try:
            logger.debug("Vector search returned %s rows; best_distance=%s", len(rows), float(rows[0].get("distance")))
        except Exception:
            logger.debug("Vector search returned %s rows", len(rows))
    return rows

def get_system_message_rag(content):
    return f"""You are a hedge fund manager focused on maximizing profits. Your role involves analyzing financial data, market trends, and investment opportunities to make informed decisions that drive substantial returns.

    Generate your response by following the steps below:
    1. Recursively break down the investment query into smaller, actionable questions.
    2. For each question/directive:
        2a. Select the most relevant information from the context in light of current market conditions and investment strategies.
    3. Generate a draft response using the selected information.
    4. Remove duplicate or redundant content from the draft response.
    5. Refine your final response to enhance accuracy, relevance, and profitability insights.
    6. Do not attempt to summarize the answers; provide thorough explanations to support investment decisions.
    7. Only present your final, polished response!

    Constraints:
    1. DO NOT PROVIDE ANY EXPLANATION OR DETAILS OR MENTION THAT YOU WERE GIVEN CONTEXT.
    2. Don't mention that you are not able to find the answer in the provided context.
    3. Don't fabricate answers; rely solely on the provided information.
    4. Strive to offer responses that align with maximizing financial gains based on the given context.

    CONTENT:
    {content}
    """

def get_ques_response_prompt(question):
    return f"""
    ==============================================================\n    Based on the above context, please provide the answer to the following question:\n    {question}
    """

def generate_rag_response(content, query):
    try:
        client = Client(host=config.OLLAMA_HOST)
        stream = client.chat(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": get_system_message_rag(content)},
                {"role": "user", "content": get_ques_response_prompt(query)}
            ],
            stream=True
        )

        logger.info("Generating RAG response...")
        full_answer = ''.join([chunk['message']['content'] for chunk in stream])

        return full_answer
    except Exception as e:
        logger.error(f"Error in generate_rag_response: {e}")
        return "Error generating response."

def Extract_context(query, limit=5):
    """
    Extract relevant context from pgvector database using semantic similarity.
    
    Args:
        query: The query string to search for
        limit: Maximum number of similar chunks to retrieve
        
    Returns:
        A string containing the concatenated relevant context chunks
    """
    logger.info(f"Extracting context for query: {query}")

    # Generate embedding for the query using the same model as ingestion
    logger.info(f"Generating embedding using model: {config.EMBEDDING_MODEL}")
    logger.debug("Embedding endpoint base_url=%s", config.OPENAI_BASE)
    response = openai_client.embeddings.create(
        input=query,
        model=config.EMBEDDING_MODEL
    )
    query_embedding = response.data[0].embedding  # This is a list of floats
    if len(query_embedding) != N_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {N_DIM}, got {len(query_embedding)}. "
            f"Check EMBEDDING_MODEL/EMBEDDING_DIM."
        )

    # Find similar embeddings using pgvector
    query_embedding_array = [float(x) for x in query_embedding]
    logger.info("Searching for similar embeddings in pgvector...")
    similar_results = find_similar_embeddings(query_embedding_array, limit=limit)

    if not similar_results:
        logger.warning("No similar embeddings found in database")
        return ""

    # Extract and insight: concat top-k chunks
    context_chunks = []
    for row in similar_results:
        content = row.get("content") or ""
        distance = row.get("distance")
        context_chunks.append(content)
        if distance is not None:
            try:
                logger.debug(f"Found similar chunk (distance: {float(distance):.4f}): {content[:100]}...")
            except Exception:
                logger.debug(f"Found similar chunk (distance: {distance}): {content[:100]}...")

    context = "\n\n".join(context_chunks)
    logger.info(f"Extracted {len(context_chunks)} context chunks, total length: {len(context)} characters")
    return context

def save_chat_history(user_id, query, response):
    logger.info(f"Saving chat history for user {user_id}: {query} -> {response}")
    pass  # Mock implementation for testing
