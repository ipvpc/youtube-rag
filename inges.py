# Reading a PDF Document
import os
import logging
import config
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ['USER_AGENT'] = 'alpha-agent'

# Setup logging
logging.basicConfig(
    level=getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

documents = []
text_loader_kwargs = {"autodetect_encoding": True}

# Define the directory paths and corresponding loaders
loaders = [
    ("/app/docs/txt/", "**/*.txt", TextLoader, {"autodetect_encoding": True})
]

# Iterate over each loader configuration
for path, glob_pattern, loader_cls, *kwargs in loaders:
    if not os.path.exists(path):
        logging.warning(f"Skipping non-existent directory: {path}")
        continue

    if any(filename.endswith(glob_pattern.split('.')[-1])
           for dirpath, _, filenames in os.walk(path)
           for filename in filenames):
        loader_kwargs = kwargs[0] if kwargs else {}
        loader = DirectoryLoader(path, glob=glob_pattern, loader_cls=loader_cls, loader_kwargs=loader_kwargs)
        documents.extend(loader.load())

# Generating Text Embeddings
import openai
from openai import OpenAI

client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE)
from langchain_openai import OpenAIEmbeddings

# IMPORTANT: Use values from config.py
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=config.OPENAI_BASE)'
# openai.api_base = config.OPENAI_BASE  # e.g. "http://embeddings.alpha5.finance:8001/v1"

def generate_embeddings(text_chunks):
    embeddings = []
    logging.getLogger(__name__).info("Generating embeddings for %s chunks (model=%s)", len(text_chunks), config.EMBEDDING_MODEL)
    for chunk in text_chunks:
        # If you want, you could also use config.EMBEDDING_MODEL here
        # Instead of "text-embedding-ada-002", e.g.: model=config.EMBEDDING_MODEL
        response = client.embeddings.create(input=chunk,
        model=config.EMBEDDING_MODEL)
        embeddings.append(response.data[0].embedding)
    return embeddings

# Storing Embeddings in PostgreSQL
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import numpy as np
from sqlalchemy.dialects.postgresql import JSON

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
from sqlalchemy import text
with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

Base.metadata.create_all(engine)

# Ensure the embedding column exists even if the table pre-dates the fix above.
with engine.begin() as conn:
    conn.execute(
        text(f'ALTER TABLE "{config.COLLECTION_NAME}" ADD COLUMN IF NOT EXISTS embedding vector({N_DIM})')
    )

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

def insert_embeddings(docs_with_chunks, embeddings):
    """
    Inserts the text (content), embeddings, and metadata into the DB if not already present.

    Args:
        docs_with_chunks (list of tuple(Document, str)):
            A list of (Document, str) tuples where each Document is a LangChain Document
            and str is the chunk from that document's content.
        embeddings (list of list of float]):
            Numeric embeddings returned by OpenAI, ordered correspondingly.
    """
    for (doc, chunk), embedding in zip(docs_with_chunks, embeddings):
        if len(embedding) != N_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {N_DIM}, got {len(embedding)}. "
                f"Check EMBEDDING_MODEL/EMBEDDING_DIM."
            )
        # Check if the content already exists in the database
        existing_entry = session.query(TextEmbedding).filter_by(content=chunk).first()

        if not existing_entry:
            new_embedding = TextEmbedding(
                content=chunk,
                embedding=embedding,
                doc_metadata=doc.metadata
            )
            session.add(new_embedding)

    session.commit()

def split_text_into_chunks(text, chunk_size=2000, chunk_overlap=20):
    """
    Splits a single string of text into smaller text chunks.

    Args:
        text (str): The complete textual content to be split.
        chunk_size (int): The maximum size (in characters) of each chunk.
        chunk_overlap (int): The overlap (in characters) between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    # You can tune chunk_size and chunk_overlap to best fit your use case.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_text(text)
    return chunks

def main():
    # documents = ...  (loaded from DirectoryLoader)

    # Prepare a list of (doc, chunk) to keep track of which chunk came from which doc
    docs_with_chunks = []

    for doc in documents:
        if doc.page_content and doc.page_content.strip():
            # Enrich metadata with video_id derived from transcript filename:
            # /app/docs/txt/<video_id>.txt  ->  video_id
            try:
                src = (doc.metadata or {}).get("source") or ""
                base = os.path.basename(src)
                video_id = os.path.splitext(base)[0]
                if video_id:
                    doc.metadata["video_id"] = video_id
            except Exception:
                pass

            sub_chunks = split_text_into_chunks(doc.page_content)
            for chunk in sub_chunks:
                docs_with_chunks.append((doc, chunk))

    if not docs_with_chunks:
        logging.info("No text content found. Ensure your .txt files exist in the directory.")
        return

    # Extract just the chunk text for embedding
    chunks_for_embedding = [chunk for _, chunk in docs_with_chunks]

    # Generate embeddings
    embeddings = generate_embeddings(chunks_for_embedding)

    # Insert embeddings and pass doc metadata
    logging.getLogger(__name__).info("Inserting embeddings into table=%s dim=%s", config.COLLECTION_NAME, N_DIM)
    insert_embeddings(docs_with_chunks, embeddings)

if __name__ == "__main__":
    main()
