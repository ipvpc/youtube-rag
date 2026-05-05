import logging

import config
from lightrag_remote import remote_query

logging.basicConfig(
    level=getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def run_rag_query(user_query: str) -> str:
    """Query the remote LightRAG service (POST /query)."""
    if not user_query or not str(user_query).strip():
        return ""
    text = remote_query(str(user_query).strip())
    return (text or "").strip()


def save_chat_history(user_id, query, response):
    logger.info("Saving chat history for user %s: %s -> %s", user_id, query, response)


__all__ = [
    "run_rag_query",
    "save_chat_history",
]
