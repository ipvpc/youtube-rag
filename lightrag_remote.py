"""
HTTP client for a remote LightRAG API server (lightrag-hku FastAPI).

Expected routes (default server layout):
  POST {LIGHTRAG_SERVICE_URL}/query
  POST {LIGHTRAG_SERVICE_URL}/documents/texts

Auth: optional X-API-Key and/or Bearer token, depending on how the remote server is configured.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import requests

import config

logger = logging.getLogger(__name__)


def _base_url() -> str:
    base = (config.LIGHTRAG_SERVICE_URL or "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "LIGHTRAG_SERVICE_URL is not set. Point it at your LightRAG API base "
            "(e.g. http://lightrag:9621)."
        )
    return base


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
    if getattr(config, "LIGHTRAG_API_KEY", None):
        h["X-API-Key"] = config.LIGHTRAG_API_KEY
    bearer = (getattr(config, "LIGHTRAG_BEARER_TOKEN", None) or "").strip()
    if bearer:
        h["Authorization"] = f"Bearer {bearer}"
    return h


def _timeout() -> float:
    return float(getattr(config, "LIGHTRAG_REQUEST_TIMEOUT", 300))


def _ensure_query_min_length(query: str) -> str:
    """LightRAG QueryRequest requires query length >= 3."""
    q = (query or "").strip()
    if len(q) >= 3:
        return q
    return (q + "   ")[:3] if q else "   "


def remote_query(user_query: str) -> str:
    """POST /query — non-streaming JSON response."""
    url = f"{_base_url()}/query"
    body: dict[str, Any] = {
        "query": _ensure_query_min_length(user_query),
        "mode": config.LIGHTRAG_QUERY_MODE,
        "stream": False,
        "include_references": False,
    }
    up = (getattr(config, "LIGHTRAG_USER_PROMPT", None) or "").strip()
    if up:
        body["user_prompt"] = up

    logger.debug("POST %s mode=%s", url, body.get("mode"))
    resp = requests.post(url, json=body, headers=_headers(), timeout=_timeout())
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        logger.error(
            "LightRAG /query failed: %s %s",
            resp.status_code,
            (resp.text or "")[:2000],
        )
        raise

    data = resp.json()
    out = (data.get("response") or "").strip()
    return out


def remote_insert_texts(texts: list[str], file_sources: Optional[list[str]]) -> dict[str, Any]:
    """POST /documents/texts — queues documents on the remote service."""
    url = f"{_base_url()}/documents/texts"
    payload: dict[str, Any] = {"texts": texts}
    if file_sources is not None:
        payload["file_sources"] = file_sources

    logger.info("POST %s (batch size=%s)", url, len(texts))
    resp = requests.post(url, json=payload, headers=_headers(), timeout=_timeout())
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        logger.error(
            "LightRAG /documents/texts failed: %s %s",
            resp.status_code,
            (resp.text or "")[:2000],
        )
        raise
    return resp.json()


def ingest_transcripts_batch(
    texts: list[str],
    file_sources: list[str],
) -> dict[str, Any]:
    """Send one batch insert to the remote LightRAG service."""
    if not texts:
        return {"status": "skipped", "message": "no texts", "track_id": ""}
    if file_sources and len(file_sources) != len(texts):
        raise ValueError("file_sources must be the same length as texts when provided")
    return remote_insert_texts(texts, file_sources if file_sources else None)


def format_ingest_result(result: dict[str, Any]) -> str:
    """Human-readable line for logging / subprocess stdout."""
    return json.dumps(result, indent=2)
