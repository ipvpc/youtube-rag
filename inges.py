"""
Ingest YouTube transcripts into a remote LightRAG API (`POST /documents/texts`).

Run via `python inges.py` or triggered from Flask `/run-ingest`.
"""
import json
import logging
import os

import chardet
import config
from lightrag_remote import format_ingest_result, ingest_transcripts_batch

os.environ["USER_AGENT"] = "alpha-agent"

logging.basicConfig(
    level=getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)

text_dir = config.TEXT_FOLDER


def _read_text_file(path: str) -> str | None:
    with open(path, "rb") as handle:
        raw = handle.read()
    if not raw.strip():
        return None
    detected = chardet.detect(raw)
    encoding = (detected or {}).get("encoding") or "utf-8"
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


def load_transcript_files(root_dir: str) -> list[tuple[str, str]]:
    """Return (absolute_path, content) for each non-empty .txt under root_dir."""
    documents: list[tuple[str, str]] = []
    if not os.path.exists(root_dir):
        logging.warning("Skipping non-existent directory: %s", root_dir)
        return documents

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(dirpath, filename)
            content = _read_text_file(path)
            if content and content.strip():
                documents.append((path, content))
    return documents


def main() -> None:
    documents = load_transcript_files(text_dir)
    if not documents:
        log.info("No text content found. Ensure .txt files exist under %s", text_dir)
        print(format_ingest_result({"status": "skipped", "message": "no documents", "track_id": ""}))
        return

    texts: list[str] = []
    file_sources: list[str] = []

    for src, page_content in documents:
        base = os.path.basename(src)
        video_id, _ext = os.path.splitext(base)
        if not video_id:
            video_id = "unknown"
        texts.append(page_content)
        # Stable source id for dedupe on the LightRAG server
        file_sources.append(f"youtube-{video_id}.txt")

    if not texts:
        log.info("No non-empty documents to ingest.")
        print(format_ingest_result({"status": "skipped", "message": "no non-empty docs", "track_id": ""}))
        return

    try:
        result = ingest_transcripts_batch(texts, file_sources)
        log.info("Remote LightRAG ingest: %s", result)
        print(format_ingest_result(result))
    except Exception as e:
        log.exception("Remote LightRAG ingest failed: %s", e)
        print(json.dumps({"error": str(e)}))
        raise


if __name__ == "__main__":
    main()
