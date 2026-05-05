"""
Ingest YouTube transcripts into a remote LightRAG API (`POST /documents/texts`).

Run via `python inges.py` or triggered from Flask `/run-ingest`.
"""
import json
import logging
import os

import config
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from lightrag_remote import format_ingest_result, ingest_transcripts_batch

os.environ["USER_AGENT"] = "alpha-agent"

logging.basicConfig(
    level=getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)

documents = []
text_dir = config.TEXT_FOLDER
glob_pattern = "**/*.txt"

if os.path.exists(text_dir):
    if any(
        filename.endswith(".txt")
        for _, _, filenames in os.walk(text_dir)
        for filename in filenames
    ):
        loader = DirectoryLoader(
            text_dir,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
        )
        documents.extend(loader.load())
else:
    logging.warning("Skipping non-existent directory: %s", text_dir)


def main() -> None:
    if not documents:
        log.info("No text content found. Ensure .txt files exist under %s", text_dir)
        print(format_ingest_result({"status": "skipped", "message": "no documents", "track_id": ""}))
        return

    texts: list[str] = []
    file_sources: list[str] = []

    for doc in documents:
        if not doc.page_content or not doc.page_content.strip():
            continue
        src = (doc.metadata or {}).get("source") or ""
        base = os.path.basename(src)
        video_id, _ext = os.path.splitext(base)
        if not video_id:
            video_id = "unknown"
        texts.append(doc.page_content)
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
