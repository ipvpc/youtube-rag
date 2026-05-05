# YouTube RAG (Retrieval-Augmented Generation)

Flask app that downloads YouTube audio, transcribes it via a Whisper-compatible HTTP API, chunks and embeds transcripts into PostgreSQL (pgvector), and answers questions with an Ollama-backed RAG flow. The default UI is a single-page experience: add a video, optionally ingest, then chat against embedded transcripts.

## What it does

1. Download audio with **yt-dlp** and extract **MP3** (192 kbps).
2. Send the MP3 to your **transcription API**; store **`.txt`** next to your configured text folder.
3. **Ingest** (`inges.py`): load all `.txt` files under the text directory (`**/*.txt`), chunk with LangChain, embed via an OpenAI-compatible API, upsert into pgvector (skips rows whose chunk `content` already exists).
4. **Query** (`query.py`): embed the user question, retrieve similar chunks (cosine distance below **0.7**), stream a completion from **Ollama**.

Only **`app.py`** needs to run as the server; `query.py` is imported as a library. Ingestion is normally triggered over HTTP (`/run-ingest` or bundled with `/download-transcribe`) which runs `inges.py` in a subprocess.

## Project layout

```
youtube-rag/
├── app.py              # Flask app, download/transcribe, ingest trigger, query API
├── inges.py            # Embedding pipeline (run standalone or via app)
├── query.py            # Semantic search + Ollama RAG (imported by app)
├── config.py           # Central defaults + env-based settings
├── requirements.txt
├── Dockerfile
├── build.sh            # Build/push image (see script for registry/tag)
├── test.sh             # Example docker run (adjust env and volumes)
├── templates/
│   └── index.html      # UI: transcribe, ingest, chat + SSE progress
└── LICENSE             # Apache-2.0
```

At runtime, ensure **`DOWNLOAD_FOLDER`** and **`TEXT_FOLDER`** exist (the app creates them on startup). Docker copies a `docs/` tree into the image; bind-mount a host `docs` for persistence if you use containers.

## Configuration

Environment variables override `config.py` defaults.

| Variable | Default | Purpose |
|----------|---------|---------|
| `COLLECTION_NAME` | `youtube_finance_docs` | Postgres table name for embedding rows |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model id for the OpenAI-compatible API |
| `EMBEDDING_DIM` | `384` | Vector size; must match the model output and the pgvector column |
| `OPENAI_API_KEY` | _(empty)_ | API key for the embedding server, if required |
| `OPENAI_BASE` | `http://embeddings.alpha5.finance:8001/v1` | Base URL for embeddings (typically includes `/v1`) |
| `MODEL` | `gemma2:27b` | Ollama model name for chat |
| `OLLAMA_HOST` | `http://ollama.alpha5.finance:11434` | Ollama HTTP API host |
| `DOWNLOAD_FOLDER` | `/app/docs/mp3` | MP3 output directory |
| `TEXT_FOLDER` | `/app/docs/txt` | Transcript `.txt` directory |
| `API_URL` | `http://whisper-api.alpha5.finance:5005/transcribe` | Whisper HTTP endpoint (`POST` multipart field `file`) |
| `POSTGRES_CONNECTION_URI` | `postgresql://postgres@127.0.0.1:5432/youtube_rag` | Postgres connection string |
| `PORT` | `5004` | Flask listen port |
| `DEBUG` | `false` | Flask debug |
| `LOG_LEVEL` | `DEBUG` if `DEBUG=true`, else `INFO` | Logging level |
| `CORS_ORIGINS` | `*` | Comma-separated origins for `flask-cors` |

**Postgres:** use a URI with credentials for non-local hosts, for example `postgresql://USER:PASSWORD@host:5432/youtube_rag`. The app and scripts expect the **pgvector** extension; `inges.py` / `query.py` run `CREATE EXTENSION IF NOT EXISTS vector` and ensure an `embedding vector(EMBEDDING_DIM)` column.

**Embedding dimension:** `EMBEDDING_DIM` must match the embedding model. If you change the model, update `EMBEDDING_DIM` and use a **new** `COLLECTION_NAME` or migrate the column type—`query.py` validates stored column dimension vs config.

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (`templates/index.html`): transcribe, ingest, chat |
| `POST` | `/` | Legacy form POST: same download+transcribe flow as JSON route, then redirect to `/success` |
| `GET` | `/success` | Plain-text success after legacy form flow |
| `GET` | `/health` | JSON `{"status":"healthy","service":"youtube-rag"}` |
| `POST` | `/download-transcribe` | JSON body: `youtube_url`, optional `ingest` (bool), optional `task_id`. Returns paths, metadata, optional ingest subprocess result, and `task_id`. |
| `GET` | `/progress/<task_id>` | **Server-Sent Events** stream of `{stage, percent, message, ...}` for long operations |
| `POST` | `/run-ingest` | Optional JSON `{"task_id": "..."}`; runs `python inges.py`, returns stdout/stderr/returncode |
| `POST` | `/query` | JSON `query`, optional `user_id`, optional `task_id`. Returns `response` and `task_id`. If no chunks pass the similarity threshold, returns a helpful message instead of an LLM answer. |

**Duplicate videos:** before download, the app resolves a stable `video_id` with yt-dlp and checks the **`youtube_videos`** table. Re-submitting the same video returns HTTP 400 with a duplicate error unless you clear that row.

## Main code paths

- **`app.py`**: CORS, directories, `youtube_videos` ORM, progress store for SSE, `_download_transcribe_flow` (metadata → download → transcribe → register row), JSON and form routes, subprocess ingest.
- **`inges.py`**: `DirectoryLoader` for `/app/docs/txt/**/*.txt`, `RecursiveCharacterTextSplitter` (2000 chars, 20 overlap), `OpenAI` client embeddings, `TextEmbedding` table named `COLLECTION_NAME`, duplicate skip by exact chunk string match.
- **`query.py`**: Query embedding via same API/model as ingest, pgvector `<=>` with threshold **0.7**, top-**k** (default 5), Ollama streaming chat with a fixed “hedge fund manager” system prompt. `save_chat_history` is currently a no-op (logging only).

## Dependencies

See `requirements.txt` for pinned and unpinned packages. Notable: **Flask**, **flask-cors**, **yt-dlp**, **moviepy**, **langchain**-community/text-splitters/openai, **openai**, **sqlalchemy**, **pgvector**, **psycopg2-binary**, **requests**, **fake-useragent**. Chat uses the **`ollama`** Python client (`from ollama import Client`); ensure it is installed (directly or as a transitive dependency of your environment).

## Local run

```bash
pip install -r requirements.txt
mkdir -p docs/mp3 docs/txt
export POSTGRES_CONNECTION_URI="postgresql://..."
export OPENAI_BASE="http://your-embeddings:8001/v1"
export API_URL="http://your-whisper:5005/transcribe"
export OLLAMA_HOST="http://localhost:11434"
python app.py
```

Open `http://localhost:5004` (or your `PORT`).

## Docker

`build.sh` and `test.sh` in this repo target **`registry.alpha5.finance/trade-system/youtube-rag:latest`**—adjust registry/tag for your environment.

Build (or run the commands inside `build.sh`):

```bash
docker build -t registry.alpha5.finance/trade-system/youtube-rag:latest .
```

Example run (from `test.sh`; add `-e POSTGRES_CONNECTION_URI=...` pointing at a reachable Postgres with pgvector):

```bash
docker run --rm -it --name youtube-rag \
  -p 5014:5004 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e OPENAI_BASE=http://embeddings.alpha5.finance:8001/v1 \
  -e API_URL=http://whisper-api.alpha5.finance:5005/transcribe \
  -e EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
  -e EMBEDDING_DIM=384 \
  -e COLLECTION_NAME=youtube_docs \
  -e MODEL=deepseek-r1:1.5b \
  -e OLLAMA_HOST=http://ollama.alpha5.finance:11434 \
  -e TZ=America/New_York \
  -v /data/ai-docs/youtube-rag/docs:/app/docs \
  registry.alpha5.finance/trade-system/youtube-rag:latest
```

The image **`CMD`** is `python app.py` on port **5004** inside the container.

## Example `curl` calls

**Download + transcribe (JSON):**

```bash
curl -sS -X POST http://localhost:5004/download-transcribe \
  -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=VIDEO_ID","ingest":false,"task_id":"my-task-1"}'
```

**Ingest all transcripts:**

```bash
curl -sS -X POST http://localhost:5004/run-ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Query:**

```bash
curl -sS -X POST http://localhost:5004/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What themes are discussed?","user_id":"user123"}'
```

## Operational notes

- **Concurrency:** at most **3** simultaneous yt-dlp downloads (semaphore), plus a random **1–3 s** delay per download.
- **FFmpeg:** required on the host or in the image for audio extraction (Dockerfile installs `ffmpeg`).
- **Similarity:** chunks with cosine distance **≥ 0.7** are excluded from RAG context.
- **Logging:** set `LOG_LEVEL` (e.g. `DEBUG`) for verbose logs.

## License

This project is licensed under the **Apache License 2.0**; see [LICENSE](LICENSE).
