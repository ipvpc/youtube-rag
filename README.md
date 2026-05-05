# YouTube RAG

Flask application that **downloads YouTube audio**, **transcribes** it via a Whisper-compatible HTTP API, **indexes transcripts** into a **remote [LightRAG](https://github.com/HKUDS/LightRAG)** service, and **answers questions** through that same LightRAG HTTP API. The bundled UI covers download, optional ingest, and chat with **Server-Sent Events** progress.

PostgreSQL is used only for the **`youtube_videos`** registry (duplicate detection and paths), not for vector RAG storage.

## Flow

1. **Download** — `yt-dlp` extracts **MP3** (192 kbps), throttled (semaphore, random delay).
2. **Transcribe** — `POST` audio to **`API_URL`** (multipart field `file`); response JSON must include **`text`**.
3. **Register** — Row in **`youtube_videos`** keyed by stable **`video_id`** (re-submitting the same video returns HTTP 400).
4. **Ingest** — `inges.py` loads `**/*.txt` under **`TEXT_FOLDER`**, batches them to LightRAG **`POST /documents/texts`** with stable **`file_sources`** like `youtube-{video_id}.txt` for deduplication on the server.
5. **Query** — **`POST /query`** on the LightRAG base URL with configurable **`LIGHTRAG_QUERY_MODE`** (`naive`, `local`, `global`, `hybrid`, `mix`, `bypass`).

Run **`python app.py`** as the main process; **`inges.py`** is invoked as a subprocess for ingest.

## Project layout

```
youtube-rag/
├── app.py                 # Flask routes, yt-dlp, transcribe, SSE, Postgres video registry
├── inges.py               # Batch send transcripts → remote LightRAG /documents/texts
├── query.py               # Calls remote LightRAG /query (imported by app)
├── lightrag_remote.py     # HTTP client (requests) for LightRAG API
├── config.py              # Environment-driven settings
├── docker-compose.yml     # Postgres + LightRAG server + youtube-rag
├── Dockerfile             # Flask app image
├── Dockerfile.lightrag    # LightRAG API (lightrag-hku) image
├── .env.example           # All variables (copy to .env)
├── requirements.txt
├── templates/index.html   # SPA-style UI + EventSource progress
├── docs/mp3, docs/txt     # Runtime audio / transcripts (created or bind-mounted)
└── LICENSE                # Apache-2.0
```

## Configuration

Copy **`.env.example`** to **`.env`** and set secrets. Docker Compose reads **`.env`** for variable substitution.

### Flask app (`config.py`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEBUG` | `false` | Flask debug |
| `LOG_LEVEL` | `INFO` (or `DEBUG` if `DEBUG=true`) | Logging |
| `PORT` | `5004` | Flask listen port |
| `CORS_ORIGINS` | `*` | Comma-separated origins for `flask-cors` |
| `DOWNLOAD_FOLDER` | `/app/docs/mp3` | MP3 output |
| `TEXT_FOLDER` | `/app/docs/txt` | Transcript `.txt` files |
| `API_URL` | Alpha5 Whisper URL | Transcription endpoint |
| `OPENAI_API_KEY` | _(empty)_ | Key for OpenAI-compatible clients used by the app |
| `OPENAI_BASE` | Alpha5 embeddings `/v1` | OpenAI-compatible base URL (include `/v1` if required) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Metadata / tooling |
| `EMBEDDING_DIM` | `384` | Same |
| `MODEL` | `gemma2:27b` | Default Ollama model name in config (unused by remote LightRAG path) |
| `OLLAMA_HOST` | Alpha5 Ollama URL | Default Ollama host in config (unused by remote LightRAG path) |
| `COLLECTION_NAME` | `youtube_finance_docs` | Legacy / docs naming |
| `POSTGRES_CONNECTION_URI` | local `youtube_rag` | SQLAlchemy URI for **`youtube_videos`** |

### Remote LightRAG client (`lightrag_remote.py`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LIGHTRAG_SERVICE_URL` | `http://127.0.0.1:9621` | LightRAG API base (no trailing slash) |
| `LIGHTRAG_API_KEY` | _(empty)_ | `X-API-Key` when the server requires it |
| `LIGHTRAG_BEARER_TOKEN` | _(empty)_ | `Authorization: Bearer …` when using JWT-style auth |
| `LIGHTRAG_REQUEST_TIMEOUT` | `300` | HTTP timeout (seconds) |
| `LIGHTRAG_QUERY_MODE` | `hybrid` | Passed as `mode` on **`POST /query`** |
| `LIGHTRAG_USER_PROMPT` | _(empty)_ | If set, sent as `user_prompt` on **`POST /query`** |

### Docker Compose (see `.env.example`)

- **Postgres**: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`.
- **LightRAG container**: `OPENAI_COMPAT_BASE`, optional `OPENAI_LLM_BASE` / `OPENAI_EMBEDDING_BASE` / `OPENAI_EMBEDDING_API_KEY`, `LLM_MODEL`, `LIGHTRAG_LLM_MODEL`, `LIGHTRAG_EMBEDDING_MODEL`, `LIGHTRAG_EMBEDDING_DIM`, shared `OPENAI_API_KEY`, optional `LIGHTRAG_API_KEY` for the LightRAG server’s API key.

The **LightRAG** service is built with **`LLM_BINDING=openai`** and **`EMBEDDING_BINDING=openai`** (OpenAI-compatible HTTPS). Point hosts at OpenAI, vLLM, LiteLLM, or your own gateway; each base URL should include **`/v1`** when your provider expects it.

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (`templates/index.html`) |
| `POST` | `/` | Legacy form: download + transcribe, redirect to `/success` |
| `GET` | `/success` | Plain text after legacy form |
| `GET` | `/health` | `{"status":"healthy","service":"youtube-rag"}` |
| `POST` | `/download-transcribe` | JSON: `youtube_url`, optional `ingest`, optional `task_id` |
| `GET` | `/progress/<task_id>` | SSE progress JSON |
| `POST` | `/run-ingest` | Optional JSON `task_id`; runs `python inges.py` |
| `POST` | `/query` | JSON: `query`, optional `user_id`, optional `task_id` → LightRAG answer |

## Local development

```bash
cp .env.example .env
# Edit .env: POSTGRES_CONNECTION_URI, LIGHTRAG_SERVICE_URL, OPENAI_*, API_URL, etc.

pip install -r requirements.txt
mkdir -p docs/mp3 docs/txt
python app.py
```

Open `http://localhost:5004` (or your `PORT`). You need a reachable **Postgres**, **LightRAG API**, **Whisper** service, and any **OpenAI-compatible** endpoints referenced in `.env`.

## Docker Compose

```bash
cp .env.example .env
# Set at minimum: OPENAI_API_KEY, OPENAI_COMPAT_BASE

docker compose up -d --build
```

Services:

- **`postgres`** — database for `youtube_videos`.
- **`lightrag`** — `lightrag-server` on port **9621**, OpenAI-compatible LLM + embeddings (see compose env).
- **`youtube-rag`** — Flask on **5004**, `LIGHTRAG_SERVICE_URL=http://lightrag:9621`, Whisper/embeddings default to **`host.docker.internal`** (see `extra_hosts`).

Ensure **`OPENAI_COMPAT_BASE`**, **`OPENAI_API_KEY`**, and the **`LLM_MODEL`** / **`LIGHTRAG_EMBEDDING_*`** values match your OpenAI-compatible providers (models and dimensions must line up).

## Single-container image (`Dockerfile`)

Build and run the **Flask app only**; supply Postgres, LightRAG, Whisper, and embedding URLs via `-e` or `.env`. Example:

```bash
docker build -t youtube-rag:local .
docker run --rm -p 5004:5004 \
  -e POSTGRES_CONNECTION_URI="postgresql://..." \
  -e LIGHTRAG_SERVICE_URL="http://lightrag.example:9621" \
  -e OPENAI_API_KEY="..." \
  -e OPENAI_BASE="http://host:8001/v1" \
  -e API_URL="http://host:5005/transcribe" \
  -v "$(pwd)/docs:/app/docs" \
  youtube-rag:local
```

`build.sh` / `test.sh` may reference a private registry; adjust tags to match your deployment.

## Example `curl`

**Download + transcribe (JSON)**

```bash
curl -sS -X POST http://localhost:5004/download-transcribe \
  -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=VIDEO_ID","ingest":false,"task_id":"task-1"}'
```

**Ingest all `.txt` under `TEXT_FOLDER`**

```bash
curl -sS -X POST http://localhost:5004/run-ingest \
  -H "Content-Type: application/json" \
  -d '{"task_id":"ingest-1"}'
```

**Query (via remote LightRAG)**

```bash
curl -sS -X POST http://localhost:5004/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What themes are discussed?","user_id":"user123"}'
```

## Operational notes

- At most **3** concurrent downloads; **1–3 s** random delay per download.
- **FFmpeg** required (installed in `Dockerfile`).
- **`save_chat_history`** is a stub (logging only); chat persistence in the UI is **localStorage**.
- LightRAG’s **`/query`** API requires query length **≥ 3** characters; the client pads very short strings minimally.

## Dependencies

See **`requirements.txt`**: Flask, flask-cors, yt-dlp, moviepy, langchain-community (loaders), sqlalchemy, psycopg2-binary, requests, openai, etc. The **LightRAG server** image installs **`lightrag-hku`** separately (`Dockerfile.lightrag`).

## License

Licensed under the **Apache License 2.0**; see [LICENSE](LICENSE).
