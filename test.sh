sudo docker run --rm -it --name youtube-qa-agent \
-p 5014:5004 \
-e OPENAI_API_KEY=alpha5cloud \
-e OPENAI_BASE=http://embeddings.alpha5.finance:8001 \
-e WHISPER_API_URL=http://whisper-api.alpha5.finance:5005/transcribe \
-e EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   \
-e EMBEDDING_DIM=384 \
-e COLLECTION_NAME=youtube_docs \
-e MODEL=deepseek-r1:1.5b \
-e OLLAMA_HOST=http://ollama.alpha5.finance:11434 \
-e TZ=America/New_York \
-v /data/ai-docs/youtube-rag/docs:/app/docs \
registry.alpha5.finance/trade-system/youtube-rag:latest bash
