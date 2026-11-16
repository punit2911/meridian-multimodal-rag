# Deployment

## Local
See the README quick start. Data (SQLite DB, page renders, figure crops) lives in `./data` (`MERIDIAN_DATA_DIR`).

## Docker
```bash
cp .env.example .env      # optional
docker compose up --build  # http://localhost:8000, data in the `meridian-data` volume
```
Compose starts Meridian with Qdrant and persists the vector index in `qdrant-data`; documents and parsed chunks remain in `meridian-data`. The container runs as a non-root user, exposes `/api/health` for health checks and loads the demo corpus on first start. Qdrant is not published to the host network by default.

The Qdrant service is free and self-hosted. No vector database account or API key is required. The default LSA embedder also remains offline and does not require an LLM API key. To use the in-memory dense index instead, set `MERIDIAN_VECTOR_STORE=memory`.

## Configuration (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `MERIDIAN_HOST`, `MERIDIAN_PORT` | 127.0.0.1, 8000 | bind address |
| `MERIDIAN_DATA_DIR` | `./data` | KB storage |
| `MERIDIAN_MAX_UPLOAD_MB` | 60 | per-file limit (larger files are rejected with an error entry) |
| `MERIDIAN_RENDER_DPI` | 170 | page render resolution |
| `MERIDIAN_OCR_LANG` | eng | Tesseract language(s), e.g. `eng+deu` (install the packs) |
| `MERIDIAN_LLM_PROVIDER` | none | `anthropic` or `openai` (OpenAI-compatible) |
| `MERIDIAN_LLM_MODEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` | – | LLM settings |
| `MERIDIAN_AGENT_MODE` | auto | `policy` forces offline |
| `MERIDIAN_VECTOR_STORE` | memory | `qdrant` to use Qdrant; Compose enables it |
| `MERIDIAN_QDRANT_URL`, `MERIDIAN_QDRANT_COLLECTION`, `MERIDIAN_QDRANT_API_KEY` | `http://localhost:6333`, `meridian_chunks`, – | Qdrant connection |

## Production notes
* Put a reverse proxy (nginx, Caddy) with TLS and authentication in front; the app itself has no login.
* The Qdrant vector collection is rebuilt from the canonical SQLite document/chunk store on startup and after each document mutation. This first integration uses Qdrant for approximate-nearest-neighbor search, but is not yet incremental ingestion or a horizontally scaled ingestion service.
* The knowledge base's lexical, graph, and fact indexes are still in-process; multi-replica coordination and tenant isolation are not implemented.
* Back up the `data` volume. The index is rebuilt from SQLite on start.
* For stronger semantic matching `pip install sentence-transformers` and set `MERIDIAN_EMBEDDINGS=st` (model via `MERIDIAN_ST_MODEL`, default `BAAI/bge-small-en-v1.5`); falls back to LSA if unavailable. This needs a one-time model download, so it is not enabled by default. Re-ingest after switching.
