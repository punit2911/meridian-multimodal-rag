"""`python -m meridian` starts the web application."""
import argparse
import logging
from pathlib import Path

import uvicorn

# Load .env from project root if present (python-dotenv is available via requirements)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from .api.app import create_app
from .config import Settings


def main() -> None:
    ap = argparse.ArgumentParser(description="Meridian multimodal agentic RAG")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--load-demo", action="store_true", help="ingest the demo corpus on start-up if the knowledge base is empty")
    args = ap.parse_args()
    s = Settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(s)
    st = app.state.meridian
    if args.load_demo and not st.kb.chunks:
        from .bootstrap import load_demo
        print("Loading the demo corpus (first start only, about 30 seconds) ...")
        load_demo(st.kb, s)
        st.rebuild_agent()
    uvicorn.run(app, host=args.host or s.host, port=args.port or s.port, log_level="info")


if __name__ == "__main__":
    main()
