import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import requests
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meridian.api.app import create_app  # noqa: E402
from meridian.bootstrap import ensure_demo_corpus  # noqa: E402
from meridian.config import Settings  # noqa: E402
from meridian.kb import KnowledgeBase  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    s = Settings()
    s.data_dir = tmp_path_factory.mktemp("kb")
    s.demo_dir = tmp_path_factory.mktemp("demo")
    s.eval_dir = tmp_path_factory.mktemp("eval")
    s.reports_dir = tmp_path_factory.mktemp("reports")
    s.llm.provider = "none"
    s.ensure_dirs()
    ensure_demo_corpus(s)
    return s


@pytest.fixture(scope="session")
def kb(settings):
    k = KnowledgeBase(settings)
    k.add_directory(settings.demo_dir)
    return k


@pytest.fixture(scope="session")
def agent(kb, settings):
    from meridian.agent.agent import Orchestrator
    return Orchestrator(kb, settings)


@pytest.fixture(scope="session")
def questions(settings):
    from meridian.eval.runner import load_golden
    return load_golden(settings.eval_dir)


@pytest.fixture(scope="session")
def server(settings):
    """A real uvicorn server on a free port, empty knowledge base."""
    import copy
    s = copy.copy(settings)
    from pathlib import Path as P
    s.data_dir = P(str(settings.data_dir)) / "api"
    s.ensure_dirs()
    app = create_app(s)
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True); th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if requests.get(base + "/api/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.1)
    yield base
    srv.should_exit = True
    th.join(timeout=10)
