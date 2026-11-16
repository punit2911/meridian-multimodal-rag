"""Starlette application: JSON + SSE API and the single-page UI."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import queue
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..agent.agent import AgentError, Orchestrator
from ..agent.factory import make_agent
from ..config import Settings
from ..ingest.pipeline import SUPPORTED, UnsupportedFileType
from ..kb import KnowledgeBase
from ..retrieval.context import ContextConfig
from ..retrieval.pipeline import HybridRetriever, preset
from ..schemas import TraceEvent

log = logging.getLogger("meridian")
STATIC = Path(__file__).parent / "static"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")
IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp", "image/gif", "image/tiff"}


def _err(status: int, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


class State:
    def __init__(self, settings: Settings):
        settings.ensure_dirs()
        self.s = settings
        self.kb = KnowledgeBase(settings)
        self.agent: Orchestrator = make_agent(self.kb, settings)
        self.lock = threading.Lock()  # serialises ingestion vs. agent rebuild
        self.eval_status: dict = {"running": False, "log": [], "error": None, "finished": None}

    def rebuild_agent(self) -> None:
        self.agent = make_agent(self.kb, self.s)


def create_app(settings: Optional[Settings] = None) -> Starlette:
    state = State(settings or Settings())
    s = state.s
    max_bytes = s.max_upload_mb * 1024 * 1024

    # ------------------------------------------------------------------------------------------ helpers
    async def read_form(request: Request):
        try:
            return await request.form(max_files=20, max_fields=50)
        except Exception as exc:
            raise HTTPException(400, f"Could not read the upload: {exc}")

    async def image_from_form(form) -> Optional[bytes]:
        up = form.get("image")
        if up is None or not hasattr(up, "read"):
            return None
        data = await up.read()
        if not data:
            return None
        if len(data) > max_bytes:
            raise HTTPException(413, f"Image larger than {s.max_upload_mb} MB.")
        return data

    def configs(form_or_json) -> tuple:
        name = str(form_or_json.get("preset") or "full")
        try:
            rcfg = preset(name)
        except KeyError:
            raise HTTPException(400, f"Unknown retrieval preset '{name}'.")
        ccfg = ContextConfig()
        k = form_or_json.get("k")
        if k:
            try:
                ccfg.k = max(1, min(12, int(k)))
            except ValueError:
                raise HTTPException(400, "k must be an integer.")
        return rcfg, ccfg

    # ------------------------------------------------------------------------------------------ meta
    async def health(request):
        return JSONResponse({"status": "ok", "documents": len(state.kb.docs), "chunks": len(state.kb.chunks)})

    async def config(request):
        return JSONResponse({"agent_mode": state.agent.mode, "llm_provider": s.llm.provider, "llm_model": s.llm.effective_model,
                             "embedding_backend": state.kb.ix.dense.backend.name if state.kb.chunks else "lsa",
                             "vector_store": state.kb.ix.dense.vector_store, "supported": sorted(SUPPORTED),
                             "presets": ["bm25", "dense", "hybrid_rrf", "hybrid+entity+fact", "+prf+graph", "+late_rerank", "full"],
                             "max_upload_mb": s.max_upload_mb})

    async def stats(request):
        counts: dict[str, int] = {}
        for c in state.kb.chunks:
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        return JSONResponse(state.kb.stats() | {"documents_list": [d.model_dump() | {"chunks": counts.get(d.id, 0)} for d in state.kb.docs.values()]})

    # ------------------------------------------------------------------------------------------ ask
    async def ask(request: Request):
        ctype = request.headers.get("content-type", "")
        image = None
        if ctype.startswith("application/json"):
            try:
                body = await request.json()
            except ValueError:
                return _err(400, "Invalid JSON body.")
            if not isinstance(body, dict):
                return _err(400, "JSON body must be an object.")
            question = str(body.get("question", ""))
            cfgsrc = body
        else:
            form = await read_form(request)
            question = str(form.get("question", ""))
            image = await image_from_form(form)
            cfgsrc = form
        rcfg, ccfg = configs(cfgsrc)
        try:
            res = await asyncio.to_thread(state.agent.answer, question, image, retrieval_cfg=rcfg, context_cfg=ccfg)
        except AgentError as exc:
            return _err(422, str(exc))
        return JSONResponse(res.model_dump(mode="json"))

    async def ask_stream(request: Request):
        form = await read_form(request)
        question = str(form.get("question", ""))
        image = await image_from_form(form)
        rcfg, ccfg = configs(form)
        q: "queue.Queue" = queue.Queue()

        def on_event(ev: TraceEvent) -> None:
            q.put(("trace", ev.model_dump(mode="json")))

        def work() -> None:
            try:
                res = state.agent.answer(question, image, retrieval_cfg=rcfg, context_cfg=ccfg, on_event=on_event)
                q.put(("result", res.model_dump(mode="json")))
            except AgentError as exc:
                q.put(("error", {"error": str(exc), "status": 422}))
            except Exception as exc:  # never leave the stream hanging
                log.exception("agent failure")
                q.put(("error", {"error": f"Internal error: {type(exc).__name__}", "status": 500}))
            q.put(("done", {}))

        threading.Thread(target=work, daemon=True).start()

        async def gen():
            while True:
                try:
                    kind, data = await asyncio.to_thread(q.get, True, 120)
                except queue.Empty:
                    yield "event: error\ndata: " + json.dumps({"error": "timeout"}) + "\n\n"
                    return
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                if kind == "done":
                    return

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------------------------------ documents
    async def list_docs(request):
        return JSONResponse({"documents": [d.model_dump() for d in state.kb.docs.values()]})

    async def upload(request: Request):
        form = await read_form(request)
        files = [v for k, v in form.multi_items() if k in ("file", "files") and hasattr(v, "read")]
        if not files:
            return _err(400, "No file uploaded (use the 'file' field).")
        results = []
        for up in files:
            name = Path(up.filename or "upload").name
            data = await up.read()
            if not data:
                results.append({"file": name, "status": "error", "error": "empty file"})
                continue
            if len(data) > max_bytes:
                results.append({"file": name, "status": "error", "error": f"larger than {s.max_upload_mb} MB"})
                continue
            with tempfile.TemporaryDirectory(dir=s.data_dir / "uploads") as tmp:
                path = Path(tmp) / name
                path.write_bytes(data)
                try:
                    r = await asyncio.to_thread(state.kb.add_file, path, original_name=name)
                    results.append({"file": name, "status": r["status"], "chunks": r["chunks"], "doc_id": r["doc"]["id"]})
                except UnsupportedFileType as exc:
                    results.append({"file": name, "status": "error", "error": str(exc)})
                except Exception as exc:
                    log.exception("ingest failed for %s", name)
                    results.append({"file": name, "status": "error", "error": f"could not parse this file ({type(exc).__name__}: {exc})"})
        state.rebuild_agent()
        ok = any(r["status"] in ("added", "replaced", "duplicate") for r in results)
        return JSONResponse({"results": results}, status_code=200 if ok else 422)

    async def delete_doc(request: Request):
        doc_id = request.path_params["doc_id"]
        if not await asyncio.to_thread(state.kb.remove, doc_id):
            return _err(404, f"No document '{doc_id}'.")
        state.rebuild_agent()
        return JSONResponse({"deleted": doc_id})

    async def load_demo(request):
        from ..bootstrap import load_demo as _load
        results = await asyncio.to_thread(_load, state.kb, s)
        state.rebuild_agent()
        return JSONResponse({"results": results, "stats": state.kb.stats()})

    async def reset(request):
        await asyncio.to_thread(state.kb.reset)
        state.rebuild_agent()
        return JSONResponse({"status": "reset"})

    # ------------------------------------------------------------------------------------------ evidence viewing
    async def chunk_info(request: Request):
        ch = state.kb.chunk(request.path_params["chunk_id"])
        if ch is None:
            return _err(404, "Unknown chunk.")
        d = state.kb.doc(ch.doc_id)
        return JSONResponse({"chunk": ch.public(), "document": d.model_dump() if d else None})

    async def page_image(request: Request):
        doc_id, page = request.path_params["doc_id"], request.path_params["page"]
        if not SAFE_ID.match(doc_id):
            return _err(400, "Bad document id.")
        path = s.data_dir / "pages" / doc_id / f"p{page}.png"
        if not path.exists():
            return _err(404, "Page image not available.")
        img = Image.open(path).convert("RGB")
        chunk_id = request.query_params.get("chunk")
        if chunk_id:
            ch = state.kb.chunk(chunk_id)
            if ch and ch.bbox and ch.page == int(page):
                b = ch.bbox
                sx, sy = img.width / b.page_width, img.height / b.page_height
                box = [b.x0 * sx, b.y0 * sy, b.x1 * sx, b.y1 * sy]
                over = Image.new("RGBA", img.size, (0, 0, 0, 0))
                dr = ImageDraw.Draw(over)
                dr.rectangle(box, fill=(255, 214, 0, 70), outline=(230, 150, 0, 255), width=3)
                img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})

    async def chunk_image(request: Request):
        ch = state.kb.chunk(request.path_params["chunk_id"])
        if ch is None or not ch.image_path:
            return _err(404, "No image for this chunk.")
        path = (s.data_dir / ch.image_path).resolve()
        if s.data_dir.resolve() not in path.parents or not path.exists():
            return _err(404, "Image file missing.")
        return FileResponse(path)

    async def graph(request: Request):
        ids = [c for c in request.query_params.get("chunks", "").split(",") if c]
        if not ids:
            return _err(400, "Pass ?chunks=<id>,<id>.")
        sub = state.kb.ix.graph.subgraph(ids, hops=int(request.query_params.get("hops", "1")))
        for n in sub.get("nodes", []):
            ch = state.kb.chunk(n["id"][2:]) if str(n.get("id", "")).startswith("c:") else None
            if ch:
                n["label"] = (ch.section or ch.modality.value)[:40]
                n["modality"] = ch.modality.value
                n["page"] = ch.page
        return JSONResponse(sub)

    # ------------------------------------------------------------------------------------------ evaluation
    async def eval_report(request):
        p = s.reports_dir / "eval_report.json"
        if not p.exists():
            return _err(404, "No evaluation report yet. Run it from the Evaluation tab or `python scripts/run_eval.py`.")
        return Response(p.read_text(encoding="utf-8"), media_type="application/json")

    async def eval_run(request):
        if state.eval_status["running"]:
            return _err(409, "An evaluation is already running.")
        if not state.kb.chunks:
            return _err(422, "Load the demo corpus first (the golden questions refer to it).")
        if not (s.eval_dir / "golden_qa.json").exists():
            return _err(422, "No golden question set found.")
        state.eval_status.update(running=True, log=[], error=None, finished=None)

        def work():
            from ..eval.report import run_full_evaluation
            try:
                run_full_evaluation(state.kb, s, train_models=False, log=lambda m: state.eval_status["log"].append(m))
                state.rebuild_agent()
            except Exception as exc:
                log.exception("evaluation failed")
                state.eval_status["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                state.eval_status.update(running=False, finished=time.time())

        threading.Thread(target=work, daemon=True).start()
        return JSONResponse({"started": True})

    async def eval_status(request):
        return JSONResponse(state.eval_status)

    async def eval_chart(request: Request):
        name = request.path_params["name"]
        if name not in ("ablation", "reliability"):
            return _err(404, "Unknown chart.")
        p = s.reports_dir / f"{name}.png"
        return FileResponse(p) if p.exists() else _err(404, "Chart not generated yet.")

    async def index(request):
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    async def http_exc(request, exc: HTTPException):
        return _err(exc.status_code, exc.detail)

    async def server_exc(request, exc: Exception):
        log.exception("unhandled error")
        return _err(500, f"Internal error: {type(exc).__name__}")

    routes = [
        Route("/", index), Route("/api/health", health), Route("/api/config", config), Route("/api/stats", stats),
        Route("/api/ask", ask, methods=["POST"]), Route("/api/ask/stream", ask_stream, methods=["POST"]),
        Route("/api/documents", list_docs), Route("/api/documents", upload, methods=["POST"]),
        Route("/api/documents/{doc_id}", delete_doc, methods=["DELETE"]),
        Route("/api/demo/load", load_demo, methods=["POST"]), Route("/api/reset", reset, methods=["POST"]),
        Route("/api/chunk/{chunk_id}", chunk_info), Route("/api/chunk/{chunk_id}/image", chunk_image),
        Route("/api/page/{doc_id}/{page:int}.png", page_image), Route("/api/graph", graph),
        Route("/api/eval/report", eval_report), Route("/api/eval/run", eval_run, methods=["POST"]), Route("/api/eval/status", eval_status),
        Route("/api/eval/chart/{name}.png", eval_chart),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ]
    app = Starlette(routes=routes, exception_handlers={HTTPException: http_exc, Exception: server_exc})
    app.state.meridian = state
    return app
