import json

import requests


def test_health_and_config(server):
    assert requests.get(server + "/api/health").json()["status"] == "ok"
    config = requests.get(server + "/api/config").json()
    assert "full" in config["presets"] and config["vector_store"] in {"memory", "qdrant"}
    assert requests.get(server + "/").status_code == 200 and "Meridian" in requests.get(server + "/").text


def test_empty_kb_is_handled(server):
    r = requests.post(server + "/api/ask", json={"question": "anything"})
    assert r.status_code == 200 and r.json()["abstained"] and "empty" in r.json()["answer"].lower()


def test_load_demo_then_ask(server):
    r = requests.post(server + "/api/demo/load", timeout=300)
    assert r.status_code == 200 and r.json()["stats"]["documents"] == 6
    a = requests.post(server + "/api/ask", json={"question": "What is the maximum fluid temperature of the HX-200?"}).json()
    assert "140" in a["answer"] and a["citations"]
    c = a["citations"][0]
    pg = requests.get(f"{server}/api/page/{c['doc_id']}/{c['page']}.png", params={"chunk": c["chunk_id"]})
    assert pg.status_code == 200 and pg.headers["content-type"] == "image/png"
    g = requests.get(server + "/api/graph", params={"chunks": c["chunk_id"]}).json()
    assert g["nodes"]


def test_stream_emits_trace_then_result(server):
    r = requests.post(server + "/api/ask/stream", data={"question": "What torque for the HX-200 impeller bolt?"}, stream=True)
    events = [l[6:].strip() for l in r.iter_lines(decode_unicode=True) if l and l.startswith("event:")]
    assert events[0] == "trace" and "result" in events and events[-1] == "done"


def test_image_question(server):
    from pathlib import Path
    img = Path(__file__).resolve().parent.parent / "eval_data" / "images" / "photo_alarm_E47_hx200.png"
    r = requests.post(server + "/api/ask", data={"question": "What does this alarm mean?"}, files={"image": ("a.png", img.read_bytes(), "image/png")})
    assert r.status_code == 200 and "suction" in r.json()["answer"].lower()


def test_validation_errors(server):
    assert requests.post(server + "/api/ask", json={"question": ""}).status_code == 422
    assert requests.post(server + "/api/ask", json={"question": "x", "preset": "nope"}).status_code == 400
    assert requests.post(server + "/api/ask", data="{bad", headers={"content-type": "application/json"}).status_code == 400
    assert requests.post(server + "/api/ask", data={"question": "what is this"}, files={"image": ("a.png", b"garbage", "image/png")}).status_code == 422
    assert requests.post(server + "/api/documents", files={"file": ("x.exe", b"zz")}).status_code == 422
    assert requests.post(server + "/api/documents", data={}).status_code == 400
    assert requests.get(server + "/api/page/../../etc/passwd/1.png").status_code in (400, 404)
    assert requests.get(server + "/api/chunk/nope").status_code == 404
    assert requests.delete(server + "/api/documents/nope").status_code == 404


def test_upload_text_and_delete(server):
    r = requests.post(server + "/api/documents", files={"file": ("note.txt", b"HX-200 field note\n\nThe HX-200 coupling guard bolt torque is 12 Nm.\n")})
    assert r.status_code == 200 and r.json()["results"][0]["status"] in ("added", "replaced")
    doc_id = r.json()["results"][0]["doc_id"]
    assert requests.delete(f"{server}/api/documents/{doc_id}").status_code == 200


def test_eval_endpoints(server):
    assert requests.get(server + "/api/eval/status").json()["running"] is False
    assert requests.get(server + "/api/eval/report").status_code in (200, 404)
