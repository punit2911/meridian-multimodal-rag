"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "n/a" : (x * 100).toFixed(1) + "%");
const num3 = (x) => (x == null ? "n/a" : Number(x).toFixed(3));
let image = null, lastResult = null, busy = false;

function toast(msg, ms = 3500) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.h); toast.h = setTimeout(() => (t.hidden = true), ms); }
async function api(path, opts) {
  let r;
  try { r = await fetch(path, opts); } catch (e) { throw new Error("Cannot reach the server. Is it still running?"); }
  let data = null; try { data = await r.json(); } catch (e) { /* non-JSON */ }
  if (!r.ok) throw new Error((data && data.error) || `Request failed (${r.status})`);
  return data;
}

/* ---------- tabs ---------- */
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "kb") refreshKb(); if (b.dataset.tab === "eval") loadEval();
}));

/* ---------- status / config ---------- */
const EXAMPLES = ["What torque should I use for the HX-200 impeller bolt?", "Is the HX-200 rated for fluid at 150 °C?", "What is the HX-500 specific power at 50% load?",
  "Which alarm code occurs most often in the maintenance log?", "What does alarm E-47 mean on the HX-200?", "How do I replace the HX-200 mechanical seal?", "Who is the CEO of Helix Dynamics?"];
async function init() {
  $("#examples").innerHTML = EXAMPLES.map((q) => `<button type="button">${esc(q)}</button>`).join("");
  $("#examples").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") { $("#question").value = e.target.textContent; $("#askForm").requestSubmit(); } });
  try {
    const c = await api("/api/config");
    $("#preset").innerHTML = c.presets.map((p) => `<option ${p === "full" ? "selected" : ""}>${esc(p)}</option>`).join("");
    $("#status").textContent = `${c.agent_mode === "llm_tools" ? "LLM · " + c.llm_provider : "offline"} agent · ${c.vector_store} · ${c.embedding_backend}`;
  } catch (e) { $("#status").textContent = "offline"; }
  await refreshStats();
}
async function refreshStats() {
  try { const s = await api("/api/stats"); $("#emptyKb").hidden = s.chunks > 0; return s; } catch (e) { return null; }
}

/* ---------- ask ---------- */
$("#imageInput").addEventListener("change", (e) => setImage(e.target.files[0]));
document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/")); if (f) setImage(f); });
function setImage(f) {
  image = f && f.type.startsWith("image/") ? f : null;
  const chip = $("#imageChip");
  chip.hidden = !image;
  if (image) { chip.innerHTML = `${esc(image.name || "pasted image")} <button type="button" class="link" id="rmImg">remove</button>`; $("#rmImg").onclick = () => { setImage(null); $("#imageInput").value = ""; }; }
  else if (f) toast("Please choose an image file.");
}
$("#askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy) return;
  const q = $("#question").value.trim();
  if (!q && !image) { toast("Type a question or attach an image."); return; }
  busy = true; $("#askBtn").disabled = true; $("#askBtn").innerHTML = '<span class="spin"></span> Working';
  $("#trace").innerHTML = ""; $("#answerArea").innerHTML = "";
  const fd = new FormData(); fd.append("question", q); fd.append("preset", $("#preset").value); if (image) fd.append("image", image);
  try {
    const r = await fetch("/api/ask/stream", { method: "POST", body: fd });
    if (!r.ok || !r.body) { let m = `Request failed (${r.status})`; try { m = (await r.json()).error || m; } catch (_) {} throw new Error(m); }
    const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i; while ((i = buf.indexOf("\n\n")) >= 0) { handleSse(buf.slice(0, i)); buf = buf.slice(i + 2); }
    }
  } catch (err) { $("#answerArea").innerHTML = `<div class="card"><div class="banner bad">${esc(err.message)}</div></div>`; }
  finally { busy = false; $("#askBtn").disabled = false; $("#askBtn").textContent = "Ask"; }
});
function handleSse(block) {
  let ev = "message", data = "";
  block.split("\n").forEach((l) => { if (l.startsWith("event:")) ev = l.slice(6).trim(); else if (l.startsWith("data:")) data += l.slice(5).trim(); });
  if (!data) return; const d = JSON.parse(data);
  if (ev === "trace") addTrace(d);
  else if (ev === "result") renderAnswer(d);
  else if (ev === "error") $("#answerArea").innerHTML = `<div class="card"><div class="banner bad">${esc(d.error)}</div></div>`;
}
function addTrace(t) {
  const li = document.createElement("li"); li.className = t.status === "warn" ? "warn" : t.status === "fail" ? "fail" : "";
  li.innerHTML = `<span class="ms">${t.ms ? Math.round(t.ms) + " ms" : ""}</span><b>${esc(t.title)}</b>${t.detail ? `<small>${esc(t.detail)}</small>` : ""}`;
  $("#trace").appendChild(li);
}
function linkCites(text, cites) {
  const max = cites.length;
  return esc(text).replace(/\[(\d+)\]/g, (m, n) => (+n >= 1 && +n <= max ? `<button class="cite" data-cite="${n}" title="Show evidence ${n}">${n}</button>` : m));
}
function renderAnswer(r) {
  lastResult = r;
  const conf = r.confidence, faith = r.faithfulness;
  const cls = (v, g, w) => (v >= g ? "good" : v >= w ? "warn" : "bad");
  let h = `<div class="card"><div class="answer ${r.abstained ? "abstained" : ""}">${r.answer.split(/\n+/).map((p) => `<p>${linkCites(p, r.citations)}</p>`).join("")}</div>`;
  h += `<div class="badges">`;
  h += r.abstained ? `<span class="badge warn">Declined · ${esc(r.abstain_reason.replace(/_/g, " "))}</span>` : `<span class="badge ${cls(conf, .75, .5)}">Confidence ${pct(conf)}</span><span class="badge ${cls(faith, .95, .7)}">Faithfulness ${pct(faith)}</span>`;
  h += `<span class="badge">${esc(r.intent.replace("_", " "))}</span><span class="badge">${Math.round(r.timings.total_ms || 0)} ms</span><span class="badge">${esc(r.mode)}</span></div>`;
  (r.warnings || []).forEach((w) => (h += `<div class="banner warn">${esc(w)}</div>`));
  (r.conflicts || []).forEach((c) => (h += `<div class="banner warn"><b>Sources disagree on “${esc(c.subject)}”.</b> ${esc(c.loser)}: ${esc(c.loser_value)} → ${esc(c.winner)}: <b>${esc(c.winner_value)}</b>. ${esc(c.rule)}.</div>`));
  h += `</div>`;
  if (r.citations.length) {
    h += `<div class="card"><h3>Evidence</h3><div class="evidence">` + r.citations.map((c) => `
      <div class="ev" id="ev-${c.index}"><header><span class="num">${c.index}</span><span class="src">${esc(c.doc_title)}</span><span class="meta">p.${c.page} · ${esc(c.modality.replace("_", " "))}${c.section ? " · " + esc(c.section) : ""}${c.via !== "retrieval" ? " · via " + esc(c.via) : ""}</span></header>
      <p>${esc(c.snippet)}</p>${c.has_image ? `<img class="thumb" loading="lazy" alt="figure from the source" src="/api/chunk/${encodeURIComponent(c.chunk_id)}/image">` : ""}
      <div class="acts"><button class="link" data-page="${c.index}">View source page</button></div></div>`).join("") + `</div></div>`;
    h += `<div class="card"><h3>Evidence graph</h3><div id="graphBox" class="muted small">Loading…</div></div>`;
  }
  if (r.claims.length) {
    h += `<div class="card"><details open><summary>Claim-level verification (${r.claims.length})</summary><ul class="claims">` + r.claims.map((c) => `<li><span class="v ${c.verdict}">${esc(c.verdict)}</span>${esc(c.text)}${c.reasons.length ? `<small class="muted"> — ${esc(c.reasons.join("; "))}</small>` : ""}</li>`).join("") + `</ul></details></div>`;
  }
  $("#answerArea").innerHTML = h;
  if (r.citations.length) drawGraph(r.citations.map((c) => c.chunk_id));
}
$("#answerArea").addEventListener("click", (e) => {
  const c = e.target.closest("[data-cite]"); if (c) { const el = $("#ev-" + c.dataset.cite); if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.style.outline = "2px solid var(--accent)"; setTimeout(() => (el.style.outline = ""), 1600); } return; }
  const p = e.target.closest("[data-page]"); if (p && lastResult) openPage(lastResult.citations[+p.dataset.page - 1]);
});
function openPage(c) {
  const modal = $("#modal");
  $("#modalBody").innerHTML = `<h2>${esc(c.doc_title)} — page ${c.page}</h2><p class="muted small">The highlighted region is the evidence the answer cites.</p>
    <img alt="source page with highlighted evidence" src="/api/page/${encodeURIComponent(c.doc_id)}/${c.page}.png?chunk=${encodeURIComponent(c.chunk_id)}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'notice',textContent:'No page image for this source type. Evidence text: '+${JSON.stringify(c.snippet)}}))">`;
  modal.hidden = false;
}
$("#modalClose").onclick = () => ($("#modal").hidden = true);
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("#modal").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#modal").hidden = true; });

async function drawGraph(ids) {
  const box = $("#graphBox"); if (!box) return;
  try {
    const g = await api("/api/graph?chunks=" + ids.map(encodeURIComponent).join(","));
    if (!g.nodes.length) { box.textContent = "No graph neighbours."; return; }
    const W = 640, H = 320, n = g.nodes.length, idx = {}; g.nodes.forEach((x, i) => (idx[x.id] = i));
    const seeds = g.nodes.filter((x) => x.seed), others = g.nodes.filter((x) => !x.seed);
    const pos = {}; seeds.forEach((x, i) => (pos[x.id] = [W / 2 + (i - (seeds.length - 1) / 2) * 90, H / 2]));
    others.forEach((x, i) => { const a = (i / Math.max(others.length, 1)) * 2 * Math.PI, r = 120 + (i % 2) * 22; pos[x.id] = [W / 2 + Math.cos(a) * r * 1.8, H / 2 + Math.sin(a) * r]; });
    const col = { mention: "#c2410c", row_of: "#0f766e", adjacent: "#9ca3af", warns: "#b42318", "refers-to": "#6d28d9" };
    let s = `<svg class="graph" viewBox="0 0 ${W} ${H}" role="img" aria-label="Evidence graph">`;
    g.edges.forEach((e) => { const a = pos[e.source], b = pos[e.target]; if (a && b) s += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${col[e.kind] || "#9ca3af"}" stroke-width="${1 + e.weight}" opacity=".6"><title>${esc(e.kind)}</title></line>`; });
    g.nodes.forEach((x) => { const p = pos[x.id]; if (!p) return; const label = x.label || x.id.replace(/^[a-z]+:/, "");
      s += `<g><circle cx="${p[0]}" cy="${p[1]}" r="${x.seed ? 9 : 6}" fill="${x.seed ? "#0f766e" : "#8aa4c8"}"><title>${esc(label)}</title></circle><text x="${p[0] + 10}" y="${p[1] + 3}">${esc(String(label).slice(0, 22))}</text></g>`; });
    box.innerHTML = s + `</svg><div class="muted small">Teal = cited evidence; grey = neighbours linked by cross-references, rows, adjacency or warnings.</div>`;
  } catch (e) { box.textContent = "Graph unavailable."; }
}

/* ---------- knowledge base ---------- */
async function refreshKb() {
  const s = await refreshStats(); if (!s) return;
  $("#kbStats").innerHTML = [["documents", s.documents], ["chunks", s.chunks], ["facts", s.facts], ["graph edges", s.graph?.edges ?? 0], ["models", (s.models || []).join(", ") || "–"]]
    .map(([k, v]) => `<div class="stat"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join("") +
    Object.entries(s.modalities || {}).map(([k, v]) => `<div class="stat"><b>${v}</b><span>${esc(k.replace("_", " "))}</span></div>`).join("");
  const counts = {}; (s.documents_list || []).forEach((d) => (counts[d.id] = d));
  $("#docTable tbody").innerHTML = (s.documents_list || []).map((d) => `<tr><td><b>${esc(d.title)}</b><div class="muted small">${esc(d.filename)}</div></td><td>${esc(d.doc_type)}</td><td>${esc((d.models || []).join(", "))}</td><td class="num">${d.pages || ""}</td><td>${esc(d.effective_date || "")}</td><td class="num">${d.chunks ?? ""}</td><td><button class="link" data-del="${esc(d.id)}">Delete</button></td></tr>`).join("") || `<tr><td colspan="7" class="muted">No documents yet.</td></tr>`;
}
$("#docTable").addEventListener("click", async (e) => { const id = e.target.dataset.del; if (!id || !confirm("Delete this document and its index entries?")) return;
  try { await api("/api/documents/" + encodeURIComponent(id), { method: "DELETE" }); toast("Deleted."); refreshKb(); } catch (err) { toast(err.message); } });
async function upload(files) {
  if (!files.length) return; const fd = new FormData(); [...files].forEach((f) => fd.append("file", f));
  $("#uploadLog").innerHTML = '<span class="spin"></span> Parsing, OCR and indexing — large PDFs can take a minute…';
  try { const r = await api("/api/documents", { method: "POST", body: fd }); $("#uploadLog").innerHTML = r.results.map((x) => `${esc(x.file)}: ${esc(x.status)}${x.error ? " — " + esc(x.error) : ""}${x.chunks ? " (" + x.chunks + " chunks)" : ""}`).join("<br>"); }
  catch (err) { $("#uploadLog").textContent = err.message; }
  refreshKb();
}
$("#fileInput").addEventListener("change", (e) => { upload(e.target.files); e.target.value = ""; });
const drop = $("#drop");
["dragover", "dragenter"].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload(e.dataTransfer.files));
async function loadDemo(btn) {
  btn.disabled = true; const old = btn.textContent; btn.innerHTML = '<span class="spin"></span> Loading…';
  try { await api("/api/demo/load", { method: "POST" }); toast("Demo corpus loaded."); refreshKb(); refreshStats(); } catch (e) { toast(e.message); } finally { btn.disabled = false; btn.textContent = old; }
}
$("#loadDemo").onclick = (e) => loadDemo(e.currentTarget); $("#loadDemoInline").onclick = (e) => loadDemo(e.currentTarget);
$("#resetKb").onclick = async () => { if (!confirm("Remove ALL documents from the knowledge base?")) return; try { await api("/api/reset", { method: "POST" }); refreshKb(); } catch (e) { toast(e.message); } };

/* ---------- evaluation ---------- */
function tbl(head, rows, bestCol) {
  return `<table class="tbl"><thead><tr>${head.map((h, i) => `<th class="${i ? "num" : ""}">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr class="${r.best ? "best" : ""}">${r.cells.map((c, i) => `<td class="${i ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
async function loadEval() {
  let r; try { r = await api("/api/eval/report"); } catch (e) { $("#evalBody").textContent = e.message; return; }
  const abl = Object.entries(r.ablation_test); const t = r.e2e_test, a = r.e2e_all, h = r.hallucination, c = r.calibration || {};
  const m = (o, k) => pct(o[k] && o[k].mean != null ? o[k].mean : o[k]);
  let html = `<p>${r.questions} golden questions (${r.splits.train} train / ${r.splits.test} held-out test), ${r.documents} documents, ${r.chunks} chunks · ${r.seconds}s</p>`;
  html += `<h3>Retrieval ablation (held-out test)</h3>` + tbl(["Configuration", "Hit@1", "Hit@5", "MRR", "nDCG@10", "ms"], abl.map(([k, s], i) => ({ best: i === abl.length - 1, cells: [esc(k), pct(s["hit@1"].mean), pct(s["hit@5"].mean), num3(s.mrr.mean), num3(s["ndcg@10"].mean), Math.round(s.ms.mean)] })));
  html += `<h3 style="margin-top:20px">End-to-end</h3>` + tbl(["Metric", "All", "Test"], [["Answer correctness", "answer_correctness"], ["Abstention recall", "abstention_recall"], ["Abstention precision", "abstention_precision"], ["False-abstention rate", "false_abstention_rate"], ["Faithfulness", "faithfulness"], ["Citation recall", "citation_recall"]].map(([l, k]) => ({ cells: [l, m(a, k), m(t, k)] })));
  html += `<p class="muted small">ECE ${num3(a.ece)} · Brier ${num3(a.brier)} · calibrated test ECE ${num3(c.calibrated_ece_test)}</p>`;
  html += `<h3 style="margin-top:20px">By question type</h3>` + tbl(["Type", "Correct"], Object.entries(a.by_type).map(([k, v]) => ({ cells: [esc(k), pct(v)] })));
  html += `<h3 style="margin-top:20px">Hallucination-injection benchmark</h3><p class="muted small">${h.clean_answers} clean answers · false-positive rate ${pct(h.false_positive_rate)} · overall detection <b>${pct(h.overall_detection_rate)}</b></p>` + tbl(["Injected error", "Count", "Detected"], Object.keys(h.injected).map((k) => ({ cells: [esc(k), h.injected[k], pct(h.detection_rate[k])] })));
  html += `<div class="charts" style="margin-top:20px"><img alt="ablation chart" src="/api/eval/chart/ablation.png?t=${Date.now()}" onerror="this.remove()"><img alt="reliability diagram" src="/api/eval/chart/reliability.png?t=${Date.now()}" onerror="this.remove()"></div>`;
  const fails = r.e2e_rows.filter((x) => !x.correct);
  html += `<details style="margin-top:16px"><summary>Failed questions (${fails.length})</summary><ul>${fails.map((f) => `<li><code>${esc(f.id)}</code> ${esc(f.question)} <span class="muted">→ ${esc((f.missing || []).join(", "))}</span></li>`).join("")}</ul></details>`;
  $("#evalBody").innerHTML = html;
}
$("#runEval").onclick = async () => {
  try { await api("/api/eval/run", { method: "POST" }); } catch (e) { toast(e.message); return; }
  $("#runEval").disabled = true; $("#evalMsg").innerHTML = '<span class="spin"></span> running…';
  const poll = setInterval(async () => { try { const s = await api("/api/eval/status"); $("#evalMsg").innerHTML = s.running ? `<span class="spin"></span> ${esc(s.log[s.log.length - 1] || "running…")}` : (s.error ? esc(s.error) : "done");
      if (!s.running) { clearInterval(poll); $("#runEval").disabled = false; if (!s.error) loadEval(); } } catch (e) { clearInterval(poll); $("#runEval").disabled = false; } }, 1500);
};
init();
