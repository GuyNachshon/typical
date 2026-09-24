"""Typical site API: POST /api/decide over inference/typical, single-KV-encode per request
(mirrors gpu-runpod-full-experiment:demo/app.py's Batch tab -- native_kv_decide, not per-query
.choice/.score/.noul). Serves site/ as static files at "/".

Run: uv run uvicorn --app-dir site server:app --port 8787
"""
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # this file lives in site/
sys.path.insert(0, str(REPO_ROOT / "inference"))

# ponytail: HF_TOKEN lives in .env, not the shell env -- same three-line loader as demo/app.py,
# not worth a python-dotenv dependency for.
_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from typical import Typical
from typical.core import to_labels
from typical.native import native_kv_decide

REPOS = {"typical-small": "OzLabs/typical-small", "typical-medium": "OzLabs/typical-medium"}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ponytail: one slot, evict-on-switch -- matches demo/app.py's _cache; a real LRU only matters
# if the site ever serves >1 model concurrently under load.
_cache = {"name": None, "model": None}
# ponytail: one lock, one GPU -- concurrent forward passes on MPS from FastAPI's threadpool crash
# Metal (MTLCommandBufferStatusCommitted assertion); serialize instead of queueing.
_lock = threading.Lock()


def _warm(m: Typical) -> None:
    """First MPS call on a new shape is slow (batch of 3: 866 ms cold -> 105 ms warm; see
    content-spec.md 'Notes for infra/site workers'). Run one 3-query batch now so the first
    real request isn't the one that pays for it."""
    native_kv_decide(
        m.head, m.model, "warmup state.",
        [("a?", ["x", "y"]), ("b?", ["x", "y"]), ("c?", ["x", "y"])],
        max_state=m.max_state, max_suffix=2048,
    )


def get_model(name: str) -> Typical:
    repo = REPOS.get(name or "typical-small")
    if repo is None:
        raise HTTPException(400, f"unknown model {name!r}")
    if _cache["name"] != name:
        model = Typical.from_pretrained(repo, device="auto")
        _warm(model)
        _cache["model"] = model
        _cache["name"] = name
    return _cache["model"]


class Query(BaseModel):
    type: str  # "choice" | "noul" | "score"
    question: str
    labels: list[str] = []


class DecideRequest(BaseModel):
    model: str = "typical-small"
    state: str
    queries: list[Query]


@app.post("/api/decide")
def decide(req: DecideRequest):
    with _lock:
        m = get_model(req.model)
    queries = [(q.type, q.question, ["no", "yes"] if q.type == "noul" else q.labels) for q in req.queries]
    for typ, question, labels in queries:
        if len(labels) < 2:
            raise HTTPException(400, f"'{typ}' query {question!r} needs >=2 labels")

    t0 = time.perf_counter()
    with _lock:  # the device->host copy in to_labels must stay inside too, or it races the next forward
        raws = native_kv_decide(m.head, m.model, req.state, [(q, l) for _, q, l in queries],
                                max_state=m.max_state, max_suffix=2048)
        labelled = [to_labels(raw, labels) for (_, _, labels), raw in zip(queries, raws)]
    ms = (time.perf_counter() - t0) * 1000

    results = []
    for (typ, _, labels), (probs, p_null) in zip(queries, labelled):
        argmax = max(probs, key=probs.get)
        entry = {"probs": probs, "p_null": p_null, "argmax": argmax}
        if typ == "score":
            entry["expected"] = sum(i * probs[lv] for i, lv in enumerate(labels))
        results.append(entry)
    return {"results": results, "ms": ms, "device": m.device, "model": req.model}


@app.get("/api/health")
def health():
    return {"model": _cache["name"], "device": _cache["model"].device if _cache["model"] else None}


app.mount("/", StaticFiles(directory=REPO_ROOT / "site", html=True), name="site")
