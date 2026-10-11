from __future__ import annotations

import asyncio
import json
import pickle
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import design, pipeline as P, serialize as S
from .feature import FeatureSpec

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
CACHE = ROOT / "results" / "cache"

SCENARIOS = {
    "ghost": dict(title="Synthetic test A · truth hidden", kind="synthetic", with_spot=False, seed=3),
    "knot": dict(title="Synthetic test B · truth hidden", kind="synthetic", with_spot=True, spot_frac=0.06, seed=5),
    "m87-2017-101": dict(title="M87* · EHT 2017 April 11 (real data)", kind="real", day=101),
}

app = FastAPI(title="Anelench")
executor = ThreadPoolExecutor(max_workers=1)
sessions: dict[str, P.Session] = {}
jobs: dict[str, dict] = {}


def build_session(sid: str) -> P.Session:
    spec = SCENARIOS[sid]
    if spec["kind"] == "real":
        s = P.real_session(spec["day"])
    else:
        s = P.synthetic_session(sid, spec["title"], P.scenario_truth(spec["with_spot"], spec.get("spot_frac", 0.06)),
                                seed=spec["seed"])
    s.id, s.title = sid, spec["title"]
    return s


def new_job(kind: str) -> dict:
    job = {"id": uuid.uuid4().hex[:10], "kind": kind, "events": [], "done": False, "result": None, "error": None,
           "t0": time.time()}
    jobs[job["id"]] = job
    return job


def emit(job: dict, ev: dict) -> None:
    images = ev.pop("images", None)
    if images is not None and (ev.get("step", 0) % 60 < 10 or ev.get("step") == ev.get("steps", 0) - 1):
        ev["previews"] = [S.img64(im) for im in images]
    job["events"].append(ev)


def run_job(job: dict, fn) -> None:
    def work():
        try:
            job["result"] = fn(lambda ev: emit(job, ev))
        except Exception as exc:
            job["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job["done"] = True

    executor.submit(work)


class SearchBody(BaseModel):
    x_uas: float
    y_uas: float
    r_uas: float = 9.0
    threshold: float = 0.2
    fwhm_uas: float = 20.0


class DesignBody(BaseModel):
    budget: float = 70.0


class RetestBody(BaseModel):
    package_id: str
    seed: int = 11


@app.get("/api/scenarios")
def scenarios():
    return [{"id": k, "title": v["title"], "kind": v["kind"], "loaded": k in sessions} for k, v in SCENARIOS.items()]


@app.post("/api/session/{sid}/load")
def load(sid: str):
    if sid not in SCENARIOS:
        raise HTTPException(404, "unknown scenario")
    job = new_job("load")

    def fn(progress):
        if f"{sid}@base" in sessions:
            sessions[sid] = sessions.pop(f"{sid}@base")
        if sid not in sessions:
            cache = CACHE / f"{sid}.pkl"
            if cache.exists():
                with open(cache, "rb") as f:
                    s = pickle.load(f)
            else:
                s = build_session(sid)
                P.run_reference(s, progress=lambda ev: progress({**ev, "stage": "reference"}))
                CACHE.mkdir(parents=True, exist_ok=True)
                with open(cache, "wb") as f:
                    pickle.dump(s, f)
            sessions[sid] = s
        return S.session_payload(sessions[sid])

    run_job(job, fn)
    return {"job": job["id"]}


@app.post("/api/session/{sid}/search")
def search(sid: str, body: SearchBody):
    s = _session(sid)
    job = new_job("search")

    def fn(progress):
        spec = FeatureSpec(body.x_uas, body.y_uas, body.r_uas, body.threshold, body.fwhm_uas)
        res = P.run_search(s, spec, frames_every=20, progress=lambda ev: progress({**ev, "stage": "search"}))
        return S.search_payload(s, res, frames=True)

    run_job(job, fn)
    return {"job": job["id"]}


@app.post("/api/session/{sid}/design")
def plan(sid: str, body: DesignBody):
    s = _session(sid)
    if s.search is None:
        raise HTTPException(400, "run a feature test first")
    job = new_job("design")

    def fn(progress):
        progress({"stage": "design", "step": 0, "steps": 1})
        evals = P.run_design(s)
        return S.design_payload(s, evals, body.budget)

    run_job(job, fn)
    return {"job": job["id"]}


@app.post("/api/session/{sid}/retest")
def retest(sid: str, body: RetestBody):
    s = _session(sid)
    ev = next((e for e in (s.evals or []) if e.package.id == body.package_id), None)
    if ev is None:
        raise HTTPException(404, "unknown package")
    if s.kind != "synthetic" and ev.package.template is None:
        raise HTTPException(400, "on real data only nights that were actually observed can be added")
    job = new_job("retest")

    def fn(progress):
        ns = P.retest(s, ev.package, seed=body.seed, progress=lambda e: progress({**e, "stage": "retest"}))
        sessions.setdefault(f"{sid}@base", s)
        sessions[sid] = ns
        return {"package": ev.package.to_dict(), "search": S.search_payload(ns, ns.search)}

    run_job(job, fn)
    return {"job": job["id"]}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404)
    return {"done": job["done"], "error": job["error"], "n_events": len(job["events"]),
            "result": job["result"] if job["done"] else None}


@app.get("/api/jobs/{jid}/stream")
async def job_stream(jid: str):
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404)

    async def gen():
        sent = 0
        while True:
            evs = job["events"]
            while sent < len(evs):
                yield f"event: progress\ndata: {json.dumps(evs[sent], default=_np)}\n\n"
                sent += 1
            if job["done"]:
                if job["error"]:
                    yield f"event: failed\ndata: {json.dumps({'error': job['error']})}\n\n"
                else:
                    yield f"event: result\ndata: {json.dumps(job['result'], default=_np)}\n\n"
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _np(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _session(sid: str) -> P.Session:
    s = sessions.get(sid)
    if s is None:
        raise HTTPException(409, "load the scenario first")
    return s


@app.get("/api/health")
def health():
    return {"ok": True, "mode": "live"}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB), name="web")


def main():
    import argparse
    import webbrowser

    import uvicorn

    ap = argparse.ArgumentParser(description="Anelench live engine")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    print(f"Anelench is running at {url}  (press Ctrl+C to stop)", flush=True)
    if a.open:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
