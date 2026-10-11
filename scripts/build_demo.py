from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anelench import pipeline as P, serialize as S
from anelench.feature import FeatureSpec
from anelench.server import CACHE, SCENARIOS, build_session

OUT = ROOT / "web" / "demo"
REGIONS = {
    "ghost": dict(x_uas=-37.6, y_uas=13.7, r_uas=9.0, threshold=0.2),
    "knot": dict(x_uas=-37.6, y_uas=13.7, r_uas=9.0, threshold=0.2),
}
BUDGET = 70.0


def _np(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def build(sid: str, region: dict) -> dict:
    t0 = time.time()
    s = build_session(sid)
    P.run_reference(s)
    print(f"[{sid}] reference done ({time.time() - t0:.0f}s)", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{sid}.pkl", "wb") as f:
        pickle.dump(s, f)
    spec = FeatureSpec(region["x_uas"], region["y_uas"], region["r_uas"], region["threshold"])
    res = P.run_search(s, spec, frames_every=20)
    print(f"[{sid}] search: {res.verdict()['code']} q=[{res.q_range()[0]:.3f},{res.q_range()[1]:.3f}] ({time.time() - t0:.0f}s)", flush=True)
    bundle = {"id": sid, "region": region, "session": S.session_payload(s), "search": S.search_payload(s, res, frames=True)}
    evals = P.run_design(s)
    bundle["design"] = S.design_payload(s, evals, BUDGET)
    print(f"[{sid}] design: {len(evals)} packages ({time.time() - t0:.0f}s)", flush=True)
    rec = bundle["design"].get("recommended")
    if rec is not None and s.kind == "synthetic":
        ev = next(e for e in evals if e.package.id == rec["id"])
        ns = P.retest(s, ev.package, seed=11)
        bundle["retest"] = {"package": ev.package.to_dict(), "search": S.search_payload(ns, ns.search)}
        v = ns.search.verdict()
        print(f"[{sid}] retest: {v['code']} q=[{v['q_min']:.3f},{v['q_max']:.3f}] ({time.time() - t0:.0f}s)", flush=True)
        if v["code"] == "undetermined":
            evals2 = P.run_design(ns)
            d2 = S.design_payload(ns, evals2, BUDGET)
            rec2 = d2.get("recommended")
            if rec2 is not None:
                ev2 = next(e for e in evals2 if e.package.id == rec2["id"])
                ns2 = P.retest(ns, ev2.package, seed=12)
                bundle["retest2"] = {"package": ev2.package.to_dict(), "design": {"recommended": {
                    k: rec2[k] for k in ("id", "label", "stations", "cost", "d_min", "ut_start", "ut_end")}},
                    "search": S.search_payload(ns2, ns2.search)}
                v2 = ns2.search.verdict()
                print(f"[{sid}] retest2: {v2['code']} q=[{v2['q_min']:.3f},{v2['q_max']:.3f}] ({time.time() - t0:.0f}s)", flush=True)
    if s.kind == "real":
        bundle["retests"] = {}
        for o in bundle["design"].get("observed", []):
            ev = next(e for e in evals if e.package.id == o["id"])
            ns = P.retest(s, ev.package)
            bundle["retests"][o["id"]] = {"package": ev.package.to_dict(), "search": S.search_payload(ns, ns.search)}
            v = ns.search.verdict()
            print(f"[{sid}] {o['id']}: {v['code']} q=[{v['q_min']:.3f},{v['q_max']:.3f}] ({time.time() - t0:.0f}s)", flush=True)
        if bundle["retests"]:
            bundle["retest"] = bundle["retests"][bundle["design"]["observed"][0]["id"]]
    return bundle


def main(ids: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        region = REGIONS.get(sid) or json.loads((OUT / f"{sid}.region.json").read_text())
        bundle = build(sid, region)
        (OUT / f"{sid}.json").write_text(json.dumps(bundle, default=_np, separators=(",", ":")))
        print(f"[{sid}] wrote {OUT / (sid + '.json')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SCENARIOS))
