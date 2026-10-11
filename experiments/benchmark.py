from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anelench import design, pipeline as P
from anelench.closures import build_closures
from anelench.data import add_systematic_noise
from anelench.feature import FeatureFunctional, FeatureSpec
from anelench.forward import ClosureForward
from anelench.skymodels import random_truth

THRESHOLD = 0.2
BUDGET = 70.0
DECISIVE = 4.0


def side(q: float) -> str:
    return "present" if q > THRESHOLD else "absent"


def run_truth(i: int, with_spot: bool, families: list[str], steps: dict, full: bool) -> dict:
    rng = np.random.default_rng(1000 + i)
    sx, sy = P.spot_xy()
    truth = random_truth(rng, with_spot, (sx, sy), spot_frac_range=(0.05, 0.10))
    s = P.synthetic_session(f"bench-{i}", truth.name, truth, seed=100 + i)
    s.families = list(families)
    spec = FeatureSpec(sx, sy, 9.0, THRESHOLD)
    feat = FeatureFunctional(spec, s.grid)
    q_true = float(feat.q_np(s.truth_img)[0])
    P.run_reference(s, steps=steps["ref"])
    res = P.run_search(s, spec, steps=steps["search"])
    v0 = res.verdict()
    q_ref0 = float(feat.q_np(s.reference_image())[0])
    rec = {"i": i, "with_spot": with_spot, "q_true": q_true, "truth": side(q_true), "initial": v0["code"],
           "initial_interval": [v0["q_min"], v0["q_max"]], "initial_standard": side(q_ref0), "strategies": {}}
    if v0["code"] != "undetermined":
        return rec
    evals = P.run_design(s)
    pairs = s.pairs
    rec["pair_kind"], rec["n_pairs"] = s.pair_kind, len(pairs)
    sign = 1.0 if rec["truth"] == "absent" else -1.0
    warm = {k: r[0].state for k, r in s.refs.items()}
    cache: dict[str, dict] = {}
    pick_rng = np.random.default_rng(7 + i)
    for k in design.STRATEGIES:
        ch = design.choose(evals, k, BUDGET, pick_rng)
        if ch is None:
            continue
        pid = ch.package.id
        if pid not in cache:
            new = P.observe_package(s, ch.package, np.random.default_rng(11 + i))
            night = add_systematic_noise(new, P.SYS_NOISE)
            fw = ClosureForward(night, build_closures(night), s.grid)
            toward = []
            for a, p in pairs:
                d = (float(fw.chi2_np(p.image)["total"][0]) - float(fw.chi2_np(a.image)["total"][0])) * fw.n_data
                toward.append(sign * d)
            ns = P.with_new_data(s, new, pid)
            P.run_reference(ns, steps=steps["ref"], init=warm)
            q_ref = float(feat.q_np(ns.reference_image())[0])
            out = {"toward_truth": toward, "worst": float(min(toward)), "median": float(np.median(toward)),
                   "all_decisive": bool(min(toward) > DECISIVE), "standard": side(q_ref), "q_ref": q_ref}
            if full:
                ns.p_eff = {kk: max(v, s.p_eff.get(kk, 0.0)) for kk, v in ns.p_eff.items()}
                ns.slack = {kk: P.acceptance_slack(p) for kk, p in ns.p_eff.items()}
                P.run_search(ns, spec, steps=steps["search"], carry=P.carried_witnesses(s.search))
                v = ns.search.verdict()
                out.update({"verdict": v["code"], "interval": [v["q_min"], v["q_max"]]})
            cache[pid] = out
        rec["strategies"][k] = {"package": pid, "cost": ch.package.cost, "d_min": ch.d_min, **cache[pid]}
    return rec


def summarise(records: list[dict]) -> dict:
    out = {}
    pending = [r for r in records if r["strategies"]]
    for k in design.STRATEGIES:
        rows = [r for r in pending if k in r["strategies"]]
        if not rows:
            continue
        st = [r["strategies"][k] for r in rows]
        out[k] = {
            "n": len(rows),
            "mean_cost": float(np.mean([x["cost"] for x in st])),
            "all_pairs_resolved": float(np.mean([x["all_decisive"] for x in st])),
            "worst_pair_toward_truth": float(np.median([x["worst"] for x in st])),
            "wrong_standard": float(np.mean([x["standard"] != r["truth"] for x, r in zip(st, rows)])),
            "per_cost_worst": float(np.median([x["worst"] / x["cost"] for x in st])),
        }
        if all("verdict" in x for x in st):
            dec = [{"supported": "present", "rejected": "absent"}.get(x["verdict"]) for x in st]
            out[k].update({
                "wrong_anelench": float(np.mean([d is not None and d != r["truth"] for d, r in zip(dec, rows)])),
                "correct_anelench": float(np.mean([d == r["truth"] for d, r in zip(dec, rows)])),
                "undetermined": float(np.mean([d is None for d in dec])),
                "mean_width": float(np.mean([x["interval"][1] - x["interval"][0] for x in st])),
                "truth_covered": float(np.mean([x["interval"][0] - 0.02 <= r["q_true"] <= x["interval"][1] + 0.02
                                                for x, r in zip(st, rows)])),
            })
    out["_initial"] = {"n": len(records),
                       "undetermined": float(np.mean([r["initial"] == "undetermined" for r in records])),
                       "wrong_standard": float(np.mean([r["initial_standard"] != r["truth"] for r in records])),
                       "decided_wrong": float(np.mean([{"supported": "present", "rejected": "absent"}.get(r["initial"], r["truth"]) != r["truth"] for r in records]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark.json"))
    ap.add_argument("--families", default="rml-mem,rml-tv,nf-smooth,dip")
    ap.add_argument("--quick", action="store_true", help="skip the counterexample search after each night")
    a = ap.parse_args()
    a.full = not a.quick
    steps = {"ref": 600, "search": 400}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out.read_text())["records"] if out.exists() else []
    done = {r["i"] for r in records}
    for i in range(a.start, a.start + a.n):
        if i in done:
            continue
        t0 = time.time()
        r = run_truth(i, with_spot=(i % 2 == 1), families=a.families.split(","), steps=steps, full=a.full)
        records.append(r)
        out.write_text(json.dumps({"threshold": THRESHOLD, "budget": BUDGET, "families": a.families.split(","),
                                   "decisive": DECISIVE, "records": records, "summary": summarise(records)}, indent=1))
        brief = {k: (v.get("verdict"), [round(x, 2) for x in v.get("interval", [])], v["standard"]) for k, v in r["strategies"].items()}
        print(f"truth {i} ({r['truth']}, q={r['q_true']:.2f}) initial={r['initial']} {brief} [{time.time() - t0:.0f}s]", flush=True)
    print(json.dumps(summarise(records), indent=1))


if __name__ == "__main__":
    main()
