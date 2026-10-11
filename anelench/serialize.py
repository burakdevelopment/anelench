from __future__ import annotations

import base64

import numpy as np

from . import design
from .constants import STATIONS
from .data import CODES
from .families import FAMILIES
from .feature import FeatureFunctional
from .forward import ImageGrid
from .geometry import elevation_deg, subsource_point
from .search import Explanation, SearchResult


def img64(img: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(img, dtype=np.float32).tobytes()).decode("ascii")


def _f(x, nd=4):
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def families_meta(keys: list[str]) -> list[dict]:
    return [{"key": k, "label": FAMILIES[k]["label"], "short": FAMILIES[k]["short"], "blurb": FAMILIES[k]["blurb"]}
            for k in keys]


def stations_meta(sefd: dict) -> list[dict]:
    out = []
    for c, s in STATIONS.items():
        lat, lon = s.lat_lon
        out.append({"code": c, "name": s.name, "location": s.location, "lat": _f(lat, 3), "lon": _f(lon, 3),
                    "sefd": _f(sefd.get(c, s.sefd), 0), "in2017": s.in_2017, "joined": s.joined,
                    "cost_weight": s.cost_weight})
    return out


def explanation(e: Explanation, grid: ImageGrid, feature: FeatureFunctional | None = None, n_data: int = 0) -> dict:
    d = {"family": e.family, "short": FAMILIES[e.family]["short"], "direction": e.direction, "start": e.start,
         "chi2": _f(e.chi2), "chi2_cp": _f(e.chi2_cp), "chi2_lca": _f(e.chi2_lca), "chi2_total": _f(e.chi2 * n_data, 2),
         "reg": _f(e.reg, 5), "q": _f(e.q), "feasible": bool(e.feasible), "image": img64(e.image)}
    return d


def closure_plot(fwd, images: dict[str, np.ndarray], top: int = 3) -> list[dict]:
    cs, data = fwd.cs, fwd.data
    tri: dict[str, list[int]] = {}
    band0 = data.band[cs.cp_idx[:, 0]] == data.band.min()
    for i, lab in enumerate(cs.cp_label):
        if band0[i]:
            tri.setdefault(lab, []).append(i)
    models = {k: fwd.closures_np(v)[0][0] for k, v in images.items()}
    keys = list(models)
    if len(keys) >= 2:
        diff = np.abs(np.angle(np.exp(1j * (models[keys[0]] - models[keys[1]])))) / fwd.cp_sigma
        score = {l: float(np.max(diff[ix])) + 0.02 * len(ix) for l, ix in tri.items()}
    else:
        score = {l: float(len(ix)) for l, ix in tri.items()}
    ranked = sorted((l for l in tri if len(tri[l]) >= 4), key=lambda l: -score[l])[:top]
    out = []
    for lab in ranked:
        ix = np.array(tri[lab])
        t = data.time[cs.cp_idx[ix, 0]]
        o = np.argsort(t)
        ix, t = ix[o], t[o]
        out.append({"triangle": lab, "time": [_f(x, 3) for x in t],
                    "obs": [_f(np.degrees(x), 2) for x in fwd.cp_obs[ix]],
                    "sigma": [_f(np.degrees(x), 2) for x in fwd.cp_sigma[ix]],
                    "band": [int(data.band[cs.cp_idx[i, 0]]) for i in ix],
                    "models": {k: [_f(np.degrees(x), 2) for x in m[ix]] for k, m in models.items()}})
    return out


def session_payload(s) -> dict:
    d, grid = s.data, s.grid
    refs = {k: [explanation(e, grid, n_data=s.fwd.n_data) for e in r] for k, r in s.refs.items()}
    stations_in = sorted(set(d.station_codes))
    t = np.linspace(d.time.min() - 1, d.time.max() + 1, 80)
    vis_track = {c: [_f(x, 1) for x in elevation_deg(np.array(STATIONS[c].xyz), d.ra_deg, d.dec_deg, d.mjd, t)]
                 for c in STATIONS}
    lat, lon = subsource_point(d.ra_deg, d.dec_deg, d.mjd, t)
    payload = {
        "id": s.id, "title": s.title, "kind": s.kind,
        "grid": {"npix": grid.npix, "fov_uas": grid.fov_uas},
        "families": families_meta(s.families),
        "default_family": s.default_family,
        "references": refs,
        "chi2_gof": _f(s.chi2_max), "p_eff": {k: _f(v, 1) for k, v in s.p_eff.items()},
        "slack": {k: _f(v, 1) for k, v in s.slack.items()},
        "n_cp": s.closures.n_cp, "n_lca": s.closures.n_lca, "n_vis": len(d),
        "uv": {"u": [_f(x / 1e9, 3) for x in d.u], "v": [_f(x / 1e9, 3) for x in d.v],
               "baseline": d.baseline_codes(), "time": [_f(x, 3) for x in d.time]},
        "time_range": [_f(d.time.min(), 3), _f(d.time.max(), 3)],
        "stations": stations_meta(s.sefd), "stations_in_data": stations_in,
        "elevation": {"t": [_f(x, 3) for x in t], "el": vis_track},
        "subsource": {"t": [_f(x, 3) for x in t], "lat": [_f(x, 2) for x in lat], "lon": [_f(x, 2) for x in lon]},
        "source": {"name": d.source, "ra": d.ra_deg, "dec": d.dec_deg, "mjd": d.mjd},
        "history": s.history,
        "notes": s.notes,
    }
    if s.kind == "synthetic" and s.truth_img is not None:
        payload["truth"] = {"image": img64(s.truth_img), "model": s.truth.to_dict(), "name": s.truth.name}
    return payload


def display_pair(s, res: SearchResult):
    from .pipeline import witness_pairs

    pairs, kind = witness_pairs(res)
    if not pairs:
        return None
    if kind == "twins":
        pref = [p for p in pairs if p[0].family == s.default_family] or pairs
        return max(pref, key=lambda ap: ap[1].q - ap[0].q)
    return res.witness_pair()


def search_payload(s, res: SearchResult, frames: bool = False) -> dict:
    grid, feat = s.grid, s.feature
    pair = display_pair(s, res)
    verdict = res.verdict()
    expl = [explanation(e, grid, feat, s.fwd.n_data) for e in res.explanations]
    by_family = {}
    for e in res.explanations:
        if e.feasible:
            lo, hi = by_family.get(e.family, (np.inf, -np.inf))
            by_family[e.family] = (min(lo, e.q), max(hi, e.q))
    ref_q = {k: _f(feat.q_np(r[0].image)[0]) for k, r in s.refs.items()}
    out = {"feature": feat.spec.to_dict(), "verdict": {k: (_f(v) if isinstance(v, float) else v) for k, v in verdict.items()},
           "explanations": expl, "intervals": {k: [_f(a), _f(b)] for k, (a, b) in by_family.items()},
           "reference_q": ref_q, "n_data": s.fwd.n_data}
    if pair is not None:
        a, p = pair
        out["pair"] = {"absent": explanation(a, grid, feat, s.fwd.n_data), "present": explanation(p, grid, feat, s.fwd.n_data),
                       "delta_chi2_total": _f((p.chi2 - a.chi2) * s.fwd.n_data, 2)}
        out["fit"] = closure_plot(s.fwd, {"absent": a.image, "present": p.image})
        if frames and a.frames and p.frames:
            out["frames"] = {"absent": [img64(f) for f in a.frames], "present": [img64(f) for f in p.frames]}
    else:
        ref = s.reference_image()
        out["fit"] = closure_plot(s.fwd, {"reference": ref})
    if s.kind == "synthetic" and s.truth_img is not None:
        out["truth_q"] = _f(feat.q_np(s.truth_img)[0])
    return out


def package_geometry(ev: design.PackageEval) -> dict:
    tpl = ev.template
    bl = sorted({(CODES[a], CODES[b]) for a, b in zip(tpl.ant1, tpl.ant2)})
    return {"baselines": [list(x) for x in bl], "u": [_f(x / 1e9, 3) for x in tpl.u], "v": [_f(x / 1e9, 3) for x in tpl.v],
            "time": [_f(x, 3) for x in tpl.time]}


def revealed_separation(s, ev: design.PackageEval) -> dict:
    from .closures import build_closures
    from .data import add_systematic_noise
    from .forward import ClosureForward

    real = add_systematic_noise(ev.package.template, 0.02)
    fw = ClosureForward(real, build_closures(real), s.grid)
    deltas, fits = [], []
    for a, p in s.pairs:
        ca = float(fw.chi2_np(a.image)["total"][0])
        cp = float(fw.chi2_np(p.image)["total"][0])
        deltas.append((cp - ca) * fw.n_data)
        fits.append([ca, cp])
    return {"delta_chi2": [_f(x, 1) for x in deltas], "median_abs": _f(float(np.median(np.abs(deltas))), 1),
            "prefers": "without" if np.median(deltas) > 0 else "with", "n_data": fw.n_data,
            "reduced_chi2": [[_f(x, 3) for x in f] for f in fits]}


def design_payload(s, evals: list, budget: float, top: int = 12, rng_seed: int = 0) -> dict:
    if not evals:
        return {"available": False, "reason": "The current data already decide this feature; no follow-up is needed."}
    rng = np.random.default_rng(rng_seed)
    picks = {}
    for k in design.STRATEGIES:
        ch = design.choose(evals, k, budget, rng)
        if ch is not None:
            picks[k] = {"id": ch.package.id, "d_min": _f(ch.d_min, 2), "cost": ch.package.cost, "label": ch.package.label}
    within = [e for e in evals if e.package.cost <= budget]
    ranked = sorted(within, key=lambda e: (e.d_min < 3.0, -e.score if e.d_min >= 3.0 else -e.d_min))[:top]
    rec = next((e for e in evals if e.package.id == picks.get("counterexample", {}).get("id")), None)
    out = {"available": True, "budget": budget, "strategies": design.STRATEGIES, "picks": picks,
           "packages": [{**e.to_dict(), "geometry": package_geometry(e)} for e in ranked],
           "all": [{"id": e.package.id, "cost": e.package.cost, "d_min": _f(e.d_min, 2), "group": e.package.group,
                    "label": e.package.label} for e in evals],
           "pool": {"pairs": len(s.pairs), "kind": s.pair_kind}}
    if rec is not None:
        a, p = s.pairs[rec.worst_pair]
        out["recommended"] = {**rec.to_dict(), "geometry": package_geometry(rec),
                              "signals": design.predicted_signals(rec, s.grid, {"absent": a.image, "present": p.image}),
                              "worst_pair": {"absent_family": a.family, "present_family": p.family}}
    observed = [e for e in evals if e.package.template is not None]
    if observed:
        out["observed"] = [{**e.to_dict(), "revealed": revealed_separation(s, e)}
                           for e in sorted(observed, key=lambda e: -e.d_min)]
    return out
