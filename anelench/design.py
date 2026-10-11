from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np

from .closures import ClosureSet, build_closures, closure_phases, closure_sigmas_model, log_closure_amps
from .constants import INTRA_SITE, STATIONS
from .data import CODE_INDEX, CODES, VisData, estimate_sefds, thermal_sigma
from .forward import ImageGrid, image_vis
from .geometry import baseline_uv, elevation_deg

CORE_2017 = ["AA", "AP", "AZ", "JC", "LM", "PV", "SM"]
LATER = ["GL", "KP", "NN"]


@dataclass
class Package:
    id: str
    label: str
    stations: list[str]
    ut_start: float
    ut_end: float
    cost: float
    group: str
    template: VisData | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "stations": self.stations, "ut_start": self.ut_start,
                "ut_end": self.ut_end, "cost": self.cost, "group": self.group, "observed": self.template is not None}


def observed_night_package(thermal: VisData, label: str, pid: str, weights: dict | None = None) -> Package:
    weights = weights or {c: s.cost_weight for c, s in STATIONS.items()}
    codes = sorted({CODES[i] for i in np.union1d(thermal.ant1, thermal.ant2)}, key=CODE_INDEX.get)
    hours = 0.0
    for c in codes:
        i = CODE_INDEX[c]
        t = np.unique(thermal.time[(thermal.ant1 == i) | (thermal.ant2 == i)])
        hours += weights.get(c, 1.0) * len(t) * float(np.median(thermal.tint)) / 3600.0
    return Package(pid, label, codes, float(thermal.time.min()), float(thermal.time.max()), round(hours, 1),
                   "observed 2017 night", template=thermal)


def station_sefds(data: VisData | None) -> dict[str, float]:
    nominal = {c: s.sefd for c, s in STATIONS.items()}
    if data is None:
        return nominal
    measured = estimate_sefds(data)
    ratios = [measured[c] / nominal[c] for c in measured if c in nominal]
    scale = float(np.median(ratios)) if ratios else 1.0
    out = {c: nominal[c] * scale for c in nominal}
    out.update({c: v for c, v in measured.items() if c in nominal})
    return out


def candidate_packages(windows=(3.0, 6.0), starts=tuple(range(0, 24, 2)), weights: dict | None = None) -> list[Package]:
    weights = weights or {c: s.cost_weight for c, s in STATIONS.items()}
    sets = {
        "2017 array": CORE_2017,
        "2017 + GLT": CORE_2017 + ["GL"],
        "2017 + Kitt Peak": CORE_2017 + ["KP"],
        "2017 + NOEMA": CORE_2017 + ["NN"],
        "2017 + GLT + KP": CORE_2017 + ["GL", "KP"],
        "2017 + GLT + NOEMA": CORE_2017 + ["GL", "NN"],
        "Full 2021+ array": CORE_2017 + LATER,
        "North America + GLT": ["AZ", "KP", "LM", "JC", "SM", "GL"],
        "Europe-Americas": ["PV", "NN", "LM", "AZ", "KP", "GL"],
        "ALMA-led southern": ["AA", "AP", "LM", "AZ", "PV", "NN"],
    }
    out = []
    for name, st in sets.items():
        for dur in windows:
            for s0 in starts:
                cost = dur * sum(weights.get(c, 1.0) for c in st)
                out.append(Package(f"{name}|{s0:02d}|{dur:g}", f"{name}, {s0:02d}:00-{(s0 + dur) % 24:02.0f}:00 UT",
                                   list(st), float(s0), float(s0 + dur), float(cost), name))
    return out


def package_template(pkg: Package, mjd: int, ra: float, dec: float, freqs: dict, bandwidth: float,
                     sefd: dict, *, scan_min: float = 6.0, cadence_min: float = 12.0, el_min: float = 10.0) -> VisData:
    times = np.arange(pkg.ut_start + 0.5 * cadence_min / 60.0, pkg.ut_end, cadence_min / 60.0)
    xyz = {c: np.array(STATIONS[c].xyz) for c in pkg.stations}
    up = {c: elevation_deg(xyz[c], ra, dec, mjd, times) >= el_min for c in pkg.stations}
    rows = []
    order = sorted(pkg.stations, key=CODE_INDEX.get)
    for si, t in enumerate(times):
        live = [c for c in order if up[c][si]]
        for a, b in combinations(live, 2):
            if frozenset({a, b}) in INTRA_SITE:
                continue
            sig = thermal_sigma(sefd[a], sefd[b], bandwidth, scan_min * 60.0)
            for band, f in freqs.items():
                u, v = baseline_uv(xyz[a], xyz[b], ra, dec, mjd, np.array([t]), f)
                rows.append((t, CODE_INDEX[a], CODE_INDEX[b], u[0], v[0], sig, band, scan_min * 60.0, si))
    if not rows:
        z = np.zeros(0)
        return VisData(z, z.astype(int), z.astype(int), z, z, z.astype(complex), z, z.astype(int), z, z.astype(int),
                       mjd, ra, dec, freqs=dict(freqs), bandwidth=bandwidth)
    r = np.array(rows, dtype=float)
    return VisData(time=r[:, 0], ant1=r[:, 1].astype(int), ant2=r[:, 2].astype(int), u=r[:, 3], v=r[:, 4],
                   vis=np.zeros(len(r), complex), sigma=r[:, 5], band=r[:, 6].astype(int), tint=r[:, 7],
                   scan=r[:, 8].astype(int), mjd=mjd, ra_deg=ra, dec_deg=dec, freqs=dict(freqs), bandwidth=bandwidth,
                   meta={"package": pkg.id})


@dataclass
class PackageEval:
    package: Package
    n_vis: int
    n_cp: int
    n_lca: int
    d_min: float
    d_median: float
    worst_pair: int | None
    snr_total: float
    uv_new: float
    spread: float
    template: VisData = field(repr=False, default=None)
    closures: ClosureSet = field(repr=False, default=None)

    @property
    def score(self) -> float:
        return self.d_min / self.package.cost if self.package.cost > 0 else 0.0

    def to_dict(self) -> dict:
        return {**self.package.to_dict(), "n_vis": self.n_vis, "n_cp": self.n_cp, "n_lca": self.n_lca,
                "d_min": self.d_min, "d_median": self.d_median, "score": self.score, "snr_total": self.snr_total,
                "uv_new": self.uv_new, "spread": self.spread}


def _closure_values(vis: np.ndarray, cs: ClosureSet):
    return closure_phases(vis, cs), log_closure_amps(vis, cs)


def separation(psi0, ell0, psi1, ell1, s_cp, s_lca) -> float:
    d2 = np.sum(2.0 * (1.0 - np.cos(psi0 - psi1)) / s_cp**2) + np.sum((ell0 - ell1) ** 2 / s_lca**2)
    return float(np.sqrt(d2))


def evaluate_packages(packages: list[Package], grid: ImageGrid, reference: np.ndarray,
                      pairs: list[tuple[np.ndarray, np.ndarray]], existing: VisData, *,
                      sefd: dict, sys_noise: float = 0.02, samples: list[np.ndarray] | None = None,
                      freqs: dict | None = None, bandwidth: float | None = None, mjd: int | None = None) -> list[PackageEval]:
    freqs = freqs or existing.freqs
    bandwidth = bandwidth or existing.bandwidth
    mjd = (existing.mjd + 1) if mjd is None else mjd
    absent = [a for a, _ in pairs]
    present = [p for _, p in pairs]
    imgs = [reference] + absent + present + list(samples or [])
    n_a, n_p = len(absent), len(present)
    stack = np.stack(imgs)
    cell = 1.0 / (grid.fov_uas * np.pi / 180 / 3600 / 1e6)
    ex_uv = np.concatenate([np.stack([existing.u, existing.v], 1), -np.stack([existing.u, existing.v], 1)])
    out = []
    for pkg in packages:
        tpl = pkg.template if pkg.template is not None else package_template(
            pkg, mjd, existing.ra_deg, existing.dec_deg, freqs, bandwidth, sefd)
        if len(tpl) == 0:
            continue
        vis_all = image_vis(stack, tpl.u, tpl.v, grid)
        amp_ref = np.abs(vis_all[0])
        sigma = np.sqrt(tpl.sigma**2 + (sys_noise * amp_ref) ** 2)
        tpl_ref = replace(tpl, vis=vis_all[0], sigma=sigma)
        cs = build_closures(tpl_ref)
        if cs.n_cp + cs.n_lca == 0:
            continue
        s_cp, s_lca = closure_sigmas_model(sigma, amp_ref, cs)
        vals = [_closure_values(v, cs) for v in vis_all]
        ds = []
        for i in range(n_a):
            a, p = vals[1 + i], vals[1 + n_a + i]
            ds.append(separation(a[0], a[1], p[0], p[1], s_cp, s_lca))
        d_min = float(min(ds)) if ds else 0.0
        worst = int(np.argmin(ds)) if ds else None
        spread = 0.0
        if samples:
            sv = vals[1 + n_a + n_p:]
            sp = [separation(x[0], x[1], y[0], y[1], s_cp, s_lca) for x, y in combinations(sv, 2)]
            spread = float(np.mean(sp)) if sp else 0.0
        new_uv = np.stack([tpl.u, tpl.v], 1)
        dmin = np.min(np.linalg.norm(new_uv[:, None, :] - ex_uv[None, :, :], axis=2), axis=1)
        uv_new = float(np.mean(np.clip(dmin / cell, 0.0, 1.0)) * len(new_uv))
        out.append(PackageEval(pkg, len(tpl), cs.n_cp, cs.n_lca, d_min, float(np.median(ds)) if ds else 0.0, worst,
                               float(np.sum(amp_ref / sigma)), uv_new, spread, tpl_ref, cs))
    return out


STRATEGIES = {
    "random": "Random feasible package",
    "max_snr": "Highest total signal-to-noise",
    "uv_coverage": "Largest new Fourier coverage",
    "single_model": "Largest single-model uncertainty",
    "counterexample": "Anelench: separate the hardest witness pair",
    "counterexample_max": "Anelench, whole budget: largest hardest-pair separation",
}


def choose(evals: list[PackageEval], strategy: str, budget: float, rng: np.random.Generator | None = None,
           d_required: float = 3.0) -> PackageEval | None:
    pool = [e for e in evals if e.package.cost <= budget and e.n_cp + e.n_lca > 0]
    if not pool:
        return None
    if strategy == "random":
        rng = rng or np.random.default_rng()
        return pool[int(rng.integers(len(pool)))]
    if strategy == "max_snr":
        return max(pool, key=lambda e: e.snr_total)
    if strategy == "uv_coverage":
        return max(pool, key=lambda e: e.uv_new)
    if strategy == "single_model":
        return max(pool, key=lambda e: e.spread)
    if strategy == "counterexample":
        decisive = [e for e in pool if e.d_min >= d_required]
        return max(decisive, key=lambda e: e.score) if decisive else max(pool, key=lambda e: e.d_min)
    if strategy == "counterexample_max":
        return max(pool, key=lambda e: e.d_min)
    raise ValueError(strategy)


def predicted_signals(ev: PackageEval, grid: ImageGrid, images: dict[str, np.ndarray], top: int = 4) -> dict:
    cs, tpl = ev.closures, ev.template
    s_cp, _ = closure_sigmas_model(tpl.sigma, np.abs(tpl.vis), cs)
    curves = {k: closure_phases(image_vis(v, tpl.u, tpl.v, grid), cs) for k, v in images.items()}
    keys = list(curves)
    diff = np.abs(np.angle(np.exp(1j * (curves[keys[0]] - curves[keys[-1]])))) / s_cp
    tri = {}
    band0 = tpl.band[cs.cp_idx[:, 0]] == tpl.band.min()
    for i, lab in enumerate(cs.cp_label):
        if band0[i]:
            tri.setdefault(lab, []).append(i)
    ranked = sorted(tri, key=lambda l: -np.max(diff[tri[l]]))[:top]
    out = []
    for lab in ranked:
        idx = tri[lab]
        t = tpl.time[cs.cp_idx[idx, 0]]
        o = np.argsort(t)
        idx = [idx[i] for i in o]
        out.append({"triangle": lab, "time": tpl.time[cs.cp_idx[idx, 0]].tolist(), "sigma_deg": np.degrees(s_cp[idx]).tolist(),
                    "curves": {k: np.degrees(curves[k][idx]).tolist() for k in keys},
                    "max_sep_sigma": float(np.max(diff[idx]))})
    return {"triangles": out}
