from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .families import FAMILIES, make_family
from .feature import FeatureFunctional
from .forward import ClosureForward, ImageGrid, chi2_acceptance

ProgressFn = Callable[[dict], None]


@dataclass
class Explanation:
    family: str
    direction: int
    seed: int
    image: np.ndarray
    chi2: float
    chi2_cp: float
    chi2_lca: float
    reg: float
    q: float = float("nan")
    feasible: bool = True
    start: int = 0
    frames: list = field(default_factory=list)
    state: dict | None = None

    def summary(self) -> dict:
        return {"family": self.family, "direction": self.direction, "seed": self.seed, "chi2": self.chi2,
                "chi2_cp": self.chi2_cp, "chi2_lca": self.chi2_lca, "reg": self.reg, "q": self.q,
                "feasible": self.feasible, "start": self.start}


def run_family(key: str, fwd: ClosureForward, grid: ImageGrid, flux: float, directions: list[int], *,
               feature: FeatureFunctional | None = None, chi2_max: float | list[float] = np.inf,
               reg_max: list[float] | None = None, steps: int = 900, warmup: int = 450, seed: int = 0,
               init_states: list[dict] | None = None, starts: list[int] | None = None, beta: float = 8.0,
               frames_every: int = 0, progress: ProgressFn | None = None, label: str = "",
               keep_state: bool = False) -> list[Explanation]:
    batch = len(directions)
    torch.manual_seed(seed)
    fam, lr = make_family(key, batch, grid, flux, seed=seed)
    if init_states is not None:
        for b, st in enumerate(init_states):
            fam.import_instance(b, st)
    lbfgs = FAMILIES[key].get("optimizer") == "lbfgs"
    inner = 10 if lbfgs else 1
    n_outer = max(1, steps // inner)
    warm_outer = warmup // inner
    sched = None
    if not lbfgs:
        warm_start = init_states is not None
        opt = torch.optim.Adam(fam.parameters(), lr=lr * (0.3 if warm_start else 1.0))
        if warm_start:
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 60.0))
    dirs = torch.tensor(directions, dtype=torch.float32)
    active = dirs != 0
    rmax = torch.tensor(reg_max if reg_max is not None else [math.inf] * batch, dtype=torch.float32)
    cmax = np.broadcast_to(np.asarray(chi2_max, dtype=float), (batch,)).copy()
    wall = torch.tensor(np.where(np.isfinite(cmax), cmax - 0.5 / fwd.n_data, np.inf), dtype=torch.float32)
    has_wall = torch.isfinite(wall) & active
    wall = torch.where(has_wall, wall, torch.zeros_like(wall))
    has_rwall = torch.isfinite(rmax) & active
    rwall = torch.where(has_rwall, rmax * 0.97, torch.ones_like(rmax))
    kappa = 50.0
    ramp_len = max(1, (n_outer - warm_outer) // 4)

    best_img: list = [None] * batch
    best = [dict(q=(-math.inf if d > 0 else math.inf), chi=math.inf, cp=math.inf, lca=math.inf, reg=math.inf)
            for d in directions]
    frames: list[list[np.ndarray]] = [[] for _ in range(batch)]
    t0 = time.time()
    ctl = {"pushing": False, "ramp": 0.0, "end_of_fit": False}

    def evaluate():
        img = fam()
        chi_cp, chi_lca = fwd.chi2(img)
        chi = (chi_cp * fwd.n_cp + chi_lca * fwd.n_lca) / fwd.n_data
        reg = fam.regulariser(img)
        q = feature.q(img) if feature is not None else torch.zeros(batch)
        loss_b = chi + reg
        if ctl["pushing"]:
            over = torch.relu((chi - wall) * fwd.n_data / 4.0) * has_wall
            rover = torch.relu((reg - rwall) / rwall) * has_rwall
            loss_b = loss_b - beta * ctl["ramp"] * dirs * q + 0.5 * kappa * (over**2 + rover**2)
        return loss_b.sum(), (img, chi_cp, chi_lca, chi, reg, q)

    def track(aux):
        img, chi_cp, chi_lca, chi, reg, q = aux
        chi_np, q_np, reg_np = chi.detach().numpy(), q.detach().numpy(), reg.detach().numpy()
        for b in range(batch):
            d = directions[b]
            ok = chi_np[b] <= cmax[b] and reg_np[b] <= float(rmax[b])
            if d == 0:
                keep = chi_np[b] < best[b]["chi"]
            elif ctl["pushing"]:
                keep = ok and ((d > 0 and q_np[b] > best[b]["q"]) or (d < 0 and q_np[b] < best[b]["q"]))
            else:
                keep = ctl["end_of_fit"] and ok
            if keep:
                best[b] = dict(q=float(q_np[b]), chi=float(chi_np[b]), cp=float(chi_cp[b]),
                               lca=float(chi_lca[b]), reg=float(reg_np[b]))
                best_img[b] = img[b].detach().numpy().copy()

    for k in range(n_outer):
        ctl["pushing"] = feature is not None and k >= warm_outer
        ctl["ramp"] = min(1.0, (k - warm_outer + 1) / ramp_len) if ctl["pushing"] else 0.0
        ctl["end_of_fit"] = k == warm_outer - 1
        if lbfgs:
            opt = torch.optim.LBFGS(fam.parameters(), lr=1.0, max_iter=inner, history_size=10,
                                    line_search_fn="strong_wolfe")
            last = {}

            def closure():
                opt.zero_grad(set_to_none=True)
                loss, aux = evaluate()
                loss.backward()
                with torch.no_grad():
                    track(aux)
                last["aux"] = aux
                return loss

            opt.step(closure)
            aux = last["aux"]
        else:
            loss, aux = evaluate()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            with torch.no_grad():
                track(aux)

        with torch.no_grad():
            img, chi_cp, chi_lca, chi, reg, q = aux
            chi, reg = chi.detach(), reg.detach()
            it = k * inner
            if frames_every and (it % frames_every < inner or k == n_outer - 1):
                for b in range(batch):
                    frames[b].append(img[b].detach().numpy().reshape(grid.npix, grid.npix).astype(np.float32))
            if progress is not None and (it % 20 < inner or k == n_outer - 1):
                progress({"family": key, "label": label, "step": it, "steps": steps, "elapsed": time.time() - t0,
                          "chi2": chi.numpy().tolist(), "q": q.detach().numpy().tolist(),
                          "directions": list(directions), "phase": "push" if ctl["pushing"] else "fit",
                          "images": img.detach().numpy().reshape(batch, grid.npix, grid.npix)})

    final = fam().detach()
    out = []
    for b in range(batch):
        d = directions[b]
        if d == 0 or best_img[b] is None:
            img_b = final[b].numpy()
            parts = fwd.chi2_np(img_b)
            with torch.no_grad():
                reg_b = float(fam.regulariser(final)[b])
            rec = dict(chi=float(parts["total"][0]), cp=float(parts["cp"][0]), lca=float(parts["lca"][0]), reg=reg_b,
                       q=float(feature.q_np(img_b)[0]) if feature is not None else float("nan"))
            feasible = d == 0 or (rec["chi"] <= cmax[b] and reg_b <= float(rmax[b]))
        else:
            img_b, rec, feasible = best_img[b], best[b], True
        out.append(Explanation(
            key, d, seed * 100 + b, img_b.reshape(grid.npix, grid.npix), rec["chi"], rec["cp"], rec["lca"], rec["reg"],
            rec["q"], feasible, (starts[b] if starts else b), frames[b],
            fam.export_instance(b) if keep_state else None))
    return out


def optimism(key: str, fwd: ClosureForward, grid: ImageGrid, flux: float, ref: Explanation, *,
             steps: int = 400, seed: int = 0) -> float:
    rng = np.random.default_rng(1000 + seed)
    psi, ell = fwd.closures_np(ref.image)
    fb = fwd.with_observed(psi[0] + rng.normal(0.0, fwd.cp_sigma), ell[0] + rng.normal(0.0, fwd.lca_sigma))
    chi_truth = float(fb.chi2_np(ref.image)["total"][0])
    refit = run_family(key, fb, grid, flux, [0], steps=steps, warmup=steps, seed=seed, init_states=[ref.state])
    return max(0.0, (chi_truth - refit[0].chi2) * fb.n_data)


def acceptance_slack(p_eff: float) -> float:
    p = max(p_eff, 2.0)
    return p + 2.0 * math.sqrt(2.0 * p)


def reference_images(fwd: ClosureForward, grid: ImageGrid, flux: float, families: list[str], *, restarts: int = 2,
                     steps: int = 900, seed: int = 0, progress: ProgressFn | None = None,
                     init: dict[str, dict] | None = None) -> dict[str, list[Explanation]]:
    refs = {}
    for i, k in enumerate(families):
        res = run_family(k, fwd, grid, flux, [0] * restarts, steps=steps, warmup=steps, seed=seed + 31 * i,
                         progress=progress, label="reference", keep_state=True)
        if init and k in init:
            res += run_family(k, fwd, grid, flux, [0], steps=steps, warmup=steps, seed=seed + 31 * i + 7,
                              progress=progress, label="reference", keep_state=True, init_states=[init[k]])
        refs[k] = sorted(res, key=lambda e: e.chi2)
    return refs


@dataclass
class SearchResult:
    explanations: list[Explanation]
    chi2_max: float
    threshold: float
    reference_chi2: float = float("nan")
    delta_chi2: float = 4.0
    n_data: int = 0

    def absent(self) -> list[Explanation]:
        return [e for e in self.explanations if e.direction < 0 and e.feasible and e.q < self.threshold]

    def present(self) -> list[Explanation]:
        return [e for e in self.explanations if e.direction > 0 and e.feasible and e.q > self.threshold]

    def q_range(self) -> tuple[float, float]:
        feas = [e.q for e in self.explanations if e.feasible and e.direction != 0]
        return (min(feas), max(feas)) if feas else (float("nan"), float("nan"))

    def verdict(self) -> dict:
        a, p = self.absent(), self.present()
        lo, hi = self.q_range()
        if a and p:
            code, text = "undetermined", "The current measurements are explained equally well with and without the feature."
        elif p:
            code, text = "supported", "No data-consistent explanation without the feature was found under any tested assumption."
        elif a:
            code, text = "rejected", "No data-consistent explanation with the feature was found under any tested assumption."
        else:
            code, text = "inconclusive", "The search did not reach acceptable fits on either side of the threshold."
        return {"code": code, "text": text, "q_min": lo, "q_max": hi, "n_absent": len(a), "n_present": len(p),
                "threshold": self.threshold, "chi2_max": self.chi2_max, "delta_chi2": self.delta_chi2,
                "n_data": self.n_data}

    def witness_pair(self) -> tuple[Explanation, Explanation] | None:
        a, p = self.absent(), self.present()
        if not (a and p):
            return None
        return min(a, key=lambda e: e.q), max(p, key=lambda e: e.q)

    def pool(self) -> list[tuple[Explanation, Explanation]]:
        return [(a, p) for a in self.absent() for p in self.present()]


SEEDED = 100
CARRIED = 200


def _soft_region(grid: ImageGrid, x: float, y: float, r: float, edge: float = 2.5) -> np.ndarray:
    xx, yy = grid.xy_uas()
    return 1.0 / (1.0 + np.exp((np.hypot(xx - x, yy - y) - r) / edge))


def counterexample_search(fwd: ClosureForward, grid: ImageGrid, flux: float, feature: FeatureFunctional,
                          refs: dict[str, list[Explanation]], chi2_gof: float, *, slack: dict[str, float],
                          starts: int = 2, steps: int = 600, warmup: int = 30, reg_slack: float = 3.0,
                          seeded: bool = True, carry: list[Explanation] | None = None, seed: int = 1,
                          frames_every: int = 0, progress: ProgressFn | None = None) -> SearchResult:
    expl: list[Explanation] = []
    n = fwd.n_data
    best_key = min(refs, key=lambda k: refs[k][0].chi2)
    ceiling = min(chi2_gof, refs[best_key][0].chi2 + slack[best_key] / n)
    for i, (k, rlist) in enumerate(refs.items()):
        chosen = [r for r in rlist[:starts] if r.chi2 <= ceiling] or rlist[:1]
        directions, states, rmax, cmax, start_ids = [], [], [], [], []
        for j, r in enumerate(chosen):
            for d in (-1, +1):
                directions.append(d)
                states.append(r.state)
                rmax.append(max(r.reg, 1e-6) * (1.0 + reg_slack) + 1e-4)
                cmax.append(ceiling)
                start_ids.append(j)
        expl += run_family(k, fwd, grid, flux, directions, feature=feature, chi2_max=cmax, reg_max=rmax,
                           steps=steps, warmup=warmup, seed=seed + 17 * i, init_states=states, starts=start_ids,
                           frames_every=frames_every, progress=progress, label="search", keep_state=True)
        old = [e for e in (carry or []) if e.family == k and e.state is not None]
        if old:
            r_lim = max(chosen[0].reg, 1e-6) * (1.0 + reg_slack) + 1e-4
            expl += run_family(k, fwd, grid, flux, [e.direction for e in old], feature=feature,
                               chi2_max=[ceiling] * len(old), reg_max=[r_lim] * len(old), steps=steps, warmup=0,
                               seed=seed + 17 * i + 9, init_states=[e.state for e in old],
                               starts=[CARRIED] * len(old), progress=progress, label="search", keep_state=True)
        if seeded and FAMILIES[k]["cls"].key == "pixel":
            ref = chosen[0]
            m = _soft_region(grid, feature.spec.x_uas, feature.spec.y_uas, feature.spec.r_uas + 3.0)
            edits = [ref.image * (1.0 - 0.95 * m), ref.image * (1.0 + 1.5 * m)]
            seeds = [{"theta": torch.log(torch.tensor((e / e.sum() * flux).ravel(), dtype=torch.float32) + 1e-9 * flux)}
                     for e in edits]
            r_lim = max(ref.reg, 1e-6) * (1.0 + reg_slack) + 1e-4
            expl += run_family(k, fwd, grid, flux, [-1, +1], feature=feature, chi2_max=[ceiling] * 2,
                               reg_max=[r_lim] * 2, steps=steps, warmup=0, seed=seed + 17 * i + 5,
                               init_states=seeds, starts=[SEEDED, SEEDED], frames_every=frames_every,
                               progress=progress, label="search", keep_state=True)
    best_ref = min(r[0].chi2 for r in refs.values())
    return SearchResult(expl, ceiling, feature.spec.threshold, best_ref, float(slack[best_key]), n)


def acceptance_from_reference(fwd: ClosureForward, refs: dict[str, list[Explanation]], n_sigma: float = 2.0) -> float:
    best = min(r[0].chi2 for r in refs.values())
    return float(chi2_acceptance(fwd.n_data, best, n_sigma))


__all__ = ["Explanation", "SearchResult", "run_family", "reference_images", "counterexample_search",
           "acceptance_from_reference", "FAMILIES"]
