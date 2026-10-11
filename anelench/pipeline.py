from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import design
from .closures import build_closures
from .data import VisData, add_systematic_noise, concat, drop_intra_site, load_eht2017, prepare_for_imaging, scan_average
from .feature import FeatureFunctional, FeatureSpec
from .forward import ClosureForward, ImageGrid
from .search import (Explanation, ProgressFn, SearchResult, acceptance_from_reference, acceptance_slack,
                     counterexample_search, optimism, reference_images)
from .simulate import observe
from .skymodels import Blob, SkyModel, polar

from .families import FAMILIES

DEFAULT_FAMILIES = ["rml-mem", "rml-tv", "nf-smooth", "nf-sharp", "dip"] + [
    k for k in ("vae-rings", "vae-knots") if k in FAMILIES]
SYS_NOISE = 0.02
FLUX = 0.6


@dataclass
class Session:
    id: str
    title: str
    kind: str
    grid: ImageGrid
    thermal: VisData
    data: VisData
    sefd: dict
    truth: SkyModel | None = None
    truth_img: np.ndarray | None = None
    families: list[str] = field(default_factory=lambda: list(DEFAULT_FAMILIES))
    refs: dict | None = None
    chi2_max: float = float("nan")
    p_eff: dict = field(default_factory=dict)
    slack: dict = field(default_factory=dict)
    pairs: list = field(default_factory=list)
    pair_kind: str = ""
    feature: FeatureFunctional | None = None
    search: SearchResult | None = None
    evals: list | None = None
    history: list = field(default_factory=list)
    seed: int = 0
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        self._refresh()

    def _refresh(self):
        self.closures = build_closures(self.data)
        self.fwd = ClosureForward(self.data, self.closures, self.grid)

    @property
    def default_family(self) -> str:
        return "rml-mem"

    def reference_image(self, family: str | None = None) -> np.ndarray:
        return self.refs[family or self.default_family][0].image


def coverage_2017(day: int = 101) -> tuple[VisData, VisData]:
    raw = load_eht2017(day)
    thermal = drop_intra_site(scan_average(raw))
    return raw, thermal


def synthetic_session(sid: str, title: str, truth: SkyModel, seed: int = 0, day: int = 101,
                      grid: ImageGrid | None = None) -> Session:
    grid = grid or ImageGrid()
    raw, thermal = coverage_2017(day)
    img = truth.render(grid)
    obs_thermal = observe(img, grid, thermal, np.random.default_rng(seed))
    data = add_systematic_noise(obs_thermal, SYS_NOISE)
    sefd = design.station_sefds(raw)
    return Session(sid, title, "synthetic", grid, obs_thermal, data, sefd, truth, img, seed=seed)


def real_session(day: int = 101, grid: ImageGrid | None = None) -> Session:
    grid = grid or ImageGrid()
    raw = load_eht2017(day)
    thermal = drop_intra_site(scan_average(raw))
    data = prepare_for_imaging(raw, SYS_NOISE)
    sefd = design.station_sefds(raw)
    dates = {95: "April 5", 96: "April 6", 100: "April 10", 101: "April 11"}
    return Session(f"m87-2017-{day}", f"M87*, EHT 2017, {dates.get(day, day)} (real data)", "real", grid,
                   thermal, data, sefd, notes={"day": day})


SPOT_PA, SPOT_R = 290.0, 40.0


def spot_xy() -> tuple[float, float]:
    return polar(SPOT_R, SPOT_PA)


def scenario_truth(with_spot: bool, spot_frac: float = 0.08) -> SkyModel:
    sx, sy = spot_xy()
    blobs = [Blob(sx, sy, 10.0, spot_frac)] if with_spot else []
    return SkyModel(blobs=blobs, name="ring + faint spot" if with_spot else "ring only")


def run_reference(s: Session, *, restarts: int = 2, steps: int = 900, progress: ProgressFn | None = None,
                  init: dict | None = None) -> None:
    s.refs = reference_images(s.fwd, s.grid, FLUX, s.families, restarts=restarts, steps=steps, seed=s.seed,
                              progress=progress, init=init)
    s.chi2_max = acceptance_from_reference(s.fwd, s.refs)
    s.p_eff = {k: optimism(k, s.fwd, s.grid, FLUX, r[0], seed=s.seed) for k, r in s.refs.items()}
    s.slack = {k: acceptance_slack(p) for k, p in s.p_eff.items()}


def run_search(s: Session, spec: FeatureSpec, *, starts: int = 2, steps: int = 600, frames_every: int = 0,
               progress: ProgressFn | None = None, carry: list | None = None) -> SearchResult:
    s.feature = FeatureFunctional(spec, s.grid)
    s.search = counterexample_search(s.fwd, s.grid, FLUX, s.feature, s.refs, s.chi2_max, slack=s.slack,
                                     starts=starts, steps=steps, frames_every=frames_every, progress=progress,
                                     carry=carry, seed=s.seed + 1)
    return s.search


def single_model_samples(s: Session, family: str | None = None) -> list[np.ndarray]:
    fam = family or s.default_family
    return [e.image for e in s.refs[fam]]


def witness_pairs(res: SearchResult, max_pairs: int = 12) -> tuple[list[tuple[Explanation, Explanation]], str]:
    t = res.threshold
    by: dict = {}
    for e in res.explanations:
        if e.feasible and e.direction != 0:
            by.setdefault((e.family, e.start), {})[e.direction] = e
    twins = [(d[-1], d[+1]) for d in by.values() if -1 in d and +1 in d and d[-1].q < t < d[+1].q]
    if twins:
        return twins[:max_pairs], "twins"
    cross = [(a, p) for a in res.absent() for p in res.present()]
    cross.sort(key=lambda ap: ap[1].q - ap[0].q)
    return cross[:max_pairs], "cross"


def run_design(s: Session, *, packages: list | None = None) -> list:
    pairs, kind = witness_pairs(s.search)
    s.pairs, s.pair_kind = pairs, kind
    if not pairs:
        s.evals = []
        return s.evals
    pk = packages or design.candidate_packages()
    if packages is None and s.kind == "real" and "day" in s.notes:
        pk = pk + observed_nights(s.notes["day"])
    s.evals = design.evaluate_packages(pk, s.grid, s.reference_image(), [(a.image, p.image) for a, p in pairs],
                                       s.data, sefd=s.sefd, sys_noise=SYS_NOISE, samples=single_model_samples(s))
    return s.evals


def observe_package(s: Session, pkg: design.Package, rng: np.random.Generator, truth_img: np.ndarray | None = None) -> VisData:
    tpl = design.package_template(pkg, s.data.mjd + 1, s.data.ra_deg, s.data.dec_deg, s.data.freqs,
                                  s.data.bandwidth, s.sefd)
    img = truth_img if truth_img is not None else s.truth_img
    return observe(img, s.grid, tpl, rng)


def with_new_data(s: Session, new_thermal: VisData, label: str) -> Session:
    thermal = concat([s.thermal, new_thermal])
    data = add_systematic_noise(thermal, SYS_NOISE)
    ns = Session(s.id, s.title, s.kind, s.grid, thermal, data, s.sefd, s.truth, s.truth_img, list(s.families),
                 seed=s.seed + 7, history=s.history + [label])
    return ns


def retest(s: Session, pkg: design.Package, *, seed: int = 99, progress: ProgressFn | None = None,
           restarts: int = 2, steps: int = 900, search_steps: int = 600) -> Session:
    new = pkg.template if pkg.template is not None and pkg.template.vis.any() else observe_package(
        s, pkg, np.random.default_rng(seed))
    ns = with_new_data(s, new, pkg.id)
    warm = {k: r[0].state for k, r in (s.refs or {}).items() if r and r[0].state is not None}
    run_reference(ns, restarts=restarts, steps=steps, progress=progress, init=warm)
    ns.p_eff = {k: max(v, s.p_eff.get(k, 0.0)) for k, v in ns.p_eff.items()}
    ns.slack = {k: acceptance_slack(p) for k, p in ns.p_eff.items()}
    run_search(ns, s.feature.spec, steps=search_steps, progress=progress, carry=carried_witnesses(s.search))
    return ns


def carried_witnesses(res: SearchResult | None) -> list[Explanation]:
    if res is None:
        return []
    best: dict = {}
    for e in res.explanations:
        if not e.feasible or e.direction == 0 or e.state is None:
            continue
        key = (e.family, e.direction)
        if key not in best or (e.q - best[key].q) * e.direction > 0:
            best[key] = e
    return list(best.values())


DAYS_2017 = {95: "April 5", 96: "April 6", 100: "April 10", 101: "April 11"}


def observed_nights(exclude: int) -> list[design.Package]:
    out = []
    for day, name in DAYS_2017.items():
        if day == exclude:
            continue
        _, thermal = coverage_2017(day)
        out.append(design.observed_night_package(thermal, f"Observed 2017 night, {name}", f"night-{day}"))
    return out
