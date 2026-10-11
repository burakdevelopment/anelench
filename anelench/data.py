from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .constants import ETA_QUANT, INTRA_SITE

CODES = ["AA", "AP", "AZ", "JC", "LM", "PV", "SM", "SR", "GL", "KP", "NN"]
CODE_INDEX = {c: i for i, c in enumerate(CODES)}


@dataclass
class VisData:
    time: np.ndarray
    ant1: np.ndarray
    ant2: np.ndarray
    u: np.ndarray
    v: np.ndarray
    vis: np.ndarray
    sigma: np.ndarray
    band: np.ndarray
    tint: np.ndarray
    scan: np.ndarray
    mjd: int
    ra_deg: float
    dec_deg: float
    source: str = "M87"
    freqs: dict = field(default_factory=dict)
    bandwidth: float = 1.856e9
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.time)

    def select(self, mask: np.ndarray) -> "VisData":
        mask = np.asarray(mask)
        arrays = {k: getattr(self, k)[mask] for k in ("time", "ant1", "ant2", "u", "v", "vis", "sigma", "band", "tint", "scan")}
        return replace(self, **arrays)

    @property
    def snr(self) -> np.ndarray:
        return np.abs(self.vis) / self.sigma

    @property
    def station_codes(self) -> list[str]:
        idx = np.union1d(self.ant1, self.ant2)
        return [CODES[i] for i in idx]

    def baseline_codes(self) -> list[str]:
        return [f"{CODES[a]}-{CODES[b]}" for a, b in zip(self.ant1, self.ant2)]

    def uvdist(self) -> np.ndarray:
        return np.hypot(self.u, self.v)


def concat(datasets: list[VisData]) -> VisData:
    base = datasets[0]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("time", "ant1", "ant2", "u", "v", "vis", "sigma", "band", "tint", "scan")}
    offset = 0
    freqs = {}
    for d in datasets:
        for k in parts:
            arr = getattr(d, k)
            parts[k].append(arr + offset if k == "scan" else arr)
        offset += int(d.scan.max()) + 1 if len(d) else 0
        freqs.update(d.freqs)
    merged = {k: np.concatenate(v) for k, v in parts.items()}
    return replace(base, freqs=freqs, **merged)


def load_uvfits(path: str | Path, band: int = 0) -> VisData:
    from astropy.io import fits

    with fits.open(path) as hdul:
        hdr = hdul[0].header
        gd = hdul[0].data
        an = hdul["AIPS AN"].data
        an_names = [n.strip() for n in an["ANNAME"]]
        nosta = list(an["NOSTA"])
        freq = float(hdr["CRVAL4"])
        bw = float(hdul["AIPS FQ"].data["TOTAL BANDWIDTH"][0])
        mjd = int(round(float(hdr["MJD"])))
        jd = np.asarray(gd.par("DATE"), dtype=float)
        uu = np.asarray(gd.par("UU---SIN"), dtype=float) * freq
        vv = np.asarray(gd.par("VV---SIN"), dtype=float) * freq
        blcode = np.asarray(gd.par("BASELINE"), dtype=float).astype(int)
        tint = np.asarray(gd.par("INTTIM"), dtype=float)
        raw = np.asarray(gd.data, dtype=float).reshape(len(jd), -1, 3)
        source = str(hdr.get("OBJECT", "M87"))
        ra, dec = float(hdr["OBSRA"]), float(hdr["OBSDEC"])

    rr = raw[:, 0, :]
    weight = rr[:, 2]
    ok = np.isfinite(weight) & (weight > 0)
    vis = np.conj(rr[:, 0] + 1j * rr[:, 1])
    sigma_rr = np.where(ok, 1.0 / np.sqrt(np.where(ok, weight, 1.0)), np.inf)
    sigma = sigma_rr / math.sqrt(2.0)

    num_to_name = {n: name for n, name in zip(nosta, an_names)}
    a1_name = [num_to_name[b // 256] for b in blcode]
    a2_name = [num_to_name[b % 256] for b in blcode]
    a1 = np.array([CODE_INDEX[n] for n in a1_name])
    a2 = np.array([CODE_INDEX[n] for n in a2_name])

    swap = a1 > a2
    a1, a2 = np.where(swap, a2, a1), np.where(swap, a1, a2)
    uu = np.where(swap, -uu, uu)
    vv = np.where(swap, -vv, vv)
    vis = np.where(swap, np.conj(vis), vis)

    time = (jd - (mjd + 2400000.5)) * 24.0
    keep = ok & (a1 != a2) & np.isfinite(vis.real)
    keep &= ~np.isin(a1, [CODE_INDEX["SR"]]) & ~np.isin(a2, [CODE_INDEX["SR"]])

    d = VisData(
        time=time[keep], ant1=a1[keep], ant2=a2[keep], u=uu[keep], v=vv[keep], vis=vis[keep],
        sigma=sigma[keep], band=np.full(keep.sum(), band), tint=tint[keep], scan=np.zeros(keep.sum(), int),
        mjd=mjd, ra_deg=ra, dec_deg=dec, source=source, freqs={band: freq}, bandwidth=bw,
        meta={"file": Path(path).name, "an_xyz": {n: tuple(map(float, xyz)) for n, xyz in zip(an_names, an["STABXYZ"])}},
    )
    return assign_scans(d)


def assign_scans(d: VisData, gap_s: float = 90.0) -> VisData:
    if len(d) == 0:
        return d
    ut = np.unique(d.time)
    breaks = np.diff(ut) * 3600.0 > gap_s
    label = np.concatenate([[0], np.cumsum(breaks)])
    scan = label[np.searchsorted(ut, d.time)]
    return replace(d, scan=scan.astype(int))


def scan_average(d: VisData) -> VisData:
    key = np.stack([d.scan, d.band, d.ant1, d.ant2], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    w = 1.0 / d.sigma**2
    sw = np.bincount(inv, w)
    vis = (np.bincount(inv, w * d.vis.real) + 1j * np.bincount(inv, w * d.vis.imag)) / sw
    u = np.bincount(inv, w * d.u) / sw
    v = np.bincount(inv, w * d.v) / sw
    tint = np.bincount(inv, d.tint)
    sigma = 1.0 / np.sqrt(sw)
    scan_t = {s: 0.5 * (d.time[d.scan == s].min() + d.time[d.scan == s].max()) for s in np.unique(d.scan)}
    time = np.array([scan_t[s] for s in uniq[:, 0]])
    return replace(
        d, time=time, ant1=uniq[:, 2], ant2=uniq[:, 3], u=u, v=v, vis=vis, sigma=sigma,
        band=uniq[:, 1], tint=tint, scan=uniq[:, 0],
    )


def drop_intra_site(d: VisData) -> VisData:
    names = [frozenset({_code(a), _code(b)}) for a, b in zip(d.ant1, d.ant2)]
    keep = np.array([n not in INTRA_SITE for n in names])
    return d.select(keep)


def add_systematic_noise(d: VisData, frac: float) -> VisData:
    sigma = np.sqrt(d.sigma**2 + (frac * np.abs(d.vis)) ** 2)
    return replace(d, sigma=sigma)


def estimate_sefds(d: VisData) -> dict[str, float]:
    target = np.log(d.sigma**2 * d.tint * ETA_QUANT**2 * 4.0 * d.bandwidth)
    pairs = sorted({(a, b) for a, b in zip(d.ant1, d.ant2)})
    stations = sorted({s for p in pairs for s in p})
    col = {s: i for i, s in enumerate(stations)}
    rows, rhs = [], []
    for a, b in pairs:
        m = (d.ant1 == a) & (d.ant2 == b)
        row = np.zeros(len(stations))
        row[col[a]] = row[col[b]] = 1.0
        rows.append(row)
        rhs.append(np.median(target[m]))
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return {_code(s): float(np.exp(x)) for s, x in zip(stations, sol)}


def thermal_sigma(sefd1: float, sefd2: float, bandwidth: float, tint: float) -> float:
    return math.sqrt(sefd1 * sefd2 / (4.0 * bandwidth * tint)) / ETA_QUANT


def _code(i: int) -> str:
    return CODES[int(i)]


def load_eht2017(day: int = 101, root: str | Path | None = None, bands: tuple[str, ...] = ("lo", "hi")) -> VisData:
    root = Path(root) if root else Path(__file__).resolve().parents[1] / "data" / "eht2017"
    parts = []
    for i, b in enumerate(bands):
        f = root / f"SR1_M87_2017_{day:03d}_{b}_hops_netcal_StokesI.uvfits"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found - run scripts/download_eht_data.py first")
        parts.append(load_uvfits(f, band=i))
    merged = concat(parts)
    return assign_scans(merged)


def prepare_for_imaging(d: VisData, sys_noise: float = 0.02) -> VisData:
    d = scan_average(d)
    d = drop_intra_site(d)
    d = add_systematic_noise(d, sys_noise)
    return d
