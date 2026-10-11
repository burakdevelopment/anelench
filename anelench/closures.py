from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .data import CODES, VisData


@dataclass
class ClosureSet:
    cp_idx: np.ndarray
    cp_sign: np.ndarray
    cp_scan: np.ndarray
    cp_label: list[str]
    lca_idx: np.ndarray
    lca_scan: np.ndarray
    lca_label: list[str]

    @property
    def n_cp(self) -> int:
        return len(self.cp_idx)

    @property
    def n_lca(self) -> int:
        return len(self.lca_idx)


def _greedy_independent(vectors: list[np.ndarray], order: np.ndarray, tol: float = 1e-8) -> list[int]:
    chosen: list[int] = []
    basis: list[np.ndarray] = []
    for i in order:
        v = vectors[i].astype(float).copy()
        for b in basis:
            v -= (v @ b) * b
        n = np.linalg.norm(v)
        if n > tol:
            basis.append(v / n)
            chosen.append(int(i))
    return chosen


def build_closures(d: VisData, snr_cut_cp: float = 2.0, snr_cut_lca: float = 3.0) -> ClosureSet:
    cp_idx, cp_sign, cp_scan, cp_label = [], [], [], []
    lca_idx, lca_scan, lca_label = [], [], []
    rel = d.sigma / np.maximum(np.abs(d.vis), 1e-30)

    groups = np.unique(np.stack([d.scan, d.band], axis=1), axis=0)
    for scan, band in groups:
        rows = np.where((d.scan == scan) & (d.band == band))[0]
        bl = {(int(d.ant1[r]), int(d.ant2[r])): int(r) for r in rows}
        col = {k: i for i, k in enumerate(bl)}
        stations = sorted({s for k in bl for s in k})

        cands, vecs, noise = [], [], []
        good = {k: r for k, r in bl.items() if d.snr[r] >= snr_cut_cp}
        for a, b, c in combinations(stations, 3):
            if (a, b) in good and (b, c) in good and (a, c) in good:
                r = [good[(a, b)], good[(b, c)], good[(a, c)]]
                vec = np.zeros(len(col))
                vec[col[(a, b)]], vec[col[(b, c)]], vec[col[(a, c)]] = 1, 1, -1
                cands.append((r, (1, 1, -1), f"{CODES[a]}-{CODES[b]}-{CODES[c]}"))
                vecs.append(vec)
                noise.append(np.sqrt(np.sum(rel[r] ** 2)))
        for i in _greedy_independent(vecs, np.argsort(noise)):
            r, s, lab = cands[i]
            cp_idx.append(r); cp_sign.append(s); cp_scan.append(scan); cp_label.append(lab)

        cands, vecs, noise = [], [], []
        good = {k: r for k, r in bl.items() if d.snr[r] >= snr_cut_lca}
        for quad in combinations(stations, 4):
            a, b, c, e = quad
            matchings = [((a, b), (c, e)), ((a, c), (b, e)), ((a, e), (b, c))]
            avail = [m for m in matchings if m[0] in good and m[1] in good]
            for num, den in combinations(avail, 2):
                r = [good[num[0]], good[num[1]], good[den[0]], good[den[1]]]
                vec = np.zeros(len(col))
                vec[col[num[0]]] += 1; vec[col[num[1]]] += 1
                vec[col[den[0]]] -= 1; vec[col[den[1]]] -= 1
                lab = f"{CODES[num[0][0]]}{CODES[num[0][1]]}·{CODES[num[1][0]]}{CODES[num[1][1]]}/{CODES[den[0][0]]}{CODES[den[0][1]]}·{CODES[den[1][0]]}{CODES[den[1][1]]}"
                cands.append((r, lab))
                vecs.append(vec)
                noise.append(np.sqrt(np.sum(rel[r] ** 2)))
        for i in _greedy_independent(vecs, np.argsort(noise)):
            r, lab = cands[i]
            lca_idx.append(r); lca_scan.append(scan); lca_label.append(lab)

    return ClosureSet(
        cp_idx=np.array(cp_idx, dtype=int).reshape(-1, 3), cp_sign=np.array(cp_sign, dtype=int).reshape(-1, 3),
        cp_scan=np.array(cp_scan, dtype=int), cp_label=cp_label,
        lca_idx=np.array(lca_idx, dtype=int).reshape(-1, 4), lca_scan=np.array(lca_scan, dtype=int), lca_label=lca_label,
    )


def closure_phases(vis: np.ndarray, cs: ClosureSet) -> np.ndarray:
    v = vis[cs.cp_idx]
    v = np.where(cs.cp_sign > 0, v, np.conj(v))
    return np.angle(np.prod(v, axis=1))


def log_closure_amps(vis: np.ndarray, cs: ClosureSet) -> np.ndarray:
    la = np.log(np.abs(vis[cs.lca_idx]))
    return la[:, 0] + la[:, 1] - la[:, 2] - la[:, 3]


def closure_sigmas(d: VisData, cs: ClosureSet) -> tuple[np.ndarray, np.ndarray]:
    rel = d.sigma / np.maximum(np.abs(d.vis), 1e-30)
    s_cp = np.sqrt(np.sum(rel[cs.cp_idx] ** 2, axis=1))
    s_lca = np.sqrt(np.sum(rel[cs.lca_idx] ** 2, axis=1))
    return s_cp, s_lca


def closure_sigmas_model(sigma: np.ndarray, amp: np.ndarray, cs: ClosureSet) -> tuple[np.ndarray, np.ndarray]:
    rel = sigma / np.maximum(amp, 1e-30)
    return np.sqrt(np.sum(rel[cs.cp_idx] ** 2, axis=1)), np.sqrt(np.sum(rel[cs.lca_idx] ** 2, axis=1))
