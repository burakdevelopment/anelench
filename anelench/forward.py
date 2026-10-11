from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .closures import ClosureSet, closure_phases, closure_sigmas, log_closure_amps
from .constants import RAD_PER_UAS
from .data import VisData

torch.set_num_threads(max(1, torch.get_num_threads()))


@dataclass(frozen=True)
class ImageGrid:
    npix: int = 64
    fov_uas: float = 128.0

    @property
    def psize_uas(self) -> float:
        return self.fov_uas / self.npix

    @property
    def psize(self) -> float:
        return self.psize_uas * RAD_PER_UAS

    def axis_uas(self) -> np.ndarray:
        return (self.npix / 2.0 - 0.5 - np.arange(self.npix)) * self.psize_uas

    def xy_uas(self) -> tuple[np.ndarray, np.ndarray]:
        a = self.axis_uas()
        y, x = np.meshgrid(a, a, indexing="ij")
        return x, y


def dft_matrix(u: np.ndarray, v: np.ndarray, grid: ImageGrid) -> tuple[np.ndarray, np.ndarray]:
    x, y = grid.xy_uas()
    x = x.ravel() * RAD_PER_UAS
    y = y.ravel() * RAD_PER_UAS
    phase = -2.0 * np.pi * (np.outer(u, x) + np.outer(v, y))
    return np.cos(phase), np.sin(phase)


def image_vis(img: np.ndarray, u: np.ndarray, v: np.ndarray, grid: ImageGrid) -> np.ndarray:
    c, s = dft_matrix(u, v, grid)
    flat = img.reshape(-1, grid.npix * grid.npix)
    out = flat @ c.T + 1j * (flat @ s.T)
    return out[0] if img.ndim == 2 else out


class ClosureForward:
    def __init__(self, data: VisData, cs: ClosureSet, grid: ImageGrid, dtype=torch.float32):
        self.grid = grid
        self.data = data
        self.cs = cs
        used = np.unique(np.concatenate([cs.cp_idx.ravel(), cs.lca_idx.ravel()]))
        remap = -np.ones(len(data), dtype=int)
        remap[used] = np.arange(len(used))
        self.used = used
        c, s = dft_matrix(data.u[used], data.v[used], grid)
        self.C = torch.tensor(c, dtype=dtype)
        self.S = torch.tensor(s, dtype=dtype)
        self.cp_idx = torch.tensor(remap[cs.cp_idx], dtype=torch.long)
        self.cp_conj = torch.tensor(cs.cp_sign < 0)
        self.lca_idx = torch.tensor(remap[cs.lca_idx], dtype=torch.long)

        s_cp, s_lca = closure_sigmas(data, cs)
        psi = closure_phases(data.vis, cs)
        ell = log_closure_amps(data.vis, cs)
        self.cp_obs = psi
        self.lca_obs = ell
        self.cp_sigma = s_cp
        self.lca_sigma = s_lca
        self.t_cos = torch.tensor(np.cos(psi), dtype=dtype)
        self.t_sin = torch.tensor(np.sin(psi), dtype=dtype)
        self.t_ell = torch.tensor(ell, dtype=dtype)
        self.t_wcp = torch.tensor(1.0 / s_cp**2, dtype=dtype)
        self.t_wlca = torch.tensor(1.0 / s_lca**2, dtype=dtype)
        self.n_cp = cs.n_cp
        self.n_lca = cs.n_lca

    @property
    def n_data(self) -> int:
        return self.n_cp + self.n_lca

    def with_observed(self, psi: np.ndarray, ell: np.ndarray) -> "ClosureForward":
        import copy

        out = copy.copy(self)
        out.cp_obs, out.lca_obs = psi, ell
        out.t_cos = torch.tensor(np.cos(psi), dtype=self.C.dtype)
        out.t_sin = torch.tensor(np.sin(psi), dtype=self.C.dtype)
        out.t_ell = torch.tensor(ell, dtype=self.C.dtype)
        return out

    def vis(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = img.reshape(img.shape[0], -1)
        return flat @ self.C.T, flat @ self.S.T

    def model_closures(self, img: torch.Tensor):
        re, im = self.vis(img)
        r = re[:, self.cp_idx]
        i = torch.where(self.cp_conj, -im[:, self.cp_idx], im[:, self.cp_idx])
        pr = r[..., 0] * r[..., 1] - i[..., 0] * i[..., 1]
        pi = r[..., 0] * i[..., 1] + i[..., 0] * r[..., 1]
        zr = pr * r[..., 2] - pi * i[..., 2]
        zi = pr * i[..., 2] + pi * r[..., 2]
        la = 0.5 * torch.log(re**2 + im**2 + 1e-20)
        ll = la[:, self.lca_idx]
        ell = ll[..., 0] + ll[..., 1] - ll[..., 2] - ll[..., 3]
        return zr, zi, ell

    def chi2(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zr, zi, ell = self.model_closures(img)
        mag = torch.sqrt(zr**2 + zi**2 + 1e-24)
        cosd = (zr * self.t_cos + zi * self.t_sin) / mag
        chi_cp = (2.0 * (1.0 - cosd) * self.t_wcp).mean(dim=1) if self.n_cp else torch.zeros(img.shape[0])
        chi_lca = ((ell - self.t_ell) ** 2 * self.t_wlca).mean(dim=1) if self.n_lca else torch.zeros(img.shape[0])
        return chi_cp, chi_lca

    def chi2_total(self, img: torch.Tensor) -> torch.Tensor:
        a, b = self.chi2(img)
        return (a * self.n_cp + b * self.n_lca) / max(self.n_data, 1)

    def closures_np(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            zr, zi, ell = self.model_closures(torch.tensor(img.reshape(-1, self.grid.npix**2), dtype=self.C.dtype))
        return np.arctan2(zi.numpy(), zr.numpy()), ell.numpy()

    def chi2_np(self, img: np.ndarray) -> dict:
        with torch.no_grad():
            t = torch.tensor(img.reshape(-1, self.grid.npix**2), dtype=self.C.dtype)
            a, b = self.chi2(t)
        tot = (a * self.n_cp + b * self.n_lca) / max(self.n_data, 1)
        return {"cp": a.numpy(), "lca": b.numpy(), "total": tot.numpy()}


def chi2_acceptance(n_data: int, chi2_best: float, n_sigma: float = 2.0) -> float:
    return max(chi2_best, 1.0) * (1.0 + n_sigma * np.sqrt(2.0 / max(n_data, 1)))
