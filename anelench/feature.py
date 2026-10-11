from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .forward import ImageGrid


@dataclass
class FeatureSpec:
    x_uas: float
    y_uas: float
    r_uas: float
    threshold: float
    fwhm_uas: float = 20.0
    kind: str = "peak_ratio"

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        if self.kind == "peak_ratio":
            return (f"mean brightness inside the {self.r_uas:.0f} uas region relative to the image peak, "
                    f"both at {self.fwhm_uas:.0f} uas resolution")
        return (f"mean brightness inside the region relative to the surrounding annulus, "
                f"at {self.fwhm_uas:.0f} uas resolution")


class FeatureFunctional:
    def __init__(self, spec: FeatureSpec, grid: ImageGrid):
        self.spec = spec
        self.grid = grid
        n = grid.npix
        x, y = grid.xy_uas()
        d = np.hypot(x - spec.x_uas, y - spec.y_uas)
        edge = 0.5 * grid.psize_uas
        self.mask = 1.0 / (1.0 + np.exp((d - spec.r_uas) / edge))
        ann = 1.0 / (1.0 + np.exp((d - 2.0 * spec.r_uas) / edge)) - self.mask
        self.annulus = np.clip(ann, 0.0, 1.0)
        self.t_mask = torch.tensor((self.mask / self.mask.sum()).ravel(), dtype=torch.float32)
        self.t_ann = torch.tensor((self.annulus / max(self.annulus.sum(), 1e-9)).ravel(), dtype=torch.float32)
        m = 2 * n
        k = np.fft.fftfreq(m) * m
        s = spec.fwhm_uas / 2.3548 / grid.psize_uas
        g1 = np.exp(-0.5 * (k / s) ** 2)
        ker = np.outer(g1, g1)
        ker /= ker.sum()
        self.t_kft = torch.tensor(np.fft.rfft2(ker), dtype=torch.complex64)

    def blur(self, img: torch.Tensor) -> torch.Tensor:
        n = self.grid.npix
        x = img.reshape(-1, n, n)
        pad = torch.zeros(x.shape[0], 2 * n, 2 * n, dtype=x.dtype)
        pad[:, :n, :n] = x
        out = torch.fft.irfft2(torch.fft.rfft2(pad) * self.t_kft, s=(2 * n, 2 * n))
        return out[:, :n, :n]

    def q(self, img: torch.Tensor) -> torch.Tensor:
        b = self.blur(img).reshape(img.shape[0], -1)
        inside = b @ self.t_mask
        if self.spec.kind == "peak_ratio":
            return inside / b.amax(dim=1)
        return inside / (b @ self.t_ann + 1e-12)

    def q_np(self, img: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.q(torch.tensor(img.reshape(-1, self.grid.npix**2), dtype=torch.float32)).numpy()

    def blur_np(self, img: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.blur(torch.tensor(img.reshape(-1, self.grid.npix**2), dtype=torch.float32)).numpy()
