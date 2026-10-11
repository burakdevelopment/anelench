from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .forward import ImageGrid


@dataclass
class Blob:
    x_uas: float
    y_uas: float
    fwhm_uas: float
    flux_frac: float


@dataclass
class SkyModel:
    diameter_uas: float = 42.0
    width_uas: float = 9.0
    asym: float = 0.55
    bright_pa_deg: float = 190.0
    floor: float = 0.04
    blobs: list[Blob] = field(default_factory=list)
    total_flux: float = 0.6
    name: str = "ring"

    def render(self, grid: ImageGrid) -> np.ndarray:
        x, y = grid.xy_uas()
        r = np.hypot(x, y)
        theta = np.arctan2(x, y)
        r0 = self.diameter_uas / 2.0
        sig = self.width_uas / 2.3548
        radial = np.exp(-0.5 * ((r - r0) / sig) ** 2)
        az = 1.0 + self.asym * np.cos(theta - np.radians(self.bright_pa_deg))
        ring = radial * az
        inside = self.floor * np.exp(-0.5 * (r / (0.8 * r0)) ** 4) * ring.max()
        base = ring + inside
        base = base / base.sum()
        blob_frac = sum(b.flux_frac for b in self.blobs)
        img = (1.0 - blob_frac) * base
        for b in self.blobs:
            s = b.fwhm_uas / 2.3548
            g = np.exp(-0.5 * ((x - b.x_uas) ** 2 + (y - b.y_uas) ** 2) / s**2)
            img = img + b.flux_frac * g / g.sum()
        return self.total_flux * img / img.sum()

    def to_dict(self) -> dict:
        return asdict(self)


def polar(r_uas: float, pa_deg: float) -> tuple[float, float]:
    return r_uas * np.sin(np.radians(pa_deg)), r_uas * np.cos(np.radians(pa_deg))


def random_truth(rng: np.random.Generator, with_spot: bool, spot_xy: tuple[float, float],
                 spot_frac_range=(0.06, 0.12)) -> SkyModel:
    m = SkyModel(
        diameter_uas=rng.uniform(38, 46),
        width_uas=rng.uniform(7, 12),
        asym=rng.uniform(0.3, 0.8),
        bright_pa_deg=rng.uniform(140, 230),
        floor=rng.uniform(0.0, 0.1),
        name="ring+spot" if with_spot else "ring",
    )
    if with_spot:
        jitter = rng.normal(0, 2.0, size=2)
        m.blobs.append(Blob(spot_xy[0] + jitter[0], spot_xy[1] + jitter[1], rng.uniform(8, 12), rng.uniform(*spot_frac_range)))
    return m
