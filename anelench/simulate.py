from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import VisData
from .forward import ImageGrid, image_vis


def observe(img: np.ndarray, grid: ImageGrid, template: VisData, rng: np.random.Generator, *,
            gain_amp: float = 0.1, nonclosing: float = 0.02, thermal: bool = True) -> VisData:
    model = image_vis(img, template.u, template.v, grid)
    keys = np.unique(np.concatenate([np.stack([template.scan, template.ant1], 1),
                                     np.stack([template.scan, template.ant2], 1)]), axis=0)
    gains = {(int(s), int(a)): (1.0 + gain_amp * rng.standard_normal()) * np.exp(2j * np.pi * rng.random())
             for s, a in keys}
    g1 = np.array([gains[(int(s), int(a))] for s, a in zip(template.scan, template.ant1)])
    g2 = np.array([gains[(int(s), int(a))] for s, a in zip(template.scan, template.ant2)])
    vis = g1 * np.conj(g2) * model
    if nonclosing:
        vis = vis * (1.0 + nonclosing * (rng.standard_normal(len(vis)) + 1j * rng.standard_normal(len(vis))))
    if thermal:
        vis = vis + template.sigma * (rng.standard_normal(len(vis)) + 1j * rng.standard_normal(len(vis)))
    return replace(template, vis=vis)
