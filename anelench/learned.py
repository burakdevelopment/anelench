from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .forward import ImageGrid
from .skymodels import Blob, SkyModel, polar

MODELS = Path(__file__).resolve().parents[1] / "models"
LATENT = 24


def training_image(rng: np.random.Generator, grid: ImageGrid, knot_prob: float) -> np.ndarray:
    m = SkyModel(diameter_uas=rng.uniform(32, 50), width_uas=rng.uniform(6, 15), asym=rng.uniform(0.0, 0.9),
                 bright_pa_deg=rng.uniform(0, 360), floor=rng.uniform(0.0, 0.15))
    if rng.random() < knot_prob:
        x, y = polar(rng.uniform(26, 52), rng.uniform(0, 360))
        m.blobs.append(Blob(x, y, rng.uniform(6, 12), rng.uniform(0.02, 0.15)))
    img = m.render(grid)
    shift = rng.normal(0, 2.0, size=2)
    img = np.roll(img, (int(round(shift[0] / grid.psize_uas)), int(round(shift[1] / grid.psize_uas))), axis=(0, 1))
    return (img / img.max()).astype(np.float32)


class VAE(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 96, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.Flatten(), nn.Linear(96 * 16, 2 * latent))
        self.fc = nn.Linear(latent, 96 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.fc(z), 0.1).view(-1, 96, 4, 4)
        return torch.sigmoid(self.dec(h)).squeeze(1)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.enc(x.unsqueeze(1)).chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


def train(name: str, knot_prob: float, *, n_images: int = 24000, epochs: int = 12, seed: int = 0,
          grid: ImageGrid | None = None, log=print) -> Path:
    grid = grid or ImageGrid()
    rng = np.random.default_rng(seed)
    data = torch.tensor(np.stack([training_image(rng, grid, knot_prob) for _ in range(n_images)]))
    torch.manual_seed(seed)
    vae = VAE()
    opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(n_images)
        tot = 0.0
        beta = min(1.0, (ep + 1) / 4) * 0.5
        for i in range(0, n_images, 128):
            x = data[perm[i:i + 128]]
            rec, mu, logvar = vae(x)
            recon = F.binary_cross_entropy(rec, x, reduction="sum") / len(x)
            kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / len(x)
            loss = recon + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sched.step()
        log(f"[{name}] epoch {ep + 1}/{epochs} loss {tot / n_images:.2f}")
    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / f"vae_{name}.pt"
    torch.save({"state": vae.state_dict(), "latent": vae.latent, "knot_prob": knot_prob, "n_images": n_images,
                "epochs": epochs}, out)
    return out


_cache: dict[str, VAE] = {}


def load(name: str) -> VAE:
    if name not in _cache:
        ck = torch.load(MODELS / f"vae_{name}.pt", map_location="cpu", weights_only=True)
        vae = VAE(ck["latent"])
        vae.load_state_dict(ck["state"])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        _cache[name] = vae
    return _cache[name]


def available(name: str) -> bool:
    return (MODELS / f"vae_{name}.pt").exists()
