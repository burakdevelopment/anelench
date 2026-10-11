from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .forward import ImageGrid


def gaussian_prior(grid: ImageGrid, fwhm_uas: float) -> np.ndarray:
    x, y = grid.xy_uas()
    s = fwhm_uas / 2.3548
    p = np.exp(-0.5 * (x**2 + y**2) / s**2)
    return p / p.sum()


def soft_window(grid: ImageGrid, radius_frac: float = 0.85, edge_uas: float = 4.0) -> np.ndarray:
    x, y = grid.xy_uas()
    r = np.hypot(x, y)
    return 1.0 / (1.0 + np.exp((r - radius_frac * grid.fov_uas / 2.0) / edge_uas))


class Family(nn.Module):
    key = "base"

    def __init__(self, batch: int, grid: ImageGrid, flux: float, reg: dict, prior_fwhm_uas: float = 50.0):
        super().__init__()
        self.batch = batch
        self.grid = grid
        self.flux = flux
        self.reg = dict(reg)
        prior = gaussian_prior(grid, prior_fwhm_uas)
        self.register_buffer("prior", torch.tensor(prior.ravel(), dtype=torch.float32))
        self.register_buffer("log_prior", torch.log(self.prior + 1e-30))

    def normalise(self, raw: torch.Tensor) -> torch.Tensor:
        return self.flux * raw / raw.sum(dim=1, keepdim=True)

    def batched_tensors(self):
        shared = {"prior", "log_prior", "window"}
        for name, t in list(self.named_parameters()) + list(self.named_buffers()):
            if name not in shared and t.dim() > 0 and t.shape[0] == self.batch:
                yield name, t

    def export_instance(self, b: int) -> dict:
        return {name: t.detach()[b].clone() for name, t in self.batched_tensors()}

    def import_instance(self, b: int, state: dict) -> None:
        with torch.no_grad():
            for name, t in self.batched_tensors():
                t[b].copy_(state[name])

    def regulariser(self, img: torch.Tensor) -> torch.Tensor:
        p = img / img.sum(dim=1, keepdim=True)
        n = self.grid.npix
        out = torch.zeros(img.shape[0])
        if self.reg.get("mem"):
            kl = (p * (torch.log(p + 1e-30) - self.log_prior)).sum(dim=1)
            out = out + self.reg["mem"] * kl
        if self.reg.get("tsv") or self.reg.get("tv"):
            q = p.view(-1, n, n)
            dx = q[:, :, 1:] - q[:, :, :-1]
            dy = q[:, 1:, :] - q[:, :-1, :]
            if self.reg.get("tsv"):
                out = out + self.reg["tsv"] * (dx.pow(2).sum(dim=(1, 2)) + dy.pow(2).sum(dim=(1, 2))) * n * n
            if self.reg.get("tv"):
                tv = torch.sqrt(dx[:, :-1, :] ** 2 + dy[:, :, :-1] ** 2 + 1e-12).sum(dim=(1, 2))
                out = out + self.reg["tv"] * tv
        return out


def smooth_noise(batch: int, grid: ImageGrid, corr_uas: float, amp: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    n = grid.npix
    k = np.fft.fftfreq(n)
    kk = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    s = corr_uas / grid.psize_uas
    filt = np.exp(-2.0 * (np.pi * s * kk) ** 2)
    out = []
    for _ in range(batch):
        f = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * filt))
        out.append(amp * f.ravel() / (f.std() + 1e-12))
    return torch.tensor(np.array(out), dtype=torch.float32)


class PixelRML(Family):
    key = "pixel"

    def __init__(self, batch, grid, flux, reg, prior_fwhm_uas=50.0, seed=0, init=None, noise=0.25):
        super().__init__(batch, grid, flux, reg, prior_fwhm_uas)
        base = self.log_prior.unsqueeze(0).repeat(batch, 1)
        if init is not None:
            base = torch.log(torch.as_tensor(init, dtype=torch.float32).reshape(batch, -1) + 1e-6 * flux)
        self.theta = nn.Parameter(base + smooth_noise(batch, grid, 20.0, noise, seed))

    def forward(self) -> torch.Tensor:
        return self.flux * torch.softmax(self.theta, dim=1)


class NeuralField(Family):
    key = "field"

    def __init__(self, batch, grid, flux, reg, prior_fwhm_uas=60.0, seed=0, scale=3.0, n_freq=48, hidden=96):
        super().__init__(batch, grid, flux, reg, prior_fwhm_uas)
        g = torch.Generator().manual_seed(seed)
        x, y = grid.xy_uas()
        coords = torch.tensor(np.stack([x.ravel(), y.ravel()], 1) / (grid.fov_uas / 2.0), dtype=torch.float32)
        bmat = torch.randn(batch, 2, n_freq, generator=g) * scale
        proj = 2.0 * math.pi * torch.einsum("pc,bcf->bpf", coords, bmat)
        self.register_buffer("feat", torch.cat([torch.cos(proj), torch.sin(proj)], dim=2) / math.sqrt(n_freq))
        self.register_buffer("window", torch.tensor(soft_window(grid).ravel(), dtype=torch.float32))
        d_in = 2 * n_freq

        def lin(i, o, s=1.0):
            return nn.Parameter(torch.randn(batch, i, o, generator=g) * s * math.sqrt(2.0 / i))

        self.w1, self.b1 = lin(d_in, hidden), nn.Parameter(torch.zeros(batch, 1, hidden))
        self.w2, self.b2 = lin(hidden, hidden), nn.Parameter(torch.zeros(batch, 1, hidden))
        self.w3, self.b3 = lin(hidden, 1, 0.02), nn.Parameter(torch.zeros(batch, 1, 1))

    def forward(self) -> torch.Tensor:
        h = F.silu(torch.bmm(self.feat, self.w1) + self.b1)
        h = F.silu(torch.bmm(h, self.w2) + self.b2)
        out = (torch.bmm(h, self.w3) + self.b3).squeeze(-1)
        raw = torch.exp(torch.clamp(out, -30, 30) + self.log_prior) * self.window
        return self.normalise(raw)


class DeepImagePrior(Family):
    key = "dip"

    def __init__(self, batch, grid, flux, reg, prior_fwhm_uas=60.0, seed=0, ch=16):
        super().__init__(batch, grid, flux, reg, prior_fwhm_uas)
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        n0 = grid.npix // 8
        self.register_buffer("z", 0.5 * torch.randn(1, batch * ch, n0, n0, generator=g))
        self.register_buffer("window", torch.tensor(soft_window(grid).ravel(), dtype=torch.float32))
        self.convs = nn.ModuleList()
        for _ in range(3):
            self.convs.append(nn.Conv2d(batch * ch, batch * ch, 3, padding=1, groups=batch))
            self.convs.append(nn.Conv2d(batch * ch, batch * ch, 3, padding=1, groups=batch))
        self.head = nn.Conv2d(batch * ch, batch, 1, groups=batch)
        self.ch = ch
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.zero_()

    def batched_tensors(self):
        b, c = self.batch, self.ch
        yield "z", self.z.view(b, c, *self.z.shape[2:])
        for i, conv in enumerate(self.convs):
            yield f"w{i}", conv.weight.view(b, c, *conv.weight.shape[1:])
            yield f"b{i}", conv.bias.view(b, c)
        yield "hw", self.head.weight.view(b, 1, c, 1, 1)
        yield "hb", self.head.bias.view(b, 1)

    def forward(self) -> torch.Tensor:
        h = self.z
        for k in range(0, len(self.convs), 2):
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            h = F.leaky_relu(self.convs[k](h), 0.1)
            h = F.leaky_relu(self.convs[k + 1](h), 0.1)
        out = self.head(h).reshape(self.batch, -1)
        raw = torch.exp(torch.clamp(out, -30, 30) + self.log_prior) * self.window
        return self.normalise(raw)


class LearnedPrior(Family):
    key = "latent"

    def __init__(self, batch, grid, flux, reg, prior_fwhm_uas=40.0, seed=0, model="rings"):
        super().__init__(batch, grid, flux, reg, prior_fwhm_uas)
        from .learned import load

        object.__setattr__(self, "_vae", load(model))
        g = torch.Generator().manual_seed(seed)
        z0 = 0.3 * torch.randn(batch, self._vae.latent, generator=g)
        self.z = nn.Parameter(z0)
        with torch.no_grad():
            d0 = self._vae.decode(z0).reshape(batch, -1) + 1e-4
        self.theta = nn.Parameter(torch.log(d0 / d0.sum(dim=1, keepdim=True)))

    def decoded(self) -> torch.Tensor:
        d = self._vae.decode(self.z).reshape(self.batch, -1) + 1e-6
        return d / d.sum(dim=1, keepdim=True)

    def forward(self) -> torch.Tensor:
        return self.flux * torch.softmax(self.theta, dim=1)

    def regulariser(self, img: torch.Tensor) -> torch.Tensor:
        p = img / img.sum(dim=1, keepdim=True)
        fid = ((p - self.decoded()) ** 2).sum(dim=1) * p.shape[1]
        lat = 0.5 * (self.z**2).sum(dim=1)
        return super().regulariser(img) + self.reg.get("fid", 1.0) * fid + self.reg.get("latent", 0.0) * lat


FAMILIES: dict[str, dict] = {
    "rml-mem": dict(cls=PixelRML, lr=1.0, optimizer="lbfgs", reg={"mem": 0.5, "tsv": 0.1}, kw=dict(prior_fwhm_uas=40.0),
                    label="Pixel RML · entropy + smoothness",
                    short="RML-MEM", blurb="Classic regularised maximum likelihood: stay close to a Gaussian prior, penalise roughness."),
    "rml-tv": dict(cls=PixelRML, lr=1.0, optimizer="lbfgs", reg={"tv": 0.6, "mem": 0.2}, kw=dict(prior_fwhm_uas=40.0),
                   label="Pixel RML · total variation",
                   short="RML-TV", blurb="Prefers piecewise-flat images with sharp edges (compressed-sensing style)."),
    "nf-smooth": dict(cls=NeuralField, lr=5e-3, reg={"mem": 0.6, "tsv": 0.1}, kw=dict(scale=1.5, prior_fwhm_uas=40.0),
                      label="Neural field · low-bandwidth Fourier features",
                      short="NF-1.5", blurb="Coordinate MLP; its spectral bias favours smooth structure at the array resolution."),
    "nf-sharp": dict(cls=NeuralField, lr=5e-3, reg={"mem": 0.5, "tsv": 0.05}, kw=dict(scale=3.0, prior_fwhm_uas=40.0),
                     label="Neural field · high-bandwidth Fourier features",
                     short="NF-3.0", blurb="Same MLP with higher-frequency features; free to place compact, super-resolved detail."),
    "dip": dict(cls=DeepImagePrior, lr=4e-3, reg={"tsv": 0.01}, kw=dict(prior_fwhm_uas=45.0),
                label="Deep image prior · CNN decoder",
                short="DIP", blurb="An untrained convolutional decoder: natural-image statistics come from the architecture."),
}


def _register_learned() -> None:
    from . import learned

    specs = {
        "rings": ("vae-rings", "Learned prior · VAE trained on rings only", "VAE-rings",
                  "A generative model trained on 24,000 simulated rings and crescents, none of which has a knot."),
        "knots": ("vae-knots", "Learned prior · VAE trained on rings with knots", "VAE-knots",
                  "The same generative model trained on images half of which carry a compact knot outside the ring."),
    }
    for model, (key, label, short, blurb) in specs.items():
        if learned.available(model):
            FAMILIES[key] = dict(cls=LearnedPrior, lr=1.0, optimizer="lbfgs", reg={"fid": 1.0, "latent": 0.02},
                                 kw=dict(model=model), label=label, short=short, blurb=blurb)


_register_learned()


def make_family(key: str, batch: int, grid: ImageGrid, flux: float, seed: int = 0, init=None) -> tuple[Family, float]:
    spec = FAMILIES[key]
    kw = dict(spec["kw"])
    if init is not None and spec["cls"] is PixelRML:
        kw["init"] = init
    fam = spec["cls"](batch, grid, flux, spec["reg"], seed=seed, **kw)
    return fam, spec["lr"]
