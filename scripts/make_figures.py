from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "web" / "demo"
OUT = ROOT / "docs" / "figures"
INK, PAPER, A, B, PLAN = "#14213d", "#eef1f4", "#2457c5", "#d4570b", "#0e7c6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#7b889c",
                     "axes.labelcolor": INK, "xtick.color": "#485870", "ytick.color": "#485870"})


def img(b64: str, n: int = 64) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(n, n)


def plate(ax, im, fov, region=None, title=None, vmax=None, color="white"):
    ax.imshow(im, cmap="afmhot", vmin=0, vmax=vmax or im.max(), extent=[fov / 2, -fov / 2, -fov / 2, fov / 2])
    if region:
        ax.add_patch(plt.Circle((region["x_uas"], region["y_uas"]), region["r_uas"], fill=False, ls="--", lw=1.2, color=color))
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def pair_figure(bundle: dict, name: str) -> None:
    s, r = bundle["session"], bundle["search"]
    fov = s["grid"]["fov_uas"]
    if "pair" not in r:
        return
    a, p = img(r["pair"]["absent"]["image"]), img(r["pair"]["present"]["image"])
    vmax = max(a.max(), p.max())
    ref = img(s["references"][s["default_family"]][0]["image"])
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), facecolor=PAPER, gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    plate(ax[0], ref, fov, r["feature"], "Reconstruction (RML-MEM)")
    plate(ax[1], a, fov, r["feature"], f"Without: q = {r['pair']['absent']['q']:.2f}", vmax)
    plate(ax[2], p, fov, r["feature"], f"With: q = {r['pair']['present']['q']:.2f}", vmax)
    for k, c in ((1, A), (2, B)):
        for sp in ax[k].spines.values():
            sp.set_color(c); sp.set_linewidth(3)
    tri = r["fit"][0]
    t = np.array(tri["time"])
    ax[3].errorbar(t, tri["obs"], yerr=tri["sigma"], fmt="o", ms=3, color=INK, lw=1, label="measured")
    for k, c, lab in (("absent", A, "without"), ("present", B, "with")):
        m = np.array(tri["models"][k])
        m = m + 360 * np.round((np.array(tri["obs"]) - m) / 360)
        ax[3].plot(t, m, color=c, lw=2, label=lab)
    ax[3].set_xlabel("UT (h)"); ax[3].set_ylabel("closure phase (deg)")
    ax[3].set_title(f"Same data, triangle {tri['triangle']}", fontsize=10, color=INK, loc="left")
    ax[3].legend(frameon=False, fontsize=9)
    ax[3].set_facecolor(PAPER)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_pair.png", dpi=160, facecolor=PAPER)
    plt.close(fig)


def interval_figure(bundle: dict, name: str) -> None:
    r = bundle["search"]
    rows = [("Before", r["verdict"])]
    for i, key in enumerate(("retest", "retest2")):
        if key in bundle:
            rows.append((f"After night {i + 1}", bundle[key]["search"]["verdict"]))
    t = r["feature"]["threshold"]
    fig, ax = plt.subplots(figsize=(7, 0.9 + 0.55 * len(rows)), facecolor=PAPER)
    for j, (lab, v) in enumerate(rows[::-1]):
        lo, hi = v["q_min"], v["q_max"]
        if lo < t:
            ax.barh(j, min(hi, t) - lo, left=lo, color=A, height=0.5)
        if hi > t:
            ax.barh(j, hi - max(lo, t), left=max(lo, t), color=B, height=0.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([x[0] for x in rows[::-1]])
    ax.axvline(t, color=INK, ls="--", lw=1)
    truth = (bundle.get("retest") or {}).get("search", {}).get("truth_q")
    if truth is not None:
        ax.plot([truth], [-0.45], marker="D", color=PLAN, ms=7, clip_on=False)
        ax.text(truth, -0.85, " truth", color=PLAN, va="center", fontsize=9)
    ax.set_xlim(0, max(0.6, max(v["q_max"] for _, v in rows) + 0.05))
    ax.set_xlabel("region brightness / ring peak (acceptable range)")
    ax.set_facecolor(PAPER)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_intervals.png", dpi=160, facecolor=PAPER)
    plt.close(fig)


def priors_figure() -> None:
    import torch

    sys.path.insert(0, str(ROOT))
    from anelench import learned

    names = [n for n in ("rings", "knots") if learned.available(n)]
    if not names:
        return
    g = torch.Generator().manual_seed(4)
    fig, ax = plt.subplots(len(names), 8, figsize=(12, 1.7 * len(names) + 0.3), facecolor=PAPER)
    ax = np.atleast_2d(ax)
    for i, n in enumerate(names):
        vae = learned.load(n)
        with torch.no_grad():
            ims = vae.decode(torch.randn(8, vae.latent, generator=g)).numpy()
        for j in range(8):
            ax[i, j].imshow(ims[j], cmap="afmhot", vmin=0)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(f"VAE-{n}", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "learned_priors.png", dpi=150, facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    priors_figure()
    for f in sorted(DEMO.glob("*.json")):
        if f.name.endswith(".region.json") or f.name == "benchmark.json":
            continue
        b = json.loads(f.read_text())
        pair_figure(b, f.stem)
        interval_figure(b, f.stem)
        print("figures for", f.stem)


if __name__ == "__main__":
    sys.exit(main())
