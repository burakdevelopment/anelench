from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anelench.closures import build_closures, closure_phases, log_closure_amps
from anelench.data import CODES, VisData, load_uvfits
from anelench.feature import FeatureFunctional, FeatureSpec
from anelench.forward import ClosureForward, ImageGrid, image_vis
from anelench.geometry import baseline_uv
from anelench.skymodels import Blob, SkyModel

DATA = Path(__file__).resolve().parents[1] / "data" / "eht2017"
GRID = ImageGrid()


def toy_scan(n_st: int = 6, seed: int = 0) -> VisData:
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(n_st, 1)
    n = len(a1)
    u, v = rng.uniform(-8e9, 8e9, n), rng.uniform(-8e9, 8e9, n)
    vis = image_vis(SkyModel(blobs=[Blob(-20, 10, 8, 0.1)]).render(GRID), u, v, GRID)
    z = np.zeros(n)
    return VisData(time=z, ant1=a1, ant2=a2, u=u, v=v, vis=vis, sigma=np.full(n, 1e-3), band=z.astype(int),
                   tint=z + 10, scan=z.astype(int), mjd=57854, ra_deg=187.7, dec_deg=12.4)


def test_closure_counts_are_maximal_independent_sets():
    for n in (4, 5, 6, 7):
        cs = build_closures(toy_scan(n), snr_cut_cp=0, snr_cut_lca=0)
        assert cs.n_cp == (n - 1) * (n - 2) // 2
        assert cs.n_lca == n * (n - 3) // 2


def test_closure_quantities_cancel_station_gains():
    d = toy_scan(6)
    cs = build_closures(d, snr_cut_cp=0, snr_cut_lca=0)
    rng = np.random.default_rng(1)
    g = (1 + 0.3 * rng.standard_normal(6)) * np.exp(2j * np.pi * rng.random(6))
    corrupted = g[d.ant1] * np.conj(g[d.ant2]) * d.vis
    assert np.allclose(np.exp(1j * closure_phases(corrupted, cs)), np.exp(1j * closure_phases(d.vis, cs)), atol=1e-9)
    assert np.allclose(log_closure_amps(corrupted, cs), log_closure_amps(d.vis, cs), atol=1e-9)


def test_forward_model_reproduces_noiseless_closures():
    d = toy_scan(6)
    cs = build_closures(d, snr_cut_cp=0, snr_cut_lca=0)
    img = SkyModel(blobs=[Blob(-20, 10, 8, 0.1)]).render(GRID)
    fwd = ClosureForward(d, cs, GRID)
    chi = fwd.chi2_np(img)
    assert chi["cp"][0] < 1e-2 and chi["lca"][0] < 1e-2
    assert fwd.chi2_np(SkyModel(bright_pa_deg=10).render(GRID))["total"][0] > 1.0


def test_feature_is_scale_invariant_and_sees_the_knot():
    spec = FeatureSpec(-37.6, 13.7, 9.0, 0.2)
    f = FeatureFunctional(spec, GRID)
    ring = SkyModel().render(GRID)
    knot = SkyModel(blobs=[Blob(-37.6, 13.7, 10, 0.08)]).render(GRID)
    assert np.isclose(f.q_np(ring)[0], f.q_np(3.0 * ring)[0], rtol=1e-5)
    assert f.q_np(knot)[0] > f.q_np(ring)[0] + 0.1


@pytest.mark.parametrize("key", ["rml-mem", "nf-smooth", "dip", "vae-rings"])
def test_family_instances_export_and_import(key):
    import torch

    from anelench.families import FAMILIES, make_family

    if key not in FAMILIES:
        pytest.skip("learned prior not trained (scripts/train_priors.py)")
    fam, _ = make_family(key, 3, GRID, 0.6, seed=0)
    with torch.no_grad():
        img = fam()
        assert torch.allclose(img.sum(dim=1), torch.full((3,), 0.6), atol=1e-4)
        assert not torch.allclose(img[0], img[2])
        fam.import_instance(2, fam.export_instance(0))
        img2 = fam()
    assert torch.allclose(img2[0], img2[2], atol=1e-6)
    assert torch.allclose(img2[1], img[1], atol=1e-6)


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/download_eht_data.py first")
def test_uv_from_earth_rotation_matches_the_release():
    d = load_uvfits(DATA / "SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits")
    xyz = d.meta["an_xyz"]
    worst = 0.0
    for a, b in sorted(set(zip(d.ant1, d.ant2))):
        m = (d.ant1 == a) & (d.ant2 == b)
        u, v = baseline_uv(np.array(xyz[CODES[a]]), np.array(xyz[CODES[b]]), d.ra_deg, d.dec_deg, d.mjd, d.time[m], d.freqs[0])
        scale = max(np.hypot(d.u[m], d.v[m]).max(), 1e6)
        worst = max(worst, np.hypot(u - d.u[m], v - d.v[m]).max() / scale)
    assert worst < 1e-3
