# Anelench

**Is that faint bright spot in the black-hole image real, or an artefact of how the image was made?**
Anelench searches for two images that explain the *same* telescope measurements, one with the
feature and one without. When both exist, it computes which feasible joint observation of
Earth's telescopes would tell them apart, and then checks its own recommendation by adding that
night and testing again.

## Quick start

| System | Command |
|---|---|
| Windows | double-click `run.bat` |
| macOS / Linux | `bash run.sh` |

The script finds Python 3.10+, installs the packages into a private `.venv` if they are missing,
downloads the public EHT 2017 data once (about 4.5 MB), starts the engine and opens

http://127.0.0.1:8000. 

The `web/` folder also runs as a static site that replays results precomputed with the same code:

`python -m http.server 8001 --directory web`, then open http://127.0.0.1:8001.

Two documents explain everything:

* [`docs/Anelench_Explained.pdf`](docs/Anelench_Explained.pdf): the project in plain words.
* [`docs/Anelench_Mathematics.pdf`](docs/Anelench_Mathematics.pdf): every formula, proof and algorithm.

## What it does

1. **Mark.** Draw a circle around the structure you doubt and state a claim: "the region is at
   least *t* of the ring's peak brightness", measured at a common 20 µas resolution.
2. **Test.** Under seven imaging assumptions (two pixel methods, two neural fields, a deep image
   prior and two variational autoencoders trained on simulated black holes) Anelench pushes the
   image to remove the feature and to strengthen it, while it keeps fitting the gain-free closure
   measurements within the noise. Both sides survive: *undetermined*, shown as two images and a
   blink comparator.
3. **Plan.** It scores 173 feasible follow-up nights (stations, UT windows, elevation limits,
   per-telescope noise, cost) by the separation of the hardest pair of competing images per unit
   of telescope time, and compares with maximum signal-to-noise, maximum Fourier coverage,
   single-model uncertainty and random choices.
4. **Check.** It adds the night (simulated from the hidden truth, or a real held-back 2017 night)
   and runs the whole search again.

## Why it matters

The EHT samples the sky's Fourier transform at a few hundred points; an algorithm fills in the
rest, and its assumptions shape the fine details (Feng, Bouman & Freeman 2024). Reading physics off
those details is a second inference step (Dahale et al. 2025: ring ellipticity follows the
emitting gas, not the spin). Anelench puts a measurable check between the two.

Uncertainty quantification for radio images (QuantifAI, Liaudat et al. 2024), hypothesis tests on
image structure and goal-oriented optimal design (Attia, Alexanderian & Saibaba 2018) exist
separately. Anelench's contribution is narrower: for one chosen feature it searches for
counterexamples under several assumptions, ties the result to a feasible new EHT measurement, and
checks that choice on measurements it was not shown, including real 2017 nights.

## Results

**Synthetic test A, a knot that is not there** (true region value 0.08, claim "above 0.20"). Five
of the seven assumptions draw a knot (0.23 to 0.37). Anelench: *undetermined*, acceptable images
span 0.04 to 0.48, shown as a twin pair from the classic method (0.12 and 0.34, total misfits
22 units apart over 392 closure measurements). It recommends a three-hour ALMA-led night for 27
weighted telescope-hours, where the maximum-signal, maximum-coverage and single-model strategies
each spend 69. After that night the classic reconstruction moves to the correct side (0.18) and the
range narrows to 0.04 to 0.41; a second planned night gives 0.04 to 0.40.

**Synthetic test B, a faint knot that is there** (true value 0.25). The five hand-built
assumptions see it (0.27 to 0.38); the two trained generative models erase it (0.14 and 0.16), the
learned-prior bias Feng, Bouman & Freeman describe. Anelench: *undetermined*, 0.14 to 0.44. After
the recommended night (2017 array plus GLT and Kitt Peak, cost 34.5) the learned priors move towards
the truth (0.19, 0.18) and the range narrows to 0.18 to 0.38. The low end is held by the knot-free
prior: whether the knot is "seen" still depends on admitting that prior, and Anelench shows it.

**Real M87\*, the western extension** (April 11 2017; claim "at least 0.30 of the ring peak").
All seven reconstructions show it (0.30 to 0.50), so any single pipeline would report it.
Anelench: *undetermined*, 0.16 to 0.71. We held back the other three 2017 nights and ranked them
by predicted twin-pair separation before opening them: April 6 (7.6σ), April 5 (7.0σ), April 10
(4.1σ). Revealed, the real data separated the explanations in exactly that order (median Δχ² 678,
602, 14), and all three favoured the explanation without the bright extension. Re-running the search
with each night narrowed the range most with April 5 (0.23 to 0.43) and least with April 10 (0.17 to
0.65). Caveats: the per-cost criterion picked April 10, the cheapest night that just clears the 3σ
floor and the least informative; and the April 5–6 separations are far larger than predicted, which
we read as the source changing over six days.

**Benchmark: eight hidden truths** (`experiments/benchmark.py`; four with a faint knot near the
jet base, four without; real April 2017 sampling; four of the assumptions, to keep the run within
a few hours; budget 70 weighted telescope-hours; after each night the full search is re-run).

* On the initial data the classic single reconstruction reported a knot in all eight cases, so it
  was wrong for all four knot-free truths. Anelench answered *undetermined* in all eight: it
  confirmed no false claim.
* One extra night of closure data rarely settles the question, whatever the strategy:

| Strategy | Mean cost | Settled correctly | Settled wrongly | Doubt left (range width) |
|---|---|---|---|---|
| Anelench, separation per cost | 29 | 1 of 8 | 0 | 0.23 |
| Anelench, whole budget | 51 | 1 of 8 | 1 of 8 | 0.19 |
| Highest signal-to-noise | 69 | 1 of 8 | 0 | 0.19 |
| Largest new Fourier coverage | 69 | 1 of 8 | 0 | 0.17 |
| Single-model uncertainty | 55 | 1 of 8 | 0 | 0.20 |
| Random feasible night | 38 | 0 of 8 | 0 | 0.28 |

  Anelench's per-cost nights matched the expensive strategies' correct settlements at 42% of their
  telescope time but left somewhat more doubt. The one wrong verdict (a "supported" knot that was
  not there) came from the search missing knot-free images that exist, and the found range
  contained the truth in only about half of the cases: the range is an inner estimate, and
  "supported" means "no counterexample found".

## How it works, briefly

* **Measurements.** Only closure phases and log closure amplitudes, in a maximal independent set
  per scan chosen greedily by noise (optimal by the matroid greedy theorem), so station gain errors
  cancel exactly.
* **Fits as well.** Flexible models also fit noise. Each model's *optimism* is measured by a
  parametric bootstrap; an image is acceptable within the best fit's χ² plus that optimism plus two
  of its standard deviations. The true image is accepted about 97% of the time, and the allowance
  does not grow with the amount of data.
* **Search.** Push the feature down and up behind one-sided penalty walls, from the references
  ("twin" pairs), from images with the region erased or brightened, and, after new data, from the
  previous counterexamples.
* **Planning.** Expected separation of two images on a night's closure data, minimum over all
  witness pairs, divided by cost, within a budget.

Details, derivations and proofs: `docs/Anelench_Mathematics.pdf`.

## Layout

```
run.bat, run.sh  one-step start
anelench/        data, closures, geometry, forward model, image models, feature,
                 search, design, pipeline, live server
web/             the interface (static demo bundles in web/demo/)
models/          the two trained generative priors
experiments/     benchmark
scripts/         data download, demo builder, figures, prior training
tests/           physics checks: python -m pytest tests
docs/            the two PDFs and figures
```

## Limits

* The search is local, so the reported range is an inner estimate: "undetermined" is proven by two
  explicit images; "supported" and "not supported" mean no counterexample was found.
* Closure quantities only: immune to calibration errors, but they discard information that careful
  amplitude calibration would add.
* The night ranking uses a fixed pool of competing images; it is a prediction, checked by re-testing.
* Different 2017 nights are treated as one static source; M87* can change over days.

## Data and references

EHT Collaboration 2019, ApJL 875 L1–L6, data release 2019-D01-01 (downloaded by
`scripts/download_eht_data.py`, not redistributed). Chael et al., eht-imaging. Tancik et al. 2020.
Ulyanov et al. 2018. Kingma & Welling 2014. Feng, Bouman & Freeman 2024. Dahale et al. 2025.
Liaudat et al. 2024. Attia, Alexanderian & Saibaba 2018.
