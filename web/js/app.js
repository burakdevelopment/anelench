import { api } from "./api.js";
import { decode, drawPlate, drawOverlay, featureQ, maxOf, toSky } from "./plates.js";
import { intervalsChart, fitCharts, scatterChart, signalsChart, beforeAfterChart, uvChart, strategyBars } from "./charts.js";
import { Globe } from "./globe.js";

const $ = (s) => document.querySelector(s);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const LEDES = {
  ghost: "A synthetic M87*-like source, measured exactly where and when the 2017 Event Horizon Telescope array measured M87*, with realistic station gains and noise. The truth stays hidden until the end.",
  knot: "A second synthetic source with the same 2017 sampling, gains and noise. Its truth also stays hidden until you have tested a feature and planned the follow-up.",
  "m87-2017-101": "The real, network-calibrated EHT data of M87* from April 11 2017 (both 2 GHz bands), imaged under every assumption with closure quantities only.",
};
const DEFAULTS = {
  ghost: { x: -37.6, y: 13.7, r: 9, t: 0.2 },
  knot: { x: -37.6, y: 13.7, r: 9, t: 0.2 },
  "m87-2017-101": { x: -36, y: -6, r: 9, t: 0.3 },
};
const DEFAULT_REGION = { x: -37.6, y: 13.7, r: 9 };

const state = {
  id: null, session: null, family: null, region: { ...DEFAULT_REGION }, threshold: 0.2,
  search: null, design: null, retest: null, blink: null, globe: null, busy: false,
};

const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
function show(id, on = true) { $(id).hidden = !on; }
function famShort(key) { return state.session.families.find((f) => f.key === key)?.short || key; }
function refImage(key) {
  const r = state.session.references[key || state.family][0];
  return decode(r.image);
}
function setBusy(btn, label) {
  state.busy = !!label;
  document.querySelectorAll("button.primary").forEach((b) => (b.disabled = !!label || b.dataset.locked === "1"));
  if (btn && label) { btn.dataset.label = btn.dataset.label || btn.textContent; btn.innerHTML = `<span class="spinner"></span>${label}`; }
  if (btn && !label && btn.dataset.label) btn.textContent = btn.dataset.label;
}
function scrollTo(id) { $(id).scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" }); }

function renderReference() {
  const s = state.session;
  const plate = $("#ref-plate");
  drawPlate(plate.querySelector("canvas"), refImage(), s.grid.npix);
  const q = featureQ(refImage(), s.grid.npix, s.grid.fov_uas, state.region);
  drawOverlay(plate.querySelector("svg"), { fov: s.grid.fov_uas, region: state.region, label: `q = ${fmt(q)}` });
  const r = s.references[state.family][0];
  $("#ref-caption").textContent = `${famShort(state.family)} reconstruction. Reduced χ² ${fmt(r.chi2, 3)} on ${s.n_cp} closure phases and ${s.n_lca} log closure amplitudes.`;
  $("#q-ref").textContent = fmt(q);
  const fam = s.families.find((f) => f.key === state.family);
  $("#fam-blurb").textContent = fam.blurb;
  $("#claim-text").innerHTML = `Claim: the region marked on the image is at least <strong>${fmt(state.threshold)}</strong> as bright as the ring's peak. In this reconstruction it is <strong>${fmt(q)}</strong>.`;
  document.querySelectorAll(".fam").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.key === state.family)));
}

function renderFamilies() {
  const s = state.session;
  const strip = $("#fam-strip");
  strip.innerHTML = "";
  for (const f of s.families) {
    const b = document.createElement("button");
    b.className = "fam";
    b.dataset.key = f.key;
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", "false");
    b.title = f.label;
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    b.appendChild(c);
    const lab = document.createElement("span");
    lab.textContent = f.short;
    b.appendChild(lab);
    b.addEventListener("click", () => { state.family = f.key; renderReference(); });
    strip.appendChild(b);
    drawPlate(c, refImage(f.key), s.grid.npix);
  }
}

function renderFacts() {
  const s = state.session;
  const box = $("#facts");
  box.innerHTML = "";
  const uv = document.createElement("div");
  uv.className = "chart";
  box.appendChild(uv);
  uvChart(uv, { u: s.uv.u, v: s.uv.v, size: 120 });
  const dl = document.createElement("dl");
  const rows = [
    ["Stations", s.stations_in_data.join(", ")],
    ["Scans", `${fmt(s.time_range[0], 1)}–${fmt(s.time_range[1], 1)} h UT`],
    ["Visibilities", `${s.n_vis} scan-averaged`],
    ["Closures", `${s.n_cp} phases, ${s.n_lca} log amplitudes`],
    ["Error budget", "thermal + 2% non-closing"],
  ];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    dl.append(dt, dd);
  }
  box.appendChild(dl);
}

function onPlateClick(ev) {
  if (state.busy) return;
  const plate = $("#ref-plate");
  const rect = plate.getBoundingClientRect();
  const [x, y] = toSky(ev.clientX - rect.left, ev.clientY - rect.top, state.session.grid.fov_uas, rect.width);
  state.region = { ...state.region, x, y };
  if (!api.live) staticRegionNote();
  renderReference();
}

function staticRegionNote() {
  $("#search-hint").textContent = "This static demo replays a precomputed test for the region along the jet. Run the server locally (python -m anelench.server) to test any region you mark.";
}

function stopBlink() { if (state.blink) { clearInterval(state.blink); state.blink = null; } }

function renderSearch(res) {
  const s = state.session;
  const npix = s.grid.npix, fov = s.grid.fov_uas;
  const v = res.verdict;
  const nf = s.families.length;
  const words = {
    undetermined: ["Undetermined.", "The measurements are explained just as well without the feature as with it."],
    supported: ["Supported.", `No acceptable explanation without the feature was found under any of the ${nf} assumptions.`],
    rejected: ["Not supported.", `No acceptable explanation with the feature was found under any of the ${nf} assumptions.`],
    inconclusive: ["Inconclusive.", "The search did not reach acceptable fits on either side of the threshold."],
  }[v.code];
  $("#verdict").innerHTML = `<span class="tag ${v.code}">${words[0]}</span>${words[1]} Acceptable explanations put the region anywhere between <strong>${fmt(v.q_min)}</strong> and <strong>${fmt(v.q_max)}</strong> of the ring peak.`;

  const region = { x: res.feature.x_uas, y: res.feature.y_uas, r: res.feature.r_uas };
  stopBlink();
  if (res.pair) {
    const A = decode(res.pair.absent.image), B = decode(res.pair.present.image);
    const vmax = Math.max(maxOf(A), maxOf(B));
    const draw = (sel, img, q) => {
      drawPlate($(sel).querySelector("canvas"), img, npix, vmax);
      drawOverlay($(sel).querySelector("svg"), { fov, region, label: `q = ${fmt(q)}`, compass: false });
    };
    const animate = (frames, sel, finalImg, q) => {
      if (!frames || reduceMotion) { draw(sel, finalImg, q); return; }
      let i = 0;
      const imgs = frames.map(decode);
      const tick = () => {
        if (i < imgs.length) { drawPlate($(sel).querySelector("canvas"), imgs[i], npix, vmax); i++; setTimeout(tick, 50); }
        else draw(sel, finalImg, q);
      };
      tick();
    };
    animate(res.frames?.absent, "#plate-a", A, res.pair.absent.q);
    animate(res.frames?.present, "#plate-b", B, res.pair.present.q);
    startBlink(A, B, vmax, region, res.pair);
    const dchi = res.pair.delta_chi2_total;
    $("#cap-a").textContent = `Without: q = ${fmt(res.pair.absent.q)}, ${res.pair.absent.short}, reduced χ² ${fmt(res.pair.absent.chi2, 3)}`;
    $("#cap-b").textContent = `With: q = ${fmt(res.pair.present.q)}, ${res.pair.present.short}, reduced χ² ${fmt(res.pair.present.chi2, 3)}`;
    $(".blink-fig").hidden = false;
    $("#plate-a").parentElement.hidden = false;
    $("#plate-b").parentElement.hidden = false;
    $("#run-design").hidden = false;
    $("#blink-btn").title = `Total χ² differs by ${fmt(Math.abs(dchi), 1)} over ${res.n_data} closure measurements.`;
  } else {
    $(".blink-fig").hidden = true;
    $("#plate-b").parentElement.hidden = true;
    const ref = refImage();
    drawPlate($("#plate-a canvas"), ref, npix);
    drawOverlay($("#plate-a svg"), { fov, region, compass: false });
    $("#cap-a").textContent = "No counterexample on the other side of the threshold was found.";
    $("#run-design").hidden = true;
  }
  intervalsChart($("#intervals"), {
    intervals: res.intervals, families: s.families, refQ: res.reference_q, threshold: res.feature.threshold,
    truthQ: state.revealed ? res.truth_q : null,
  });
  fitCharts($("#fit"), res.fit);
  show("#step-2");
}

function startBlink(A, B, vmax, region, pair) {
  const s = state.session;
  const plate = $("#plate-blink");
  const tag = $("#blink-tag");
  let on = false;
  const flip = () => {
    on = !on;
    drawPlate(plate.querySelector("canvas"), on ? B : A, s.grid.npix, vmax);
    drawOverlay(plate.querySelector("svg"), { fov: s.grid.fov_uas, region, compass: false });
    tag.textContent = on ? "With the feature" : "Without the feature";
    tag.style.background = on ? "var(--b)" : "var(--a)";
  };
  flip();
  const btn = $("#blink-btn");
  const run = () => { stopBlink(); state.blink = setInterval(flip, 800); btn.textContent = "Pause blink comparator"; btn.setAttribute("aria-pressed", "true"); };
  const pause = () => { stopBlink(); btn.textContent = "Blink the two explanations"; btn.setAttribute("aria-pressed", "false"); };
  btn.onclick = () => (state.blink ? pause() : run());
  if (reduceMotion) { pause(); btn.onclick = () => flip(); btn.textContent = "Switch explanation"; } else run();
}

function renderDesign(d) {
  const s = state.session;
  if (!d.available) {
    $("#plan-lede").textContent = d.reason;
    show("#step-3");
    return;
  }
  const rec = d.recommended;
  const hours = rec.ut_end - rec.ut_start;
  const pairWord = d.pool.kind === "twins" ? "twin witness pairs (each pair grown from one reconstruction, so they differ mainly in the region)" : "witness pairs";
  $("#plan-lede").textContent = `Anelench predicted what each of ${d.pool.pairs} ${pairWord} would make the array measure on ${d.all.length} feasible follow-up nights, and ranked the nights by the separation of the hardest pair per unit of telescope time.`;
  $("#rec").innerHTML = `
    <div class="rec-title">${rec.label}</div>
    <div class="rec-meta">Stations: ${rec.stations.join(", ")}. ${fmt(hours, 0)} hours, cost ${fmt(rec.cost, 1)} weighted station-hours of a ${fmt(d.budget, 0)} budget.</div>
    <div class="rec-meta"><span class="big">${fmt(rec.d_min, 1)}σ</span> expected separation for the hardest pair, across ${rec.n_cp} closure phases and ${rec.n_lca} log closure amplitudes.</div>`;
  const order = ["counterexample", "counterexample_max", "max_snr", "uv_coverage", "single_model", "random"];
  const rows = order.filter((k) => d.picks[k]).map((k) => ({ k, label: d.strategies[k], p: d.picks[k] }));
  const eff = (p) => (p.cost > 0 ? (p.d_min || 0) / p.cost : 0);
  const emax = Math.max(...rows.map((r) => eff(r.p)), 1e-9);
  $("#picks").innerHTML = `<thead><tr><th>Strategy</th><th>Chosen night</th><th class="num">Cost</th><th class="num">Hardest pair</th><th class="num">Per station-hour</th></tr></thead><tbody>${rows
    .map((r) => `<tr class="${r.k.startsWith("counterexample") ? "ours" : ""}"><td>${r.label}</td><td>${r.p.label}</td><td class="num">${fmt(r.p.cost, 1)}</td><td class="num">${fmt(r.p.d_min, 1)}σ</td><td class="num">${fmt(eff(r.p), 2)}σ<span class="bar" style="width:${Math.round((54 * eff(r.p)) / emax)}px"></span></td></tr>`)
    .join("")}</tbody>`;
  scatterChart($("#scatter"), { all: d.all, picks: d.picks, strategies: d.strategies, budget: d.budget, recId: rec.id });
  signalsChart($("#signals"), rec.signals);

  state.globe = new Globe($("#globe"), s.stations, s.source);
  const mid = (rec.ut_start + rec.ut_end) / 2;
  const ut = $("#ut-in");
  ut.min = rec.ut_start; ut.max = rec.ut_end; ut.value = mid;
  const draw = () => {
    $("#ut-out").textContent = `${(+ut.value % 24).toFixed(1)} h`;
    state.globe.render({ ut: +ut.value, mjd: s.source.mjd + 1, active: rec.stations, baselines: rec.geometry.baselines });
  };
  ut.oninput = draw;
  state.globe.init().then(draw);
  renderObserved(d, false);
  const best = d.observed?.[0];
  $("#run-retest").hidden = !(s.kind === "synthetic" || best);
  $("#run-retest").textContent = s.kind === "synthetic" ? "Simulate the follow-up night and test again"
    : `Reveal the real ${best.label.replace("Observed 2017 night, ", "")} data and test again`;
  $("#retest-hint").textContent = s.kind === "synthetic"
    ? "The follow-up is simulated from the hidden truth with fresh gains and noise; then every assumption is re-imaged and the search runs again."
    : "No one can observe M87* in 2017 again, so on real data the planner is checked retrospectively: the best-ranked held-back night is revealed, added, and the search runs again.";
  show("#step-3");
}

function renderObserved(d, revealed) {
  const box = $("#observed");
  if (!d.observed) { box.hidden = true; return; }
  box.hidden = false;
  const head = revealed
    ? `<tr><th>Night</th><th class="num">Predicted</th><th class="num">Revealed |Δχ²|</th><th>Real data prefer</th></tr>`
    : `<tr><th>Night</th><th class="num">Stations</th><th class="num">Predicted separation</th></tr>`;
  const rows = d.observed.map((o, i) => {
    const name = o.label.replace("Observed 2017 night, ", "");
    if (!revealed) return `<tr class="${i === 0 ? "ours" : ""}"><td>${name}</td><td class="num">${o.stations.length}</td><td class="num">${fmt(o.d_min, 1)}σ</td></tr>`;
    const r = o.revealed;
    return `<tr class="${i === 0 ? "ours" : ""}"><td>${name}</td><td class="num">${fmt(o.d_min, 1)}σ, |Δχ²| ≈ ${fmt(o.d_min ** 2, 0)}</td><td class="num">${fmt(r.median_abs, 0)}</td><td>${r.prefers === "with" ? "the explanation with the feature" : "the explanation without it"}</td></tr>`;
  });
  $("#observed-table").innerHTML = `<thead>${head}</thead><tbody>${rows.join("")}</tbody>`;
}

function renderRetest(results) {
  const r = results[results.length - 1];
  const before = state.search.verdict, after = r.search.verdict;
  const words = { undetermined: "Still undetermined.", supported: "Supported.", rejected: "Not supported.", inconclusive: "Inconclusive." };
  const nights = results.length === 1 ? "the new night" : `${results.length} new nights`;
  $("#verdict-after").innerHTML = `<span class="tag ${after.code}">${words[after.code]}</span> With ${nights} added, acceptable explanations put the region between <strong>${fmt(after.q_min)}</strong> and <strong>${fmt(after.q_max)}</strong> of the ring peak; the range shrank from ${fmt(before.q_max - before.q_min)} to ${fmt(after.q_max - after.q_min)}.`;
  state.revealed = true;
  const rows = [["Before", [before.q_min, before.q_max]],
    ...results.map((x, i) => [`After night ${i + 1}`, [x.search.verdict.q_min, x.search.verdict.q_max]])];
  beforeAfterChart($("#before-after"), { rows, threshold: state.search.feature.threshold, truthQ: r.search.truth_q });
  $("#round-note").textContent = results.map((x, i) => `Night ${i + 1}: ${x.package.label}, ${fmt(x.package.cost, 1)} weighted station-hours`).join(". ") + ".";
  const more = after.code === "undetermined" && results.length < 2 && state.session.kind === "synthetic"
    && (api.live || !!api.bundles[state.id]?.retest2);
  $("#run-round2").hidden = !more;
  const s = state.session;
  const box = $("#truth-reveal");
  if (!s.truth) {
    renderObserved(state.design, true);
    const o = state.design.observed?.[0];
    box.innerHTML = o ? `<p>The planner ranked the held-back nights before their data were opened. Revealed, the real measurements separate the twin explanations in the order shown in step 3. Caution: M87* can change over days, so nights further from April 11 also test the static-source assumption.</p>` : "";
    show("#step-4");
    return;
  }
  box.innerHTML = `<div class="plate"><canvas width="300" height="300"></canvas><svg class="overlay" viewBox="0 0 300 300"></svg></div><p></p>`;
  if (s.truth) {
    const T = decode(s.truth.image);
    drawPlate(box.querySelector("canvas"), T, s.grid.npix);
    drawOverlay(box.querySelector("svg"), { fov: s.grid.fov_uas, region: { x: state.search.feature.x_uas, y: state.search.feature.y_uas, r: state.search.feature.r_uas }, compass: false });
    const has = (s.truth.model.blobs || []).length > 0;
    box.querySelector("p").textContent = `The hidden truth: ${has ? "a ring with a faint knot" : "a ring with no knot at all"}. Its region value is ${fmt(r.search.truth_q)}, so the claim is ${r.search.truth_q > state.search.feature.threshold ? "true" : "false"}.`;
  }
  intervalsChart($("#intervals"), { intervals: state.search.intervals, families: s.families, refQ: state.search.reference_q, threshold: state.search.feature.threshold, truthQ: r.search.truth_q });
  show("#step-4");
}

function progressText(ev) {
  const stage = { reference: "Imaging under every assumption", search: "Searching for counterexamples", retest: "Re-imaging with the new night", design: "Scoring feasible nights" }[ev.stage] || "Working";
  const fam = ev.family ? `, ${famShort(ev.family)}` : "";
  const step = ev.steps ? `, step ${Math.min(ev.step + 1, ev.steps)} of ${ev.steps}` : "";
  return `${stage}${fam}${step}`;
}

async function loadScenario(id) {
  stopBlink();
  state.id = id; state.search = state.design = null; state.retests = []; state.revealed = false;
  $("#run-round2").hidden = true;
  document.querySelectorAll(".ds").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  ["#step-2", "#step-3", "#step-4"].forEach((s) => show(s, false));
  $("#ds-lede").textContent = LEDES[id];
  $("#search-hint").textContent = "";
  const status = $("#ref-status");
  try {
    let session;
    if (api.live) {
      status.textContent = "Loading…";
      session = await api.job(`api/session/${id}/load`, {}, (ev) => {
        status.textContent = progressText(ev);
        if (ev.previews && state.session == null) {
          drawPlate($("#ref-plate canvas"), decode(ev.previews[0]), 64);
        }
      });
    } else {
      session = (await api.bundle(id)).session;
    }
    status.textContent = "";
    state.session = session;
    state.family = session.default_family;
    if (!api.live) {
      const b = api.bundles[id];
      state.region = { x: b.region.x_uas, y: b.region.y_uas, r: b.region.r_uas };
      state.threshold = b.region.threshold;
    } else {
      const d = DEFAULTS[id] || { ...DEFAULT_REGION, t: 0.2 };
      state.region = { x: d.x, y: d.y, r: d.r };
      state.threshold = d.t;
    }
    $("#r-in").value = state.region.r; $("#r-out").textContent = `${state.region.r} µas`;
    $("#t-in").value = state.threshold; $("#t-out").textContent = fmt(state.threshold);
    renderFamilies();
    renderReference();
    renderFacts();
    $("#run-search").disabled = false;
    if (!api.live) staticRegionNote();
  } catch (err) {
    status.textContent = err.message;
  }
}

async function runSearch() {
  const btn = $("#run-search");
  setBusy(btn, "Searching for counterexamples…");
  const status = $("#ref-status");
  try {
    let res;
    if (api.live) {
      show("#step-2");
      $("#verdict").textContent = "Pushing every assumption to remove and to strengthen the feature while the fit stays within the noise.";
      res = await api.job(`api/session/${state.id}/search`, {
        x_uas: state.region.x, y_uas: state.region.y, r_uas: state.region.r, threshold: state.threshold, fwhm_uas: 20,
      }, (ev) => {
        status.textContent = progressText(ev);
        if (ev.previews && ev.previews.length >= 2) {
          const A = decode(ev.previews[0]), B = decode(ev.previews[1]);
          const vmax = Math.max(maxOf(A), maxOf(B));
          drawPlate($("#plate-a canvas"), A, 64, vmax);
          drawPlate($("#plate-b canvas"), B, 64, vmax);
          $("#cap-a").textContent = `Removing the feature under ${famShort(ev.family)}…`;
          $("#cap-b").textContent = `Strengthening it under ${famShort(ev.family)}…`;
        }
      });
    } else {
      res = (await api.bundle(state.id)).search;
    }
    status.textContent = "";
    state.search = res;
    renderSearch(res);
    scrollTo("#step-2");
  } catch (err) {
    status.textContent = err.message;
  } finally {
    setBusy(btn, null);
  }
}

async function runDesign() {
  const btn = $("#run-design");
  setBusy(btn, "Scoring feasible nights…");
  try {
    const d = api.live ? await api.job(`api/session/${state.id}/design`, { budget: 70 }, () => {}) : (await api.bundle(state.id)).design;
    state.design = d;
    renderDesign(d);
    scrollTo("#step-3");
  } catch (err) {
    $("#plan-lede").textContent = err.message;
    show("#step-3");
  } finally {
    setBusy(btn, null);
  }
}

async function runRetest() {
  const btn = $("#run-retest");
  const rec = state.session.kind === "synthetic" ? state.design?.recommended : state.design?.observed?.[0];
  if (!rec) return;
  setBusy(btn, "Observing and re-imaging…");
  try {
    let r;
    if (api.live) {
      r = await api.job(`api/session/${state.id}/retest`, { package_id: rec.id }, (ev) => { $("#retest-hint").textContent = progressText(ev); });
    } else {
      r = (await api.bundle(state.id)).retest;
    }
    state.retests = [r];
    renderRetest(state.retests);
    scrollTo("#step-4");
  } catch (err) {
    $("#retest-hint").textContent = err.message;
  } finally {
    setBusy(btn, null);
  }
}

async function runRound2() {
  const btn = $("#run-round2");
  setBusy(btn, "Planning and observing another night…");
  try {
    let r;
    if (api.live) {
      const d = await api.job(`api/session/${state.id}/design`, { budget: 70 }, () => {});
      if (!d.available || !d.recommended) throw new Error(d.reason || "No further night separates the remaining explanations.");
      r = await api.job(`api/session/${state.id}/retest`, { package_id: d.recommended.id, seed: 12 }, (ev) => { $("#round-note").textContent = progressText(ev); });
    } else {
      r = (await api.bundle(state.id)).retest2;
    }
    state.retests.push(r);
    renderRetest(state.retests);
  } catch (err) {
    $("#round-note").textContent = err.message;
  } finally {
    setBusy(btn, null);
  }
}

async function renderRealValidation() {
  let m;
  try { m = await api.bundle("m87-2017-101"); } catch { return; }
  const obs = m.design?.observed;
  if (!obs || !m.retests) return;
  const before = m.search.verdict;
  $("#val-real-note").textContent = `We tested the western extension in the April 11 data (claim: at least ${fmt(m.region.threshold)} of the ring peak; acceptable range [${fmt(before.q_min)}, ${fmt(before.q_max)}]) and ranked the three held-back 2017 nights before opening them. Then each night's real measurements were revealed, added, and the whole search re-run.`;
  const rows = obs.map((o) => {
    const rt = m.retests[o.id]?.search?.verdict;
    const name = o.label.replace("Observed 2017 night, ", "");
    return `<tr><td>${name}</td><td class="num">${fmt(o.d_min, 1)}σ</td><td class="num">${fmt(o.revealed.median_abs, 0)}</td><td class="num">${rt ? `[${fmt(rt.q_min)}, ${fmt(rt.q_max)}]` : "–"}</td><td class="num">${rt ? fmt(rt.q_max - rt.q_min) : "–"}</td></tr>`;
  });
  $("#val-real-table").innerHTML = `<thead><tr><th>Held-back night</th><th class="num">Predicted separation</th><th class="num">Revealed median |Δχ²|</th><th class="num">Range after adding it</th><th class="num">Width</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  show("#val-real");
  show("#validation");
}

async function renderValidation() {
  renderRealValidation();
  let b;
  try {
    const r = await fetch("demo/benchmark.json", { cache: "no-store" });
    if (!r.ok) return;
    b = await r.json();
  } catch { return; }
  const sum = b.summary;
  const labels = {
    counterexample: "Anelench, best separation per cost", counterexample_max: "Anelench, whole budget",
    max_snr: "Highest signal-to-noise", uv_coverage: "Largest new Fourier coverage",
    single_model: "Single-model uncertainty", random: "Random feasible night",
  };
  const keys = Object.keys(labels).filter((k) => sum[k]);
  if (!keys.length) return;
  const n0 = sum._initial.n, nu = Math.round(sum._initial.undetermined * n0);
  const pct = (v) => `${Math.round(100 * v)}%`;
  $("#val-lede").textContent = `${n0} random M87*-like truths (half with a faint knot near the jet base, half without), measured with the real April 2017 sampling. ${nu} were undetermined from the 2017 data alone; for those, each strategy scheduled one night within a budget of ${b.budget} weighted station-hours, simulated from the truth with fresh gains and noise. From the initial data, the default single reconstruction got the claim wrong in ${pct(sum._initial.wrong_standard)} of all cases.`;
  show("#val-grid");
  strategyBars($("#val-width"), keys.map((k) => ({ key: k, label: labels[k], value: sum[k].mean_width ?? 0 })));
  $("#val-table").innerHTML = `<thead><tr><th>Strategy</th><th class="num">Mean cost</th><th class="num">Settled correctly</th><th class="num">Settled wrongly</th><th class="num">Single pipeline wrong</th></tr></thead><tbody>${keys
    .map((k) => `<tr class="${k.startsWith("counterexample") ? "ours" : ""}"><td>${labels[k]}</td><td class="num">${fmt(sum[k].mean_cost, 0)}</td><td class="num">${pct(sum[k].correct_anelench ?? 0)}</td><td class="num">${pct(sum[k].wrong_anelench ?? 0)}</td><td class="num">${pct(sum[k].wrong_standard)}</td></tr>`)
    .join("")}</tbody>`;
  show("#validation");
}

async function boot() {
  const live = await api.init();
  document.querySelectorAll(".ds").forEach((b) => b.addEventListener("click", () => !state.busy && loadScenario(b.dataset.id)));
  $("#ref-plate").addEventListener("click", onPlateClick);
  $("#ref-plate").addEventListener("keydown", (e) => {
    const step = 2;
    const k = { ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[e.key];
    if (!k || state.busy) return;
    e.preventDefault();
    state.region = { ...state.region, x: state.region.x + k[0], y: state.region.y + k[1] };
    renderReference();
  });
  $("#r-in").addEventListener("input", (e) => { state.region.r = +e.target.value; $("#r-out").textContent = `${e.target.value} µas`; renderReference(); });
  $("#t-in").addEventListener("input", (e) => { state.threshold = +e.target.value; $("#t-out").textContent = fmt(state.threshold); renderReference(); });
  $("#run-search").addEventListener("click", runSearch);
  $("#run-design").addEventListener("click", runDesign);
  $("#run-retest").addEventListener("click", runRetest);
  $("#run-round2").addEventListener("click", runRound2);
  renderValidation();
  const params = new URLSearchParams(location.search);
  const start = ["ghost", "knot", "m87-2017-101"].includes(params.get("s")) ? params.get("s") : "ghost";
  await loadScenario(start);
  const step = Number(params.get("step") || 1);
  if (api.live || step < 2) return;
  await runSearch();
  if (step >= 3 && !$("#run-design").hidden) await runDesign();
  if (step >= 4 && !$("#run-retest").hidden) await runRetest();
  if (step >= 5 && !$("#run-round2").hidden) await runRound2();
  const target = { 2: "#step-2", 3: "#step-3", 4: "#step-4", 5: "#step-4", 6: "#validation" }[Math.min(step, 6)];
  if (target && !$(target).hidden) $(target).scrollIntoView({ block: "start" });
}

boot();
