const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = () => ({ ink: css("--ink"), ink2: css("--ink-2"), ink3: css("--ink-3"), rule: css("--rule"), a: css("--a"), b: css("--b"), plan: css("--plan"), paper: css("--paper") });

function frame(el, width, height, margin) {
  el.innerHTML = "";
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  return { svg, g, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
}

export function intervalsChart(el, { intervals, families, refQ, threshold, truthQ, labelFor }) {
  const c = C();
  const keys = families.filter((f) => intervals[f.key]);
  const rowH = 30;
  const { svg, g, w, h } = frame(el, 460, 40 + rowH * keys.length + 26, { top: 16, right: 18, bottom: 34, left: 74 });
  const hi = Math.max(0.5, ...keys.map((f) => intervals[f.key][1] + 0.05), threshold + 0.1, truthQ ?? 0);
  const x = d3.scaleLinear().domain([0, Math.min(1, hi)]).range([0, w]);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  g.append("text").attr("x", w).attr("y", h + 30).attr("text-anchor", "end").text("region brightness / ring peak");
  keys.forEach((f, i) => {
    const y = i * rowH + 10;
    const [lo, hi2] = intervals[f.key];
    g.append("text").attr("x", -10).attr("y", y + 4).attr("text-anchor", "end").style("fill", c.ink).text(labelFor ? labelFor(f) : f.short);
    g.append("line").attr("x1", 0).attr("x2", w).attr("y1", y).attr("y2", y).attr("stroke", c.rule).attr("stroke-dasharray", "1 3");
    const segA = [lo, Math.min(hi2, threshold)], segB = [Math.max(lo, threshold), hi2];
    if (segA[1] > segA[0]) g.append("rect").attr("x", x(segA[0])).attr("y", y - 5).attr("width", x(segA[1]) - x(segA[0])).attr("height", 10).attr("fill", c.a);
    if (segB[1] > segB[0]) g.append("rect").attr("x", x(segB[0])).attr("y", y - 5).attr("width", x(segB[1]) - x(segB[0])).attr("height", 10).attr("fill", c.b);
    if (refQ && refQ[f.key] != null) g.append("line").attr("x1", x(refQ[f.key])).attr("x2", x(refQ[f.key])).attr("y1", y - 9).attr("y2", y + 9).attr("stroke", c.ink).attr("stroke-width", 2);
  });
  const tx = x(threshold);
  g.append("line").attr("x1", tx).attr("x2", tx).attr("y1", -6).attr("y2", h).attr("stroke", c.ink).attr("stroke-dasharray", "4 3");
  g.append("text").attr("x", tx + 4).attr("y", -4).style("fill", c.ink).text(`claim threshold ${threshold.toFixed(2)}`);
  if (truthQ != null) {
    g.append("path").attr("d", d3.symbol(d3.symbolDiamond, 70)).attr("transform", `translate(${x(truthQ)},${h - 8})`).attr("fill", c.plan);
    g.append("text").attr("x", x(truthQ) + 7).attr("y", h - 4).style("fill", c.plan).text("truth");
  }
  svg.append("desc").text("Horizontal bars: range of the feature value allowed by the data under each imaging assumption. Ticks: the reconstruction's own value.");
}

function wrapNear(model, obs) {
  let m = model;
  while (m - obs > 180) m -= 360;
  while (m - obs < -180) m += 360;
  return m;
}

export function fitCharts(el, fit, colors = { absent: "a", present: "b", reference: "ink" }) {
  const c = C();
  el.innerHTML = "";
  for (const tri of fit) {
    const box = document.createElement("div");
    const title = document.createElement("div");
    title.className = "sm-title";
    title.textContent = tri.triangle.replaceAll("-", "–");
    box.appendChild(title);
    el.appendChild(box);
    const holder = document.createElement("div");
    box.appendChild(holder);
    const { g, w, h } = frame(holder, 300, 190, { top: 8, right: 8, bottom: 26, left: 38 });
    const pts = tri.time.map((t, i) => ({ t, o: tri.obs[i], s: tri.sigma[i] }));
    const models = Object.entries(tri.models).map(([k, arr]) => ({ k, pts: arr.map((v, i) => ({ t: tri.time[i], v: wrapNear(v, tri.obs[i]) })) }));
    const all = pts.flatMap((p) => [p.o - p.s, p.o + p.s]).concat(models.flatMap((m) => m.pts.map((p) => p.v)));
    const y = d3.scaleLinear().domain(d3.extent(all)).nice().range([h, 0]);
    const x = d3.scaleLinear().domain(d3.extent(tri.time)).nice().range([0, w]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(4).tickFormat((d) => `${d}h`).tickSizeOuter(0));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat((d) => `${d}°`).tickSizeOuter(0));
    for (const m of models) {
      const col = c[colors[m.k] || "ink"];
      g.append("path").datum(m.pts).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2).attr("stroke-opacity", 0.9)
        .attr("d", d3.line().x((p) => x(p.t)).y((p) => y(p.v)).curve(d3.curveMonotoneX));
    }
    g.selectAll(".eb").data(pts).enter().append("line").attr("x1", (p) => x(p.t)).attr("x2", (p) => x(p.t))
      .attr("y1", (p) => y(p.o - p.s)).attr("y2", (p) => y(p.o + p.s)).attr("stroke", c.ink).attr("stroke-width", 1);
    g.selectAll(".pt").data(pts).enter().append("circle").attr("cx", (p) => x(p.t)).attr("cy", (p) => y(p.o)).attr("r", 2.6).attr("fill", c.ink);
  }
}

export function scatterChart(el, { all, picks, strategies, budget, recId }) {
  const c = C();
  const { g, w, h } = frame(el, 520, 300, { top: 14, right: 16, bottom: 40, left: 46 });
  const dmax = Math.max(10, ...all.map((p) => p.d_min || 0));
  const x = d3.scaleLinear().domain([0, d3.max(all, (p) => p.cost) * 1.05]).range([0, w]);
  const y = d3.scaleSymlog().constant(1).domain([0, dmax * 1.1]).range([h, 0]);
  g.append("rect").attr("x", x(budget)).attr("y", 0).attr("width", Math.max(0, w - x(budget))).attr("height", h).attr("fill", c.rule).attr("opacity", 0.35);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickValues([0, 1, 3, 10, 30, 100].filter((v) => v <= dmax * 1.1)).tickFormat((d) => `${d}σ`).tickSizeOuter(0));
  g.append("text").attr("x", w).attr("y", h + 34).attr("text-anchor", "end").text("cost, weighted station-hours");
  g.append("line").attr("x1", 0).attr("x2", w).attr("y1", y(3)).attr("y2", y(3)).attr("stroke", c.ink3).attr("stroke-dasharray", "4 3");
  g.append("text").attr("x", 4).attr("y", y(3) - 5).text("3σ: decisive");
  g.selectAll(".p").data(all).enter().append("circle").attr("cx", (p) => x(p.cost)).attr("cy", (p) => y(p.d_min || 0))
    .attr("r", 3).attr("fill", c.ink3).attr("opacity", (p) => (p.cost <= budget ? 0.7 : 0.25));
  const order = Object.keys(strategies);
  order.forEach((k, i) => {
    const p = picks[k];
    if (!p) return;
    const ours = k === "counterexample";
    const px = x(p.cost), py = y(p.d_min || 0);
    g.append("circle").attr("cx", px).attr("cy", py).attr("r", ours ? 7 : 5).attr("fill", ours ? c.plan : "none").attr("stroke", ours ? c.plan : c.ink).attr("stroke-width", 2);
  });
}

export function signalsChart(el, signals) {
  const c = C();
  el.innerHTML = "";
  for (const tri of signals.triangles) {
    const box = document.createElement("div");
    const title = document.createElement("div");
    title.className = "sm-title";
    title.textContent = `${tri.triangle.replaceAll("-", "–")}, up to ${tri.max_sep_sigma.toFixed(1)}σ apart`;
    box.appendChild(title);
    el.appendChild(box);
    const holder = document.createElement("div");
    box.appendChild(holder);
    const { g, w, h } = frame(holder, 300, 190, { top: 8, right: 8, bottom: 26, left: 40 });
    const ref = tri.curves.absent;
    const series = Object.entries(tri.curves).map(([k, arr]) => ({ k, pts: arr.map((v, i) => ({ t: tri.time[i], v: wrapNear(v, ref[i]), s: tri.sigma_deg[i] })) }));
    const all = series.flatMap((s) => s.pts.flatMap((p) => [p.v - p.s, p.v + p.s]));
    const x = d3.scaleLinear().domain(d3.extent(tri.time)).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(all)).nice().range([h, 0]);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(4).tickFormat((d) => `${d}h`).tickSizeOuter(0));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat((d) => `${d}°`).tickSizeOuter(0));
    for (const s of series) {
      const col = s.k === "absent" ? c.a : c.b;
      g.append("path").datum(s.pts).attr("fill", col).attr("opacity", 0.16)
        .attr("d", d3.area().x((p) => x(p.t)).y0((p) => y(p.v - p.s)).y1((p) => y(p.v + p.s)).curve(d3.curveMonotoneX));
      g.append("path").datum(s.pts).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2)
        .attr("d", d3.line().x((p) => x(p.t)).y((p) => y(p.v)).curve(d3.curveMonotoneX));
    }
  }
}

export function beforeAfterChart(el, { rows, threshold, truthQ }) {
  const c = C();
  const { g, w, h } = frame(el, 520, 76 + rows.length * 38, { top: 20, right: 18, bottom: 34, left: 96 });
  const hi = Math.max(0.5, ...rows.map((r) => r[1][1] + 0.05), threshold + 0.1);
  const x = d3.scaleLinear().domain([0, Math.min(1, hi)]).range([0, w]);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  rows.forEach(([lab, [lo, hi2]], i) => {
    const y = 14 + i * 38;
    g.append("text").attr("x", -10).attr("y", y + 5).attr("text-anchor", "end").style("fill", c.ink).style("font-size", "13px").text(lab);
    const segA = [lo, Math.min(hi2, threshold)], segB = [Math.max(lo, threshold), hi2];
    if (segA[1] > segA[0]) g.append("rect").attr("x", x(segA[0])).attr("y", y - 7).attr("width", x(segA[1]) - x(segA[0])).attr("height", 14).attr("fill", c.a);
    if (segB[1] > segB[0]) g.append("rect").attr("x", x(segB[0])).attr("y", y - 7).attr("width", x(segB[1]) - x(segB[0])).attr("height", 14).attr("fill", c.b);
  });
  const tx = x(threshold);
  g.append("line").attr("x1", tx).attr("x2", tx).attr("y1", -8).attr("y2", h).attr("stroke", c.ink).attr("stroke-dasharray", "4 3");
  g.append("text").attr("x", tx + 4).attr("y", -6).style("fill", c.ink).text(`claim threshold ${threshold.toFixed(2)}`);
  if (truthQ != null) {
    g.append("path").attr("d", d3.symbol(d3.symbolDiamond, 80)).attr("transform", `translate(${x(truthQ)},${h - 8})`).attr("fill", c.plan);
    g.append("text").attr("x", x(truthQ) + 7).attr("y", h - 4).style("fill", c.plan).text("truth");
  }
}

export function strategyBars(el, rows, { fmt = (v) => v.toFixed(2), ours = ["counterexample", "counterexample_max"] } = {}) {
  const c = C();
  const { g, w, h } = frame(el, 520, 36 + rows.length * 34, { top: 8, right: 60, bottom: 24, left: 210 });
  const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.value) * 1.1 || 1]).range([0, w]);
  rows.forEach((r, i) => {
    const y = i * 34 + 6;
    const mine = ours.includes(r.key);
    g.append("text").attr("x", -10).attr("y", y + 13).attr("text-anchor", "end").style("fill", c.ink).style("font-size", "12px")
      .style("font-weight", mine ? 650 : 400).text(r.label);
    g.append("rect").attr("x", 0).attr("y", y).attr("width", x(r.value)).attr("height", 18).attr("fill", mine ? c.plan : c.ink3);
    g.append("text").attr("x", x(r.value) + 6).attr("y", y + 13).style("fill", c.ink).style("font-size", "12px").text(fmt(r.value));
  });
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(4).tickSizeOuter(0));
}

export function uvChart(el, { u, v, nu, nv, size = 180 }) {
  const c = C();
  const { g, w, h } = frame(el, size, size, { top: 6, right: 6, bottom: 6, left: 6 });
  const lim = Math.max(9, ...u.map(Math.abs), ...v.map(Math.abs), ...(nu || []).map(Math.abs), ...(nv || []).map(Math.abs));
  const x = d3.scaleLinear().domain([lim, -lim]).range([0, w]);
  const y = d3.scaleLinear().domain([-lim, lim]).range([h, 0]);
  g.append("circle").attr("cx", x(0)).attr("cy", y(0)).attr("r", x(0) - x(lim * 0.5) ).attr("fill", "none").attr("stroke", c.rule);
  g.append("circle").attr("cx", x(0)).attr("cy", y(0)).attr("r", x(0) - x(lim)).attr("fill", "none").attr("stroke", c.rule);
  const dot = (uu, vv, col, r) => {
    for (let i = 0; i < uu.length; i++) {
      g.append("circle").attr("cx", x(uu[i])).attr("cy", y(vv[i])).attr("r", r).attr("fill", col);
      g.append("circle").attr("cx", x(-uu[i])).attr("cy", y(-vv[i])).attr("r", r).attr("fill", col);
    }
  };
  dot(u, v, c.ink, 1.3);
  if (nu) dot(nu, nv, c.plan, 1.6);
}
