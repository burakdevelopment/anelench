const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const LAND_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-110m.json";
let landCache = null;

export function gmstDeg(mjd, ut) {
  const jd = mjd + 2400000.5 + ut / 24;
  const g = 280.46061837 + 360.98564736629 * (jd - 2451545.0);
  return ((g % 360) + 360) % 360;
}

export function subsourceLon(ra, mjd, ut) {
  let lon = ra - gmstDeg(mjd, ut);
  lon = ((lon + 540) % 360) - 180;
  return lon;
}

export function elevation(lat, lon, dec, lonS) {
  const r = Math.PI / 180;
  const s = Math.sin(lat * r) * Math.sin(dec * r) + Math.cos(lat * r) * Math.cos(dec * r) * Math.cos((lon - lonS) * r);
  return Math.asin(Math.max(-1, Math.min(1, s))) / r;
}

export class Globe {
  constructor(el, stations, source) {
    this.el = el;
    this.stations = stations;
    this.source = source;
    this.size = 520;
    el.innerHTML = "";
    this.svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${this.size} ${this.size}`);
    this.proj = d3.geoOrthographic().scale(this.size / 2 - 12).translate([this.size / 2, this.size / 2]).clipAngle(90);
    this.path = d3.geoPath(this.proj);
  }

  async init() {
    if (!landCache) {
      try {
        const topo = await (await fetch(LAND_URL)).json();
        landCache = topojson.feature(topo, topo.objects.land);
      } catch {
        landCache = { type: "FeatureCollection", features: [] };
      }
    }
    this.land = landCache;
  }

  render({ ut, mjd, active = [], baselines = [] }) {
    const { ra, dec } = this.source;
    const lonS = subsourceLon(ra, mjd, ut);
    this.proj.rotate([-lonS, -dec]);
    const svg = this.svg;
    svg.selectAll("*").remove();
    svg.append("path").datum({ type: "Sphere" }).attr("class", "sphere").attr("d", this.path);
    svg.append("path").datum(d3.geoGraticule10()).attr("class", "grat").attr("d", this.path);
    if (this.land) svg.append("path").datum(this.land).attr("class", "land").attr("d", this.path);
    svg.append("path").datum(d3.geoCircle().center([lonS, dec]).radius(80)()).attr("fill", "none")
      .attr("stroke", css("--ink-3")).attr("stroke-dasharray", "3 4").attr("d", this.path);

    const byCode = Object.fromEntries(this.stations.map((s) => [s.code, s]));
    const up = (s) => elevation(s.lat, s.lon, dec, lonS) >= 10;
    for (const [a, b] of baselines) {
      const sa = byCode[a], sb = byCode[b];
      if (!sa || !sb) continue;
      const live = up(sa) && up(sb);
      svg.append("path").datum({ type: "LineString", coordinates: [[sa.lon, sa.lat], [sb.lon, sb.lat]] })
        .attr("class", live ? "bl" : "bl off").attr("d", this.path);
    }
    const g = svg.append("g");
    for (const s of this.stations) {
      const p = this.proj([s.lon, s.lat]);
      const vis = d3.geoDistance([s.lon, s.lat], [lonS, dec]) < Math.PI / 2;
      if (!p || !vis) continue;
      const inPkg = active.includes(s.code);
      const node = g.append("g").attr("class", "st").attr("transform", `translate(${p[0]},${p[1]})`);
      node.append("circle").attr("r", inPkg ? 5.5 : 3.5)
        .attr("fill", inPkg ? (up(s) ? css("--plan") : css("--paper")) : css("--ink-3"))
        .attr("stroke", inPkg ? css("--plan") : css("--paper"));
      node.append("text").attr("x", 8).attr("y", 4).text(s.code).style("opacity", inPkg ? 1 : 0.6);
      node.append("title").text(`${s.name}, ${s.location}${s.in2017 ? "" : ` (joined ${s.joined})`}`);
    }
    return lonS;
  }
}
