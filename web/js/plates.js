export function decode(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

const LUT = (() => {
  const t = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    t[i * 3] = 255 * Math.min(1, 2 * x);
    t[i * 3 + 1] = 255 * Math.min(1, Math.max(0, 2 * x - 0.5));
    t[i * 3 + 2] = 255 * Math.min(1, Math.max(0, 2 * x - 1));
  }
  return t;
})();

export function maxOf(img) {
  let m = 0;
  for (const v of img) if (v > m) m = v;
  return m;
}

export function drawPlate(canvas, img, npix, vmax) {
  const off = document.createElement("canvas");
  off.width = off.height = npix;
  const ctx = off.getContext("2d");
  const data = ctx.createImageData(npix, npix);
  const m = vmax || maxOf(img) || 1;
  for (let i = 0; i < npix * npix; i++) {
    const k = Math.max(0, Math.min(255, Math.round((img[i] / m) * 255)));
    data.data[i * 4] = LUT[k * 3];
    data.data[i * 4 + 1] = LUT[k * 3 + 1];
    data.data[i * 4 + 2] = LUT[k * 3 + 2];
    data.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
  const c = canvas.getContext("2d");
  c.imageSmoothingEnabled = true;
  c.imageSmoothingQuality = "high";
  c.fillStyle = "#000";
  c.fillRect(0, 0, canvas.width, canvas.height);
  c.drawImage(off, 0, 0, canvas.width, canvas.height);
}

export function toPx(x, y, fov, size) {
  return [(0.5 - x / fov) * size, (0.5 - y / fov) * size];
}
export function toSky(px, py, fov, size) {
  return [(0.5 - px / size) * fov, (0.5 - py / size) * fov];
}

const NS = "http://www.w3.org/2000/svg";
function el(name, attrs, parent) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

export function drawOverlay(svg, { fov, region, label, compass = true, scale = true }) {
  const size = svg.viewBox.baseVal.width;
  svg.innerHTML = "";
  if (region) {
    const [cx, cy] = toPx(region.x, region.y, fov, size);
    const r = (region.r / fov) * size;
    el("circle", { cx, cy, r, fill: "none", stroke: "#fff", "stroke-width": size / 256, "stroke-dasharray": `${size / 80} ${size / 120}`, opacity: 0.9 }, svg);
    if (label) {
      const right = cx + r + size * 0.2 < size;
      const t = el("text", {
        x: right ? cx + r + size / 60 : cx - r - size / 60, y: cy + size / 110, fill: "#fff",
        "font-size": size / 26, "font-family": "Archivo, sans-serif", "font-weight": 600, "text-anchor": right ? "start" : "end",
      }, svg);
      t.textContent = label;
    }
  }
  const s = size / 512;
  if (compass) {
    const g = el("g", { transform: `translate(${size - 46 * s}, ${46 * s})`, stroke: "#c9ced6", "stroke-width": 1.4 * s, fill: "none" }, svg);
    el("path", { d: `M0 0 V${-26 * s} M${-4 * s} ${-20 * s} L0 ${-26 * s} L${4 * s} ${-20 * s}` }, g);
    el("path", { d: `M0 0 H${-26 * s} M${-20 * s} ${-4 * s} L${-26 * s} 0 L${-20 * s} ${4 * s}` }, g);
    const n = el("text", { x: -4 * s, y: -32 * s, fill: "#c9ced6", stroke: "none", "font-size": 13 * s, "font-family": "Archivo, sans-serif" }, g);
    n.textContent = "N";
    const e = el("text", { x: -42 * s, y: 5 * s, fill: "#c9ced6", stroke: "none", "font-size": 13 * s, "font-family": "Archivo, sans-serif" }, g);
    e.textContent = "E";
  }
  if (scale) {
    const len = (20 / fov) * size;
    const x0 = size - 24 * s - len, y0 = size - 26 * s;
    el("line", { x1: x0, y1: y0, x2: x0 + len, y2: y0, stroke: "#c9ced6", "stroke-width": 2 * s }, svg);
    const t = el("text", { x: x0 + len / 2, y: y0 - 7 * s, fill: "#c9ced6", "font-size": 13 * s, "text-anchor": "middle", "font-family": "Archivo, sans-serif" }, svg);
    t.textContent = "20 µas";
  }
}

export function blur(img, npix, fwhmPx) {
  const s = fwhmPx / 2.3548;
  const rad = Math.ceil(3 * s);
  const k = [];
  let sum = 0;
  for (let i = -rad; i <= rad; i++) { const v = Math.exp(-0.5 * (i / s) ** 2); k.push(v); sum += v; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(npix * npix);
  const out = new Float32Array(npix * npix);
  for (let r = 0; r < npix; r++)
    for (let c = 0; c < npix; c++) {
      let acc = 0;
      for (let j = -rad; j <= rad; j++) { const cc = c + j; if (cc >= 0 && cc < npix) acc += img[r * npix + cc] * k[j + rad]; }
      tmp[r * npix + c] = acc;
    }
  for (let r = 0; r < npix; r++)
    for (let c = 0; c < npix; c++) {
      let acc = 0;
      for (let j = -rad; j <= rad; j++) { const rr = r + j; if (rr >= 0 && rr < npix) acc += tmp[rr * npix + c] * k[j + rad]; }
      out[r * npix + c] = acc;
    }
  return out;
}

export function featureQ(img, npix, fov, region, fwhm = 20) {
  const ps = fov / npix;
  const b = blur(img, npix, fwhm / ps);
  let peak = 0, wsum = 0, acc = 0;
  for (let r = 0; r < npix; r++) {
    const y = (npix / 2 - 0.5 - r) * ps;
    for (let c = 0; c < npix; c++) {
      const x = (npix / 2 - 0.5 - c) * ps;
      const v = b[r * npix + c];
      if (v > peak) peak = v;
      const d = Math.hypot(x - region.x, y - region.y);
      const w = 1 / (1 + Math.exp((d - region.r) / (0.5 * ps)));
      wsum += w; acc += w * v;
    }
  }
  return peak > 0 ? acc / wsum / peak : 0;
}
