export const api = {
  live: false,
  bundles: {},

  async init() {
    try {
      const r = await fetch("api/health", { cache: "no-store" });
      this.live = r.ok && (await r.json()).mode === "live";
    } catch {
      this.live = false;
    }
    return this.live;
  },

  async bundle(id) {
    if (!this.bundles[id]) {
      const r = await fetch(`demo/${id}.json`);
      if (!r.ok) throw new Error(`Precomputed demo for ${id} is missing (demo/${id}.json).`);
      this.bundles[id] = await r.json();
    }
    return this.bundles[id];
  },

  async job(path, body, onProgress) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${r.status})`);
    }
    const { job } = await r.json();
    return new Promise((resolve, reject) => {
      const es = new EventSource(`api/jobs/${job}/stream`);
      es.addEventListener("progress", (e) => onProgress && onProgress(JSON.parse(e.data)));
      es.addEventListener("result", (e) => { es.close(); resolve(JSON.parse(e.data)); });
      es.addEventListener("failed", (e) => { es.close(); reject(new Error(JSON.parse(e.data).error)); });
      es.onerror = () => { es.close(); reject(new Error("Lost connection to the Anelench server.")); };
    });
  },
};
