/* ============================================================================
   core.js: the UI toolkit. DOM helpers, theme-aware colours, heatmaps, charts,
   controls, browser storage, widget registry, router.
   ========================================================================== */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}
const fmt = (x, d = 2) => (x === -Infinity ? "−∞" : x === Infinity ? "∞" : Number.isFinite(x) ? (Math.abs(x) >= 1e4 ? x.toExponential(1) : x.toFixed(d)).replace("-", "−") : String(x));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

/* ---------- storage (per-viewer conveniences only) ---------- */
const store = {
  get(k, def) { try { const v = localStorage.getItem("itf:" + k); return v === null ? def : JSON.parse(v); } catch (e) { return def; } },
  set(k, v) { try { localStorage.setItem("itf:" + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
};

/* ---------- theme-aware colours ---------- */
const Theme = {
  cache: {}, subs: new Set(),
  v(name) { if (!(name in this.cache)) this.cache[name] = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); return this.cache[name]; },
  rgb(name) { return hexToRgb(this.v(name)); },
  changed() { this.cache = {}; this.subs.forEach((f) => { try { f(); } catch (e) { console.error(e); } }); },
  on(f) { this.subs.add(f); return () => this.subs.delete(f); },
};
function hexToRgb(h) {
  h = h.replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const mix = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
const rgbStr = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
function seqColor(t) { // 0..1 -> heat ramp
  t = clamp(t, 0, 1);
  const c0 = Theme.rgb("--heat-0"), c1 = Theme.rgb("--heat-1"), c2 = Theme.rgb("--heat-2");
  return t < 0.5 ? mix(c0, c1, t * 2) : mix(c1, c2, (t - 0.5) * 2);
}
function divColor(t) { // -1..1 -> diverging ramp
  t = clamp(t, -1, 1);
  return t < 0 ? mix(Theme.rgb("--mid"), Theme.rgb("--neg"), -t) : mix(Theme.rgb("--mid"), Theme.rgb("--pos"), t);
}
const lum = (c) => (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
(() => {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener && mq.addEventListener("change", () => Theme.changed());
  new MutationObserver(() => Theme.changed()).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();

/* ---------- tooltip ---------- */
const Tip = {
  node: null,
  show(x, y, text) {
    if (!this.node) { this.node = el("div", { id: "tip", role: "tooltip" }); document.body.append(this.node); }
    this.node.textContent = text;
    this.node.classList.add("show");
    const r = this.node.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    this.node.style.left = Math.max(4, left) + "px";
    this.node.style.top = Math.max(4, top) + "px";
  },
  hide() { this.node && this.node.classList.remove("show"); },
};

/* ---------- heatmap (canvas) ----------
   data: array of rows. opts: rowLabels, colLabels, mode 'seq'|'div', vmin, vmax, cell, maxWidth, annotate, tip(i,j,v), onClick(i,j), highlight {rows:[], cols:[]} */
function heatmap(data, opts = {}) {
  const wrap = el("div", { class: "scroll-x" });
  const canvas = el("canvas", { class: "hm" });
  wrap.append(canvas);
  let state = { data, opts };
  function draw() {
    const { data, opts } = state;
    const R = data.length, C = data[0] ? data[0].length : 0;
    const rl = opts.rowLabels, cl = opts.colLabels;
    const ctx = canvas.getContext("2d");
    ctx.font = "11px " + getComputedStyle(document.body).getPropertyValue("--f-mono");
    const padL = rl ? Math.min(90, Math.max(...rl.map((s) => ctx.measureText(String(s)).width)) + 10) : 0;
    const longCol = cl ? Math.max(...cl.map((s) => String(s).length)) : 0;
    const padT = cl ? (longCol > 2 ? Math.min(70, longCol * 7 + 8) : 18) : 0;
    let cell = opts.cell || 22;
    if (opts.maxWidth) cell = Math.max(opts.minCell || 3, Math.min(cell, Math.floor((opts.maxWidth - padL) / Math.max(1, C))));
    const cw = cell, ch = opts.cellH || cell;
    const W = padL + C * cw + 2, H = padT + R * ch + 2;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.width = W + "px"; canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    let vmin = opts.vmin, vmax = opts.vmax;
    if (vmin === undefined || vmax === undefined) {
      let lo = Infinity, hi = -Infinity;
      data.forEach((r) => r.forEach((v) => { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
      if (opts.mode === "div") { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; vmin = vmin ?? -m; vmax = vmax ?? m; }
      else { vmin = vmin ?? lo; vmax = vmax ?? hi; }
    }
    state.vmin = vmin; state.vmax = vmax; state.padL = padL; state.padT = padT; state.cw = cw; state.ch = ch;
    ctx.font = (opts.annotateFont || Math.min(11, cw * 0.36)) + "px " + getComputedStyle(document.body).getPropertyValue("--f-mono");
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      const v = data[i][j];
      let col;
      if (v === null || v === undefined || Number.isNaN(v)) col = Theme.rgb("--sunk");
      else if (v === -Infinity) col = Theme.rgb("--sunk");
      else if (opts.mode === "div") col = divColor(v / Math.max(Math.abs(vmin), Math.abs(vmax)));
      else col = seqColor((v - vmin) / (vmax - vmin || 1));
      ctx.fillStyle = rgbStr(col);
      ctx.fillRect(padL + j * cw, padT + i * ch, cw - (cw > 6 ? 1 : 0), ch - (ch > 6 ? 1 : 0));
      if (v === -Infinity && cw >= 14) { ctx.fillStyle = Theme.v("--faint"); ctx.fillText("−∞", padL + j * cw + cw / 2, padT + i * ch + ch / 2); }
      else if (opts.annotate && cw >= 26 && Number.isFinite(v)) {
        ctx.fillStyle = lum(col) > 0.55 ? "#141a26" : "#ffffff";
        ctx.fillText(fmt(v, opts.digits ?? 2), padL + j * cw + cw / 2, padT + i * ch + ch / 2);
      }
    }
    const hl = opts.highlight || {};
    ctx.strokeStyle = Theme.v("--bad"); ctx.lineWidth = 2;
    (hl.rows || []).forEach((i) => ctx.strokeRect(padL + 1, padT + i * ch + 1, C * cw - 2, ch - 2));
    (hl.cols || []).forEach((j) => ctx.strokeRect(padL + j * cw + 1, padT + 1, cw - 2, R * ch - 2));
    (hl.cells || []).forEach(([i, j]) => ctx.strokeRect(padL + j * cw + 1, padT + i * ch + 1, cw - 3, ch - 3));
    ctx.fillStyle = Theme.v("--muted");
    ctx.font = "11px " + getComputedStyle(document.body).getPropertyValue("--f-mono");
    if (rl) { ctx.textAlign = "right"; rl.forEach((s, i) => { if (ch >= 9 || i % Math.ceil(9 / ch) === 0) ctx.fillText(String(s), padL - 5, padT + i * ch + ch / 2); }); }
    if (cl) {
      cl.forEach((s, j) => {
        if (cw < 8 && j % Math.ceil(10 / cw) !== 0) return;
        const x = padL + j * cw + cw / 2;
        if (longCol > 2) { ctx.save(); ctx.translate(x, padT - 4); ctx.rotate(-Math.PI / 2); ctx.textAlign = "left"; ctx.fillText(String(s), 0, 0); ctx.restore(); }
        else { ctx.textAlign = "center"; ctx.fillText(String(s), x, padT - 8); }
      });
    }
  }
  function cellAt(ev) {
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left - state.padL, y = ev.clientY - r.top - state.padT;
    const j = Math.floor(x / state.cw), i = Math.floor(y / state.ch);
    if (i < 0 || j < 0 || i >= state.data.length || j >= state.data[0].length) return null;
    return [i, j];
  }
  canvas.addEventListener("mousemove", (ev) => {
    const c = cellAt(ev);
    if (!c) return Tip.hide();
    const [i, j] = c, v = state.data[i][j], o = state.opts;
    const txt = o.tip ? o.tip(i, j, v) : `${o.rowLabels ? "row " + o.rowLabels[i] : "row " + i}, ${o.colLabels ? "col " + o.colLabels[j] : "col " + j}\nvalue ${fmt(v, 4)}`;
    Tip.show(ev.clientX, ev.clientY, txt);
    o.onHover && o.onHover(i, j);
  });
  canvas.addEventListener("mouseleave", () => { Tip.hide(); state.opts.onHover && state.opts.onHover(null, null); });
  canvas.addEventListener("click", (ev) => { const c = cellAt(ev); if (c && state.opts.onClick) state.opts.onClick(c[0], c[1]); });
  const off = Theme.on(draw);
  wrap.update = (d, o) => { state.data = d; if (o) state.opts = { ...state.opts, ...o }; draw(); };
  wrap.destroy = off;
  requestAnimationFrame(draw);
  draw();
  return wrap;
}

/* ---------- bar list (probabilities) ---------- */
function bars(items, opts = {}) { // items: [{label, value, cls}]
  const max = opts.max ?? Math.max(...items.map((d) => d.value), 1e-9);
  return el("div", { class: "bars" }, items.map((d) =>
    el("div", { class: "bar " + (d.cls || ""), title: d.title || "" },
      el("span", { class: "lab" }, d.label),
      el("span", { class: "track" }, el("span", { class: "fill", style: { width: (100 * d.value) / max + "%" } })),
      el("span", { class: "val" }, opts.pct ? (100 * d.value).toFixed(1) + "%" : fmt(d.value, opts.digits ?? 3)))));
}

/* ---------- line chart (SVG) ---------- */
function lineChart(series, opts = {}) {
  // series: [{name, values:[y] or points:[[x,y]], color}]
  const W = opts.width || 560, H = opts.height || 220, m = { l: 44, r: 12, t: 12, b: 28 };
  const pts = series.map((s) => s.points || s.values.map((y, i) => [i * (opts.xStep || 1), y]));
  const xs = pts.flat().map((p) => p[0]), ys = pts.flat().map((p) => p[1]).filter(Number.isFinite);
  const tx = opts.logx ? Math.log10 : (x) => x, ty = opts.logy ? Math.log10 : (y) => y;
  const x0 = tx(opts.xmin ?? Math.min(...xs)), x1 = tx(opts.xmax ?? Math.max(...xs));
  const y0 = ty(opts.ymin ?? Math.min(...ys)), y1 = ty(opts.ymax ?? Math.max(...ys));
  const X = (x) => m.l + ((tx(x) - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (y) => H - m.b - ((ty(y) - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "chart"); svg.style.width = "100%"; svg.style.maxWidth = W + "px";
  const add = (tag, a, txt) => { const n = document.createElementNS(ns, tag); for (const k in a) n.setAttribute(k, a[k]); if (txt !== undefined) n.textContent = txt; svg.append(n); return n; };
  const yt = opts.yTicks || niceTicks(opts.ymin ?? Math.min(...ys), opts.ymax ?? Math.max(...ys), 4, opts.logy);
  yt.forEach((v) => { add("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }); add("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, opts.yFmt ? opts.yFmt(v) : fmtTick(v)); });
  const xt = opts.xTicks || niceTicks(opts.xmin ?? Math.min(...xs), opts.xmax ?? Math.max(...xs), 5, opts.logx);
  xt.forEach((v) => add("text", { x: X(v), y: H - 8, "text-anchor": "middle" }, opts.xFmt ? opts.xFmt(v) : fmtTick(v)));
  add("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" });
  if (opts.xLabel) add("text", { x: W - m.r, y: H - m.b - 6, "text-anchor": "end" }, opts.xLabel);
  series.forEach((s, si) => {
    const p = pts[si].filter((q) => Number.isFinite(q[1]));
    const color = s.color || Theme.v(si === 0 ? "--accent" : "--neg");
    add("path", { d: p.map((q, i) => (i ? "L" : "M") + X(q[0]).toFixed(1) + "," + Y(q[1]).toFixed(1)).join(""), fill: "none", stroke: color, "stroke-width": 2 });
    if (s.dots) p.forEach((q) => add("circle", { cx: X(q[0]), cy: Y(q[1]), r: 3.5, fill: color }));
  });
  (opts.markers || []).forEach((mk) => { add("line", { x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: H - m.b, stroke: Theme.v("--bad"), "stroke-dasharray": "4 3" }); if (mk.label) add("text", { x: X(mk.x) + 4, y: m.t + 10 }, mk.label); });
  if (series.length > 1) series.forEach((s, si) => add("text", { x: m.l + 8 + si * 130, y: m.t + 10, fill: s.color || Theme.v(si === 0 ? "--accent" : "--neg") }, "■ " + s.name));
  return svg;
}
function niceTicks(lo, hi, n, log) {
  if (log) { const out = []; for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) out.push(10 ** e); return out.filter((v) => v >= lo * 0.999 && v <= hi * 1.001); }
  const span = hi - lo || 1, step = 10 ** Math.floor(Math.log10(span / n)), err = (n * step) / span;
  const s = step * (err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1);
  const out = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(10));
  return out;
}
const fmtTick = (v) => (Math.abs(v) >= 1000 ? (v / 1000) + "k" : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(0) : String(+v.toFixed(3)));

/* ---------- controls ---------- */
let _uid = 0;
const uid = (p = "c") => p + "-" + ++_uid;
function slider(label, { min, max, step = 1, value, fmt: f = (v) => v, id }, onInput) {
  id = id || uid("s");
  const out = el("output", { for: id }, f(value));
  const inp = el("input", { type: "range", id, min, max, step, value, "aria-label": label });
  inp.addEventListener("input", () => { out.textContent = f(+inp.value); onInput(+inp.value); });
  const box = el("div", { class: "ctl" }, el("label", { for: id }, label), inp, out);
  box.set = (v) => { inp.value = v; out.textContent = f(+v); };
  box.input = inp;
  return box;
}
function seg(options, value, onChange, label) {
  const box = el("div", { class: "ctl" }, label ? el("label", {}, label) : null);
  const s = el("div", { class: "seg", role: "group" });
  const btns = options.map((o) => {
    const [val, text] = Array.isArray(o) ? o : [o, String(o)];
    const b = el("button", { type: "button", class: val === value ? "on" : "", "aria-pressed": String(val === value) }, text);
    b.addEventListener("click", () => { btns.forEach((x) => { x.classList.remove("on"); x.setAttribute("aria-pressed", "false"); }); b.classList.add("on"); b.setAttribute("aria-pressed", "true"); onChange(val); });
    return b;
  });
  s.append(...btns); box.append(s);
  return box;
}
function select(label, options, value, onChange, id) {
  id = id || uid("sel");
  const s = el("select", { id }, options.map((o) => { const [v, t] = Array.isArray(o) ? o : [o, o]; return el("option", { value: v, selected: String(v) === String(value) }, t); }));
  s.addEventListener("change", () => onChange(s.value));
  const box = el("div", { class: "ctl" }, el("label", { for: id }, label), s);
  box.sel = s;
  return box;
}
const button = (text, onClick, cls = "") => el("button", { type: "button", class: "btn " + cls, onclick: onClick }, text);
const shapeChip = (text, kind = "") => el("span", { class: "shape " + kind }, text);

/* ---------- widget frame ---------- */
function frame(host, { title, tag, sub }) {
  host.classList.add("widget");
  const body = el("div", { class: "w-body" });
  host.replaceChildren(el("div", { class: "w-head" }, tag ? el("span", { class: "w-tag" }, tag) : null, el("h3", {}, title), sub ? el("div", { class: "w-sub" }, sub) : null), body);
  return body;
}

/* ---------- widget registry + lazy mounting ---------- */
const Widgets = {
  defs: {},
  register(name, fn) { this.defs[name] = fn; },
  mountIn(root) {
    $$("[data-widget]", root).forEach((host) => {
      if (host.dataset.mounted) return;
      host.dataset.mounted = "1";
      const fn = this.defs[host.dataset.widget];
      if (!fn) { host.textContent = "Missing widget: " + host.dataset.widget; return; }
      const opts = host.dataset.opts ? JSON.parse(host.dataset.opts) : {};
      try { const r = fn(host, opts); if (r && r.catch) r.catch((e) => { console.error(e); host.append(el("p", { class: "loading" }, "This figure failed to load: " + e.message)); }); }
      catch (e) { console.error(e); host.append(el("p", { class: "loading" }, "This figure failed to load: " + e.message)); }
    });
  },
};

/* ---------- progress ---------- */
const Progress = {
  get() { return store.get("progress", {}); },
  mark(key) { const p = this.get(); if (!p[key]) { p[key] = Date.now(); store.set("progress", p); Router.refreshNav(); } },
};

/* ---------- router ---------- */
const Router = {
  views: [], current: null,
  init() {
    this.views = $$(".view").map((v) => v.id.replace("view-", ""));
    $("#menu-btn").addEventListener("click", () => $(".rail").classList.toggle("open"));
    window.addEventListener("hashchange", () => this.go(location.hash.slice(1)));
    const start = location.hash.slice(1) || "start";
    this.go(start, true);
  },
  go(name, first) {
    let step = null;
    const m = (name || "").match(/^(l\d+)-(\d+)$/);
    if (m) { name = m[1]; step = +m[2] - 1; }
    if (!this.views.includes(name)) name = "start";
    if (typeof Lessons !== "undefined" && Lessons.byId[name]) Lessons.render(name, step);
    $$(".view").forEach((v) => (v.hidden = v.id !== "view-" + name));
    const view = $("#view-" + name);
    Widgets.mountIn(view);
    $$(".nav-group a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + name));
    $(".rail").classList.remove("open");
    if (!first) window.scrollTo({ top: 0 });
    this.current = name;
    if (/^l\d+$/.test(name)) Progress.mark("visit:" + name);
    document.title = (view.dataset.title ? view.dataset.title + " · " : "") + "Inside the Transformer";
    this.refreshNav();
  },
  refreshNav() {
    const p = Progress.get();
    $$(".nav-group a").forEach((a) => {
      const k = a.getAttribute("href").slice(1), d = a.querySelector(".done");
      if (d) d.textContent = p["quiz:" + k] ? "✓" : "";
    });
    const chapters = ["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8", "l9"], done = chapters.filter((c) => p["quiz:" + c]).length;
    const bar = $("#progress-fill");
    if (bar) bar.style.width = (100 * done) / chapters.length + "%";
    const t = $("#progress-text");
    if (t) t.textContent = `${done} of ${chapters.length} lessons finished`;
  },
};
