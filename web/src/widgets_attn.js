/* ============================================================================
   widgets_attn.js: Chapter 2 (attention) and Chapter 3 (multi-head, positions).
   ========================================================================== */

/* editable matrix */
function matInput(M, onChange, { step = 0.5 } = {}) {
  const g = el("div", { class: "mat", style: { gridTemplateColumns: `repeat(${M[0].length}, auto)` } });
  M.forEach((row, i) => row.forEach((v, j) => {
    const inp = el("input", { type: "number", step, value: v, "aria-label": `row ${i} col ${j}` });
    inp.addEventListener("input", () => { const x = parseFloat(inp.value); if (Number.isFinite(x)) { M[i][j] = x; onChange(); } });
    g.append(inp);
  }));
  return g;
}
const attnMath = {
  scores(Q, K, scale) { const dk = Q[0].length; return Q.map((q) => K.map((k) => TF.dot(q, k) / (scale ? Math.sqrt(dk) : 1))); },
  mask(S, M) { return S.map((r, i) => r.map((v, j) => (M[i][j] ? v : -Infinity))); },
  softmaxRows(S) { return S.map((r) => (r.every((v) => v === -Infinity) ? r.map(() => NaN) : TF.softmax(r))); },
};

/* ===== 2.1 the attention calculator ===== */
Widgets.register("attention-calc", (host, opts) => {
  const body = frame(host, { tag: opts.lab ? "Lab" : "equation 1", title: "Attention, one step at a time", sub: "Edit any number in Q, K or V. Hover the scores to see each dot product; hover the output to see the weighted sum. Click a weight cell to block it (mask)." });
  const presets = {
    "notebook example": { Q: [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]], K: [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]], V: [[10, 0], [0, 10], [5, 5]] },
    "everyone wants token 1": { Q: [[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]], K: [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]], V: [[10, 0], [0, 10], [5, 5]] },
    "zero queries → uniform": { Q: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], K: [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]], V: [[10, 0], [0, 10], [5, 5]] },
    "very sharp (big numbers)": { Q: [[4, 4, 0, 0], [0, 0, 4, 4], [4, 0, 4, 0]], K: [[2, 2, 0, 0], [0, 0, 2, 2], [2, 0, 2, 0]], V: [[10, 0], [0, 10], [5, 5]] },
  };
  let st = JSON.parse(JSON.stringify(presets["notebook example"])), scale = true, maskMode = "none", custom = [[1, 1, 1], [1, 1, 1], [1, 1, 1]], hover = null;
  const toks = ["tok0", "tok1", "tok2"];
  const inputs = el("div"), pipe = el("div"), explain = el("div", { class: "w-caption", style: { minHeight: "2.6em" } });
  function maskM() { return maskMode === "causal" ? toks.map((_, i) => toks.map((_, j) => j <= i)) : maskMode === "custom" ? custom.map((r) => r.map(Boolean)) : toks.map(() => toks.map(() => true)); }
  function drawInputs() {
    inputs.replaceChildren(el("div", { class: "mat-row", style: { flexWrap: "wrap", gap: "1.2rem" } },
      el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, "Q  (3, d_k=4)"), matInput(st.Q, drawPipe)),
      el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, "K  (3, 4)"), matInput(st.K, drawPipe)),
      el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, "V  (3, d_v=2)"), matInput(st.V, drawPipe, { step: 1 }))));
  }
  function drawPipe() {
    const raw = attnMath.scores(st.Q, st.K, false), S = attnMath.scores(st.Q, st.K, scale), M = maskM(), Sm = attnMath.mask(S, M), W = attnMath.softmaxRows(Sm);
    const out = W.map((w) => [0, 1].map((c) => w.reduce((s, x, j) => s + x * st.V[j][c], 0)));
    const dk = st.Q[0].length;
    const hm = (data, o) => heatmap(data, { rowLabels: toks, colLabels: o.cols || toks, annotate: true, cell: 40, ...o });
    pipe.replaceChildren(el("div", { class: "pipe" },
      el("div", { class: "stage" }, el("span", { class: "st-title" }, "1 · scores = Q Kᵀ"), shapeChip("(3, 3)", "attn"),
        hm(raw, { mode: "div", onHover: (i, j) => { hover = i === null ? null : ["s", i, j]; drawExplain(raw, W, out); }, tip: (i, j) => `q${i}·k${j} = ${st.Q[i].map((q, c) => `${q}×${st.K[j][c]}`).join(" + ")} = ${fmt(raw[i][j], 2)}` })),
      el("span", { class: "arrow" }, "→"),
      el("div", { class: "stage" }, el("span", { class: "st-title" }, scale ? `2 · ÷ √d_k = ÷ ${fmt(Math.sqrt(dk), 0)}` : "2 · (scaling OFF)"), shapeChip("(3, 3)", "attn"), hm(S, { mode: "div" })),
      el("span", { class: "arrow" }, "→"),
      el("div", { class: "stage" }, el("span", { class: "st-title" }, "3 · mask → −∞"), shapeChip("(3, 3)", "attn"), hm(Sm, { mode: "div" })),
      el("span", { class: "arrow" }, "→"),
      el("div", { class: "stage" }, el("span", { class: "st-title" }, "4 · softmax each row"), shapeChip("(3, 3)", "soft"),
        hm(W, { vmin: 0, vmax: 1, onClick: (i, j) => { maskMode = "custom"; custom[i][j] = custom[i][j] ? 0 : 1; segMask.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.textContent === "custom (click)")); drawPipe(); }, tip: (i, j, v) => `weight tok${i} → tok${j} = ${fmt(v, 3)}\n(click to ${M[i][j] ? "block" : "unblock"})` }),
        el("span", { class: "w-caption" }, "row sums: " + W.map((r) => fmt(r.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0), 2)).join(", "))),
      el("span", { class: "arrow" }, "→"),
      el("div", { class: "stage" }, el("span", { class: "st-title" }, "5 · out = weights @ V"), shapeChip("(3, 2)", "attn"),
        hm(out, { cols: ["v[0]", "v[1]"], vmin: 0, vmax: 10, onHover: (i) => { hover = i === null ? null : ["o", i]; drawExplain(raw, W, out); }, tip: (i, j) => `out${i}[${j}] = ${W[i].map((w, k) => `${fmt(w, 2)}×${st.V[k][j]}`).join(" + ")} = ${fmt(out[i][j], 2)}` }))));
    drawExplain(raw, W, out);
  }
  function drawExplain(raw, W, out) {
    if (!hover) { explain.textContent = `Read each row as one token asking "who's relevant to me?". Row i of the weights says how much of each value token i takes. Try turning scaling off with the "very sharp" preset: the softmax becomes one-hot.`; return; }
    if (hover[0] === "s") { const [, i, j] = hover; explain.textContent = `Score tok${i}→tok${j}: the dot product of query ${i} with key ${j}. Large when they point the same way. Score ${fmt(raw[i][j], 2)} → weight ${fmt(W[i][j], 3)} after scaling and softmax.`; }
    else { const i = hover[1]; explain.textContent = `Output for tok${i} = ${W[i].map((w, k) => `${fmt(w, 2)}·v${k}`).join(" + ")} = [${out[i].map((x) => fmt(x, 2)).join(", ")}]: a blend of the values, mixed by row ${i} of the weights.`; }
  }
  const segMask = seg([["none", "no mask"], ["causal", "causal"], ["custom", "custom (click)"]], maskMode, (v) => { maskMode = v; if (v === "custom") custom = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]; drawPipe(); }, "mask");
  body.append(el("div", { class: "controls" },
    select("preset", Object.keys(presets), "notebook example", (v) => { st = JSON.parse(JSON.stringify(presets[v])); drawInputs(); drawPipe(); }),
    seg([[true, "÷ √d_k on"], [false, "off"]], true, (v) => { scale = v; drawPipe(); }, "scaling"), segMask), inputs, el("div", { style: { height: ".8rem" } }), pipe, explain);
  drawInputs(); drawPipe();
});

/* ===== 2.2 Q·K geometry: a soft dictionary lookup ===== */
Widgets.register("qk-geometry", (host, opts) => {
  const body = frame(host, { tag: opts.lab ? "Lab" : "intuition", title: "A soft dictionary lookup you can drag", sub: "Four keys, each carrying a coloured value. Drag the query (and the keys). Weights = softmax(q·k / √2 × sharpness). The output is the weight-blended colour." });
  const keys = [{ k: [2.2, 0.6], c: [230, 80, 60], n: "k0" }, { k: [-1.2, 2.0], c: [60, 160, 90], n: "k1" }, { k: [-1.8, -1.4], c: [60, 100, 230], n: "k2" }, { k: [1.0, -2.2], c: [230, 180, 40], n: "k3" }];
  let q = [1.6, 1.2], sharp = 1, hard = false, drag = null;
  const S = 320, sc = 45, ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${S} ${S}`); svg.style.width = "100%"; svg.style.maxWidth = "360px"; svg.style.touchAction = "none";
  const side = el("div", { class: "w-col w-grow" });
  const P = (v) => [S / 2 + v[0] * sc, S / 2 - v[1] * sc];
  function draw() {
    const scores = keys.map((k) => (TF.dot(q, k.k) / Math.SQRT2) * sharp);
    let w = TF.softmax(scores);
    if (hard) { const m = scores.indexOf(Math.max(...scores)); w = w.map((_, i) => (i === m ? 1 : 0)); }
    const out = [0, 1, 2].map((c) => Math.round(keys.reduce((s, k, i) => s + w[i] * k.c[c], 0)));
    svg.replaceChildren();
    const mk = (tag, at, txt) => { const n = document.createElementNS(ns, tag); for (const kk in at) n.setAttribute(kk, at[kk]); if (txt) n.textContent = txt; svg.append(n); return n; };
    mk("line", { x1: 0, x2: S, y1: S / 2, y2: S / 2, stroke: Theme.v("--line") }); mk("line", { y1: 0, y2: S, x1: S / 2, x2: S / 2, stroke: Theme.v("--line") });
    keys.forEach((k, i) => {
      const [x, y] = P(k.k);
      mk("line", { x1: S / 2, y1: S / 2, x2: x, y2: y, stroke: rgbStr(k.c), "stroke-width": 1.5, "stroke-dasharray": "3 3" });
      const c = mk("circle", { cx: x, cy: y, r: 6 + 22 * w[i], fill: rgbStr(k.c), "fill-opacity": 0.85, style: "cursor:grab" });
      c.addEventListener("pointerdown", (e) => { drag = i; svg.setPointerCapture(e.pointerId); });
      mk("text", { x: x + 10, y: y + 26 + 10 * w[i], fill: Theme.v("--ink"), "font-size": 12 }, `${k.n}  ${fmt(w[i] * 100, 0)}%`);
    });
    const [qx, qy] = P(q);
    mk("line", { x1: S / 2, y1: S / 2, x2: qx, y2: qy, stroke: Theme.v("--ink"), "stroke-width": 3 });
    const qc = mk("circle", { cx: qx, cy: qy, r: 9, fill: Theme.v("--ink"), style: "cursor:grab" });
    qc.addEventListener("pointerdown", (e) => { drag = "q"; svg.setPointerCapture(e.pointerId); });
    mk("text", { x: qx + 12, y: qy - 8, fill: Theme.v("--ink"), "font-size": 14, "font-weight": 700 }, "q");
    side.replaceChildren(el("span", { class: "w-label" }, "weights"), bars(keys.map((k, i) => ({ label: `${k.n} (${fmt(scores[i], 2)})`, value: w[i] })), { max: 1, pct: true }),
      el("div", { class: "mat-row" }, el("span", { class: "w-label" }, "output value"), el("span", { style: { width: "3rem", height: "2rem", borderRadius: "6px", background: rgbStr(out), border: "1px solid var(--line)" } })),
      el("p", { class: "w-caption" }, hard ? "Hard lookup (like a Python dict): only the best key counts. No gradient flows to the other keys, so it can't be learned with backprop." : "The query points towards the keys it wants. The more aligned (and the larger the sharpness), the more weight. Raise sharpness to see why scores must be kept small: the softmax becomes a hard lookup."));
  }
  svg.addEventListener("pointermove", (e) => {
    if (drag === null) return;
    const r = svg.getBoundingClientRect(), k = S / r.width;
    const v = [clamp(((e.clientX - r.left) * k - S / 2) / sc, -3.4, 3.4), clamp((S / 2 - (e.clientY - r.top) * k) / sc, -3.4, 3.4)];
    if (drag === "q") q = v; else keys[drag].k = v;
    draw();
  });
  svg.addEventListener("pointerup", () => (drag = null));
  Theme.on(draw);
  body.append(el("div", { class: "controls" }, slider("sharpness", { min: 0.1, max: 6, step: 0.1, value: 1, fmt: (v) => "×" + v.toFixed(1) }, (v) => { sharp = v; draw(); }),
    seg([[false, "soft (attention)"], [true, "hard (dict lookup)"]], false, (v) => { hard = v; draw(); })), el("div", { class: "w-row" }, svg, side));
  draw();
});

/* ===== 2.3 why √d_k ===== */
Widgets.register("sqrt-dk", (host) => {
  const body = frame(host, { tag: "§3.2.1", title: "Why divide by √d_k? Measure it", sub: "q and k have d_k random components (mean 0, variance 1). Slide d_k and watch the dot products spread out, and the unscaled softmax collapse to one-hot." });
  let e = 6, seed = 1;
  const view = el("div"), curve = el("div");
  function stats(dk, scale, r, trials = 300) {
    let maxw = 0, gn = 0;
    for (let t = 0; t < trials; t++) {
      const q = Array.from({ length: dk }, r.normal);
      const s = Array.from({ length: 10 }, () => TF.dot(q, Array.from({ length: dk }, r.normal)) / (scale ? Math.sqrt(dk) : 1));
      const p = TF.softmax(s), g = Array.from({ length: 10 }, r.normal);
      const pg = TF.dot(p, g);
      maxw += Math.max(...p) / trials;
      gn += Math.sqrt(p.reduce((acc, pi, i) => acc + (pi * (g[i] - pg)) ** 2, 0)) / trials;  // |J^T g| for softmax
    }
    return [maxw, gn];
  }
  function draw() {
    const dk = 2 ** e, r = TF.rng(seed);
    const dots = Array.from({ length: 1500 }, () => { let s = 0; for (let i = 0; i < dk; i++) s += r.normal() * r.normal(); return s; });
    const sd = Math.sqrt(dots.reduce((a, b) => a + b * b, 0) / dots.length);
    const q = Array.from({ length: dk }, r.normal), ks = Array.from({ length: 10 }, () => Array.from({ length: dk }, r.normal));
    const raw = ks.map((k) => TF.dot(q, k)), pu = TF.softmax(raw), ps = TF.softmax(raw.map((v) => v / Math.sqrt(dk)));
    const hist = (vals, lo, hi, n = 24) => { const b = new Array(n).fill(0); vals.forEach((v) => { const i = Math.floor(((v - lo) / (hi - lo)) * n); if (i >= 0 && i < n) b[i]++; }); return b; };
    const span = Math.max(4 * sd, 4);
    const hu = hist(dots, -span, span), hs = hist(dots.map((v) => v / Math.sqrt(dk)), -span, span);
    const histRow = (h, label, color) => el("div", { class: "w-col" }, el("span", { class: "w-label" }, label),
      el("div", { style: { display: "flex", alignItems: "flex-end", gap: "1px", height: "70px", width: "260px", borderBottom: "1px solid var(--line)" } },
        ...h.map((c) => el("span", { style: { flex: 1, background: `var(${color})`, height: (100 * c) / Math.max(...h, 1) + "%" } }))),
      el("span", { class: "w-caption", style: { marginTop: 0 } }, `axis −${fmt(span, 0)} … +${fmt(span, 0)}`));
    view.replaceChildren(
      el("p", {}, `d_k = ${dk}: q·k has variance ≈ ${dk} (measured std ${fmt(sd, 2)} ≈ √${dk} = ${fmt(Math.sqrt(dk), 2)}). After ÷√d_k the std is back to ≈ 1.`),
      el("div", { class: "w-row" }, histRow(hu, "q·k (unscaled)", "--neg"), histRow(hs, "q·k / √d_k", "--accent")),
      el("div", { class: "w-row", style: { marginTop: ".8rem" } },
        el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, `one query, 10 keys: unscaled · max weight ${fmt(Math.max(...pu), 2)}`), bars(pu.map((p, i) => ({ label: "k" + i, value: p, cls: "alt" })), { max: 1, pct: true })),
        el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, `scaled · max weight ${fmt(Math.max(...ps), 2)}`), bars(ps.map((p, i) => ({ label: "k" + i, value: p })), { max: 1, pct: true }))));
  }
  function drawCurve() {
    const dims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], r = TF.rng(42);
    const u = dims.map((d) => stats(d, false, r)), s = dims.map((d) => stats(d, true, r));
    curve.replaceChildren(el("div", { class: "w-row" },
      el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, "average largest weight (1 = one-hot)"), lineChart([{ name: "÷√d_k", points: dims.map((d, i) => [d, s[i][0]]), dots: true }, { name: "no scaling", points: dims.map((d, i) => [d, u[i][0]]), dots: true }], { logx: true, ymin: 0, ymax: 1, width: 380, height: 190 })),
      el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, "gradient size through the softmax (log scale)"), lineChart([{ name: "÷√d_k", points: dims.map((d, i) => [d, s[i][1]]), dots: true }, { name: "no scaling", points: dims.map((d, i) => [d, Math.max(u[i][1], 1e-6)]), dots: true }], { logx: true, logy: true, width: 380, height: 190 }))),
      el("p", { class: "w-caption" }, "Without scaling, bigger heads give bigger scores. The softmax saturates (one key gets everything) and the gradient flowing back shrinks by orders of magnitude, so learning stalls. With ÷√d_k, both curves stay flat whatever the head size."));
  }
  body.append(el("div", { class: "controls" }, slider("d_k", { min: 0, max: 10, value: e, fmt: (v) => String(2 ** v) }, (v) => { e = v; draw(); }), button("Resample", () => { seed++; draw(); })), view, el("div", { style: { height: "1rem" } }), curve);
  draw(); drawCurve();
});

/* ===== 2.4 masks ===== */
Widgets.register("masks", (host) => {
  const body = frame(host, { tag: "masks", title: "Padding masks, causal masks, and why −∞", sub: "A batch of two sentences padded to length 6. Pick the mask, then compare masking the scores (right) with zeroing weights after the softmax (wrong)." });
  let len = [4, 2], mode = "both", which = 1, method = "scores";
  const T = 6, r = TF.rng(5);
  const S0 = Array.from({ length: T }, () => Array.from({ length: T }, () => +(r.normal() * 1.2).toFixed(2)));
  const view = el("div");
  function draw() {
    const L = len[which];
    const real = (j) => j < L;
    const M = Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => (mode !== "causal" ? real(j) : true) && (mode !== "padding" ? j <= i : true)));
    let W;
    if (method === "scores") W = attnMath.softmaxRows(attnMath.mask(S0, M));
    else W = TF.softmax ? S0.map((row, i) => TF.softmax(row).map((p, j) => (M[i][j] ? p : 0))) : null;
    const labels = Array.from({ length: T }, (_, j) => (real(j) ? "w" + j : "<pad>"));
    view.replaceChildren(el("div", { class: "w-row" },
      el("div", { class: "w-col" }, el("span", { class: "w-label" }, `mask for sentence ${which} (True = may look)`), heatmap(M.map((r) => r.map((v) => (v ? 1 : 0))), { rowLabels: labels, colLabels: labels, vmin: 0, vmax: 1, cell: 34, tip: (i, j, v) => `query ${labels[i]} → key ${labels[j]}: ${v ? "allowed" : "blocked"}` })),
      el("div", { class: "w-col" }, el("span", { class: "w-label" }, method === "scores" ? "weights = softmax(masked scores)" : "weights = softmax(scores), then zeroed"),
        heatmap(W, { rowLabels: labels, colLabels: labels, vmin: 0, vmax: 1, annotate: true, cell: 40 }),
        el("span", { class: "w-caption" }, "row sums: ", ...W.map((r) => { const s = r.reduce((a, b) => a + b, 0); return el("span", { class: "badge " + (Math.abs(s - 1) < 1e-6 ? "good" : "bad"), style: { marginRight: ".2rem" } }, fmt(s, 2)); })))),
      el("p", { class: "w-caption" }, method === "scores" ? "Blocked scores become −∞, so exp(−∞) = 0: blocked weights are exactly 0 and every row still sums to 1. Pad queries (bottom rows) still see the real tokens, so no row is fully masked." : "Zeroing after the softmax leaves rows that sum to less than 1, and the blocked tokens still took part in the normalisation. That's why masks act on the scores."));
  }
  body.append(el("div", { class: "controls" },
    seg([[0, "sentence 0"], [1, "sentence 1"]], which, (v) => { which = v; draw(); }, "show"),
    slider("len 0", { min: 1, max: 6, value: len[0] }, (v) => { len[0] = v; draw(); }), slider("len 1", { min: 1, max: 6, value: len[1] }, (v) => { len[1] = v; draw(); }),
    seg([["padding", "padding"], ["causal", "causal"], ["both", "causal & padding"]], mode, (v) => { mode = v; draw(); }, "mask"),
    seg([["scores", "−∞ on scores"], ["weights", "0 on weights (wrong)"]], method, (v) => { method = v; draw(); })), view);
  draw();
});

/* ===== 2.5 the cost of attention ===== */
Widgets.register("attn-cost", (host) => {
  const body = frame(host, { tag: "O(T²)", title: "How big is the attention matrix?", sub: "One T×T matrix of float32 per head per layer, if you store it naively." });
  let e = 12, heads = 32, layers = 32;
  const out = el("div");
  function draw() {
    const T = 2 ** e, one = T * T * 4, all = one * heads * layers;
    const h = (b) => (b >= 1e12 ? fmt(b / 1e12, 1) + " TB" : b >= 1e9 ? fmt(b / 1e9, 1) + " GB" : b >= 1e6 ? fmt(b / 1e6, 1) + " MB" : fmt(b / 1e3, 1) + " KB");
    out.replaceChildren(el("dl", { class: "kv" }, el("dt", {}, "tokens T"), el("dd", {}, T.toLocaleString()), el("dt", {}, "entries T²"), el("dd", {}, (T * T).toLocaleString()),
      el("dt", {}, "one matrix"), el("dd", {}, h(one)), el("dt", {}, `× ${heads} heads × ${layers} layers`), el("dd", { style: { fontWeight: 700 } }, h(all)), el("dt", {}, "QKᵀ multiply-adds"), el("dd", {}, (T * T * 128 * heads * layers).toExponential(2) + "  (d_k = 128)")),
      el("p", { class: "w-caption" }, "Double T and every number quadruples. FlashAttention computes exactly the same result in tiles that fit in fast on-chip memory, so the full matrix is never stored. It's still O(T²) compute, but no longer O(T²) memory."));
  }
  body.append(el("div", { class: "controls" }, slider("T", { min: 6, max: 17, value: e, fmt: (v) => (2 ** v).toLocaleString() }, (v) => { e = v; draw(); }),
    slider("heads", { min: 1, max: 64, value: heads }, (v) => { heads = v; draw(); }), slider("layers", { min: 1, max: 96, value: layers }, (v) => { layers = v; draw(); })), out);
  draw();
});

/* ===== 3.1 why projections ===== */
Widgets.register("no-projection", (host) => {
  const body = frame(host, { tag: "W_Q W_K W_V", title: "Without projections, every token stares at itself", sub: "6 random token vectors (d = 16). Compare attention with Q = K = V = X against learned-style projections Q = XW_Q, K = XW_K." });
  let seed = 1, mode = "raw";
  const view = el("div");
  function draw() {
    const r = TF.rng(seed), d = 16, X = Array.from({ length: 6 }, () => Array.from({ length: d }, r.normal));
    const Wq = Array.from({ length: d }, () => Array.from({ length: d }, () => r.normal() / 4)), Wk = Array.from({ length: d }, () => Array.from({ length: d }, () => r.normal() / 4));
    const Q = mode === "raw" ? X : TF.matmul(X, Wq), K = mode === "raw" ? X : TF.matmul(X, Wk);
    const W = attnMath.softmaxRows(attnMath.scores(Q, K, true));
    let asym = 0; W.forEach((row, i) => row.forEach((v, j) => (asym = Math.max(asym, Math.abs(v - W[j][i])))));
    const diag = W.reduce((s, row, i) => s + row[i], 0) / 6;
    const toks = ["t0", "t1", "t2", "t3", "t4", "t5"];
    view.replaceChildren(el("div", { class: "w-row" }, heatmap(W, { rowLabels: toks, colLabels: toks, vmin: 0, vmax: 1, annotate: true, cell: 44 }),
      el("dl", { class: "kv" }, el("dt", {}, "avg weight on itself"), el("dd", {}, fmt(diag * 100, 0) + "%"), el("dt", {}, "scores symmetric?"), el("dd", {}, mode === "raw" ? "yes: x_i·x_j = x_j·x_i" : "no (W_Q ≠ W_K)"), el("dt", {}, "max |A − Aᵀ|"), el("dd", {}, fmt(asym, 3)))),
      el("p", { class: "w-caption" }, mode === "raw" ? "x·x = |x|² is almost always the biggest dot product, so the diagonal wins. Every token mostly attends to itself, and the pattern is forced to be symmetric." : "With separate W_Q and W_K, what a token looks for differs from how it is found. Attention can now point anywhere, and training decides where."));
  }
  body.append(el("div", { class: "controls" }, seg([["raw", "Q = K = V = X"], ["proj", "Q = XW_Q, K = XW_K"]], mode, (v) => { mode = v; draw(); }), button("New random tokens", () => { seed++; draw(); })), view);
  draw();
});

/* ===== 3.2 multi-head shapes and parameter count ===== */
Widgets.register("mha-shapes", (host) => {
  const body = frame(host, { tag: "§3.2.2", title: "Inside multi-head attention: every shape", sub: "Change the sizes. The parameter count never depends on the number of heads." });
  let B = 2, T = 10, d = 512, h = 8;
  const view = el("div");
  function draw() {
    if (d % h) h = 1;
    const dk = d / h, p = 4 * (d * d + d);
    const step = (code, shape, kind, note) => el("tr", {}, el("td", {}, el("code", {}, code)), el("td", {}, shapeChip(shape, kind)), el("td", { class: "w-caption", style: { marginTop: 0 } }, note));
    view.replaceChildren(el("div", { class: "table-wrap" }, el("table", { class: "t" },
      step("x", `(${B}, ${T}, ${d})`, "emb", "B sentences, T tokens, d_model numbers each"),
      step("q = w_q(x)", `(${B}, ${T}, ${d})`, "lin", "one Linear(d_model, d_model) for ALL heads at once"),
      step("q.view(B, T, h, d_k)", `(${B}, ${T}, ${h}, ${dk})`, "", `cut each token's ${d} numbers into ${h} slices of d_k = ${dk}`),
      step(".transpose(1, 2)", `(${B}, ${h}, ${T}, ${dk})`, "attn", "heads become a batch dim"),
      step("q @ k.transpose(-2, -1)", `(${B}, ${h}, ${T}, ${T})`, "attn", `${B * h} score matrices computed by ONE batched matmul`),
      step("softmax(…) @ v", `(${B}, ${h}, ${T}, ${dk})`, "soft", "each head's weighted average of its values"),
      step(".transpose(1, 2).contiguous().view(B, T, d)", `(${B}, ${T}, ${d})`, "", "concatenate the heads again"),
      step("w_o(…)", `(${B}, ${T}, ${d})`, "lin", "mix information across heads"))),
      el("dl", { class: "kv" }, el("dt", {}, "parameters"), el("dd", {}, `4 × (d² + d) = 4 × (${d}² + ${d}) = ${p.toLocaleString()}`),
        el("dt", {}, "with 1 head"), el("dd", {}, p.toLocaleString() + "  (same)"), el("dt", {}, "score FLOPs"), el("dd", {}, `${h} heads × T² × d_k = ${(h * T * T * dk).toLocaleString()} = T² × d_model (same as 1 head)`)));
  }
  const divs = (n) => [1, 2, 4, 8, 16, 32, 64].filter((x) => n % x === 0);
  const hSel = select("heads h", divs(d), h, (v) => { h = +v; draw(); });
  body.append(el("div", { class: "controls" }, slider("B", { min: 1, max: 8, value: B }, (v) => { B = v; draw(); }), slider("T", { min: 2, max: 64, value: T }, (v) => { T = v; draw(); }),
    select("d_model", [64, 128, 256, 512, 768, 1024], d, (v) => { d = +v; hSel.sel.replaceChildren(...divs(d).map((x) => el("option", { value: x, selected: x === h }, x))); if (d % h) h = 1; draw(); }), hSel), view);
  draw();
});

/* ===== 3.3 permutation equivariance ===== */
Widgets.register("permutation", (host) => {
  const body = frame(host, { tag: "order", title: "Shuffle the words: does attention notice?", sub: "A random multi-head attention layer (d = 8, 2 heads) over 4 words. Each row is a word's output vector. Shuffle the order and compare." });
  const words = ["dog", "bites", "man", "today"], d = 8, r = TF.rng(21);
  const emb = Object.fromEntries(words.map((w) => [w, Array.from({ length: d }, r.normal)]));
  const W = Array.from({ length: 4 }, () => Array.from({ length: d }, () => Array.from({ length: d }, () => r.normal() / Math.sqrt(d))));
  const pe = TF.sinusoidalPE(8, d);
  let order = words.slice(), usePE = false;
  function mha(X) {
    const proj = (M) => TF.matmul(X, M), Q = proj(W[0]), K = proj(W[1]), V = proj(W[2]), out = X.map(() => new Array(d).fill(0));
    for (let hh = 0; hh < 2; hh++) {
      const sl = (A) => A.map((row) => row.slice(hh * 4, hh * 4 + 4));
      const Wt = attnMath.softmaxRows(attnMath.scores(sl(Q), sl(K), true));
      const Vs = sl(V);
      Wt.forEach((wr, i) => { for (let c = 0; c < 4; c++) out[i][hh * 4 + c] = wr.reduce((s, w, j) => s + w * Vs[j][c], 0); });
    }
    return TF.matmul(out, W[3]);
  }
  const run = (ord) => mha(ord.map((w, i) => emb[w].map((v, c) => v + (usePE ? pe[i * d + c] : 0))));
  const view = el("div");
  function draw() {
    const base = run(words), now = run(order);
    const same = order.map((w, i) => { const a = now[i], b = base[words.indexOf(w)]; return Math.max(...a.map((v, c) => Math.abs(v - b[c]))) < 1e-9; });
    view.replaceChildren(el("p", {}, "sentence: ", ...order.map((w) => el("span", { class: "chip", style: { marginRight: ".3rem" } }, w))),
      el("div", { class: "w-row" },
        el("div", { class: "w-col" }, el("span", { class: "w-label" }, "output vectors (rows in current order)"), heatmap(now, { rowLabels: order, mode: "div", cell: 26 })),
        el("div", { class: "w-col" }, el("span", { class: "w-label" }, "same vector as in the original order?"),
          ...order.map((w, i) => el("div", {}, el("span", { class: "chip", style: { minWidth: "4.5rem" } }, w), " ", same[i] ? el("span", { class: "badge good" }, "identical") : el("span", { class: "badge bad" }, "different"))))),
      el("p", { class: "w-caption" }, usePE ? "With positional encodings added, a word's output depends on where it sits, so word order now matters." : "Without positions, every word gets exactly the same output vector wherever it is. The rows just move around. Attention treats the sentence as a set: 'dog bites man' and 'man bites dog' look identical to it."));
  }
  body.append(el("div", { class: "controls" }, button("Shuffle order", () => { const rr = TF.rng(Date.now() % 1e6); for (let i = 3; i > 0; i--) { const j = rr.int(i + 1); [order[i], order[j]] = [order[j], order[i]]; } draw(); }, "primary"),
    button("Original order", () => { order = words.slice(); draw(); }),
    seg([[false, "no positions"], [true, "+ positional encoding"]], usePE, (v) => { usePE = v; draw(); })), view);
  draw();
});

/* ===== 3.4 positional encoding explorer ===== */
Widgets.register("pe-explorer", (host, opts) => {
  const body = frame(host, { tag: opts.lab ? "Lab" : "§3.5", title: "Sinusoidal positional encoding, taken apart", sub: "PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(…). Each pair of dimensions is a clock hand turning at its own speed." });
  let d = 64, p = 10, k = 3;
  const maxLen = 64;
  const heat = el("div"), clocks = el("div"), sim = el("div"), shift = el("div");
  function pe() { const f = TF.sinusoidalPE(maxLen, d); return Array.from({ length: maxLen }, (_, i) => Array.from(f.subarray(i * d, (i + 1) * d))); }
  function draw() {
    const P = pe();
    heat.replaceChildren(el("span", { class: "w-label" }, `PE table: ${maxLen} positions × d = ${d}  (hover for values; red row = position ${p})`),
      heatmap(P, { mode: "div", vmin: -1, vmax: 1, cell: Math.max(4, Math.min(12, Math.floor(620 / d))), cellH: 6, maxWidth: 640, highlight: { rows: [p] }, tip: (i, j, v) => `pos ${i}, dim ${j} (${j % 2 ? "cos" : "sin"}, pair ${j >> 1})\nfrequency 1/10000^(${2 * (j >> 1)}/${d}) = ${(1 / Math.pow(10000, (2 * (j >> 1)) / d)).toExponential(2)}\nvalue ${fmt(v, 3)}`, onClick: (i) => { p = i; pSl.set(p); draw(); } }));
    // clocks for the first 8 pairs (and the slowest)
    const pairs = [0, 1, 2, 3, 4, 6, 8, d / 2 - 1].filter((x, i, a) => x < d / 2 && a.indexOf(x) === i);
    clocks.replaceChildren(el("span", { class: "w-label" }, `position ${p} as clock hands (one per dimension pair)`), el("div", { class: "w-row", style: { gap: ".6rem" } }, ...pairs.map((i) => {
      const w = 1 / Math.pow(10000, (2 * i) / d), ang = p * w, R = 26, ns = "http://www.w3.org/2000/svg", s = document.createElementNS(ns, "svg");
      s.setAttribute("viewBox", "0 0 64 76"); s.setAttribute("width", "64");
      const mk = (tag, at, txt) => { const n = document.createElementNS(ns, tag); for (const kk in at) n.setAttribute(kk, at[kk]); if (txt) n.textContent = txt; s.append(n); };
      mk("circle", { cx: 32, cy: 32, r: R, fill: "none", stroke: Theme.v("--line"), "stroke-width": 2 });
      mk("line", { x1: 32, y1: 32, x2: 32 + R * Math.sin(ang), y2: 32 - R * Math.cos(ang), stroke: Theme.v("--accent"), "stroke-width": 3, "stroke-linecap": "round" });
      mk("text", { x: 32, y: 72, "text-anchor": "middle", fill: Theme.v("--muted"), "font-size": 9 }, `pair ${i} · T≈${fmt((2 * Math.PI) / w, 0)}`);
      return s;
    })), el("p", { class: "w-caption" }, "sin = horizontal reach of the hand, cos = vertical. Pair 0 turns once every 6.3 positions (good for telling neighbours apart). The last pair needs tens of thousands of positions per turn (coarse position). Together the hands give a unique code for every position, like a smooth binary counter."));
    // similarity
    const dots = P.map((row) => TF.dot(row, P[p]));
    sim.replaceChildren(el("div", { class: "w-row" },
      el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, `PE[${p}] · PE[q] for every q`), lineChart([{ name: "dot", values: dots }], { width: 380, height: 180, markers: [{ x: p, label: "p" }] })),
      el("div", { class: "w-col" }, el("span", { class: "w-label" }, "PE · PEᵀ (first 32 positions)"), heatmap(P.slice(0, 32).map((a) => P.slice(0, 32).map((b) => TF.dot(a, b))), { cell: 7, tip: (i, j, v) => `PE[${i}]·PE[${j}] = ${fmt(v, 2)}  (|p−q| = ${Math.abs(i - j)})` }))),
      el("p", { class: "w-caption" }, "Same shape wherever p is: similarity depends (almost) only on the distance |p − q|. That's the band along the diagonal."));
    // relative shift = rotation
    let err = 0;
    for (let i = 0; i < d / 2; i++) {
      const w = 1 / Math.pow(10000, (2 * i) / d), s0 = P[p][2 * i], c0 = P[p][2 * i + 1], ck = Math.cos(k * w), sk = Math.sin(k * w);
      const s1 = s0 * ck + c0 * sk, c1 = c0 * ck - s0 * sk;
      if (p + k < maxLen) err = Math.max(err, Math.abs(s1 - P[p + k][2 * i]), Math.abs(c1 - P[p + k][2 * i + 1]));
    }
    shift.replaceChildren(el("p", {}, `Rotate every pair of PE[${p}] by the angle k·ω_i with k = ${k}: the result equals PE[${p + k}] with max error `, el("b", {}, err.toExponential(1)), ". So 'k positions later' is the same linear map (a rotation) wherever you are. This is the paper's reason for the design, and the seed of RoPE."));
  }
  const pSl = slider("position p", { min: 0, max: maxLen - 1, value: p }, (v) => { p = v; draw(); });
  body.append(el("div", { class: "controls" }, seg([16, 32, 64, 128], d, (v) => { d = v; draw(); }, "d_model"), pSl, slider("shift k", { min: 1, max: 20, value: k }, (v) => { k = v; draw(); })), heat, el("div", { style: { height: ".8rem" } }), clocks, sim, shift);
  draw();
  Theme.on(draw);
});
