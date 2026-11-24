/* ============================================================================
   widgets_basics.js: Chapter 0 (tensors) and Chapter 1 (embeddings) figures.
   ========================================================================== */

/* ---------- tiny grid of numbers (DOM, for small matrices) ---------- */
function numGrid(M, { cls = () => "", fmtv = (v) => fmt(v, 1), title = () => "", style = () => ({}), cellW } = {}) {
  const g = el("div", { class: "mat", style: { gridTemplateColumns: `repeat(${M[0].length}, auto)` } });
  M.forEach((row, i) => row.forEach((v, j) => g.append(el("div", { class: "cell " + cls(i, j), title: title(i, j), style: { ...(cellW ? { width: cellW } : {}), ...style(i, j, v) } }, fmtv(v, i, j)))));
  return g;
}
const hiStyle = (on, kind = "--accent-soft") => (on ? { background: `var(${kind})`, outline: "1.5px solid var(--accent)" } : { background: "var(--bg)" });

/* ===== 0.1 indexing a (B, T, d) tensor ===== */
Widgets.register("index-explorer", (host) => {
  const body = frame(host, { tag: "tensors", title: "Pick pieces out of a (B, T, d) tensor", sub: "x holds 3 sentences × 4 tokens × 5 numbers. Choose an expression and see exactly which numbers it selects." });
  const B = 3, T = 4, D = 5;
  const exprs = {
    "x[0]": { f: (b) => b === 0, shape: "(4, 5)", say: "sentence 0, every token, every number" },
    "x[:, -1]": { f: (b, t) => t === T - 1, shape: "(3, 5)", say: "the LAST token of EVERY sentence" },
    "x[1, 2]": { f: (b, t) => b === 1 && t === 2, shape: "(5,)", say: "one token's vector: sentence 1, token 2" },
    "x[:, :, 0]": { f: (b, t, d) => d === 0, shape: "(3, 4)", say: "number 0 of every token" },
    "x[..., :2]": { f: (b, t, d) => d < 2, shape: "(3, 4, 2)", say: "the first 2 numbers of every token (… means 'all leading dims')" },
    "x[-1, 1:3]": { f: (b, t) => b === 2 && t >= 1 && t < 3, shape: "(2, 5)", say: "last sentence, tokens 1 and 2" },
  };
  let cur = "x[:, -1]";
  const grids = el("div", { class: "w-row" }), info = el("div", { class: "w-caption" });
  function draw() {
    const e = exprs[cur];
    grids.replaceChildren(...Array.from({ length: B }, (_, b) => el("div", { class: "w-col" }, el("span", { class: "w-label" }, `sentence ${b}  (x[${b}])`),
      numGrid(Array.from({ length: T }, (_, t) => Array.from({ length: D }, (_, d) => b * 100 + t * 10 + d)), { fmtv: (v) => String(v).padStart(3, "0"), style: (t, d) => hiStyle(e.f(b, t, d)), title: (t, d) => `x[${b}, ${t}, ${d}]` }))));
    info.replaceChildren(el("code", {}, cur), " → shape ", shapeChip(e.shape), " : ", e.say, ".  (Each cell shows its own index: sentence·token·number.)");
  }
  body.append(el("div", { class: "controls" }, seg(Object.keys(exprs), cur, (v) => { cur = v; draw(); }, "expression")), grids, info);
  draw();
});

/* ===== 0.2 matrix multiplication ===== */
Widgets.register("matmul", (host) => {
  const body = frame(host, { tag: "@", title: "Matrix multiplication = many dot products", sub: "Hover a cell of the result to see the row and column it comes from. Change the shapes to see when @ works." });
  let n = 3, k = 4, m = 2, kB = 4, r = TF.rng(3);
  let A, Bm;
  const rand = (R, C) => Array.from({ length: R }, () => Array.from({ length: C }, () => r.int(7) - 3));
  const regen = () => { A = rand(n, k); Bm = rand(kB, m); };
  regen();
  const view = el("div"), eq = el("div", { class: "w-caption" });
  function draw(hi = null) {
    view.replaceChildren();
    eq.replaceChildren();
    if (k !== kB) {
      view.append(el("div", { class: "mat-row" }, el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, `A (${n}, ${k})`), numGrid(A, { fmtv: (v) => v })), el("span", { class: "op" }, "@"),
        el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, `B (${kB}, ${m})`), numGrid(Bm, { fmtv: (v) => v }))));
      eq.append(el("span", { class: "badge bad" }, "RuntimeError"), ` mat1 and mat2 shapes cannot be multiplied (${n}x${k} and ${kB}x${m}). A's LAST dim (${k}) must equal B's SECOND-TO-LAST dim (${kB}).`);
      return;
    }
    const C = TF.matmul(A, Bm);
    view.append(el("div", { class: "mat-row", style: { flexWrap: "wrap" } },
      el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, `A (${n}, ${k})`), numGrid(A, { fmtv: (v) => v, style: (i) => hiStyle(hi && hi[0] === i) })),
      el("span", { class: "op" }, "@"),
      el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, `B (${k}, ${m})`), numGrid(Bm, { fmtv: (v) => v, style: (i, j) => hiStyle(hi && hi[1] === j) })),
      el("span", { class: "op" }, "="),
      el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, `C (${n}, ${m})`), (() => {
        const g = numGrid(C, { fmtv: (v) => v, style: (i, j) => hiStyle(hi && hi[0] === i && hi[1] === j, "--c-attn") });
        [...g.children].forEach((c, idx) => { const i = Math.floor(idx / m), j = idx % m; c.style.cursor = "pointer"; c.addEventListener("mouseenter", () => draw([i, j])); });
        return g;
      })())));
    if (hi) {
      const [i, j] = hi;
      const terms = A[i].map((a, p) => `(${a})×(${Bm[p][j]})`);
      eq.append(`C[${i}, ${j}] = row ${i} of A · column ${j} of B = `, el("code", {}, terms.join(" + ") + " = " + C[i][j]));
    } else eq.append("Hover any cell of C. ", el("b", {}, "Rule: "), el("code", {}, "(n, k) @ (k, m) → (n, m)"), ". In attention, Q @ Kᵀ is exactly this: every query dotted with every key.");
  }
  body.append(el("div", { class: "controls" },
    slider("n (rows of A)", { min: 1, max: 5, value: n }, (v) => { n = v; regen(); draw(); }),
    slider("k (cols of A)", { min: 1, max: 5, value: k }, (v) => { k = v; regen(); draw(); }),
    slider("rows of B", { min: 1, max: 5, value: kB }, (v) => { kB = v; regen(); draw(); }),
    slider("m (cols of B)", { min: 1, max: 4, value: m }, (v) => { m = v; regen(); draw(); }),
    button("New numbers", () => { regen(); draw(); })), view, eq);
  draw();
});

/* ===== 0.3 broadcasting ===== */
Widgets.register("broadcast", (host) => {
  const body = frame(host, { tag: "broadcasting", title: "Broadcasting: line the shapes up from the right", sub: "Each column is one dim. Equal dims match; a 1 (or a missing dim) is stretched. Anything else is an error." });
  const cases = [
    ["(3, 4) + (4,)", [3, 4], [4], "a bias added to every row"],
    ["(3, 1) + (1, 4)", [3, 1], [1, 4], "a column plus a row → a grid"],
    ["(B,T,d) + (T,d)", [2, 6, 4], [6, 4], "positional encodings added to every sentence"],
    ["scores × mask", [2, 4, 5, 5], [2, 1, 1, 5], "a padding mask applied to (B, heads, T_q, T_k) scores"],
    ["(3, 4) + (3,)", [3, 4], [3], "a common mistake"],
  ];
  let cur = 1;
  const out = el("div");
  function draw() {
    const [, a, b, say] = cases[cur];
    const n = Math.max(a.length, b.length), pa = Array(n - a.length).fill(null).concat(a), pb = Array(n - b.length).fill(null).concat(b);
    let ok = true;
    const res = pa.map((x, i) => { const y = pb[i]; if (x === null) return y; if (y === null) return x; if (x === y) return x; if (x === 1) return y; if (y === 1) return x; ok = false; return "✗"; });
    const cell = (v, other, i) => {
      const stretched = v === null || (v === 1 && other !== 1 && other !== null);
      const bad = res[i] === "✗";
      return el("td", { class: "num", style: { background: bad ? "var(--bad-soft)" : stretched ? "var(--c-norm)" : "transparent" } }, v === null ? "·" : String(v));
    };
    out.replaceChildren(el("div", { class: "table-wrap" }, el("table", { class: "t", style: { width: "auto" } },
      el("tr", {}, el("th", {}, ""), ...pa.map((_, i) => el("th", {}, "dim " + (i - n)))),
      el("tr", {}, el("th", {}, "left"), ...pa.map((v, i) => cell(v, pb[i], i))),
      el("tr", {}, el("th", {}, "right"), ...pb.map((v, i) => cell(v, pa[i], i))),
      el("tr", {}, el("th", {}, "result"), ...res.map((v) => el("td", { class: "num", style: { fontWeight: 700 } }, String(v)))))),
      el("p", {}, ok ? el("span", { class: "badge good" }, "works") : el("span", { class: "badge bad" }, "error"), " ",
        ok ? `Result shape (${res.join(", ")}): ${say}. Yellow cells were stretched.` : `Dims ${pa[n - 1]} and ${pb[n - 1]} don't match and neither is 1. PyTorch lines shapes up from the RIGHT, so (3,) is compared with the 4, not the 3. Fix: reshape to (3, 1).`));
  }
  body.append(el("div", { class: "controls" }, seg(cases.map((c, i) => [i, c[0]]), cur, (v) => { cur = v; draw(); })), out);
  draw();
});

/* ===== 0.4 splitting into heads ===== */
const HEAD_COLORS = ["--c-attn", "--c-ffn", "--c-soft", "--c-lin", "--c-emb", "--c-norm", "--c-pos", "--accent-soft"];
Widgets.register("split-heads", (host) => {
  const body = frame(host, { tag: "reshape", title: "Splitting d_model into heads: view + transpose", sub: "3 tokens, d_model = 8. Colour = the head each number SHOULD belong to. The wrong reshape has the right shape but scrambles the colours." });
  const T = 3, D = 8;
  let h = 2, wrong = false;
  const x = Array.from({ length: T }, (_, t) => Array.from({ length: D }, (_, i) => t * D + i));
  const out = el("div");
  function draw() {
    const dk = D / h;
    const headOf = (v) => Math.floor((v % D) / dk);
    const tokOf = (v) => Math.floor(v / D);
    const panels = [];
    for (let hh = 0; hh < h; hh++) {
      const M = Array.from({ length: T }, (_, t) => Array.from({ length: dk }, (_, c) => {
        if (!wrong) return x[t][hh * dk + c];
        const flat = hh * T * dk + t * dk + c; return x[Math.floor(flat / D)][flat % D];
      }));
      const mixedTok = M.some((row, t) => row.some((v) => tokOf(v) !== t)) || M.some((row) => row.some((v) => headOf(v) !== hh));
      panels.push(el("div", { class: "w-col" }, el("span", { class: "w-label" }, `head ${hh}  (${T}, ${dk})`, mixedTok ? el("span", { class: "badge bad", style: { marginLeft: ".4rem" } }, "mixed up") : ""),
        numGrid(M, { fmtv: (v) => "t" + tokOf(v) + "·" + (v % D), style: (i, j, v) => ({ background: `var(${HEAD_COLORS[headOf(v)]})`, color: "var(--blk-ink)", outline: tokOf(v) !== i ? "2px solid var(--bad)" : "none" }), cellW: "3.1rem" })));
    }
    out.replaceChildren(
      el("div", { class: "w-col" }, el("span", { class: "w-label" }, `x  (T=${T}, d_model=${D}): each row is a token`),
        numGrid(x, { fmtv: (v) => "t" + tokOf(v) + "·" + (v % D), style: (i, j, v) => ({ background: `var(${HEAD_COLORS[headOf(v)]})`, color: "var(--blk-ink)" }), cellW: "3.1rem" })),
      el("p", {}, el("code", {}, wrong ? `x.view(h, T, d_k)   # WRONG` : `x.view(T, h, d_k).transpose(0, 1)   # right`), " → ", shapeChip(`(${h}, ${T}, ${dk})`, "attn")),
      el("div", { class: "w-row" }, panels),
      el("p", { class: "w-caption" }, wrong ? "Red outlines: a number that ended up in the wrong token's row. Every head now mixes pieces of different tokens, so attention would compare nonsense. The shape is still correct, so nothing crashes. That's what makes this bug dangerous."
        : `Each head gets a contiguous slice of d_k = ${dk} numbers from every token. Batched matmul then runs all ${h} heads at once.`));
  }
  body.append(el("div", { class: "controls" }, seg([1, 2, 4, 8], h, (v) => { h = v; draw(); }, "heads h"), seg([[false, "view + transpose"], [true, "view only (bug)"]], wrong, (v) => { wrong = v; draw(); }, "reshape")), out);
  draw();
});

/* ===== 0.5 softmax ===== */
Widgets.register("softmax", (host, opts) => {
  const body = frame(host, { tag: "softmax", title: "Softmax: scores in, weights out", sub: "Drag the scores. Weights are always positive and sum to 1; the exponential exaggerates gaps. Lower the temperature (or multiply the scores) and it becomes winner-takes-all." });
  const labels = opts.labels || ["the", "cat", "sat", "on", "mat"];
  let z = opts.scores || [1, 2, 3, 0.5, -1], temp = 1, mult = 1;
  const out = el("div", { class: "w-grow" });
  const sliders = labels.map((l, i) => slider(l, { min: -5, max: 5, step: 0.1, value: z[i], fmt: (v) => v.toFixed(1) }, (v) => { z[i] = v; draw(); }));
  function draw() {
    const zz = z.map((v) => v * mult);
    const p = TF.softmax(zz, temp);
    const H = -p.reduce((s, q) => s + (q > 0 ? q * Math.log(q) : 0), 0);
    out.replaceChildren(bars(p.map((q, i) => ({ label: labels[i], value: q, cls: q === Math.max(...p) ? "hi" : "" })), { max: 1, pct: true }),
      el("p", { class: "w-caption" }, `scores used: [${zz.map((v) => fmt(v, 1)).join(", ")}] / T=${temp}  ·  exp: [${zz.map((v) => fmt(Math.exp((v - Math.max(...zz)) / temp), 3)).join(", ")}] (after subtracting the max)  ·  sum of weights = ${fmt(p.reduce((a, b) => a + b), 3)}  ·  entropy ${fmt(H, 2)} (max ${fmt(Math.log(p.length), 2)})`));
  }
  body.append(el("div", { class: "w-row" }, el("div", { class: "w-col" }, el("span", { class: "w-label" }, "scores"), ...sliders,
    slider("temperature", { min: 0.1, max: 5, step: 0.1, value: 1, fmt: (v) => v.toFixed(1) }, (v) => { temp = v; draw(); }),
    seg([[1, "scores ×1"], [10, "×10"], [0.1, "×0.1"]], 1, (v) => { mult = v; draw(); })), out));
  draw();
});

/* ===== 0.6 gradient descent on one parameter ===== */
Widgets.register("gd-ball", (host) => {
  const body = frame(host, { tag: "training loop", title: "What loss.backward() and optimizer.step() do", sub: "Loss L(w) = (w − 2)² + 0.5. backward() computes the slope dL/dw; step() moves w a little downhill: w ← w − lr · dL/dw." });
  let w = -2.5, lr = 0.1, hist = [w];
  const L = (w) => (w - 2) ** 2 + 0.5, dL = (w) => 2 * (w - 2);
  const plot = el("div", { class: "w-grow" }), info = el("div", { class: "w-col", style: { minWidth: "15rem" } });
  function draw() {
    const xs = []; for (let x = -3; x <= 7; x += 0.1) xs.push([x, L(x)]);
    const ch = lineChart([{ name: "L(w)", points: xs }, { name: "steps taken", points: hist.map((x) => [x, L(x)]), dots: true }], { width: 520, height: 230, ymin: 0, ymax: 30, xmin: -3, xmax: 7 });
    plot.replaceChildren(ch);
    const g = dL(w);
    info.replaceChildren(
      el("pre", {}, `logits = model(x)          # forward\nloss = loss_fn(...)        # L(w) = ${fmt(L(w), 3)}\noptimizer.zero_grad()\nloss.backward()            # dL/dw = 2(w−2) = ${fmt(g, 3)}\noptimizer.step()           # w ← ${fmt(w, 3)} − ${lr}×${fmt(g, 3)}\n                           #   = ${fmt(w - lr * g, 3)}`),
      el("p", { class: "w-caption" }, lr > 1 ? "lr > 1 overshoots more every step: the loss explodes. Too large a learning rate makes training diverge." : lr > 0.5 ? "With a big lr, w jumps across the valley and oscillates." : "The slope tells you which way is downhill and how steep it is."));
  }
  const step = () => { w = w - lr * dL(w); hist.push(w); if (Math.abs(w) > 1e6) w = NaN; draw(); };
  body.append(el("div", { class: "controls" },
    slider("learning rate", { min: 0.01, max: 1.1, step: 0.01, value: lr }, (v) => { lr = v; }),
    button("step()", step, "primary"), button("20 steps", () => { for (let i = 0; i < 20; i++) step(); }),
    button("Reset", () => { w = -2.5; hist = [w]; draw(); })), el("div", { class: "w-row" }, plot, info));
  draw();
});

/* ===== 1.1 one-hot vs embedding lookup ===== */
Widgets.register("onehot-lookup", (host) => {
  const body = frame(host, { tag: "lookup", title: "An embedding is a lookup, and a lookup is a matrix multiplication", sub: "Click a word. one_hot(i) @ E picks out exactly row i of E. Then compare how similar the words look under each representation." });
  const words = ["the", "king", "queen", "man", "woman", "apple"];
  const dims = ["royal", "female", "person", "food"];
  const E = [[0.2, 0.3, 0.1, 0.2], [0.9, 0.1, 0.8, 0.0], [0.9, 0.9, 0.8, 0.0], [0.1, 0.1, 0.9, 0.1], [0.1, 0.9, 0.9, 0.1], [0.0, 0.1, 0.0, 0.9]];
  let sel = 2, rep = "emb";
  const top = el("div"), bottom = el("div");
  function draw() {
    const oh = words.map((_, i) => (i === sel ? 1 : 0));
    top.replaceChildren(el("div", { class: "controls" }, ...words.map((w, i) => el("span", { class: "chip click " + (i === sel ? "on" : ""), onclick: () => { sel = i; draw(); } }, `${i}: ${w}`))),
      el("div", { class: "mat-row", style: { flexWrap: "wrap" } },
        el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, `one_hot(${sel})  (1, 6)`), numGrid([oh], { fmtv: (v) => v, style: (i, j) => hiStyle(j === sel, "--c-emb") })),
        el("span", { class: "op" }, "@"),
        el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, "E  (V=6, d=4)"), numGrid(E, { style: (i) => hiStyle(i === sel, "--c-emb"), title: (i, j) => `${words[i]} · ${dims[j]}` })),
        el("span", { class: "op" }, "="),
        el("div", { class: "mat-wrap" }, el("span", { class: "cap" }, `E[${sel}] = "${words[sel]}"  (1, 4)`), numGrid([E[sel]], { style: () => hiStyle(true, "--c-emb") }))),
      el("p", { class: "w-caption" }, `Every 0 in the one-hot vector wipes out a row; the single 1 copies row ${sel}. nn.Embedding skips the multiplication and indexes E[${sel}] directly. Same answer, far less work. (Columns here are labelled royal / female / person / food to make it readable. A trained model's dimensions have no names.)`));
    const R = rep === "onehot" ? words.map((_, i) => words.map((_, j) => (i === j ? 1 : 0))) : words.map((_, i) => words.map((_, j) => TF.cosine(E[i], E[j])));
    bottom.replaceChildren(el("div", { class: "controls" }, seg([["onehot", "one-hot vectors"], ["emb", "embeddings"]], rep, (v) => { rep = v; draw(); }, "cosine similarity between")),
      heatmap(R, { rowLabels: words, colLabels: words, vmin: 0, vmax: 1, annotate: true, cell: 44, tip: (i, j, v) => `cos(${words[i]}, ${words[j]}) = ${fmt(v, 3)}` }),
      el("p", { class: "w-caption" }, rep === "onehot" ? "One-hot: every word is exactly as different from every other word (cosine 0). King is as far from queen as from apple." : "Embeddings: king–queen and man–woman are close, apple is far from everyone. Similar meaning, similar direction."));
  }
  body.append(top, bottom);
  draw();
});

/* ===== 1.2 cosine similarity in 2-D ===== */
Widgets.register("cosine-2d", (host) => {
  const body = frame(host, { tag: "similarity", title: "Dot product vs cosine similarity", sub: "Drag the tips of a and b. The dot product grows with length; cosine only cares about direction." });
  let a = [3, 1], b = [2, 2.5];
  const S = 300, sc = 30, ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${S} ${S}`); svg.style.width = "100%"; svg.style.maxWidth = S + "px"; svg.style.touchAction = "none";
  const info = el("div", { class: "w-col w-grow" });
  const P = (v) => [S / 2 + v[0] * sc, S / 2 - v[1] * sc];
  let drag = null;
  function draw() {
    svg.replaceChildren();
    const mk = (tag, at) => { const n = document.createElementNS(ns, tag); for (const k in at) n.setAttribute(k, at[k]); svg.append(n); return n; };
    for (let i = -5; i <= 5; i++) { mk("line", { x1: P([i, -5])[0], y1: 0, x2: P([i, 5])[0], y2: S, stroke: Theme.v("--line") }); mk("line", { y1: P([-5, i])[1], x1: 0, y2: P([5, i])[1], x2: S, stroke: Theme.v("--line") }); }
    [[a, "--accent", "a"], [b, "--neg", "b"]].forEach(([v, c, name]) => {
      const [x, y] = P(v);
      mk("line", { x1: S / 2, y1: S / 2, x2: x, y2: y, stroke: Theme.v(c), "stroke-width": 3 });
      const h = mk("circle", { cx: x, cy: y, r: 9, fill: Theme.v(c), style: "cursor:grab" });
      h.addEventListener("pointerdown", (e) => { drag = name; svg.setPointerCapture(e.pointerId); });
      const t = mk("text", { x: x + 10, y: y - 10, fill: Theme.v("--ink"), "font-size": 14, "font-weight": 700 }); t.textContent = name;
    });
    const d = TF.dot(a, b), na = TF.norm(a), nb = TF.norm(b), c = d / (na * nb);
    info.replaceChildren(el("dl", { class: "kv" },
      el("dt", {}, "a · b"), el("dd", {}, `${fmt(a[0])}×${fmt(b[0])} + ${fmt(a[1])}×${fmt(b[1])} = ${fmt(d)}`),
      el("dt", {}, "|a|, |b|"), el("dd", {}, `${fmt(na)}, ${fmt(nb)}`),
      el("dt", {}, "cos θ"), el("dd", { style: { fontWeight: 700 } }, fmt(c, 3)),
      el("dt", {}, "angle"), el("dd", {}, fmt((Math.acos(clamp(c, -1, 1)) * 180) / Math.PI, 1) + "°")),
      el("div", { class: "controls" }, button("Double a", () => { a = a.map((x) => clamp(x * 2, -5, 5)); draw(); }), button("Halve a", () => { a = a.map((x) => x / 2); draw(); }), button("Make b ⟂ a", () => { b = [-a[1], a[0]]; draw(); }), button("b = −a", () => { b = [-a[0], -a[1]]; draw(); })),
      el("p", { class: "w-caption" }, "Doubling a doubles the dot product but leaves the cosine unchanged. Word vectors are compared by cosine so that frequent words, which often get longer vectors, don't look 'similar to everything'."));
  }
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const r = svg.getBoundingClientRect(), k = S / r.width;
    const v = [clamp(((e.clientX - r.left) * k - S / 2) / sc, -5, 5), clamp((S / 2 - (e.clientY - r.top) * k) / sc, -5, 5)];
    if (drag === "a") a = v; else b = v;
    draw();
  });
  svg.addEventListener("pointerup", () => (drag = null));
  Theme.on(draw);
  body.append(el("div", { class: "w-row" }, svg, info));
  draw();
});

/* ===== 1.3 word2vec, trained live ===== */
Widgets.register("w2v", (host, opts) => {
  const body = frame(host, { tag: opts.lab ? "Lab" : "word2vec", title: "Train word2vec in your browser", sub: "Skip-gram with a full softmax, trained with Adam on 30,000 synthetic sentences. Press Train and watch the words organise themselves." });
  const sents = TF.makeCorpus(30000, 0), voc = TF.buildVocab(sents), V = voc.itos.length;
  let d = 16, windowSize = 2, lr = 0.01, epochs = 3;
  let sg, pairs, N, order, pos, epoch, losses, running = false, timer = null;
  const nouns = [...Object.keys(TF.PEOPLE), ...Object.keys(TF.ANIMALS), ...Object.keys(TF.FOODS)];
  const catColor = { person: "--accent", animal: "--neg", food: "--good" };
  let peopleOnly = true, selWord = "king";

  // corpus + pair explorer
  const sample = el("div", { class: "w-col" });
  const sampleR = TF.rng(9);
  let sampleSents = Array.from({ length: 5 }, () => sents[sampleR.int(sents.length)]), focus = [0, 1];
  function drawSample() {
    sample.replaceChildren(el("span", { class: "w-label" }, `corpus sample · click a word to see its (center, context) pairs, window = ${windowSize}`),
      ...sampleSents.map((s, si) => el("div", { class: "mono", style: { fontSize: ".9rem" } }, ...s.flatMap((w, wi) => {
        const isC = focus[0] === si && focus[1] === wi, isCtx = focus[0] === si && Math.abs(focus[1] - wi) <= windowSize && !isC;
        return [el("span", { class: "chip click", style: { background: isC ? "var(--c-emb)" : isCtx ? "var(--c-attn)" : "", color: isC || isCtx ? "var(--blk-ink)" : "", margin: "1px" }, onclick: () => { focus = [si, wi]; drawSample(); } }, w), " "];
      }))));
    const s = sampleSents[focus[0]], c = s[focus[1]];
    const ctx = s.filter((_, j) => j !== focus[1] && Math.abs(j - focus[1]) <= windowSize);
    sample.append(el("p", { class: "w-caption" }, "training pairs from the pink word: ", ...ctx.map((w) => el("code", { style: { marginRight: ".3rem" } }, `(${c}, ${w})`)), el("br"), "The model learns to predict the orange words from the pink one."));
  }

  const lossBox = el("div"), scatter = el("div", { class: "w-grow" }), status = el("div", { class: "status-line" }), nn = el("div", { class: "w-col" }), an = el("div", { class: "w-col" });
  const trainBtn = button("Train", () => (running ? pause() : start()), "primary");
  function reset() {
    pause();
    sg = new TF.SkipGram(V, d, 0);
    const P = [];
    sents.forEach((s) => TF.skipgramPairs(s.map((w) => voc.stoi[w]), windowSize).forEach((p) => P.push(p[0], p[1])));
    pairs = Int32Array.from(P); N = pairs.length / 2;
    shuffle(); pos = 0; epoch = 0; losses = [];
    status.textContent = `vocabulary ${V} words · ${N.toLocaleString()} (center, context) pairs · ${sg.V * sg.d * 2} parameters · untrained`;
    trainBtn.textContent = "Train";
    drawAll();
  }
  function shuffle() {
    const r = TF.rng(epoch + 7), idx = Int32Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) { const j = r.int(i + 1); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    order = new Int32Array(2 * N); for (let i = 0; i < N; i++) { order[2 * i] = pairs[2 * idx[i]]; order[2 * i + 1] = pairs[2 * idx[i] + 1]; }
  }
  function start() { if (epoch >= epochs) reset(); running = true; trainBtn.textContent = "Pause"; tick(); }
  function pause() { running = false; trainBtn.textContent = "Resume"; clearTimeout(timer); }
  function tick() {
    if (!running) return;
    const t0 = performance.now();
    let L = 0, nb = 0;
    while (performance.now() - t0 < 30 && epoch < epochs) {
      const end = Math.min(N, pos + 512);
      L += sg.step(order, pos, end, lr); nb++;
      pos = end;
      if (pos >= N) { epoch++; pos = 0; if (epoch < epochs) shuffle(); }
    }
    if (nb) losses.push(L / nb);
    const done = epoch >= epochs;
    status.textContent = done ? `done: ${epochs} epochs · final loss ${fmt(losses[losses.length - 1], 3)}` : `epoch ${epoch + 1}/${epochs} · ${Math.round((100 * pos) / N)}% · loss ${fmt(losses[losses.length - 1], 3)}`;
    drawAll();
    if (done) { running = false; trainBtn.textContent = "Train again"; return; }
    timer = setTimeout(tick, 0);
  }
  function drawAll() {
    lossBox.replaceChildren(losses.length > 1 ? lineChart([{ name: "loss", values: losses }], { width: 360, height: 150, xFmt: () => "" }) : el("p", { class: "w-caption" }, `Loss starts near ln(${V}) = ${fmt(Math.log(V), 2)}: a random guess among ${V} words.`));
    const words = peopleOnly ? Object.keys(TF.PEOPLE) : nouns;
    const P2 = TF.pca2(words.map((w) => sg.vec(voc.stoi[w])));
    const S = 340, ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${S} ${S}`); svg.style.width = "100%"; svg.style.maxWidth = "420px";
    const mx = Math.max(...P2.map((p) => Math.abs(p[0])), 1e-6), my = Math.max(...P2.map((p) => Math.abs(p[1])), 1e-6);
    const mk = (tag, at, txt) => { const n = document.createElementNS(ns, tag); for (const k in at) n.setAttribute(k, at[k]); if (txt) n.textContent = txt; svg.append(n); return n; };
    mk("line", { x1: S / 2, x2: S / 2, y1: 10, y2: S - 10, stroke: Theme.v("--line") }); mk("line", { y1: S / 2, y2: S / 2, x1: 10, x2: S - 10, stroke: Theme.v("--line") });
    words.forEach((w, i) => {
      const x = S / 2 + (P2[i][0] / mx) * (S / 2 - 50), y = S / 2 - (P2[i][1] / my) * (S / 2 - 30);
      const col = Theme.v(catColor[TF.categoryOf(w)]);
      const c = mk("circle", { cx: x, cy: y, r: w === selWord ? 7 : 5, fill: col, style: "cursor:pointer" });
      c.addEventListener("click", () => { selWord = w; drawAll(); });
      mk("text", { x: x + 7, y: y - 6, fill: Theme.v("--ink"), "font-size": 12, "font-weight": w === selWord ? 700 : 400 }, w);
    });
    scatter.replaceChildren(el("span", { class: "w-label" }, `embedding space (PCA of ${d}-D vectors → 2-D)`), svg,
      el("div", { class: "legend" }, el("span", {}, el("i", { style: { background: "var(--accent)" } }), "people"), el("span", {}, el("i", { style: { background: "var(--neg)" } }), "animals"), el("span", {}, el("i", { style: { background: "var(--good)" } }), "foods")));
    const sims = sg.mostSimilar(voc.stoi[selWord], 6);
    nn.replaceChildren(el("span", { class: "w-label" }, `nearest to "${selWord}" (cosine)`), select("word", nouns, selWord, (v) => { selWord = v; drawAll(); }),
      bars(sims.map(([j, s]) => ({ label: voc.itos[j], value: Math.max(0, s), title: fmt(s, 3) })), { max: 1, digits: 2 }));
    drawAnalogy();
  }
  let ana = ["man", "king", "woman"];
  function drawAnalogy() {
    const [a, b, c] = ana, res = sg.analogy(voc.stoi[a], voc.stoi[b], voc.stoi[c], 3);
    const pick = (i) => select(["a", "b", "c"][i], nouns, ana[i], (v) => { ana[i] = v; drawAnalogy(); });
    an.replaceChildren(el("span", { class: "w-label" }, "analogy: a is to b as c is to ?"), el("div", { class: "controls", style: { marginBottom: 0 } }, pick(0), pick(1), pick(2)),
      el("p", { style: { margin: ".2rem 0" } }, el("code", {}, `E[${b}] − E[${a}] + E[${c}]`), " ≈ ", el("b", { style: { fontSize: "1.2rem" } }, voc.itos[res[0][0]]), `  (cos ${fmt(res[0][1], 2)}; then ${res.slice(1).map(([j, s]) => voc.itos[j] + " " + fmt(s, 2)).join(", ")})`),
      el("div", { class: "controls" }, ...[["man", "king", "woman"], ["boy", "prince", "girl"], ["man", "boy", "woman"], ["prince", "king", "princess"], ["dog", "cat", "cow"]].map((t) => button(`${t[0]}:${t[1]} :: ${t[2]}:?`, () => { ana = t.slice(); drawAnalogy(); }))));
  }
  body.append(
    el("div", { class: "controls" }, trainBtn, button("Reset", reset),
      seg([8, 16, 32], d, (v) => { d = v; reset(); }, "dimensions d"),
      seg([1, 2, 3], windowSize, (v) => { windowSize = v; drawSample(); reset(); }, "window"),
      seg([[0.003, "0.003"], [0.01, "0.01"], [0.03, "0.03"]], lr, (v) => { lr = v; }, "lr"),
      el("label", { class: "ctl" }, el("input", { type: "checkbox", id: uid("po"), checked: true, onchange: (e) => { peopleOnly = e.target.checked; drawAll(); } }), "people only (untick to add animals and foods)")),
    status,
    el("div", { class: "w-row", style: { marginTop: ".8rem" } }, scatter, el("div", { class: "w-col w-grow" }, nn, an, el("span", { class: "w-label" }, "training loss"), lossBox)),
    el("details", { class: "reveal" }, el("summary", {}, "The corpus and the (center, context) pairs"), sample));
  drawSample();
  reset();
  Theme.on(drawAll);
});
