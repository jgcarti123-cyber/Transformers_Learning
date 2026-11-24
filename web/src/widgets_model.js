/* ============================================================================
   widgets_model.js: figures that run the REAL trained date Transformer
   (Chapters 3–5, the home page and the Lab).
   ========================================================================== */
const MODEL_LABELS = { baseline: "Full Transformer", no_positions: "No positional encoding", one_head: "1 head", one_layer: "1 layer", tiny: "d_model = 16" };
const EXAMPLES = ["tuesday, november 24, 2026", "24/11/2026", "2026/11/24", "the 3rd of july, 1969", "sat 1 jan 2000", "4.7.1776"];
const MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
const WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
function randomDateText(r = TF.rng(Date.now() % 1e9)) {
  const t0 = Date.UTC(1900, 0, 1), t1 = Date.UTC(2099, 11, 31), dt = new Date(t0 + r() * (t1 - t0));
  const d = dt.getUTCDate(), m = dt.getUTCMonth(), y = dt.getUTCFullYear(), M = MONTHS[m], W = WEEKDAYS[(dt.getUTCDay() + 6) % 7];
  const ord = (n) => (n % 100 >= 10 && n % 100 <= 20 ? n + "th" : n + ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th"));
  return r.pick([`${M} ${d}, ${y}`, `${M.slice(0, 3)} ${d} ${y}`, `${d} ${M} ${y}`, `${d} ${M.slice(0, 3)} ${y}`, `${W}, ${M} ${d}, ${y}`, `${W.slice(0, 3)} ${d} ${M.slice(0, 3)} ${y}`, `the ${ord(d)} of ${M}, ${y}`, `${d}/${m + 1}/${y}`, `${d}.${m + 1}.${y}`, `${y}/${m + 1}/${d}`]);
}
const loadingNote = (t = "Loading the trained model…") => el("p", { class: "loading" }, t);
const mean = (arrs) => arrs[0].map((_, i) => arrs.reduce((s, a) => s + a[i], 0) / arrs.length);
const rowsOf = (flat, R, C) => Array.from({ length: R }, (_, i) => Array.from(flat.subarray ? flat.subarray(i * C, (i + 1) * C) : flat.slice(i * C, (i + 1) * C)));
function attnText(chars, w, { onHover } = {}) {
  const mx = Math.max(...w, 1e-9);
  return el("div", { class: "attn-text" }, ...chars.map((c, i) => {
    const col = seqColor(w[i] / mx);
    return el("span", { style: { background: rgbStr(col), color: lum(col) > 0.55 ? "#141a26" : "#fff" }, title: `${c}: ${fmt(w[i] * 100, 1)}%` }, c === " " ? "␣" : c);
  }));
}
function modelPicker(value, onChange, { withSteps = false, step = null, onStep } = {}) {
  const box = el("div", { class: "controls", style: { marginBottom: 0 } }, select("model", Object.entries(MODEL_LABELS), value, onChange));
  if (withSteps && TF.manifest) {
    const steps = TF.manifest.models.baseline.checkpoints.map((c) => c.step);
    box.append(select("training step", [["final", "final (1500)"], ...steps.filter((s) => s !== 1500).map((s) => [s, "step " + s])], step === null ? "final" : step, (v) => onStep(v === "final" ? null : +v)));
  }
  return box;
}

/* ===== the model inspector (centrepiece) ===== */
Widgets.register("inspector", async (host, opts) => {
  const body = frame(host, { tag: opts.lab ? "Lab · the real model" : "look inside", title: "Model inspector: follow one date through the whole Transformer", sub: "This is the actual trained model (241,964 parameters) running in your browser. Every number below is computed live from the weights." });
  body.append(loadingNote());
  await TF.loadManifest();
  let name = opts.model || "baseline", step = null, text = opts.text || "tuesday, november 24, 2026", forced = {}, cur = 0;
  let encLayer = 0, crossLayer = -1, crossHead = "mean", embView = "sum", model, res;
  const input = el("input", { type: "text", value: text, id: uid("insp"), style: { width: "22rem", maxWidth: "100%", fontFamily: "var(--f-mono)" }, "aria-label": "date to translate" });
  const top = el("div"), secTok = el("div"), secEmb = el("div"), secEnc = el("div"), secDec = el("div"), secMaps = el("div"), badge = el("span");
  const H = (n, t, s) => el("h4", { style: { margin: "1.4rem 0 .4rem", fontFamily: "var(--f-display)" } }, el("span", { class: "shape", style: { marginRight: ".5rem" } }, n), t, s ? el("span", { class: "w-caption", style: { fontWeight: 400, marginLeft: ".6rem" } }, s) : null);
  let deb;
  input.addEventListener("input", () => { clearTimeout(deb); deb = setTimeout(() => { text = input.value; forced = {}; cur = 0; run(); }, 200); });
  function drawTop() {
    top.replaceChildren(el("div", { class: "controls" }, el("div", { class: "ctl" }, el("label", { for: input.id }, "date"), input),
      button("Random date", () => { input.value = text = randomDateText(); forced = {}; cur = 0; run(); }),
      modelPicker(name, async (v) => { name = v; step = null; forced = {}; await run(); drawTop(); }, { withSteps: name === "baseline", step, onStep: async (s) => { step = s; forced = {}; await run(); } }), badge),
      el("div", { class: "controls" }, ...EXAMPLES.map((ex) => el("span", { class: "chip click", onclick: () => { input.value = text = ex; forced = {}; cur = 0; run(); } }, ex))));
  }
  async function run() {
    model = await TF.loadModel(name, step);
    if (name === "baseline" && step === null && !badge.dataset.done) {
      const m = TF.manifest.models.baseline, src = model.encodeText(TF.manifest.samples[0]).ids, tgt = [1, ...model.encodeText("2026-11-24").ids];
      const e = model.encode(src), d = model.decode(tgt, e.memory, e.T);
      let mx = 0; m.ref_logits_first3.forEach((row, t) => row.forEach((v, i) => (mx = Math.max(mx, Math.abs(v - d.logits[t * 44 + i])))));
      badge.replaceChildren(el("span", { class: "badge " + (mx < 0.05 ? "good" : "bad"), title: "Logits from this browser engine vs PyTorch on the same input" }, mx < 0.05 ? `matches PyTorch (max diff ${fmt(mx, 3)})` : `differs from PyTorch by ${fmt(mx, 3)}`));
      badge.dataset.done = "1";
    }
    const { ids, bad } = model.encodeText(text.toLowerCase());
    if (bad.length || !ids.length) {
      secTok.replaceChildren(H("1", "Tokens"), el("p", {}, el("span", { class: "badge bad" }, "can't tokenize"), ` Characters ${bad.map((c) => `"${c}"`).join(", ") || "(empty)"} aren't in the vocabulary (lowercase a–z, digits, space , . / -). The model has no id for them, so it can't read them at all.`));
      [secEmb, secEnc, secDec, secMaps].forEach((s) => s.replaceChildren());
      return;
    }
    res = model.generate(ids, { forced });
    cur = Math.min(cur, res.steps.length - 1);
    drawAll();
  }
  function drawAll() {
    const enc = res.enc, S = enc.T, chars = enc.ids.map((i) => model.tok(i)), d = model.d;
    // 1 tokens
    secTok.replaceChildren(H("1", "Tokens", `(S = ${S}) characters → ids`), el("div", { class: "w-row", style: { gap: ".25rem" } }, ...enc.ids.map((id) => el("div", { class: "w-col", style: { alignItems: "center", gap: 0 } }, el("span", { class: "chip" }, model.tok(id)), el("span", { class: "w-caption", style: { margin: 0, fontSize: ".72rem" } }, id)))));
    // 2 embeddings
    const tok = rowsOf(enc.tok, S, d), pe = rowsOf(model.pe, S, d), emb = rowsOf(enc.emb, S, d);
    const showE = embView === "tok" ? tok : embView === "pe" ? pe : emb;
    secEmb.replaceChildren(H("2", "Embeddings", `src_embed(ids) × √${d}  +  positional encoding  →  ${shapeStr(S, d)}`),
      el("div", { class: "controls" }, seg([["tok", "token embedding × √d"], ["pe", "positional encoding"], ["sum", "sum (encoder input)"]], embView, (v) => { embView = v; drawAll(); })),
      heatmap(showE, { rowLabels: chars, mode: "div", cell: 9, cellH: 14, maxWidth: 680, tip: (i, j, v) => `"${chars[i]}" (position ${i}), dim ${j}: ${fmt(v, 3)}` }),
      el("p", { class: "w-caption" }, embView === "tok" ? "Identical characters have identical rows: a static lookup, exactly like word2vec." : embView === "pe" ? "The positional encoding depends only on the position, not on the character." : name === "no_positions" ? "This model has no positional encoding, so the encoder input is just the token embeddings and order is invisible." : "Their sum is what enters the encoder: what the token is, plus where it is."));
    // 3 encoder
    const L = enc.layers[encLayer], heads = L.attn.map((W) => rowsOf(W, S, S));
    const outRows = rowsOf(enc.memory, S, d), tokRows = tok;
    const cosM = (R) => R.map((a) => R.map((b) => TF.cosine(a, b)));
    secEnc.replaceChildren(H("3", "Encoder self-attention", `${model.L} layer(s) × ${model.h} heads · each map is (S, S) = (${S}, ${S}) · colour scaled to each map's max; hover for exact %`),
      el("div", { class: "controls" }, seg(Array.from({ length: model.L }, (_, i) => [i, "layer " + i]), encLayer, (v) => { encLayer = v; drawAll(); })),
      el("div", { class: "w-row" }, ...heads.map((Hm, h) => el("div", { class: "w-col" }, el("span", { class: "w-label" }, `head ${h}`), heatmap(Hm, { rowLabels: chars, colLabels: chars, vmin: 0, cell: S > 20 ? 11 : 15, tip: (i, j, v) => `"${chars[i]}"@${i} looks at "${chars[j]}"@${j}: ${fmt(v * 100, 1)}%` })))),
      el("details", { class: "reveal" }, el("summary", {}, "Static vs contextual: compare each position's vector before and after the encoder"),
        el("div", { class: "w-row" }, el("div", { class: "w-col" }, el("span", { class: "w-label" }, "cosine of token embeddings (no position)"), heatmap(cosM(tokRows), { rowLabels: chars, colLabels: chars, vmin: -1, vmax: 1, mode: "div", cell: 12 })),
          el("div", { class: "w-col" }, el("span", { class: "w-label" }, "cosine of encoder outputs (contextual)"), heatmap(cosM(outRows), { rowLabels: chars, colLabels: chars, vmin: -1, vmax: 1, mode: "div", cell: 12 }))),
        el("p", { class: "w-caption" }, "Left: two copies of the same character are identical (cosine 1). Right: after attention they differ, because each copy has absorbed information about its neighbours and its role (a day digit, a year digit…).")));
    // 4 decoding, step by step
    const st = res.steps[cur], srcChars = enc.ids.map((i) => model.vocab[i]);
    const cl = crossLayer < 0 ? model.L + crossLayer : crossLayer;
    const cw = crossHead === "mean" ? mean(st.cross[cl]) : st.cross[cl][crossHead];
    const top8 = st.probs.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 8);
    const outChars = res.ys.slice(1);
    secDec.replaceChildren(H("4", "Decoding, one character at a time", "greedy: pick the most likely next character, append it, repeat"),
      el("div", { class: "controls" }, button("◀", () => { cur = Math.max(0, cur - 1); drawAll(); }), button("▶ next step", () => { cur = Math.min(res.steps.length - 1, cur + 1); drawAll(); }, "primary"),
        button("Play", () => { cur = 0; drawAll(); const t = setInterval(() => { if (cur >= res.steps.length - 1) return clearInterval(t); cur++; drawAll(); }, 650); }),
        Object.keys(forced).length ? button("Clear forced choices", () => { forced = {}; run(); }) : null),
      el("div", { class: "out-text" }, ...outChars.map((id, i) => el("span", { class: (i === cur ? "cur " : "") + (res.steps[i] && res.steps[i].forced ? "forced" : ""), onclick: () => { cur = i; drawAll(); }, title: `step ${i}` }, id === 2 ? "⏎" : model.tok(id)))),
      el("p", { class: "w-caption" }, `Step ${cur}: decoder input = `, el("code", {}, "<bos>" + st.prefix.slice(1).map((i) => model.vocab[i]).join("")), ` → it predicts the next character. Output so far: "${res.text}"`),
      el("div", { class: "w-row" },
        el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, "next-character probabilities (click one to force it)"),
          (() => { const b = bars(top8.map(([i, p]) => ({ label: i === 2 ? "<eos>" : model.tok(i), value: p, cls: i === st.choice ? "hi" : "" })), { max: 1, pct: true });
            [...b.children].forEach((row, k) => { row.style.cursor = "pointer"; row.addEventListener("click", () => { forced = { ...forced, [cur]: top8[k][0] }; Object.keys(forced).forEach((s) => { if (+s > cur) delete forced[s]; }); run(); }); });
            return b; })()),
        el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, "cross-attention at this step: where the decoder looks in the input"),
          el("div", { class: "controls", style: { marginBottom: ".3rem" } }, seg(Array.from({ length: model.L }, (_, i) => [i, "layer " + i]), cl, (v) => { crossLayer = v; drawAll(); }),
            seg([["mean", "mean"], ...Array.from({ length: model.h }, (_, i) => [i, "h" + i])], crossHead, (v) => { crossHead = v; drawAll(); })),
          attnText(srcChars, cw), el("p", { class: "w-caption" }, "Darker = more attention. Try the year digits, then the month digits, and see which input characters light up."))),
      st.forced ? el("p", {}, el("span", { class: "badge bad" }, "forced"), " You overrode the model's choice here. Everything after this step is conditioned on your choice. That's exposure bias in action: the model never saw its own mistakes during training.") : null);
    // 5 full maps
    const T = res.full.T, outLabels = res.ys.slice(1, T + 1).map((i) => "→" + (i === 2 ? "⏎" : model.tok(i)));
    const crossFull = res.full.layers[cl].cross.map((W) => rowsOf(W, T, S));
    const selfFull = mean(res.full.layers[cl].self.map((W) => Array.from(W)));
    secMaps.replaceChildren(H("5", "Every attention map at once", `decoder layer ${cl}: rows = output positions, columns = what they look at`),
      el("div", { class: "w-row" }, ...crossFull.map((Hm, h) => el("div", { class: "w-col" }, el("span", { class: "w-label" }, `cross-attention head ${h}`), heatmap(Hm, { rowLabels: outLabels, colLabels: chars, vmin: 0, cell: S > 20 ? 10 : 14, tip: (i, j, v) => `${outLabels[i]} looks at "${chars[j]}"@${j}: ${fmt(v * 100, 1)}%` }))),
        el("div", { class: "w-col" }, el("span", { class: "w-label" }, "decoder self-attention (mean of heads)"), heatmap(rowsOf(selfFull, T, T), { rowLabels: outLabels, colLabels: ["<bos>", ...outLabels.slice(0, T - 1).map((s) => s.slice(1))], vmin: 0, cell: 14 }))),
      el("p", { class: "w-caption" }, "The decoder self-attention is a lower triangle: the causal mask at work."));
  }
  const shapeStr = (a, b) => `(${a}, ${b})`;
  body.replaceChildren(top, secTok, secEmb, secEnc, secDec, secMaps);
  drawTop();
  await run();
  Theme.on(() => res && drawAll());
});

/* ===== hero: the model translating, with attention highlighted ===== */
Widgets.register("hero-demo", async (host) => {
  host.classList.add("widget");
  const body = el("div", { class: "w-body" });
  host.append(body);
  body.append(loadingNote("Loading the trained model (0.6 MB)…"));
  const model = await TF.loadModel("baseline");
  const input = el("input", { type: "text", id: "hero-input", value: EXAMPLES[0], style: { width: "100%", fontFamily: "var(--f-mono)", fontSize: "1.05rem" }, "aria-label": "Type a date" });
  const srcBox = el("div"), outBox = el("div", { class: "out-text", style: { minHeight: "2.6rem" } }), note = el("p", { class: "w-caption" });
  let timer, auto = true, exi = 0;
  function play(text) {
    clearInterval(timer);
    const r = model.translate(text.toLowerCase());
    if (r.error) { srcBox.replaceChildren(); outBox.replaceChildren(el("span", { class: "badge bad" }, `can't read ${r.error.map((c) => `"${c}"`).join(" ")}`)); note.textContent = "Only lowercase letters, digits, space and , . / - are in the vocabulary."; return; }
    const chars = r.enc.ids.map((i) => model.vocab[i]);
    let k = 0;
    const show = () => {
      const st = r.steps[Math.min(k, r.steps.length - 1)];
      srcBox.replaceChildren(attnText(chars, mean(st.cross[model.L - 1])));
      outBox.replaceChildren(...r.ys.slice(1).map((id, i) => el("span", { class: i === k ? "cur" : i > k ? "pending" : "" }, id === 2 ? "" : model.tok(id))));
      note.textContent = `writing "${st.choice === 2 ? "end" : model.vocab[st.choice]}" (${fmt(st.probs[st.choice] * 100, 1)}% sure) while looking at the highlighted input`;
      k++;
      if (k >= r.steps.length) { clearInterval(timer); note.textContent = `done: "${r.text}". The highlight shows where the decoder's cross-attention looked for each character.`; if (auto) setTimeout(() => { if (auto) { exi = (exi + 1) % EXAMPLES.length; input.value = EXAMPLES[exi]; play(EXAMPLES[exi]); } }, 1800); }
    };
    show();
    timer = setInterval(show, 420);
  }
  input.addEventListener("input", () => { auto = false; clearTimeout(input._d); input._d = setTimeout(() => play(input.value), 220); });
  body.replaceChildren(el("label", { class: "w-label", for: "hero-input" }, "type any date"), input, el("div", { style: { height: ".8rem" } }), el("span", { class: "w-label" }, "input, shaded by attention"), srcBox, el("div", { style: { height: ".6rem" } }), el("span", { class: "w-label" }, "output"), outBox, note);
  play(EXAMPLES[0]);
});

/* ===== real attention heads (Chapter 3) ===== */
Widgets.register("real-heads", async (host) => {
  const body = frame(host, { tag: "real heads", title: "What do the 4 heads of the trained encoder do?", sub: "Encoder self-attention of the real model. Each head learned its own pattern. Change the date and the layer." });
  body.append(loadingNote());
  const model = await TF.loadModel("baseline");
  let text = "24/11/2026", layer = 0;
  const inp = el("input", { type: "text", value: text, id: uid("rh"), style: { fontFamily: "var(--f-mono)" } }), view = el("div");
  inp.addEventListener("input", () => { text = inp.value; draw(); });
  function draw() {
    const { ids, bad } = model.encodeText(text.toLowerCase());
    if (bad.length || !ids.length) return view.replaceChildren(el("p", { class: "loading" }, "Only lowercase letters, digits, space and , . / - are allowed."));
    const e = model.encode(ids), S = e.T, chars = ids.map((i) => model.tok(i));
    view.replaceChildren(el("div", { class: "w-row" }, ...e.layers[layer].attn.map((W, h) => el("div", { class: "w-col" }, el("span", { class: "w-label" }, "head " + h), heatmap(rowsOf(W, S, S), { rowLabels: chars, colLabels: chars, vmin: 0, cell: 16, tip: (i, j, v) => `"${chars[i]}"@${i} → "${chars[j]}"@${j}: ${fmt(v * 100, 1)}%` })))),
      el("p", { class: "w-caption" }, "Look for heads that attend to the previous or next character (a diagonal stripe just off the centre), heads that gather at separators like '/', and heads that link digits of the same number. Nobody programmed these patterns; they emerged from training."));
  }
  body.replaceChildren(el("div", { class: "controls" }, el("div", { class: "ctl" }, el("label", { for: inp.id }, "date"), inp), seg([[0, "layer 0"], [1, "layer 1"]], layer, (v) => { layer = v; draw(); })), view);
  draw();
});

/* ===== architecture + parameter calculator ===== */
Widgets.register("architecture", (host, opts) => {
  const body = frame(host, { tag: opts.lab ? "Lab" : "figure 1", title: "The Transformer, block by block", sub: "Click any block to see what it does, its input and output shapes, its parameters and the code. Change the sizes to recompute everything." });
  const presets = { "our date model": { V: 44, d: 64, h: 4, N: 2, dff: 256 }, "paper: base": { V: 37000, d: 512, h: 8, N: 6, dff: 2048 }, "paper: big": { V: 37000, d: 1024, h: 16, N: 6, dff: 4096 } };
  let cfg = { ...presets["our date model"] }, sel = "mha";
  const mhaP = (d) => 4 * (d * d + d), ffnP = (d, f) => 2 * d * f + f + d, lnP = (d) => 2 * d;
  const B = {
    emb: { name: "Input / Output Embedding", color: "--c-emb", what: "Looks up one learned vector per token id and multiplies it by √d_model. Static: the same id always gives the same vector.", io: (c) => [`(B, S) ids`, `(B, S, ${c.d})`], p: (c) => c.V * c.d, code: "x = self.src_embed(src) * math.sqrt(self.d_model)" },
    pe: { name: "Positional Encoding", color: "--c-pos", what: "Adds a fixed sin/cos pattern that depends only on the position. Without it, attention can't tell word order (Lesson 5). It has no parameters.", io: (c) => [`(B, S, ${c.d})`, `(B, S, ${c.d})`], p: () => 0, code: "x = x + self.pe[:S]          # (S, d) broadcasts over B" },
    mha: { name: "Multi-Head (self-)Attention", color: "--c-attn", what: "Every token builds a query, key and value. Each head compares queries with keys (softmax(QKᵀ/√d_k)) and averages values. This is the ONLY place tokens exchange information.", io: (c) => [`(B, S, ${c.d})`, `(B, S, ${c.d})`], p: (c) => mhaP(c.d), code: "a = self.self_attn(x, x, src_mask)   # q, k, v all from x" },
    masked: { name: "Masked Multi-Head Attention", color: "--c-attn", what: "Decoder self-attention with the causal mask: position t may look at positions ≤ t only, so training with the full target can't cheat.", io: (c) => [`(B, T, ${c.d})`, `(B, T, ${c.d})`], p: (c) => mhaP(c.d), code: "a = self.self_attn(y, y, tgt_mask)   # causal & padding" },
    cross: { name: "Cross-Attention", color: "--c-attn", what: "Queries come from the decoder, keys and values from the encoder output (memory). For each output position it asks: which input tokens matter now?", io: (c) => [`(B, T, ${c.d}) + memory (B, S, ${c.d})`, `(B, T, ${c.d})`], p: (c) => mhaP(c.d), code: "a = self.cross_attn(y, memory, src_mask)" },
    norm: { name: "Add & Norm", color: "--c-norm", what: "Residual connection (x + sublayer(x)) followed by LayerNorm over each token's vector. The residual gives gradients a straight path; LayerNorm keeps the scale stable.", io: (c) => [`(B, ·, ${c.d})`, `(B, ·, ${c.d})`], p: (c) => lnP(c.d), code: "x = self.norm1(x + sublayer_output)" },
    ffn: { name: "Feed Forward", color: "--c-ffn", what: "Two linear layers with a ReLU, applied to every token separately: d → d_ff → d. It processes what attention gathered, and holds most of each block's parameters.", io: (c) => [`(B, ·, ${c.d})`, `(B, ·, ${c.d}) via (B, ·, ${c.dff})`], p: (c) => ffnP(c.d, c.dff), code: "x = self.lin2(torch.relu(self.lin1(x)))" },
    linear: { name: "Linear", color: "--c-lin", what: "Projects each decoder output vector to one score (logit) per vocabulary token.", io: (c) => [`(B, T, ${c.d})`, `(B, T, ${c.V})`], p: (c) => c.d * c.V + c.V, code: "logits = self.out_proj(y)" },
    softmax: { name: "Softmax", color: "--c-soft", what: "Turns the logits into a probability for every possible next token. Training maximises the probability of the correct one (cross-entropy).", io: (c) => [`(B, T, ${c.V})`, `(B, T, ${c.V}) probabilities`], p: () => 0, code: "probs = torch.softmax(logits, dim=-1)" },
  };
  const ns = "http://www.w3.org/2000/svg", svgWrap = el("div", { class: "arch" }), panel = el("div", { class: "w-col w-grow", style: { minWidth: "16rem" } }), table = el("div");
  function draw() {
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 560 640"); svg.style.width = "100%"; svg.style.maxWidth = "520px";
    const mk = (tag, at, txt, parent = svg) => { const n = document.createElementNS(ns, tag); for (const k in at) n.setAttribute(k, at[k]); if (txt !== undefined) n.textContent = txt; parent.append(n); return n; };
    const blk = (key, x, y, w, h, label) => {
      const g = mk("g", { class: "blk" + (sel === key ? " sel" : ""), tabindex: 0, role: "button", "aria-label": B[key].name });
      mk("rect", { x, y, width: w, height: h, rx: 7, fill: Theme.v(B[key].color) }, undefined, g);
      label.split("\n").forEach((ln, i, a) => mk("text", { x: x + w / 2, y: y + h / 2 + 4 + (i - (a.length - 1) / 2) * 13, "text-anchor": "middle" }, ln, g));
      const go = () => { sel = key; draw(); };
      g.addEventListener("click", go); g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    };
    const wire = (x1, y1, x2, y2) => mk("path", { d: `M${x1},${y1} L${x2},${y2}`, class: "wire", "marker-end": "url(#arr)" });
    const defs = mk("defs", {}); const m = mk("marker", { id: "arr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, undefined, defs); mk("path", { d: "M0,0 L10,5 L0,10 z", fill: Theme.v("--muted") }, undefined, m);
    // encoder column (x 30..230)
    mk("rect", { x: 20, y: 236, width: 220, height: 219, rx: 12, fill: "none", stroke: Theme.v("--faint"), "stroke-dasharray": "5 4" });
    mk("text", { x: 14, y: 360, class: "lbl", "text-anchor": "end" }, "");
    mk("text", { x: 30, y: 252, class: "lbl" }, `encoder block × N = ${cfg.N}`);
    blk("emb", 45, 560, 170, 34, "Input Embedding"); mk("text", { x: 130, y: 622, class: "lbl", "text-anchor": "middle" }, `source ids (B, S)`); wire(130, 610, 130, 596);
    blk("pe", 45, 505, 170, 30, "+ Positional Encoding"); wire(130, 560, 130, 537);
    wire(130, 505, 130, 452);
    blk("mha", 45, 410, 170, 40, "Multi-Head\nAttention"); blk("norm", 45, 370, 170, 28, "Add & Norm"); wire(130, 410, 130, 400);
    blk("ffn", 45, 315, 170, 36, "Feed Forward"); wire(130, 370, 130, 353); blk("norm", 45, 275, 170, 28, "Add & Norm"); wire(130, 315, 130, 305);
    mk("text", { x: 130, y: 228, class: "lbl", "text-anchor": "middle" }, `memory (B, S, ${cfg.d})`);
    // decoder column (x 300..520)
    mk("rect", { x: 290, y: 150, width: 250, height: 305, rx: 12, fill: "none", stroke: Theme.v("--faint"), "stroke-dasharray": "5 4" });
    mk("text", { x: 300, y: 166, class: "lbl" }, `decoder block × N = ${cfg.N}`);
    blk("emb", 330, 560, 180, 34, "Output Embedding"); mk("text", { x: 420, y: 622, class: "lbl", "text-anchor": "middle" }, `target so far, shifted right (B, T)`); wire(420, 610, 420, 596);
    blk("pe", 330, 505, 180, 30, "+ Positional Encoding"); wire(420, 560, 420, 537); wire(420, 505, 420, 452);
    blk("masked", 330, 410, 180, 40, "Masked Multi-Head\nAttention"); blk("norm", 330, 372, 180, 26, "Add & Norm"); wire(420, 410, 420, 400);
    blk("cross", 330, 300, 180, 40, "Cross-Attention\n(Q dec · K,V enc)"); wire(420, 372, 420, 342); blk("norm", 330, 262, 180, 26, "Add & Norm"); wire(420, 300, 420, 290);
    blk("ffn", 330, 214, 180, 32, "Feed Forward"); wire(420, 262, 420, 248); blk("norm", 330, 176, 180, 26, "Add & Norm"); wire(420, 214, 420, 204);
    mk("path", { d: "M130,275 L130,244 L265,244 L265,320 L328,320", class: "wire", "marker-end": "url(#arr)" });
    blk("linear", 360, 110, 120, 30, "Linear"); wire(420, 176, 420, 142);
    blk("softmax", 360, 60, 120, 30, "Softmax"); wire(420, 110, 420, 92);
    mk("text", { x: 420, y: 40, class: "lbl", "text-anchor": "middle" }, `P(next token) (B, T, ${cfg.V})`); wire(420, 60, 420, 46);
    svgWrap.replaceChildren(svg);
    const b = B[sel], [i, o] = b.io(cfg);
    panel.replaceChildren(el("h4", { style: { fontFamily: "var(--f-display)" } }, el("span", { class: "shape", style: { background: `var(${b.color})`, marginRight: ".4rem" } }, " "), b.name), el("p", {}, b.what),
      el("dl", { class: "kv" }, el("dt", {}, "input"), el("dd", {}, i), el("dt", {}, "output"), el("dd", {}, o), el("dt", {}, "parameters"), el("dd", {}, b.p(cfg).toLocaleString() + (["mha", "masked", "cross", "norm", "ffn"].includes(sel) ? "  (per block)" : ""))),
      el("pre", {}, b.code));
    const enc = cfg.N * (mhaP(cfg.d) + ffnP(cfg.d, cfg.dff) + 2 * lnP(cfg.d)), dec = cfg.N * (2 * mhaP(cfg.d) + ffnP(cfg.d, cfg.dff) + 3 * lnP(cfg.d));
    const rows = [["source embedding", cfg.V * cfg.d], ["target embedding", cfg.V * cfg.d], [`encoder (${cfg.N} blocks)`, enc], [`decoder (${cfg.N} blocks)`, dec], ["output linear", cfg.d * cfg.V + cfg.V]];
    const tot = rows.reduce((s, r) => s + r[1], 0);
    table.replaceChildren(el("div", { class: "table-wrap" }, el("table", { class: "t" }, el("tr", {}, el("th", {}, "part"), el("th", {}, "parameters"), el("th", {}, "share")),
      ...rows.map(([n, v]) => el("tr", {}, el("td", {}, n), el("td", { class: "num" }, v.toLocaleString()), el("td", {}, el("div", { class: "track", style: { height: "10px", width: "120px", background: "var(--sunk)", borderRadius: "4px", overflow: "hidden" } }, el("div", { style: { height: "100%", width: (100 * v) / tot + "%", background: "var(--accent)" } }))))),
      el("tr", {}, el("td", {}, el("b", {}, "total")), el("td", { class: "num" }, el("b", {}, tot.toLocaleString())), el("td", {}, "")))),
      el("p", { class: "w-caption" }, cfg.V > 1000 ? `The paper shares one embedding matrix between the source, the target and the output layer, which saves ${(2 * cfg.V * cfg.d).toLocaleString()} parameters → about ${((tot - 2 * cfg.V * cfg.d) / 1e6).toFixed(0)}M, the "65M" the paper reports for base.` : "Our model keeps separate embeddings for source, target and output (like minitf/transformer.py)."));
  }
  const ctl = el("div", { class: "controls" });
  function drawCtl() {
    const divs = [1, 2, 4, 8, 16, 32].filter((x) => cfg.d % x === 0);
    ctl.replaceChildren(select("preset", Object.keys(presets), "", (v) => { cfg = { ...presets[v] }; drawCtl(); draw(); }),
      select("d_model", [16, 32, 64, 128, 256, 512, 768, 1024], cfg.d, (v) => { cfg.d = +v; if (cfg.d % cfg.h) cfg.h = 1; drawCtl(); draw(); }),
      select("heads", divs, cfg.h, (v) => { cfg.h = +v; draw(); }),
      slider("N", { min: 1, max: 12, value: cfg.N }, (v) => { cfg.N = v; draw(); }),
      select("d_ff", [64, 128, 256, 512, 1024, 2048, 3072, 4096], cfg.dff, (v) => { cfg.dff = +v; draw(); }),
      select("vocab V", [44, 1000, 8000, 32000, 37000, 50257], cfg.V, (v) => { cfg.V = +v; draw(); }));
    ctl.firstChild.sel.value = Object.keys(presets).find((k) => JSON.stringify(presets[k]) === JSON.stringify(cfg)) || "";
  }
  body.append(ctl, el("div", { class: "w-row" }, el("div", { class: "w-grow", style: { flexBasis: "22rem" } }, svgWrap), panel), table);
  drawCtl(); draw();
  Theme.on(draw);
});

/* ===== LayerNorm ===== */
Widgets.register("layernorm", (host) => {
  const body = frame(host, { tag: "Add & Norm", title: "LayerNorm on one token's vector", sub: "Shift and stretch the input as much as you like: LayerNorm brings it back to mean 0 and variance 1, then applies its learned scale γ and shift β." });
  const r = TF.rng(4), base = Array.from({ length: 12 }, r.normal);
  let sc = 1, sh = 0, g = 1, b = 0;
  const view = el("div");
  function draw() {
    const x = base.map((v) => v * sc + sh), m = x.reduce((a, c) => a + c, 0) / x.length, v = x.reduce((a, c) => a + (c - m) ** 2, 0) / x.length;
    const y = x.map((c) => ((c - m) / Math.sqrt(v + 1e-5)) * g + b);
    const strip = (vals, lim) => el("div", { style: { display: "flex", gap: "2px", alignItems: "center", height: "90px" } }, ...vals.map((c) => el("div", { style: { width: "18px", display: "flex", flexDirection: "column", justifyContent: "center", height: "100%" } },
      el("div", { title: fmt(c, 3), style: { height: Math.min(45, (45 * Math.abs(c)) / lim) + "px", background: c >= 0 ? "var(--pos)" : "var(--neg)", alignSelf: "stretch", transform: c >= 0 ? "translateY(-50%)" : "translateY(50%)" } }))));
    const lim = Math.max(...x.map(Math.abs), ...y.map(Math.abs), 1);
    view.replaceChildren(el("div", { class: "w-row" },
      el("div", { class: "w-col" }, el("span", { class: "w-label" }, `input x · mean ${fmt(m, 2)} · std ${fmt(Math.sqrt(v), 2)}`), strip(x, lim)),
      el("div", { class: "w-col" }, el("span", { class: "w-label" }, `LayerNorm(x) · mean ${fmt(y.reduce((a, c) => a + c, 0) / 12, 2)} · std ${fmt(Math.sqrt(y.reduce((a, c) => a + (c - b) ** 2, 0) / 12), 2)}`), strip(y, lim))),
      el("p", { class: "w-caption" }, "LayerNorm normalises across the d_model numbers of ONE token (not across the batch, unlike BatchNorm). Deep stacks stay numerically stable whatever the previous layers did to the scale."));
  }
  body.append(el("div", { class: "controls" }, slider("input × scale", { min: 0.1, max: 20, step: 0.1, value: 1 }, (v) => { sc = v; draw(); }), slider("input + shift", { min: -20, max: 20, step: 0.5, value: 0 }, (v) => { sh = v; draw(); }),
    slider("γ", { min: 0, max: 3, step: 0.1, value: 1 }, (v) => { g = v; draw(); }), slider("β", { min: -2, max: 2, step: 0.1, value: 0 }, (v) => { b = v; draw(); })), view);
  draw();
});

/* ===== teacher forcing + causality, with the real model ===== */
Widgets.register("teacher-forcing", async (host) => {
  const body = frame(host, { tag: "teacher forcing", title: "One forward pass trains every position at once", sub: "The decoder gets the correct target shifted right and predicts the next character at every position, in parallel. Edit the target and see which predictions change." });
  body.append(loadingNote());
  const model = await TF.loadModel("baseline");
  let src = "24 nov 2026", tgt = "2026-11-24", prev = null;
  const si = el("input", { type: "text", value: src, id: uid("tf"), style: { fontFamily: "var(--f-mono)" } }), ti = el("input", { type: "text", value: tgt, id: uid("tf"), style: { fontFamily: "var(--f-mono)", width: "9rem" } });
  const view = el("div");
  function draw() {
    const s = model.encodeText(src.toLowerCase()), t = model.encodeText(tgt.toLowerCase());
    if (s.bad.length || t.bad.length || !s.ids.length) return view.replaceChildren(el("p", { class: "loading" }, "Use lowercase letters, digits, space and , . / - only."));
    const full = [1, ...t.ids, 2], tin = full.slice(0, -1), tout = full.slice(1);
    const e = model.encode(s.ids), d = model.decode(tin, e.memory, e.T);
    const rows = tin.map((_, i) => { const p = TF.softmax(model.logitsRow(d, i)); let best = 0; p.forEach((x, j) => { if (x > p[best]) best = j; }); return { p, best, pc: p[tout[i]] }; });
    const loss = rows.reduce((a, r) => a - Math.log(r.pc), 0) / rows.length;
    const name = (i) => (i === 1 ? "<bos>" : i === 2 ? "<eos>" : model.tok(i));
    const changed = prev ? rows.map((r, i) => !prev[i] || Math.abs(prev[i].pc - r.pc) > 1e-6 || prev[i].best !== r.best) : rows.map(() => false);
    view.replaceChildren(el("div", { class: "table-wrap" }, el("table", { class: "t" },
      el("tr", {}, el("th", {}, "position"), ...tin.map((_, i) => el("th", { class: "num" }, i))),
      el("tr", {}, el("th", {}, "tgt_in (decoder sees)"), ...tin.map((id) => el("td", { class: "num" }, name(id)))),
      el("tr", {}, el("th", {}, "tgt_out (must predict)"), ...tout.map((id) => el("td", { class: "num", style: { fontWeight: 700 } }, name(id)))),
      el("tr", {}, el("th", {}, "model's top guess"), ...rows.map((r, i) => el("td", { class: "num", style: { color: r.best === tout[i] ? "var(--good)" : "var(--bad)" } }, name(r.best)))),
      el("tr", {}, el("th", {}, "P(correct)"), ...rows.map((r) => el("td", { class: "num" }, fmt(r.pc * 100, 0) + "%"))),
      el("tr", {}, el("th", {}, "changed by your edit?"), ...changed.map((c) => el("td", {}, c ? el("span", { class: "badge bad" }, "yes") : el("span", { class: "badge neutral" }, "no")))))),
      el("p", {}, `loss = mean of −log P(correct) = `, el("b", {}, fmt(loss, 4)), ` over ${rows.length} positions, all computed in ONE decoder pass.`),
      el("p", { class: "w-caption" }, "Change a character in the target (e.g. make the year 2036). Only the positions AFTER your edit change: the causal mask guarantees position t never sees tgt_in[t+1:]. Without it, training could just copy the answer from the input."));
    prev = rows;
  }
  si.addEventListener("input", () => { src = si.value; prev = null; draw(); });
  ti.addEventListener("input", () => { tgt = ti.value; draw(); });
  body.replaceChildren(el("div", { class: "controls" }, el("div", { class: "ctl" }, el("label", { for: si.id }, "source"), si), el("div", { class: "ctl" }, el("label", { for: ti.id }, "target"), ti), button("Reset target", () => { ti.value = tgt = model.translate(src.toLowerCase()).text || tgt; prev = null; draw(); })), view);
  draw();
});

/* ===== training timeline ===== */
Widgets.register("timeline", async (host, opts) => {
  const body = frame(host, { tag: opts.lab ? "Lab" : "training", title: "Watch the model learn: 9 snapshots of training", sub: "The real weights saved during training. Drag through the steps: loss, accuracy, translations and attention all come from that snapshot." });
  body.append(loadingNote());
  const M = await TF.loadManifest(), base = M.models.baseline, cks = base.checkpoints;
  let idx = 4, text = "tuesday, november 24, 2026";
  const view = el("div"), live = el("div");
  const inp = el("input", { type: "text", value: text, id: uid("tl"), style: { fontFamily: "var(--f-mono)", width: "18rem", maxWidth: "100%" } });
  inp.addEventListener("input", () => { text = inp.value; drawLive(); });
  function draw() {
    const c = cks[idx];
    const truth = { "tuesday, november 24, 2026": "2026-11-24", "24/11/2026": "2026-11-24", "2026/11/24": "2026-11-24", "the 3rd of july, 1969": "1969-07-03", "1 jan 2000": "2000-01-01" };
    view.replaceChildren(el("div", { class: "w-row" },
      el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, "training loss (log scale)"), lineChart([{ name: "loss", points: base.loss_curve_every5.map((v, i) => [i * 5 + 1, v]) }], { logy: true, width: 460, height: 200, markers: [{ x: Math.max(1, c.step), label: "step " + c.step }], xLabel: "step" })),
      el("div", { class: "w-col", style: { minWidth: "16rem" } },
        el("dl", { class: "kv" }, el("dt", {}, "step"), el("dd", {}, c.step + " / 1500"), el("dt", {}, "loss"), el("dd", {}, fmt(c.loss, 4) + (c.step === 0 ? `  ≈ ln 44 = ${fmt(Math.log(44), 2)}` : "")), el("dt", {}, "test accuracy"), el("dd", { style: { fontWeight: 700 } }, fmt(c.accuracy * 100, 1) + "%")),
        el("div", { class: "table-wrap" }, el("table", { class: "t" }, ...Object.entries(c.samples).map(([s, o]) => el("tr", {}, el("td", { class: "mono", style: { fontSize: ".8rem" } }, s), el("td", { class: "num", style: { color: o === truth[s] ? "var(--good)" : "var(--bad)" } }, o || "(nothing)"))))))),
      el("span", { class: "w-label" }, "accuracy on each output character (Y Y Y Y - M M - D D)"),
      el("div", { class: "w-row", style: { gap: ".3rem", alignItems: "flex-end", height: "70px" } }, ...c.per_position.map((p, i) => el("div", { class: "w-col", style: { alignItems: "center", gap: "2px" } }, el("div", { style: { width: "26px", height: 50 * p + "px", background: "var(--accent)", borderRadius: "3px" }, title: fmt(p * 100, 1) + "%" }), el("span", { class: "mono", style: { fontSize: ".75rem" } }, "YYYY-MM-DD"[i])))),
      el("p", { class: "w-caption" }, c.step === 0 ? "Untrained: random weights, so the output is gibberish and the loss sits at ln(vocabulary size)." : c.step <= 50 ? "First it learns the FORMAT: dashes in the right places, years starting with 19 or 20. That's the easy statistics." : c.step <= 200 ? "Then it learns to copy digits, but sometimes from the wrong place (look for swapped digits like 2062). Attention isn't pointing at the right characters yet." : "Attention locks on to the right input characters and accuracy jumps to ~100%."));
    drawLive();
  }
  async function drawLive() {
    const c = cks[idx], m = await TF.loadModel("baseline", c.step === 1500 ? null : c.step), r = m.translate(text.toLowerCase());
    if (r.error) return live.replaceChildren(el("p", { class: "loading" }, "Only lowercase letters, digits, space and , . / - are allowed."));
    const S = r.enc.T, T = r.full.T, chars = r.enc.ids.map((i) => m.tok(i)), out = r.ys.slice(1, T + 1).map((i) => "→" + (i === 2 ? "⏎" : m.tok(i)));
    const W = mean(r.full.layers[m.L - 1].cross.map((x) => Array.from(x)));
    live.replaceChildren(el("div", { class: "controls" }, el("div", { class: "ctl" }, el("label", { for: inp.id }, "your date"), inp), el("span", {}, "→ ", el("b", { class: "mono" }, r.text || "(nothing)"))),
      el("span", { class: "w-label" }, `cross-attention at step ${c.step} (last layer, mean of heads)`), heatmap(rowsOf(W, T, S), { rowLabels: out, colLabels: chars, vmin: 0, cell: 15 }));
  }
  body.replaceChildren(el("div", { class: "controls" }, slider("snapshot", { min: 0, max: cks.length - 1, value: idx, fmt: (v) => "step " + cks[v].step }, (v) => { idx = v; draw(); })), view, el("div", { style: { height: "1rem" } }), live);
  draw();
});

/* ===== ablation bench ===== */
Widgets.register("ablations", async (host, opts) => {
  const body = frame(host, { tag: opts.lab ? "Lab" : "break it", title: "Ablation bench: remove a part, measure the damage", sub: "Five real models trained with the same data, steps and seed; only the architecture differs. Type a date to run all five." });
  body.append(loadingNote());
  const M = await TF.loadManifest(), names = Object.keys(MODEL_LABELS);
  const models = Object.fromEntries(await Promise.all(names.map(async (n) => [n, await TF.loadModel(n)])));
  let text = "tuesday, november 24, 2026";
  const inp = el("input", { type: "text", value: text, id: uid("ab"), style: { fontFamily: "var(--f-mono)", width: "18rem", maxWidth: "100%" } });
  const live = el("div"), tbl = el("div"), pos = el("div");
  inp.addEventListener("input", () => { text = inp.value; drawLive(); });
  function drawLive() {
    const ref = models.baseline.translate(text.toLowerCase()).text;
    live.replaceChildren(el("div", { class: "table-wrap" }, el("table", { class: "t" }, el("tr", {}, el("th", {}, "model"), el("th", {}, "params"), el("th", {}, "output"), el("th", {}, "")),
      ...names.map((n) => { const r = models[n].translate(text.toLowerCase()); return el("tr", {}, el("td", {}, MODEL_LABELS[n]), el("td", { class: "num" }, M.models[n].n_params.toLocaleString()), el("td", { class: "num", style: { fontSize: "1.05rem" } }, r.error ? "can't read input" : r.text),
        el("td", {}, r.error ? "" : n === "baseline" ? el("span", { class: "badge neutral" }, "reference") : r.text === ref ? el("span", { class: "badge good" }, "same") : el("span", { class: "badge bad" }, "differs"))); }))));
  }
  const fmts = M.formats;
  tbl.replaceChildren(el("span", { class: "w-label" }, "test accuracy by input format (100 dates each)"),
    heatmap(names.map((n) => fmts.map((f) => M.models[n].per_format[f])), { rowLabels: names.map((n) => MODEL_LABELS[n]), colLabels: fmts, vmin: 0, vmax: 1, annotate: true, cell: 46, digits: 2, tip: (i, j, v) => `${MODEL_LABELS[names[i]]} on "${fmts[j]}": ${fmt(v * 100, 0)}%` }),
    el("p", {}, ...names.map((n) => el("span", { class: "chip", style: { marginRight: ".4rem" } }, `${MODEL_LABELS[n]}: ${fmt(M.models[n].accuracy * 100, 1)}%`))));
  pos.replaceChildren(el("span", { class: "w-label" }, "accuracy per output character"), lineChart(names.map((n, i) => ({ name: MODEL_LABELS[n], values: M.models[n].per_position, dots: true, color: Theme.v(["--accent", "--neg", "--good", "--warn", "--faint"][i]) })), { ymin: 0, ymax: 1, width: 560, height: 210, xTicks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], xFmt: (v) => "YYYY-MM-DD"[v] }));
  body.replaceChildren(el("div", { class: "controls" }, el("div", { class: "ctl" }, el("label", { for: inp.id }, "date"), inp),
    button("Same characters, different order", () => { const a = ["24/11/2026", "2026/11/24", "11/24/2026"]; inp.value = text = a[(a.indexOf(text) + 1) % a.length]; drawLive(); }),
    button("Random date", () => { inp.value = text = randomDateText(); drawLive(); })), live, tbl, pos,
    el("p", { class: "w-caption" }, "Without positions, the encoder can't distinguish 24/11/2026 from 2026/11/24: same characters, so the same set of vectors. It still gets the dashes and the '19'/'20' right, because those don't depend on order. Shrinking d_model to 16 costs capacity everywhere; one head or one layer turns out to be almost enough for this simple copying task."));
  drawLive();
});

/* ===== limits / out of distribution ===== */
Widgets.register("limits", async (host) => {
  const body = frame(host, { tag: "limits", title: "Push it outside what it was trained on", sub: "Trained on years 1900–2099 and 10 formats. What happens outside that?" });
  body.append(loadingNote());
  const model = await TF.loadModel("baseline");
  const cases = ["24 nov 2150", "24 nov 1850", "nov 24th 2026", "2026 nov 24", "24-11-2026", "24 Nov 2026!", "november 31, 2026"];
  body.replaceChildren(el("div", { class: "table-wrap" }, el("table", { class: "t" }, el("tr", {}, el("th", {}, "input"), el("th", {}, "output"), el("th", {}, "least-confident character")),
    ...cases.map((c) => {
      const r = model.translate(c.toLowerCase());
      if (r.error) return el("tr", {}, el("td", { class: "mono" }, c), el("td", { colspan: 2 }, el("span", { class: "badge bad" }, "can't tokenize"), ` "${r.error.join("")}" has no token id.`));
      const conf = r.steps.map((s) => s.probs[s.choice]), lo = conf.indexOf(Math.min(...conf));
      return el("tr", {}, el("td", { class: "mono" }, c), el("td", { class: "num" }, r.text), el("td", {}, `"${model.vocab[r.steps[lo].choice] === undefined ? "" : model.tok(r.steps[lo].choice)}" at ${fmt(conf[lo] * 100, 0)}% (avg ${fmt((conf.reduce((a, b) => a + b, 0) / conf.length) * 100, 0)}%)`));
    }))),
    el("p", { class: "w-caption" }, "It only learned patterns it saw. 2150 isn't a year it can write, so it falls back to something familiar, often with surprisingly high confidence. 'November 31' doesn't exist, but it copies the digits anyway: it learned to copy, not what a calendar is. And '!' and capitals can't even be read. That's a tokenization limit, which byte-level BPE removes."));
});
