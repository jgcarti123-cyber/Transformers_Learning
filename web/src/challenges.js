/* ============================================================================
   challenges.js: hands-on challenges. Compute-it (type the numbers yourself),
   bug hunts (diagnose a broken model from its symptoms), be-the-model puzzles
   (set the numbers to hit a goal), and a sentence sandbox.
   Each challenge appears at the end of its lesson AND in the Challenges hub.
   ========================================================================== */
KIND.challenge = ["Challenge", "--c-pos"];
const Challenges = {
  list: [],
  add(c) { this.list.push(c); },
  solved(id) { return !!store.get("ch:" + id, false); },
  solve(id) { if (!this.solved(id)) { store.set("ch:" + id, true); this.refresh(); } },
  refresh() { const n = this.list.filter((c) => this.solved(c.id)).length; const b = $("#ch-count"); if (b) b.textContent = `${n}/${this.list.length}`; },
};
const TYPE_LABEL = { compute: "🧮 Compute it", bug: "🐞 Bug hunt", puzzle: "🧩 Be the model" };
function challengeCard(c) {
  const status = el("span", { class: "badge " + (Challenges.solved(c.id) ? "good" : "neutral") }, Challenges.solved(c.id) ? "solved ✓" : "not solved yet");
  const body = el("div", { class: "ch-body" });
  const done = () => { Challenges.solve(c.id); status.className = "badge good"; status.textContent = "solved ✓"; };
  const card = el("section", { class: "ch-card" }, el("div", { class: "ch-head" }, el("span", { class: "ch-type" }, TYPE_LABEL[c.type]), el("b", {}, c.title), status), c.intro ? el("div", { class: "step-text", html: c.intro }) : null, body);
  c.mount(body, done);
  return card;
}

/* ---------- 🧮 compute it: you type the numbers ---------- */
function computeIt(body, done, { fields, hints = [], explain }) {
  let tries = 0;
  const inputs = fields.map((f) => el("input", { type: "text", inputmode: "decimal", id: uid("cf"), autocomplete: "off", style: { width: "6rem", fontFamily: "var(--f-mono)" } }));
  const marks = fields.map(() => el("span"));
  const fb = el("div", { class: "ch-feedback" });
  const parse = (s) => parseFloat(String(s).replace("−", "-").replace(",", "."));
  const check = () => {
    let all = true;
    fields.forEach((f, i) => { const v = parse(inputs[i].value), ok = Number.isFinite(v) && Math.abs(v - f.answer) <= (f.tol ?? 0.02); marks[i].replaceChildren(el("span", { class: "badge " + (ok ? "good" : "bad") }, ok ? "✓" : "✗")); if (!ok) all = false; });
    if (all) { fb.replaceChildren(el("p", { class: "ok-line" }, "🎉 Correct! "), el("div", { html: explain })); done(); return; }
    tries++;
    const h = hints[Math.min(tries - 1, hints.length - 1)];
    fb.replaceChildren(el("p", {}, el("b", {}, `Not yet (try ${tries}). `), h ? el("span", { html: "Hint: " + h }) : ""),
      tries >= 3 ? button("Show me the answer", () => { fields.forEach((f, i) => (inputs[i].value = f.answer.toFixed(f.digits ?? 2))); check(); }) : "");
  };
  inputs.forEach((inp) => inp.addEventListener("keydown", (e) => { if (e.key === "Enter") check(); }));
  body.append(el("div", { class: "ch-fields" }, ...fields.map((f, i) => el("div", { class: "ctl" }, el("label", { for: inputs[i].id, html: f.label }), inputs[i], marks[i]))),
    el("div", { class: "controls" }, button("Check", check, "primary")), fb);
}

/* ---------- 🐞 bug hunt: diagnose from symptoms ---------- */
const BUGS = { axis: "Softmax over the wrong axis (columns instead of rows)", scale: "Forgot to divide by √d_k", nomask: "No causal mask", after: "Masked AFTER the softmax (set weights to 0 instead of scores to −∞)", nopos: "No positional encoding", reshape: "Wrong reshape when splitting heads" };
function bugHunt(body, done, { symptom, options, answer, explain, fixed }) {
  const sym = el("div", { class: "symptom" }), fix = el("div", { hidden: true });
  symptom(sym);
  const q = predictBox({ q: "What's the bug?", options: options.map((k) => BUGS[k]), answer: options.indexOf(answer), why: explain }, (ok) => { if (ok) done(); fix.hidden = false; });
  body.append(el("span", { class: "w-label" }, "the symptoms"), sym, q, fix);
  if (fixed) { fix.append(el("span", { class: "w-label" }, "after the fix"), el("div", { class: "symptom ok" })); fixed(fix.lastChild); }
}
function sums(W) { return { rows: W.map((r) => r.reduce((a, b) => a + b, 0)), cols: W[0].map((_, j) => W.reduce((a, r) => a + r[j], 0)) }; }
function gridWithSums(W, labels) {
  const s = sums(W);
  return el("div", { class: "w-row" }, heatmap(W, { rowLabels: labels, colLabels: labels, vmin: 0, vmax: 1, annotate: true, cell: 38, digits: 2 }),
    el("div", { class: "w-col" }, el("span", { class: "w-label" }, "row sums"), el("code", {}, s.rows.map(f2).join("  ")), el("span", { class: "w-label" }, "column sums"), el("code", {}, s.cols.map(f2).join("  "))));
}
// a head where every content word asks something, so every row of the grid is interesting
const HALL = { WQ: [[1.5, 0], [0, 1.5], [1.5, 0], [2, 0]], WK: [[2, 0], [0, 2], [0, 0], [0, 0]], qLabels: ["", ""], kLabels: ["", ""] };
const SHORT = ["the", "cat", "sat", "on", "the", "mat"];
const scoresOf = (words, head, scale = 1) => words.map((_, i) => TOY.attend(words, i, head).scaled.map((s) => s * scale));
const rowSoftmax = (S) => S.map((r) => TOY.softmax(r));

/* ---------- puzzles: sliders + levels ---------- */
function levels(body, done, { lv, controls, view }) {
  let cur = 0;
  const lvBox = el("div", { class: "levels" }), out = el("div"), status = el("div", { class: "ch-feedback" });
  const won = new Set();
  const draw = () => {
    const state = view(out), L = lv[cur], ok = L.goal(state);
    if (ok && !won.has(cur)) { won.add(cur); if (won.size === lv.length) done(); }
    lvBox.replaceChildren(...lv.map((l, i) => el("button", { type: "button", class: "chip click" + (i === cur ? " on" : ""), onclick: () => { cur = i; draw(); } }, (won.has(i) ? "✓ " : "") + "Level " + (i + 1))));
    status.replaceChildren(el("p", {}, el("b", {}, `Level ${cur + 1}: `), el("span", { html: L.text })), ok ? el("p", { class: "ok-line" }, "🎉 Level complete! ", cur < lv.length - 1 ? button("Next level →", () => { cur++; draw(); }, "primary") : el("b", {}, "All levels done.")) : el("p", { class: "w-caption" }, L.hint ? "Hint: " + L.hint : ""));
  };
  body.append(lvBox, status, controls(draw), out);
  draw();
}

/* =================== the challenges =================== */
Challenges.add({ id: "dot", lesson: "l1", type: "compute", title: "Dot product by hand",
  intro: `<p>Compute the dot product of <code>[1, 0, 2]</code> and <code>[3, 1, 1]</code>: multiply position by position, then add.</p>`,
  mount: (b, d) => computeIt(b, d, { fields: [{ label: "[1, 0, 2] · [3, 1, 1] =", answer: 5, tol: 0.001, digits: 0 }], hints: ["1×3 + 0×1 + 2×1", "3 + 0 + 2"], explain: "1×3 + 0×1 + 2×1 = 3 + 0 + 2 = <b>5</b>. Attention does this hundreds of times per sentence." }) });

Challenges.add({ id: "cattired", lesson: "l2", type: "compute", title: "How similar are 'cat' and 'tired'?",
  intro: `<p>cat = <code>[1.0, 0.1, 0.0, 0.0]</code>, tired = <code>[0.6, 0.0, 0.8, 0.0]</code>. Compute their dot product.</p>`,
  mount: (b, d) => computeIt(b, d, { fields: [{ label: "cat · tired =", answer: 0.6 }], hints: ["Multiply living×living, object×object, …", "1.0×0.6 + 0.1×0 + 0×0.8 + 0×0"], explain: "1.0×0.6 = 0.6, and every other product is 0. So <b>0.6</b>: they share 'living', nothing else." }) });

Challenges.add({ id: "kitten", lesson: "l2", type: "puzzle", title: "Invent new words",
  intro: `<p>Set the four meaning numbers of a <b>new word</b> to hit each goal. The model finds its nearest neighbours by cosine similarity.</p>`,
  mount: (b, d) => {
    const v = [0.5, 0.5, 0.5, 0];
    const words = Object.keys(TOY.EMB);
    levels(b, d, {
      lv: [
        { text: `Make <b>"kitten"</b>: its nearest word must be <b>cat</b>, with cosine ≥ 0.95.`, goal: (s) => s.top[0][0] === "cat" && s.top[0][1] >= 0.95, hint: "kittens are alive, not objects, not actions." },
        { text: `Make <b>"sofa"</b>: nearest word <b>mat</b> or <b>rug</b>, cosine ≥ 0.95.`, goal: (s) => ["mat", "rug"].includes(s.top[0][0]) && s.top[0][1] >= 0.95 },
        { text: `Make <b>"sleepy"</b>: nearest word <b>tired</b>, cosine ≥ 0.97.`, goal: (s) => s.top[0][0] === "tired" && s.top[0][1] >= 0.97, hint: "tired = living 0.6, action 0.8. Only the direction matters, not the size." },
        { text: `Make a word <b>exactly between</b> cat and mat: cosine to both ≥ 0.6 and within 0.02 of each other.`, goal: (s) => Math.abs(s.cc - s.cm) <= 0.02 && s.cc >= 0.6 && s.cm >= 0.6, hint: "equal living and object, nothing else." },
      ],
      controls: (draw) => el("div", { class: "controls" }, ...TOY.FEATURES.map((f, i) => slider(f, { min: 0, max: 1, step: 0.05, value: v[i], fmt: (x) => x.toFixed(2) }, (x) => { v[i] = x; draw(); }))),
      view: (out) => {
        const top = words.map((w) => [w, TOY.cosine(v, TOY.EMB[w])]).sort((a, c) => c[1] - a[1]);
        out.replaceChildren(el("div", { class: "w-row" }, vecBars(v, { title: `your word = [${v.map(f2).join(", ")}]` }), el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, "nearest words (cosine)"), bars(top.slice(0, 5).map(([w, c], i) => ({ label: w, value: Math.max(0, c), cls: i === 0 ? "hi" : "" })), { max: 1, digits: 2 }))));
        return { top, cc: TOY.cosine(v, TOY.EMB.cat), cm: TOY.cosine(v, TOY.EMB.mat) };
      },
    });
  } });

Challenges.add({ id: "score", lesson: "l3", type: "compute", title: "Score 'it' against 'tired'",
  intro: `<p>"it"'s query is <code>[2, 0]</code>. "tired"'s key is <code>[1.2, 0]</code>. Compute the raw score, then the scaled score (÷ √2 ≈ 1.414).</p>`,
  mount: (b, d) => computeIt(b, d, { fields: [{ label: "query · key =", answer: 2.4 }, { label: "÷ √2 =", answer: 2.4 / Math.SQRT2 }], hints: ["2×1.2 + 0×0", "then 2.4 ÷ 1.414"], explain: "2×1.2 + 0×0 = <b>2.4</b>, and 2.4 ÷ 1.414 = <b>1.70</b>. That's exactly the 'tired' row you saw in lesson 3." }) });

Challenges.add({ id: "softmax3", lesson: "l3", type: "compute", title: "Softmax three scores",
  intro: `<p>Turn the scores <code>[2, 1, 0]</code> into attention weights. Use a calculator for e<sup>x</sup> (e ≈ 2.718).</p>`,
  mount: (b, d) => { const e = [Math.exp(2), Math.E, 1], s = e[0] + e[1] + e[2];
    computeIt(b, d, { fields: [0, 1, 2].map((i) => ({ label: `weight ${i + 1}`, answer: e[i] / s, tol: 0.01 })), hints: ["Step 1: e² = 7.39, e¹ = 2.72, e⁰ = 1", "Step 2: total = 7.39 + 2.72 + 1 = 11.11. Divide each by it."], explain: `7.39 / 11.11 = <b>0.67</b>, 2.72 / 11.11 = <b>0.24</b>, 1 / 11.11 = <b>0.09</b>. They add up to 1. A gap of 1 in the scores became almost a 3× gap in the weights.` }); } });

Challenges.add({ id: "query", lesson: "l3", type: "puzzle", title: "Be the query",
  intro: `<p>You control what "it" asks for, and how much action words look "living" to the keys. Hit each target.</p>`,
  mount: (b, d) => {
    let q0 = 0.5, q1 = 0.5, a = 0;
    const head = () => ({ WQ: [[0, 0], [0, 0], [0, 0], [q0, q1]], WK: [[2, 0], [0, 2], [a, 0], [0, 0]] });
    levels(b, d, {
      lv: [
        { text: `Make "it" give <b>≥ 60%</b> of its attention to <b>cat</b>.`, goal: (w) => w[1] >= 0.6, hint: "ask strongly for 'living'." },
        { text: `Now make "it" look at <b>mat</b>, ≥ 60%.`, goal: (w) => w[5] >= 0.6 },
        { text: `Make "it" pay <b>equal-ish attention to everyone</b>: no word above 12%.`, goal: (w) => Math.max(...w) <= 0.12, hint: "what if the query asks for nothing?" },
        { text: `Make <b>tired</b> win: give it more attention than cat.`, goal: (w) => w[9] > w[1] && w[9] > 0.3, hint: "make action words look more 'living' to the keys." },
        { text: `Laser focus: <b>≥ 95%</b> on cat.`, goal: (w) => w[1] >= 0.95, hint: "a bigger query sharpens the softmax, but keep tired out of the way." },
      ],
      controls: (draw) => el("div", { class: "controls" },
        slider("query: wants living", { min: -2, max: 6, step: 0.1, value: q0, fmt: (x) => x.toFixed(1) }, (x) => { q0 = x; draw(); }),
        slider("query: wants object", { min: -2, max: 6, step: 0.1, value: q1, fmt: (x) => x.toFixed(1) }, (x) => { q1 = x; draw(); }),
        slider("W_K: action words look living", { min: 0, max: 3, step: 0.1, value: a, fmt: (x) => x.toFixed(1) }, (x) => { a = x; draw(); })),
      view: (out) => { const r = TOY.attend(S, IT, head()); out.replaceChildren(wordRow(S, { weights: r.w, max: 1 }), el("p", { class: "w-caption" }, `query = [${q0.toFixed(1)}, ${q1.toFixed(1)}] · scores ÷√2 = [${r.scaled.map(f1).join(", ")}]`)); return r.w; },
    });
  } });

Challenges.add({ id: "see", lesson: "l4", type: "puzzle", title: "What can 'it' see?",
  intro: `<p>A causal mask is on (the model predicts the next word). Click every word that "it" (position 7) <b>is allowed to attend to</b>, then press Check.</p>`,
  mount: (b, d) => {
    const pick = new Set(); const out = el("div"), fb = el("div", { class: "ch-feedback" });
    const draw = () => out.replaceChildren(wordRow(S, { sel: IT, sub: (i) => (pick.has(i) ? "✓ picked" : "pos " + i), onClick: (i) => { pick.has(i) ? pick.delete(i) : pick.add(i); draw(); }, weights: S.map((_, i) => (pick.has(i) ? 0.8 : 0)), max: 1 }));
    b.append(out, el("div", { class: "controls" }, button("Check", () => {
      const right = [0, 1, 2, 3, 4, 5, 6, 7], ok = right.every((i) => pick.has(i)) && pick.size === 8;
      fb.replaceChildren(ok ? el("p", { class: "ok-line", html: `🎉 Right: positions 0–7, <b>including itself</b>. "was" and "tired" are in the future.` }) : el("p", { html: pick.has(8) || pick.has(9) ? `"was" and "tired" come <b>after</b> "it": they're the future.` : !pick.has(7) ? `Can a word look at <b>itself</b>? (Yes: the mask is j ≤ i.)` : `Every <b>earlier</b> word is allowed.` }));
      if (ok) d();
    }, "primary")), fb); draw();
  } });

Challenges.add({ id: "bug-axis", lesson: "l4", type: "bug", title: "The weights look strange",
  intro: `<p>Someone's attention code produced this grid for "the cat sat on the mat".</p>`,
  mount: (b, d) => bugHunt(b, d, { options: ["scale", "axis", "nomask", "after"], answer: "axis",
    symptom: (s) => { const Sc = scoresOf(SHORT, HALL), colSoft = rowSoftmax(Sc[0].map((_, j) => Sc.map((r) => r[j]))), Wt = colSoft[0].map((_, j) => colSoft.map((r) => r[j])); s.append(gridWithSums(Wt, SHORT), el("p", { class: "w-caption" }, "Look at the row sums and column sums.")); },
    explain: `Each <b>row</b> (one word's attention) should sum to 1, but here the <b>columns</b> do. The softmax ran over the wrong dimension: <code>softmax(scores, dim=0)</code> instead of <code>dim=-1</code>.`,
    fixed: (s) => s.append(gridWithSums(rowSoftmax(scoresOf(SHORT, HALL)), SHORT)) }) });

Challenges.add({ id: "bug-nomask", lesson: "l4", type: "bug", title: "Perfect in training, useless in use",
  intro: `<p>A next-word model reached a training loss of <b>0.01</b> after just a few steps. But when it writes text on its own, it produces gibberish. Here's one of its attention grids.</p>`,
  mount: (b, d) => bugHunt(b, d, { options: ["scale", "axis", "nomask", "nopos"], answer: "nomask",
    symptom: (s) => s.append(gridWithSums(rowSoftmax(scoresOf(SHORT, HALL)), SHORT), el("p", { class: "w-caption" }, "Look at the top-right triangle.")),
    explain: `Words are attending to words <b>after</b> them (top-right triangle). During training it could see the answer and just copy it, which explains the near-zero loss. When writing new text, the future doesn't exist, so it fails. The fix is the causal mask.`,
    fixed: (s) => { const W = SHORT.map((_, i) => TOY.attend(SHORT, i, HALL, { mask: SHORT.map((_, j) => j <= i) }).w); s.append(gridWithSums(W, SHORT)); } }) });

Challenges.add({ id: "bug-scale", lesson: "l4", type: "bug", title: "Training is stuck",
  intro: `<p>A big model (d<sub>k</sub> = 256) won't learn: the loss barely moves and the gradients are around 10<sup>−9</sup>. Its attention looks like this.</p>`,
  mount: (b, d) => bugHunt(b, d, { options: ["scale", "axis", "after", "reshape"], answer: "scale",
    symptom: (s) => s.append(gridWithSums(rowSoftmax(scoresOf(SHORT, HALL, Math.sqrt(256 / 2) * 1.414)), SHORT), el("p", { class: "w-caption" }, "Every row is ~100% on one word.")),
    explain: `With 256 numbers per query, dot products get huge (their spread grows like √256 = 16). The softmax saturates into hard 0/1 choices, where its slope is ≈ 0, so no learning signal flows back. Dividing by √d<sub>k</sub> brings the scores back to a sensible size.`,
    fixed: (s) => s.append(gridWithSums(rowSoftmax(scoresOf(SHORT, HALL)), SHORT)) }) });

Challenges.add({ id: "bug-after", lesson: "l4", type: "bug", title: "The output keeps shrinking",
  intro: `<p>With the causal mask added, this model's outputs get smaller and smaller for early words. Here are its weights.</p>`,
  mount: (b, d) => bugHunt(b, d, { options: ["after", "nomask", "axis", "scale"], answer: "after",
    symptom: (s) => { const W = rowSoftmax(scoresOf(SHORT, HALL)).map((r, i) => r.map((w, j) => (j <= i ? w : 0))); s.append(gridWithSums(W, SHORT), el("p", { class: "w-caption" }, "The future is masked… but check the row sums.")); },
    explain: `The future <i>is</i> blocked, but the rows no longer add up to 1: the weights were zeroed <b>after</b> the softmax, so the remaining weights weren't re-normalised. Early words lose most of their weight, and the outputs shrink. Mask the <b>scores</b> with −∞ before the softmax.`,
    fixed: (s) => { const W = SHORT.map((_, i) => TOY.attend(SHORT, i, HALL, { mask: SHORT.map((_, j) => j <= i) }).w); s.append(gridWithSums(W, SHORT)); } }) });

Challenges.add({ id: "pe5", lesson: "l5", type: "compute", title: "Meaning + position",
  intro: `<p>"mat" = <code>[0, 1, 0, 0]</code> sits at position 5, whose position code is <code>[−0.96, 0.28, 0.05, 1.00]</code>. What goes into the model?</p>`,
  mount: (b, d) => computeIt(b, d, { fields: [-0.96, 1.28, 0.05, 1.0].map((x, i) => ({ label: `number ${i + 1}`, answer: x, tol: 0.011 })), hints: ["Just add, number by number.", "0 + (−0.96), 1 + 0.28, 0 + 0.05, 0 + 1.00"], explain: `<b>[−0.96, 1.28, 0.05, 1.00]</b>. The same word at another position would get a different code, so a different input.` }) });

Challenges.add({ id: "bug-nopos", lesson: "l5", type: "bug", title: "Order doesn't matter?!",
  intro: `<p>A translator gives the same answer for "dog bites man" and "man bites dog". Here are "it"'s attention and output for two different word orders.</p>`,
  mount: (b, d) => bugHunt(b, d, { options: ["nopos", "nomask", "scale", "axis"], answer: "nopos",
    symptom: (s) => { const r1 = TOY.attend(S, IT, TOY.HEAD1), r2 = TOY.attend(SHUF, IT, TOY.HEAD1); s.append(wordRow(S, { weights: r1.w, max: 0.6 }), wordRow(SHUF, { weights: r2.w, max: 0.6 }), mathTable(["order", "output of 'it'"], [["original", r1.out.map(f2).join(", ")], ["shuffled", r2.out.map(f2).join(", ")]])); },
    explain: `Identical outputs for different orders means attention sees a bag of words: there's nothing that encodes position. Add positional encodings to the inputs.`,
    fixed: (s) => { const r1 = TOY.attend(S, IT, TOY.HEAD1, { embs: withPE(S) }), r2 = TOY.attend(SHUF, IT, TOY.HEAD1, { embs: withPE(SHUF) }); s.append(mathTable(["order", "output of 'it'"], [["original", r1.out.map(f2).join(", ")], ["shuffled", r2.out.map(f2).join(", ")]])); } }) });

Challenges.add({ id: "bug-reshape", lesson: "l6", type: "bug", title: "Heads that make no sense",
  intro: `<p>3 tokens × 8 numbers were split into 2 heads. Each cell shows which token (t0, t1, t2) the number originally came from.</p>`,
  mount: (b, d) => bugHunt(b, d, { options: ["reshape", "axis", "nopos", "scale"], answer: "reshape",
    symptom: (s) => { const flat = Array.from({ length: 24 }, (_, k) => k), head = (h) => [0, 1, 2].map((t) => [0, 1, 2, 3].map((c) => flat[h * 12 + t * 4 + c]));
      s.append(el("div", { class: "w-row" }, ...[0, 1].map((h) => el("div", { class: "w-col" }, el("span", { class: "w-label" }, `head ${h}, row t = token t?`), numGrid(head(h), { fmtv: (v) => "t" + Math.floor(v / 8), style: (i, j, v) => ({ outline: Math.floor(v / 8) !== i ? "2px solid var(--bad)" : "none" }) }))))); },
    explain: `Head 0's row "t1" contains numbers from t0, and so on: each head mixes pieces of different tokens. The code did <code>x.view(B, h, T, d_k)</code> directly. The right way is <code>x.view(B, T, h, d_k).transpose(1, 2)</code>. The shape is identical, so nothing crashes. That's what makes this bug dangerous.`,
    fixed: (s) => { const head = (h) => [0, 1, 2].map((t) => [0, 1, 2, 3].map((c) => t * 8 + h * 4 + c)); s.append(el("div", { class: "w-row" }, ...[0, 1].map((h) => el("div", { class: "w-col" }, el("span", { class: "w-label" }, `head ${h}`), numGrid(head(h), { fmtv: (v) => "t" + Math.floor(v / 8) }))))); } }) });

Challenges.add({ id: "assemble", lesson: "l7", type: "puzzle", title: "Assemble a Transformer",
  intro: `<p>Tap the pieces in the order data flows through a (decoder-only) Transformer, from the input token to the next-word probabilities. A block appears once here; real models repeat it.</p>`,
  mount: (b, d) => {
    const right = ["Input embedding", "+ Positional encoding", "Multi-head attention", "Add & Norm", "Feed-forward", "Add & Norm", "Linear", "Softmax"];
    const kinds = { "Input embedding": "emb", "+ Positional encoding": "pos", "Multi-head attention": "attn", "Add & Norm": "norm", "Feed-forward": "ffn", Linear: "lin", Softmax: "soft" };
    let pool = [5, 2, 7, 0, 3, 6, 1, 4].map((i) => right[i]), seq = [];
    const out = el("div"), fb = el("div", { class: "ch-feedback" });
    const draw = () => {
      out.replaceChildren(el("span", { class: "w-label" }, "your Transformer (tap a piece to remove it)"),
        el("div", { class: "flow assemble" }, ...(seq.length ? seq.map((p, i) => el("button", { type: "button", class: "shape " + kinds[p], onclick: () => { pool.push(p); seq.splice(i, 1); draw(); } }, `${i + 1}. ${p}`)) : [el("span", { class: "w-caption" }, "empty")])),
        el("span", { class: "w-label" }, "pieces"), el("div", { class: "flow" }, ...pool.map((p, i) => el("button", { type: "button", class: "shape " + kinds[p], onclick: () => { seq.push(p); pool.splice(i, 1); draw(); } }, p))));
      if (seq.length === right.length) {
        const bad = seq.findIndex((p, i) => p !== right[i]);
        if (bad < 0) { fb.replaceChildren(el("p", { class: "ok-line", html: "🎉 That's a Transformer: embed, add positions, then attention → add &amp; norm → feed-forward → add &amp; norm (×N), then Linear + Softmax for next-word probabilities." })); d(); }
        else fb.replaceChildren(el("p", { html: `Piece ${bad + 1} ("${seq[bad]}") is out of place. ${{ 0: "Everything starts by turning token IDs into vectors.", 1: "Position goes in right after the embedding, before any attention.", 2: "Words must exchange information first: the meeting comes before the thinking.", 3: "Every sublayer is followed by Add & Norm.", 4: "After the first Add & Norm, each word 'thinks alone'.", 5: "The feed-forward also gets its own Add & Norm.", 6: "Linear turns vectors into one score per vocabulary word…", 7: "…and Softmax turns scores into probabilities." }[bad]}` }));
      } else fb.replaceChildren();
    };
    b.append(out, fb, el("div", { class: "controls" }, button("Start over", () => { pool = [5, 2, 7, 0, 3, 6, 1, 4].map((i) => right[i]); seq = []; draw(); })));
    draw();
  } });

Challenges.add({ id: "layernorm", lesson: "l7", type: "compute", title: "LayerNorm by hand",
  intro: `<p>Normalise the vector <code>[1, 2, 3, 6]</code>: find its average, its spread (std = √(average of squared differences)), and the first normalised value.</p>`,
  mount: (b, d) => computeIt(b, d, { fields: [{ label: "average", answer: 3 }, { label: "std", answer: Math.sqrt(3.5), tol: 0.02 }, { label: "first value: (1 − avg) ÷ std", answer: -2 / Math.sqrt(3.5), tol: 0.02 }],
    hints: ["average = (1+2+3+6) ÷ 4", "differences from 3: −2, −1, 0, 3 → squares 4, 1, 0, 9 → average 3.5 → std = √3.5"], explain: `average <b>3</b>; squared differences 4, 1, 0, 9 average to 3.5, so std = <b>1.87</b>; first value (1 − 3) ÷ 1.87 = <b>−1.07</b>.` }) });

Challenges.add({ id: "loss", lesson: "l8", type: "compute", title: "How wrong is it?",
  intro: `<p>The model gave the right answer a probability of 25% in one example and 90% in another. What's the loss (−ln p) for each?</p>`,
  mount: (b, d) => computeIt(b, d, { fields: [{ label: "p = 0.25: loss =", answer: -Math.log(0.25) }, { label: "p = 0.90: loss =", answer: -Math.log(0.9) }], hints: ["−ln(0.25) = ln(4)", "ln 4 ≈ 1.386; −ln(0.9) ≈ 0.105"], explain: `−ln 0.25 = <b>1.39</b>, −ln 0.9 = <b>0.11</b>. Being unsure costs a lot more than being nearly right.` }) });

Challenges.add({ id: "lr", lesson: "l8", type: "puzzle", title: "Tune the learning rate",
  intro: `<p>Pick a learning rate and train the "who is it?" model for <b>40 steps</b>. Goal: <b>100% on the unseen test sentences</b>. Too small and it's too slow; too big and it overshoots.</p>`,
  mount: (b, d) => {
    let lr = 0.0005, timer = null;
    const out = el("div"), fb = el("div", { class: "ch-feedback" }), tried = {};
    const run = () => {
      clearInterval(timer);
      const m = new TOY.ItModel(16, 6), losses = []; let s = 0;
      timer = setInterval(() => {
        for (let k = 0; k < 2 && s < 40; k++, s++) losses.push(Math.min(10, m.step(IT_DATA.train, lr).loss));
        const te = m.evaluate(IT_DATA.test);
        out.replaceChildren(el("p", { class: "status-line" }, `lr ${lr} · step ${s}/40 · test accuracy ${Math.round(te.acc * 100)}%`), lineChart([{ name: "loss", values: losses }], { width: 420, height: 150, ymin: 0 }));
        if (s >= 40) {
          clearInterval(timer); tried[lr] = te.acc;
          if (te.acc === 1) { fb.replaceChildren(el("p", { class: "ok-line", html: `🎉 lr = ${lr} works! Too small (0.0005) crawls; too big (0.1, 0.5) jumps over the valley and the loss explodes. Picking the learning rate is one of the most important choices in training.` })); d(); }
          else fb.replaceChildren(el("p", { html: Math.max(...losses.slice(-5)) > losses[0] * 0.9 && lr >= 0.1 ? `The loss is bouncing or exploding: steps are <b>too big</b>.` : `Learning, but <b>too slowly</b> to finish in 40 steps.` }));
        }
      }, 30);
    };
    b.append(el("div", { class: "controls" }, seg([0.0005, 0.002, 0.02, 0.1, 0.5], lr, (v) => { lr = v; }, "learning rate"), button("Train 40 steps", run, "primary")), out, fb);
  } });

/* ---------- inject a Challenge step into each lesson (before a final Code step) ---------- */
["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"].forEach((lid) => {
  const L = Lessons.byId[lid], mine = Challenges.list.filter((c) => c.lesson === lid);
  if (!mine.length) return;
  const step = { kind: "challenge", title: mine.length > 1 ? `Challenges: ${mine.length} to solve` : "Challenge: " + mine[0].title,
    html: `<p>Use what you just learned. Solved challenges are ticked here and in the <a href="#challenges">Challenges hub</a>.</p>`,
    render(a) { mine.forEach((c) => a.append(challengeCard(c))); },
    recap: `Doing the numbers yourself is what makes them stick. Stuck? Every challenge has hints, and an answer after 3 tries.` };
  const last = L.steps[L.steps.length - 1];
  if (last.kind === "code") L.steps.splice(L.steps.length - 1, 0, step); else L.steps.push(step);
});

/* ---------- hub ---------- */
function renderHub() {
  const v = $("#view-challenges");
  if (!v) return;
  const n = Challenges.list.filter((c) => Challenges.solved(c.id)).length;
  v.replaceChildren(el("header", { class: "lesson-head" }, el("div", { class: "eyebrow" }, el("span", { class: "pill" }, "Practice")), el("h1", {}, "Challenges"),
    el("p", { class: "lede" }, `${n} of ${Challenges.list.length} solved. Compute the numbers yourself, diagnose broken models, and set the numbers to make the model do what you want.`),
    el("div", { class: "controls" }, ...Object.entries(TYPE_LABEL).map(([t, l]) => el("a", { class: "chip click", href: "#challenges", onclick: (e) => { e.preventDefault(); $("#hub-" + t).scrollIntoView({ behavior: "smooth" }); } }, `${l} (${Challenges.list.filter((c) => c.type === t).length})`)))),
    ...Object.entries(TYPE_LABEL).map(([t, l]) => el("div", { class: "hub-group", id: "hub-" + t }, el("h2", {}, l), ...Challenges.list.filter((c) => c.type === t).map((c) => el("div", {}, el("div", { class: "w-caption" }, `from lesson ${c.lesson.slice(1)}`), challengeCard(c))))));
}

/* ---------- 🧪 sentence sandbox ---------- */
const SANDBOX_EXTRA = { kitten: [0.95, 0.1, 0, 0], puppy: [0.9, 0.15, 0, 0], fish: [0.8, 0.1, 0, 0], she: [0.5, 0, 0, 1], he: [0.5, 0, 0, 1], box: [0, 1, 0, 0], sofa: [0, 0.95, 0, 0], ate: [0.1, 0, 1, 0], slept: [0.1, 0, 1, 0], ran: [0.1, 0, 1, 0], hungry: [0.6, 0, 0.8, 0], happy: [0.6, 0, 0.7, 0], soft: [0, 0.6, 0.5, 0], a: [0.1, 0.1, 0, 0], and: [0, 0, 0.2, 0], under: [0, 0.2, 0.1, 0], with: [0, 0.1, 0.1, 0] };
Widgets.register("sandbox", (host) => {
  const body = frame(host, { tag: "Sandbox", title: "Your own sentence", sub: "Build any sentence from the word bank, pick a word, and watch its attention recompute. Drag any word's meaning bars and see what changes." });
  let E = JSON.parse(JSON.stringify({ ...TOY.EMB, ...SANDBOX_EXTRA }));
  let words = "the puppy slept on the sofa because it was tired".split(" "), q = 7, headSel = "1", scale = true, mask = false, pos = false, editW = "it";
  const sentBox = el("div"), bank = el("div"), view = el("div"), editor = el("div");
  const heads = { 1: TOY.HEAD1, 2: TOY.HEAD2 };
  const embs = () => words.map((w, i) => E[w].map((v, c) => v + (pos ? TOY.pe(i)[c] : 0)));
  const run = (hk) => TOY.attend(words, q, heads[hk], { scale, mask: mask ? words.map((_, j) => j <= q) : null, embs: embs() });
  function draw() {
    q = Math.min(q, words.length - 1);
    sentBox.replaceChildren(el("span", { class: "w-label" }, "your sentence (click a word to ask from it; × removes it)"),
      el("div", { class: "wordrow" }, ...words.map((w, i) => el("span", { class: "sb-word" }, el("button", { type: "button", class: "wbox" + (i === q ? " sel" : ""), onclick: () => { q = i; draw(); } }, el("span", {}, w), el("small", {}, "pos " + i)),
        el("button", { type: "button", class: "sb-x", "aria-label": "remove " + w, onclick: () => { if (words.length > 2) { words.splice(i, 1); if (q >= i && q > 0) q--; draw(); } } }, "×")))));
    bank.replaceChildren(el("span", { class: "w-label" }, "word bank (click to add at the end)"), el("div", { class: "wordrow" }, ...Object.keys(E).map((w) => el("button", { type: "button", class: "chip click", onclick: () => { if (words.length < 14) { words.push(w); draw(); } } }, w))));
    const rs = headSel === "both" ? [run(1), run(2)] : [run(headSel)];
    const out = rs[0].out.map((v, c) => rs.reduce((s, r) => s + r.out[c], 0)), x = E[words[q]], after = x.map((v, c) => v + out[c]);
    const near = (v) => Object.keys(E).filter((w) => w !== words[q]).map((w) => [w, TOY.cosine(v, E[w])]).sort((a, c) => c[1] - a[1]).slice(0, 3);
    view.replaceChildren(...rs.map((r, k) => el("div", {}, el("span", { class: "w-label" }, `${headSel === "both" ? "head " + (k + 1) : "head " + headSel}: where "${words[q]}" looks`), wordRow(words, { weights: r.w, max: 1, sel: q }))),
      el("div", { class: "w-row" }, vecBars(x, { title: `"${words[q]}" before` }), vecBars(after, { title: `"${words[q]}" after attention`, max: 2 })),
      mathTable(["", "closest words"], [["before", near(x).map(([w, c]) => `${w} ${f2(c)}`).join(", ")], ["after", near(after).map(([w, c]) => `${w} ${f2(c)}`).join(", ")]]),
      el("details", { class: "reveal" }, el("summary", {}, "Show every number for this word"), mathTable(["word", "key", "score", "scaled", "weight"], words.map((w, j) => [w, `[${rs[0].K[j].map(f1).join(", ")}]`, f2(rs[0].raw[j]), f2(rs[0].masked[j]), Math.round(rs[0].w[j] * 100) + "%"]))));
    editor.replaceChildren(el("div", { class: "controls" }, select("edit the meaning of", Object.keys(E), editW, (v) => { editW = v; draw(); }), button("Reset all meanings", () => { E = JSON.parse(JSON.stringify({ ...TOY.EMB, ...SANDBOX_EXTRA })); draw(); })),
      el("div", { class: "controls" }, ...TOY.FEATURES.map((f, i) => slider(f, { min: 0, max: 1, step: 0.05, value: E[editW][i], fmt: (v) => v.toFixed(2) }, (v) => { E[editW][i] = v; draw(); }))));
  }
  body.append(el("div", { class: "controls" },
    select("example", ["the puppy slept on the sofa because it was tired", "the cat sat on the mat because it was tired", "the fish was under the box because it was hungry", "she ate the fish because she was hungry", "the dog ran with the cat and it was happy"], words.join(" "), (v) => { words = v.split(" "); q = words.indexOf("it") >= 0 ? words.indexOf("it") : words.length - 1; draw(); }),
    seg([["1", "head 1: who is living?"], ["2", "head 2: what action?"], ["both", "both"]], headSel, (v) => { headSel = v; draw(); }, "head"),
    seg([[true, "÷√d on"], [false, "off"]], scale, (v) => { scale = v; draw(); }, "scale"),
    seg([[false, "no mask"], [true, "causal"]], mask, (v) => { mask = v; draw(); }, "mask"),
    seg([[false, "no positions"], [true, "+ positions"]], pos, (v) => { pos = v; draw(); }, "positions")),
    sentBox, view, el("h4", {}, "Change what words mean"), el("p", { class: "w-caption" }, "Try: make 'sofa' a little 'living' and watch 'it' get confused. Or make 'tired' an object."), editor, bank);
  draw();
});
Lessons.byId.l6.steps.splice(Lessons.byId.l6.steps.length - 1, 0, { kind: "try", title: "Your own sentence",
  html: `<p>Everything so far, on any sentence you like. Pick a word to ask from, switch heads, turn the mask, scaling and positions on and off, and even change what words mean.</p>`,
  render(a) { mount(a, "sandbox"); }, recap: `Attention is just arithmetic on meanings: change the meanings or the questions and the attention changes with them.` });
