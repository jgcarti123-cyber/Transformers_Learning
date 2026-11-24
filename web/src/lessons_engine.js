/* ============================================================================
   lessons_engine.js: the guided-lesson player and the beginner building blocks
   (word rows, feature bars, worked-math tables, an animation player, predict boxes).
   ========================================================================== */
const f2 = (x) => (x === -Infinity ? "−∞" : (Math.abs(x) < 0.005 ? 0 : x).toFixed(2).replace("-", "−"));
const f1 = (x) => (x === -Infinity ? "−∞" : (Math.abs(x) < 0.05 ? 0 : x).toFixed(1).replace("-", "−"));
const FEAT_COLORS = ["--c-emb", "--c-ffn", "--c-attn", "--c-lin"];
const KIND = { learn: ["Learn", "--c-ffn"], try: ["Try it", "--c-soft"], predict: ["Predict", "--c-attn"], watch: ["Watch", "--c-lin"], math: ["The math", "--c-norm"], code: ["Code it", "--c-emb"], check: ["Check", "--c-attn"] };

/* ---------- a row of words, optionally shaded by weights ---------- */
function wordRow(words, { weights = null, sel = null, onClick = null, sub = null, dim = null, max = null } = {}) {
  const mx = max ?? (weights ? Math.max(...weights.filter(Number.isFinite), 1e-9) : 1);
  return el("div", { class: "wordrow" }, ...words.map((w, i) => {
    const wt = weights ? weights[i] : null;
    const col = wt !== null && Number.isFinite(wt) ? seqColor(clamp(wt / mx, 0, 1)) : null;
    const box = el("button", { type: "button", class: "wbox" + (sel === i ? " sel" : "") + (dim && dim(i) ? " dim" : "") + (onClick ? "" : " static"),
      style: col ? { background: rgbStr(col), color: lum(col) > 0.55 ? "#141a26" : "#fff" } : {}, onclick: onClick ? () => onClick(i) : null, tabindex: onClick ? 0 : -1 },
      el("span", { class: "wt" }, w), sub ? el("small", {}, sub(i)) : wt !== null ? el("small", {}, Number.isFinite(wt) ? Math.round(wt * 100) + "%" : "–") : null);
    return box;
  }));
}

/* ---------- a vector as labelled bars ---------- */
function vecBars(vec, { labels = TOY.FEATURES, max = 1.2, title = null, colors = FEAT_COLORS, compare = null } = {}) {
  return el("div", { class: "vec" }, title ? el("div", { class: "vec-title" }, title) : null, ...vec.map((v, i) =>
    el("div", { class: "vrow" }, el("span", { class: "vlab" }, labels[i]),
      el("span", { class: "vtrack" }, compare ? el("span", { class: "vfill ghost", style: { width: (100 * clamp(Math.abs(compare[i]) / max, 0, 1)) + "%" } }) : null,
        el("span", { class: "vfill", style: { width: (100 * clamp(Math.abs(v) / max, 0, 1)) + "%", background: `var(${v < 0 ? "--neg" : colors[i % colors.length]})` } })),
      el("span", { class: "vval" }, f2(v)))));
}

/* ---------- worked-math table ---------- */
function mathTable(head, rows, { hi = null, foot = null } = {}) {
  return el("div", { class: "table-wrap" }, el("table", { class: "t mt" },
    el("tr", {}, ...head.map((h) => el("th", {}, h))),
    ...rows.map((r, i) => el("tr", { class: hi && hi(i) ? "hi" : "" }, ...r.map((c, j) => el("td", { class: j === 0 ? "" : "num" }, c)))),
    foot ? el("tr", { class: "foot" }, ...foot.map((c, j) => el("td", { class: j === 0 ? "" : "num" }, c))) : null));
}

/* ---------- a small matrix with a caption ---------- */
function smallMat(M, cap, { rowLabels = null, colLabels = null, hiRow = null, hiCol = null, digits = 1 } = {}) {
  const g = el("table", { class: "smat" });
  if (colLabels) g.append(el("tr", {}, rowLabels ? el("th", {}) : null, ...colLabels.map((c) => el("th", {}, c))));
  M.forEach((row, i) => g.append(el("tr", { class: hiRow === i ? "hi" : "" }, rowLabels ? el("th", {}, rowLabels[i]) : null, ...row.map((v, j) => el("td", { class: hiCol === j ? "hi" : "" }, digits === 1 ? f1(v) : f2(v))))));
  return el("div", { class: "mat-wrap" }, cap ? el("span", { class: "cap" }, cap) : null, g);
}

/* ---------- animation player: frames you can play, pause and scrub ---------- */
function player(nFrames, render, { interval = 1100, caption = null, autoplay = false } = {}) {
  const stage = el("div", { class: "stage-area" }), cap = el("div", { class: "player-cap" });
  let i = 0, timer = null;
  const scrub = el("input", { type: "range", min: 0, max: nFrames - 1, value: 0, "aria-label": "animation position" });
  const playBtn = button("▶ Play", () => (timer ? stop() : play()), "primary");
  function show(k) { i = clamp(k, 0, nFrames - 1); scrub.value = i; const r = render(i); stage.replaceChildren(r.node || r); cap.replaceChildren(el("span", { class: "badge neutral" }, `${i + 1} / ${nFrames}`), " ", r.caption || (caption ? caption(i) : "")); }
  function play() { if (i >= nFrames - 1) show(0); playBtn.textContent = "⏸ Pause"; timer = setInterval(() => { if (i >= nFrames - 1) return stop(); show(i + 1); }, interval); }
  function stop() { clearInterval(timer); timer = null; playBtn.textContent = "▶ Play"; }
  scrub.addEventListener("input", () => { stop(); show(+scrub.value); });
  const box = el("div", { class: "player" }, el("div", { class: "controls" }, button("⏮", () => { stop(); show(0); }), button("◀", () => { stop(); show(i - 1); }), playBtn, button("▶|", () => { stop(); show(i + 1); }), scrub), cap, stage);
  show(0);
  if (autoplay) setTimeout(play, 400);
  box.stop = stop;
  return box;
}

/* ---------- predict-then-reveal ---------- */
function predictBox({ q, options, answer, why }, onDone) {
  const box = el("div", { class: "predict" }, el("div", { class: "pq" }, "🤔 ", q));
  const whyBox = el("div", { class: "pwhy", hidden: true });
  const btns = options.map((o, j) => el("button", { type: "button", onclick: () => {
    btns.forEach((b) => b.disabled = true);
    btns[answer].classList.add("right");
    if (j !== answer) btns[j].classList.add("wrong");
    whyBox.hidden = false;
    whyBox.replaceChildren(el("b", {}, j === answer ? "Yes! " : "Not quite. "), typeof why === "function" ? why() : el("span", { html: why }));
    onDone && onDone(j === answer);
  } }, o));
  box.append(el("div", { class: "popts" }, btns), whyBox);
  return box;
}

/* ---------- the lesson player ---------- */
const Lessons = {
  list: [], byId: {},
  add(lesson) { this.list.push(lesson); this.byId[lesson.id] = lesson; },
  render(id, stepIdx) {
    const L = this.byId[id], view = $("#view-" + id);
    const idx = clamp(stepIdx ?? store.get("step:" + id, 0), 0, L.steps.length - 1);
    store.set("step:" + id, idx);
    const st = L.steps[idx], li = this.list.indexOf(L);
    const seen = store.get("seen:" + id, {}); seen[idx] = 1; store.set("seen:" + id, seen);
    if (idx === L.steps.length - 1) Progress.mark("quiz:" + id);
    const dots = el("div", { class: "dots", role: "tablist" }, ...L.steps.map((s, k) => el("button", { type: "button", class: "dot" + (k === idx ? " on" : seen[k] ? " seen" : ""), title: `${k + 1}. ${s.title}`, "aria-label": `step ${k + 1}: ${s.title}`, onclick: () => go(k) })));
    const [kname, kcol] = KIND[st.kind] || KIND.learn;
    const area = el("div", { class: "step-area" });
    const recap = st.recap ? el("div", { class: "recap" }, el("b", {}, "What just happened? "), el("span", { html: st.recap })) : null;
    const reveal = () => { area.hidden = false; if (!area.dataset.done) { area.dataset.done = 1; st.render && st.render(area); } };
    const showRecap = () => { if (recap) recap.hidden = false; };
    const card = el("article", { class: "step-card" },
      el("div", { class: "step-kind", style: { background: `var(${kcol})` } }, kname),
      el("h2", {}, st.title),
      st.html ? el("div", { class: "step-text", html: st.html }) : null,
      st.predict ? predictBox(st.predict, () => { if (st.gate) reveal(); showRecap(); }) : null,
      area, recap);
    const go = (k) => { location.hash = id + "-" + (k + 1); };
    const next = idx < L.steps.length - 1 ? el("a", { class: "btn primary", href: `#${id}-${idx + 2}` }, "Next: " + L.steps[idx + 1].title + " →")
      : this.list[li + 1] ? el("a", { class: "btn primary", href: `#${this.list[li + 1].id}-1` }, `Lesson ${li + 2}: ${this.list[li + 1].title} →`) : el("a", { class: "btn primary", href: "#lab-inspector" }, "Explore the Lab →");
    const back = idx > 0 ? el("a", { class: "btn", href: `#${id}-${idx}` }, "← Back") : li > 0 ? el("a", { class: "btn", href: `#${this.list[li - 1].id}` }, "← Previous lesson") : el("span");
    view.replaceChildren(
      el("header", { class: "lesson-head" }, el("div", { class: "eyebrow" }, el("span", { class: "pill" }, `Lesson ${li + 1} of ${this.list.length}`), el("span", { class: "pill" }, `step ${idx + 1} of ${L.steps.length}`), L.notebook ? el("span", { class: "pill" }, L.notebook) : null),
        el("h1", {}, L.title), dots),
      card, el("nav", { class: "step-nav" }, back, next));
    if (st.predict && recap) recap.hidden = true;
    if (st.predict && st.gate) area.hidden = true; else reveal();
  },
};
document.addEventListener("keydown", (e) => {
  const m = (location.hash.slice(1) || "").match(/^(l\d+)(?:-(\d+))?$/);
  if (!m || /input|textarea|select/i.test(document.activeElement.tagName)) return;
  const L = Lessons.byId[m[1]], cur = m[2] ? +m[2] - 1 : store.get("step:" + m[1], 0);
  if (e.key === "ArrowRight" && cur < L.steps.length - 1) location.hash = `${m[1]}-${cur + 2}`;
  if (e.key === "ArrowLeft" && cur > 0) location.hash = `${m[1]}-${cur}`;
});
