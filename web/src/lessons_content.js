/* ============================================================================
   lessons_content.js: the beginner course. One sentence, followed all the way:
       "the cat sat on the mat because it was tired"
   ========================================================================== */
const S = TOY.SENT, IT = 7;
const VOC9 = ["the", "cat", "sat", "on", "mat", "because", "it", "was", "tired"];
const embOf = (w) => TOY.EMB[w];
const mount = (area, name, opts) => { const f = el("figure", { "data-widget": name, "data-opts": opts ? JSON.stringify(opts) : null }); area.append(f); Widgets.mountIn(area); };
const P = (html) => el("p", { html });
const mulStr = (a, b) => a.map((x, i) => `${f1(x)}×${f1(b[i])}`).join(" + ");

/* small 2-D scatter of words */
function scatter(words, xs, ys, { xLabel, yLabel, size = 320, hi = [] } = {}) {
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg"), pad = 40;
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`); svg.style.width = "100%"; svg.style.maxWidth = size + 40 + "px";
  const mk = (t, a, txt) => { const n = document.createElementNS(ns, t); for (const k in a) n.setAttribute(k, a[k]); if (txt !== undefined) n.textContent = txt; svg.append(n); return n; };
  const X = (v) => pad + v * (size - 2 * pad), Y = (v) => size - pad - v * (size - 2 * pad);
  mk("line", { x1: pad, y1: size - pad, x2: size - 10, y2: size - pad, stroke: Theme.v("--faint") }); mk("line", { x1: pad, y1: size - pad, x2: pad, y2: 10, stroke: Theme.v("--faint") });
  mk("text", { x: size - 10, y: size - pad + 26, "text-anchor": "end", fill: Theme.v("--muted"), "font-size": 12 }, xLabel + " →");
  mk("text", { x: pad - 8, y: 18, "text-anchor": "start", fill: Theme.v("--muted"), "font-size": 12, transform: `rotate(-90 ${pad - 26} 40)` }, yLabel + " →");
  [0, 0.5, 1].forEach((t) => { mk("text", { x: X(t), y: size - pad + 14, "text-anchor": "middle", fill: Theme.v("--faint"), "font-size": 10 }, t); mk("text", { x: pad - 6, y: Y(t) + 3, "text-anchor": "end", fill: Theme.v("--faint"), "font-size": 10 }, t); });
  words.forEach((w, i) => { const on = hi.includes(w); mk("circle", { cx: X(xs[i]), cy: Y(ys[i]), r: on ? 7 : 5, fill: Theme.v(on ? "--neg" : "--accent") }); mk("text", { x: X(xs[i]) + 8, y: Y(ys[i]) - 6 - (i % 2) * 8, fill: Theme.v("--ink"), "font-size": 13, "font-weight": on ? 700 : 400 }, w); });
  return svg;
}

/* =============================== LESSON 1 =============================== */
Lessons.add({ id: "l1", title: "Computers only see numbers", notebook: "notebooks 00–01", steps: [
  { kind: "learn", title: "One sentence, one mystery",
    html: `<p>This is the sentence we'll follow through every lesson:</p>`,
    render(a) {
      a.append(el("div", { class: "big-sentence" }, wordRow(S, { sel: IT })),
        P(`You know instantly that <b>"it"</b> means the <b>cat</b>, not the mat: a mat can't be tired. A computer knows nothing about cats or mats. All it can do is arithmetic.`),
        P(`By the end of this course you'll see, <b>number by number</b>, how a Transformer (the "T" in GPT) works out what "it" refers to. Every step is one small idea. Look for these labels:`),
        el("div", { class: "legend-kinds" }, ...Object.values(KIND).map(([n, c]) => el("span", { class: "step-kind", style: { background: `var(${c})`, position: "static" } }, n))),
        P(`Use the <b>Next</b> button, or the ← → keys.`));
    },
    recap: `We have one sentence and one question: how can a machine that only does arithmetic figure out what "it" means?` },
  { kind: "try", title: "Step 1: cut the text into tokens",
    html: `<p>A model can't read a sentence in one go. First the text is cut into small pieces called <b>tokens</b>. Click the buttons to try three ways of cutting it.</p>`,
    render(a) {
      const out = el("div");
      const ways = {
        words: [S, "Whole words: 10 tokens. Simple, and what we'll use in these lessons."],
        "word pieces": [["the", "cat", "sat", "on", "the", "mat", "be", "cause", "it", "was", "tir", "ed"], "Word pieces (used by GPT-style models): common words stay whole, rare words split into pieces it knows. Even a brand-new word can be spelled out of pieces."],
        characters: ["the cat sat on the mat because it was tired".split("").map((c) => (c === " " ? "␣" : c)), "Single characters: 43 tokens. Any text works, but sequences get long. The date translator in the Lab uses characters."],
      };
      const draw = (k) => out.replaceChildren(wordRow(ways[k][0]), P(ways[k][1]));
      a.append(seg(Object.keys(ways), "words", draw, "cut into"), out); draw("words");
    },
    recap: `Text → a list of tokens. From now on we use whole words, so our sentence is 10 tokens.` },
  { kind: "predict", title: "Step 2: give every token an ID number", gate: true,
    html: `<p>Computers store numbers, not words. So we make a <b>vocabulary</b>, a numbered list of every word we know, and replace each token with its number (its <b>ID</b>).</p>`,
    predict: { q: `"the" appears twice in our sentence (positions 0 and 4). Will both copies get the same ID?`, options: ["Yes: same word, same ID", "No: each position gets its own ID"], answer: 0,
      why: `The ID names the <i>word</i>, like an entry number in a dictionary. Where the word sits in the sentence is tracked separately, in Lesson 5.` },
    render(a) {
      a.append(mathTable(["word", "ID"], VOC9.map((w, i) => [w, String(i)])),
        P("Our sentence becomes:"), wordRow(S, { sub: (i) => "ID " + VOC9.indexOf(S[i]) }),
        P(`<code>[${S.map((w) => VOC9.indexOf(w)).join(", ")}]</code>: this list of integers is all the model receives.`));
    },
    recap: `Each token becomes an integer ID from the vocabulary. Same word, same ID.` },
  { kind: "predict", title: "Can the ID be the meaning?", gate: true,
    html: `<p>In our vocabulary, cat = 1, mat = 4 and tired = 8.</p>`,
    predict: { q: `Going by the ID numbers, is "cat" (1) closer to "mat" (4) or to "tired" (8)?`, options: ["mat, since 1 is closer to 4", "tired", "Neither: the IDs carry no meaning"], answer: 2,
      why: `IDs are just labels in whatever order we happened to list the words. List them alphabetically and cat = 1, mat = 4, tired = 7: completely different "distances". Doing maths on IDs gives nonsense.` },
    render(a) {
      a.append(mathTable(["calculation", "result", "as a word"], [["cat + mat = 1 + 4", "5", "because 🙃"], ["tired − cat = 8 − 1", "7", "was 🙃"], ["it ÷ sat = 6 ÷ 2", "3", "on 🙃"]]),
        P(`So we need a way to turn each word into numbers that <b>do</b> carry meaning. First, the simplest attempt.`));
    },
    recap: `An ID is a name, not a meaning. Arithmetic on IDs is meaningless.` },
  { kind: "try", title: "One-hot: one switch per word",
    html: `<p>A first idea: give each word a row of switches, one per vocabulary word, all off except <b>its own</b>. This list of 0s and a single 1 is called a <b>one-hot vector</b>. Click a word.</p>`,
    render(a) {
      const out = el("div");
      const draw = (k) => out.replaceChildren(
        wordRow(VOC9, { sel: k, onClick: draw }),
        el("div", { class: "onehot" }, ...VOC9.map((w, j) => el("div", { class: "sw" + (j === k ? " on" : "") }, el("span", {}, j === k ? "1" : "0"), el("small", {}, w)))),
        P(`one-hot("${VOC9[k]}") = <code>[${VOC9.map((_, j) => (j === k ? 1 : 0)).join(", ")}]</code>: ${VOC9.length} numbers, because the vocabulary has ${VOC9.length} words.`));
      a.append(out); draw(1);
    },
    recap: `A one-hot vector has one number per vocabulary word: a single 1 at the word's ID, 0 everywhere else.` },
  { kind: "math", title: "Comparing words: the dot product",
    html: `<p>To compare two lists of numbers we use the <b>dot product</b>: multiply them position by position, then add everything up. A big result means the lists "agree". We'll use it constantly, because it's the heart of attention.</p>`,
    render(a) {
      let A = "cat", B = "tired";
      const out = el("div");
      const oh = (w) => VOC9.map((v) => (v === w ? 1 : 0));
      const draw = () => {
        const a1 = oh(A), b1 = oh(B), prods = a1.map((x, i) => x * b1[i]);
        out.replaceChildren(mathTable(["position", `one-hot("${A}")`, `one-hot("${B}")`, "multiplied"], VOC9.map((w, i) => [w, String(a1[i]), String(b1[i]), String(prods[i])]), { hi: (i) => prods[i] > 0, foot: ["sum = dot product", "", "", String(prods.reduce((s, x) => s + x, 0))] }),
          P(A === B ? `A word compared with itself gives 1.` : `Two different words always give <b>0</b>: their single 1s are in different places.`));
      };
      a.append(el("div", { class: "controls" }, select("word A", VOC9, A, (v) => { A = v; draw(); }), select("word B", VOC9, B, (v) => { B = v; draw(); })), out); draw();
    },
    recap: `Dot product = multiply position by position, then add. For one-hot vectors it's 1 for the same word and 0 for any two different words.` },
  { kind: "check", title: "What's wrong with one-hot?",
    predict: { q: `With one-hot vectors, how similar is "cat" to "dog", compared with "cat" to "mat"?`, options: ["cat–dog is more similar", "Exactly equal: both give 0", "cat–mat is more similar"], answer: 1,
      why: `Every pair of different words gives 0, so one-hot can't express that cats and dogs are alike. It's also huge: with 50,000 words, every word is a list of 50,000 numbers, almost all 0. Next lesson: short lists of numbers that <b>do</b> capture meaning.` },
    recap: `Text → tokens → IDs → one-hot vectors. None of these carries meaning yet. That's what embeddings fix.` },
] });

/* =============================== LESSON 2 =============================== */
const WORDS11 = Object.keys(TOY.EMB);
Lessons.add({ id: "l2", title: "Embeddings: meaning as numbers", notebook: "notebook 01", steps: [
  { kind: "learn", title: "Describe a word with a few numbers",
    html: `<p>Imagine describing every word by answering four questions, each with a number from 0 (no) to 1 (yes):</p><ol><li>Is it a <b>living thing</b>?</li><li>Is it an <b>object or place</b>?</li><li>Is it an <b>action or feeling</b>?</li><li>Does it <b>point back</b> to another word (like "it", "she", "this")?</li></ol><p>Click a word to see its answers.</p>`,
    render(a) {
      const out = el("div");
      const draw = (k) => { const w = WORDS11[k]; out.replaceChildren(wordRow(WORDS11, { sel: k, onClick: draw }), el("div", { class: "split" }, vecBars(embOf(w), { title: `"${w}" = [${embOf(w).map(f1).join(", ")}]` }),
        P({ cat: "Cat: very much alive (1.0), a tiny bit object-like, not an action, doesn't point anywhere.", it: "'it' is a pointer word (1.0), and only 0.3 living / 0.3 object: by itself it could be either. Remember this!", tired: "'tired' is a feeling (0.8) that only living things have, so it's a bit 'living' too (0.6).", mat: "Mat: an object (1.0), nothing else.", sat: "'sat' is an action." }[w] || "Grammar words like 'the', 'on', 'because' carry little meaning of their own, so their numbers are small.")));
      };
      a.append(out); draw(1);
    },
    recap: `A word's <b>embedding</b> is a short list of numbers (a <b>vector</b>) where each number describes something about it. Here: 4 numbers per word.` },
  { kind: "try", title: "The embedding table",
    html: `<p>Put every word's numbers in one table: one row per word. This table <i>is</i> the model's embedding layer. Hover a row.</p>`,
    render(a) {
      a.append(heatmap(WORDS11.map((w) => embOf(w)), { rowLabels: WORDS11, colLabels: TOY.FEAT_SHORT, vmin: 0, vmax: 1, annotate: true, cell: 58, cellH: 30, digits: 1, tip: (i, j, v) => `${WORDS11[i]} · ${TOY.FEATURES[j]} = ${f1(v)}` }),
        el("div", { class: "note key" }, P(`<b>Honest note.</b> We wrote these numbers by hand and gave the columns names so you can read them. A real model has 64 to 12,288 numbers per word, starts them as <b>random</b>, and learns them during training (Lesson 8). Its columns have no names: the model invents its own "questions".`)));
    },
    recap: `The embedding table has one row per vocabulary word. Looking a word up means reading its row.` },
  { kind: "try", title: "Words as points on a map",
    html: `<p>Two numbers make a point on a map. Plot every word using two of its numbers. Change the axes.</p>`,
    render(a) {
      let fx = 0, fy = 1;
      const out = el("div");
      const draw = () => out.replaceChildren(scatter(WORDS11, WORDS11.map((w) => embOf(w)[fx]), WORDS11.map((w) => embOf(w)[fy]), { xLabel: TOY.FEATURES[fx], yLabel: TOY.FEATURES[fy], hi: ["it"] }));
      a.append(el("div", { class: "controls" }, select("across", TOY.FEATURES.map((f, i) => [i, f]), fx, (v) => { fx = +v; draw(); }), select("up", TOY.FEATURES.map((f, i) => [i, f]), fy, (v) => { fy = +v; draw(); })), out,
        P(`Cat and dog land next to each other; mat and rug too. <b>Similar meaning = nearby points.</b> And look where "it" (red) sits: halfway between the animals and the objects.`));
      draw();
    },
    recap: `With meaningful numbers, similar words are close together. Real embeddings do the same, just in hundreds of dimensions.` },
  { kind: "math", title: "Measuring similarity, every number",
    html: `<p>Same tool as Lesson 1, the dot product, now on embeddings. Then we divide by the two lengths to get the <b>cosine similarity</b>: 1 = pointing the same way, 0 = unrelated.</p>`,
    render(a) {
      let A = "cat", B = "dog";
      const out = el("div");
      const draw = () => {
        const x = embOf(A), y = embOf(B), pr = x.map((v, i) => v * y[i]), d = pr.reduce((s, v) => s + v, 0);
        const nx = Math.sqrt(TOY.dot(x, x)), ny = Math.sqrt(TOY.dot(y, y));
        out.replaceChildren(mathTable(["feature", A, B, `${A} × ${B}`], TOY.FEATURES.map((f, i) => [f, f1(x[i]), f1(y[i]), f2(pr[i])]), { foot: ["dot product (sum)", "", "", f2(d)] }),
          el("div", { class: "worked" },
            el("div", {}, `length of ${A} = √(${x.map((v) => f1(v) + "²").join(" + ")}) = `, el("b", {}, f2(nx))),
            el("div", {}, `length of ${B} = √(${y.map((v) => f1(v) + "²").join(" + ")}) = `, el("b", {}, f2(ny))),
            el("div", {}, `cosine = ${f2(d)} ÷ (${f2(nx)} × ${f2(ny)}) = `, el("b", { class: "big" }, f2(d / (nx * ny) || 0)))));
      };
      a.append(el("div", { class: "controls" }, select("word A", WORDS11, A, (v) => { A = v; draw(); }), select("word B", WORDS11, B, (v) => { B = v; draw(); })), out); draw();
    },
    recap: `cat · dog = 0.92 (similar), cat · mat = 0.10 (not). Dividing by the lengths gives the cosine similarity, which only cares about direction.` },
  { kind: "predict", title: "Which word is most like 'cat'?", gate: true,
    predict: { q: `Using the embedding table, which word has the highest cosine similarity with "cat"?`, options: ["dog", "mat", "tired", "it"], answer: 0, why: `Dog's numbers point almost exactly the same way as cat's. Here's every word, ranked:` },
    render(a) {
      const r = WORDS11.filter((w) => w !== "cat").map((w) => [w, TOY.cosine(embOf("cat"), embOf(w))]).sort((x, y) => y[1] - x[1]);
      a.append(bars(r.map(([w, c]) => ({ label: w, value: Math.max(0, c), cls: w === "dog" ? "hi" : "" })), { max: 1, digits: 2 }));
    },
    recap: `Similarity is now meaningful: dog ≈ cat, tired is partly similar (it's about living things), mat is far away.` },
  { kind: "watch", title: "A lookup is a multiplication",
    html: `<p>How does the model get "cat"'s row out of the table? Multiply cat's <b>one-hot</b> vector by the table. Press play: each row of the table gets multiplied by one of the switches.</p>`,
    render(a) {
      const words = VOC9, oh = words.map((w) => (w === "cat" ? 1 : 0));
      a.append(player(words.length + 1, (k) => {
        const rows = words.map((w, i) => (i < k ? embOf(w).map((v) => v * oh[i]) : null));
        const sum = [0, 1, 2, 3].map((c) => rows.reduce((s, r) => s + (r ? r[c] : 0), 0));
        return { node: el("div", { class: "w-row" },
          mathTable(["row", "switch", "row × switch"], words.map((w, i) => [w, String(oh[i]), rows[i] ? `[${rows[i].map(f1).join(", ")}]` : "…"]), { hi: (i) => i === k - 1 }),
          el("div", {}, vecBars(sum, { title: "running total" }))),
          caption: k === 0 ? "one-hot(cat) has a 1 only in cat's position." : k <= words.length ? `${words[k - 1]}'s row × ${oh[k - 1]} = ${oh[k - 1] ? "the row itself" : "all zeros"}` : "" };
      }, { interval: 700 }));
      a.append(P(`Every row except cat's is multiplied by 0. So <code>one_hot(cat) × table = cat's row</code>. An embedding layer is a matrix multiplication that real code does as a fast lookup.`));
    },
    recap: `Embedding lookup = one-hot vector × embedding table. That's why embeddings can be learned exactly like any other weights.` },
  { kind: "learn", title: "Where do real embeddings come from?",
    html: `<p>Nobody hand-writes real embeddings. One classic method, <b>word2vec</b>, uses one idea: <i>words that appear in similar sentences mean similar things</i>. It starts with random numbers and nudges them so that each word predicts its neighbours. Press <b>Train</b> below: 30,000 sentences, about 7 seconds, in your browser.</p>`,
    render(a) { mount(a, "w2v"); a.append(P(`Nobody told it which words are royal or young. It discovered those directions just from which words appear near which. Try the analogy man : king :: woman : ?`)); },
    recap: `Embeddings are learned from data: words used in similar contexts end up with similar vectors.` },
  { kind: "predict", title: "The problem: what does 'it' mean?", gate: true,
    html: `<p>Back to our sentence. Here are "it"'s four numbers: living 0.3, object 0.3, action 0, pointer 1.0.</p>`,
    predict: { q: `Using ONLY its own numbers, does "it" mean the cat or the mat?`, options: ["the cat", "the mat", "Can't tell: it's equally close to both"], answer: 2, why: `"it" is exactly as similar to cat as to mat:` },
    render(a) {
      a.append(mathTable(["comparison", "cosine similarity"], [["it vs cat", f2(TOY.cosine(embOf("it"), embOf("cat")))], ["it vs mat", f2(TOY.cosine(embOf("it"), embOf("mat")))]]),
        P(`The table gives "it" the same numbers in <b>every</b> sentence, so it can't know what "it" refers to <i>here</i>. The fix: let "it" <b>look at the other words in this sentence</b> and update its own numbers. That's <b>attention</b>, the next lesson.`));
    },
    recap: `An embedding table gives each word ONE fixed vector (a <b>static</b> embedding). Words like "it", or "bank" in river bank vs bank account, need context.` },
  { kind: "code", title: "Code it (optional)",
    html: `<p>Write the lookup as a one-hot multiplication, and the cosine similarity, in real Python with NumPy. It runs in your browser and checks itself.</p>`,
    render(a) { mount(a, "py-exercise", { id: "embedding" }); },
    recap: `You've written an embedding lookup and a similarity measure, the two operations this lesson was about.` },
] });

/* =============================== LESSON 3 =============================== */
const H1 = TOY.HEAD1, att = () => TOY.attend(S, IT, H1);
Lessons.add({ id: "l3", title: "Attention: 'it' looks at the other words", notebook: "notebook 02", steps: [
  { kind: "learn", title: "The idea: a question, name tags and messages",
    html: `<p>Picture the 10 words at a party. Every word carries three cards:</p>`,
    render(a) {
      a.append(el("div", { class: "cards3" },
        el("div", { class: "card3", style: { borderColor: "var(--c-attn)" } }, el("b", {}, "❓ Query: my question"), el("span", {}, `What am I looking for? "it" asks: "which word here is a living thing?"`)),
        el("div", { class: "card3", style: { borderColor: "var(--c-ffn)" } }, el("b", {}, "🏷️ Key: my name tag"), el("span", {}, `How others can find me. "cat"'s tag says: "I'm a living thing!"`)),
        el("div", { class: "card3", style: { borderColor: "var(--c-soft)" } }, el("b", {}, "💬 Value: my message"), el("span", {}, `What I pass on if someone listens to me. "cat" passes on its meaning.`))),
        P(`"it" compares its <b>question</b> with every word's <b>name tag</b>, listens mostly to the best matches, and mixes their <b>messages</b> into its own meaning. That's all attention is. The next steps do it with real numbers.`));
    },
    recap: `Attention = compare my question (query) with everyone's name tag (key), then take a weighted mix of their messages (values).` },
  { kind: "math", title: "'it' writes its question (the query)",
    html: `<p>The query is made by multiplying the word's 4 numbers by a small table of weights called <b>W<sub>Q</sub></b>. Think of W<sub>Q</sub> as a recipe that turns "what I am" into "what I'm looking for".</p>`,
    render(a) {
      const x = embOf("it"), q = TOY.vecMat(x, H1.WQ);
      a.append(el("div", { class: "mat-row", style: { flexWrap: "wrap", gap: "1rem" } }, vecBars(x, { title: `"it" = [${x.map(f1).join(", ")}]` }), el("span", { class: "op" }, "×"), smallMat(H1.WQ, "W_Q (4 × 2)", { rowLabels: TOY.FEAT_SHORT, colLabels: H1.qLabels }), el("span", { class: "op" }, "="), smallMat([q], "query (2 numbers)", { colLabels: H1.qLabels })),
        el("div", { class: "worked" }, ...H1.qLabels.map((lab, c) => el("div", {}, `${lab}: ${mulStr(x, H1.WQ.map((r) => r[c]))} = `, el("b", {}, f1(q[c]))))),
        P(`Only the "points back" row of W<sub>Q</sub> has a non-zero number, so the recipe says: <i>"if I'm a pointer word, I want a living thing"</i>. "it" ends up asking <b>[2, 0]</b>: strongly "wants living", not "wants object". (In a real model nobody writes W<sub>Q</sub>; it's learned. You'll watch that happen in Lesson 8.)`));
    },
    recap: `query = word's numbers × W<sub>Q</sub>. For "it": [0.3, 0.3, 0, 1] × W<sub>Q</sub> = [2, 0], meaning "I'm looking for a living thing".` },
  { kind: "math", title: "Every word shows a name tag (the key)",
    html: `<p>Every word makes its key the same way, with a different recipe, <b>W<sub>K</sub></b>: key = word × W<sub>K</sub>. Click a word to see its multiplication.</p>`,
    render(a) {
      const out = el("div");
      const draw = (k) => {
        const x = embOf(S[k]), key = TOY.vecMat(x, H1.WK);
        out.replaceChildren(wordRow(S, { sel: k, onClick: draw }),
          el("div", { class: "mat-row", style: { flexWrap: "wrap", gap: "1rem", marginTop: ".6rem" } }, smallMat([x], `"${S[k]}"`, { colLabels: TOY.FEAT_SHORT }), el("span", { class: "op" }, "×"), smallMat(H1.WK, "W_K (4 × 2)", { rowLabels: TOY.FEAT_SHORT, colLabels: H1.kLabels }), el("span", { class: "op" }, "="), smallMat([key], "key", { colLabels: H1.kLabels })),
          el("div", { class: "worked" }, ...H1.kLabels.map((lab, c) => el("div", {}, `${lab}: ${mulStr(x, H1.WK.map((r) => r[c]))} = `, el("b", {}, f1(key[c]))))),
          mathTable(["word", "key: is living", "key: is object"], S.map((w, i) => { const kk = TOY.vecMat(embOf(w), H1.WK); return [w, f1(kk[0]), f1(kk[1])]; }), { hi: (i) => i === k }));
      };
      a.append(out); draw(1);
    },
    recap: `key = word × W<sub>K</sub>. cat's tag is [2.0, 0.2] ("very living"), mat's is [0, 2.0] ("very object").` },
  { kind: "predict", title: "Which name tag matches best?", gate: true,
    predict: { q: `"it" asks [2, 0] ("wants living"). Which word's key will match it best?`, options: ["cat", "mat", "tired", "it (itself)"], answer: 0, why: `cat's key [2.0, 0.2] points the same way as the question. Next step: the exact numbers.` },
    render(a) { a.append(wordRow(S, { weights: att().raw.map((v) => v / 4), sub: (i) => "match " + f1(att().raw[i]) })); },
    recap: `The best match is the word whose name tag points the same way as the question.` },
  { kind: "math", title: "Score = question · name tag",
    html: `<p>"How well does the tag match the question?" is a <b>dot product</b> again: query · key.</p>`,
    render(a) {
      const r = att();
      a.append(mathTable(["word", "key", "query · key", "score"], S.map((w, i) => [w, `[${r.K[i].map(f1).join(", ")}]`, `2×${f1(r.K[i][0])} + 0×${f1(r.K[i][1])}`, f2(r.raw[i])]), { hi: (i) => i === 1 }),
        P(`cat scores <b>4.0</b>, tired 2.4, it 1.2, and objects 0. These raw <b>scores</b> say how relevant each word is to "it".`));
    },
    recap: `score = query · key, one number per word. Bigger = more relevant.` },
  { kind: "math", title: "Scale: divide by √2",
    html: `<p>Before going further, every score is divided by <b>√d<sub>k</sub></b>, where d<sub>k</sub> is how many numbers a query has (here 2, so √2 ≈ 1.41). Big vectors give big dot products, and Lesson 4 shows why too-big scores break learning. This keeps them moderate.</p>`,
    render(a) {
      const r = att();
      a.append(mathTable(["word", "score", "÷ √2", "scaled score"], S.map((w, i) => [w, f2(r.raw[i]), `${f2(r.raw[i])} ÷ 1.41`, f2(r.scaled[i])]), { hi: (i) => i === 1 }));
    },
    recap: `scaled score = score ÷ √d<sub>k</sub>. cat: 4.0 ÷ 1.41 = 2.83.` },
  { kind: "math", title: "Softmax: scores become percentages",
    html: `<p>We want "how much should 'it' listen to each word?" as percentages that add up to 100%. <b>Softmax</b> does it in two moves: raise e (≈ 2.718) to the power of each score, which makes every number positive and stretches the gaps, then divide by the total.</p>`,
    render(a) {
      const r = att(), tot = r.sumRaw;
      const out = el("div");
      a.append(mathTable(["word", "scaled score", "e^score", "÷ total", "attention"], S.map((w, i) => [w, f2(r.scaled[i]), f2(r.expsRaw[i]), `${f2(r.expsRaw[i])} ÷ ${f2(tot)}`, Math.round(r.w[i] * 100) + "%"]), { hi: (i) => i === 1, foot: ["total", "", f2(tot), "", "100%"] }),
        P(`Example for cat: e<sup>2.83</sup> = 16.9, and 16.9 ÷ ${f2(tot)} = <b>${Math.round(r.w[1] * 100)}%</b>. "it" spends about half its attention on "cat".`),
        el("h4", {}, "Try it: sharpness"), P(`Multiply the scores by a number before the softmax. Watch the percentages.`), out);
      const draw = (m) => { const w = TOY.softmax(r.scaled.map((s) => s * m)); out.replaceChildren(wordRow(S, { weights: w, max: 1 }), P(`×${m}: cat gets ${Math.round(w[1] * 100)}%. ${m > 2 ? "Bigger scores → winner-takes-all." : m < 1 ? "Smaller scores → attention spreads evenly." : ""}`)); };
      out.before(slider("multiply scores by", { min: 0.1, max: 5, step: 0.1, value: 1, fmt: (v) => "×" + v.toFixed(1) }, draw)); draw(1);
    },
    recap: `attention weights = softmax(scaled scores): positive, adding up to 100%. "it": cat 52%, tired 17%, the rest small.` },
  { kind: "watch", title: "Mixing the messages (values)",
    html: `<p>Each word's <b>value</b> (its message) here is simply its own 4 numbers. "it" takes 52% of cat's message, 17% of tired's, and so on, and adds them all up. Press play.</p>`,
    render(a) {
      const r = att();
      const order = S.map((_, i) => i).sort((x, y) => r.w[y] - r.w[x]);
      a.append(player(order.length + 1, (k) => {
        const used = order.slice(0, k), tot = [0, 1, 2, 3].map((c) => used.reduce((s, j) => s + r.w[j] * r.V[j][c], 0));
        const j = order[k - 1];
        return { node: el("div", { class: "w-row" }, el("div", { class: "w-col" }, wordRow(S, { weights: r.w, dim: (i) => !used.includes(i), max: 0.6 }),
          k ? el("div", { class: "worked" }, `${Math.round(r.w[j] * 100)}% × "${S[j]}" [${r.V[j].map(f1).join(", ")}] = [${r.V[j].map((v) => f2(v * r.w[j])).join(", ")}]`) : null),
          vecBars(tot, { title: "running total (the attention output)" })),
          caption: k === 0 ? "Start from zero." : `Add ${Math.round(r.w[j] * 100)}% of "${S[j]}".` };
      }, { interval: 900 }));
    },
    recap: `attention output = Σ weight × value. For "it": [0.65, 0.12, 0.19, 0.07], mostly "living", with a bit of "feeling" from "tired".` },
  { kind: "try", title: "'it' after attention",
    html: `<p>The output is <b>added</b> to "it"'s original numbers (keeping the original is called a <i>residual connection</i>). Compare before and after, then change the sentence.</p>`,
    render(a) {
      let animal = "cat", place = "mat";
      const out = el("div");
      const draw = () => {
        const words = S.map((w) => (w === "cat" ? animal : w === "mat" ? place : w));
        const r = TOY.attend(words, IT, H1), x = embOf("it"), after = x.map((v, i) => v + r.out[i]);
        out.replaceChildren(wordRow(words, { weights: r.w, max: 0.6 }),
          el("div", { class: "w-row" }, vecBars(x, { title: `"it" before: [${x.map(f2).join(", ")}]` }), vecBars(after, { title: `"it" after: [${after.map(f2).join(", ")}]`, max: 1.6 })),
          mathTable(["", `similarity to "${animal}"`, `similarity to "${place}"`], [["before attention", f2(TOY.cosine(x, embOf(animal))), f2(TOY.cosine(x, embOf(place)))], ["after attention", f2(TOY.cosine(after, embOf(animal))), f2(TOY.cosine(after, embOf(place)))]]),
          P(`"it" now leans clearly towards <b>${animal}</b>. Its new vector depends on the sentence, so it's a <b>contextual embedding</b>.`));
      };
      a.append(el("div", { class: "controls" }, seg(["cat", "dog"], animal, (v) => { animal = v; draw(); }, "animal"), seg(["mat", "rug"], place, (v) => { place = v; draw(); }, "place")), out); draw();
    },
    recap: `Attention changed "it" from 'could be anything' (0.30 similar to both) to 'clearly the cat' (0.66 vs 0.38). That's how context gets into a word.` },
  { kind: "check", title: "Quick check",
    predict: { q: `Why does "it" give 17% of its attention to "tired"?`, options: [`"tired"'s name tag says "living" a little (0.6 × 2 = 1.2)`, `Because "tired" is the last word`, "It's random noise"], answer: 0,
      why: `Feelings like "tired" belong to living things, so its embedding has some "living" in it, its key matches the question somewhat, and it gets some attention. Attention follows the numbers, nothing else, so position plays no part (yet).` },
    recap: `You've now done attention by hand: query, keys, scores, ÷√d<sub>k</sub>, softmax, weighted sum of values. The paper writes it as softmax(QKᵀ/√d<sub>k</sub>)V. Next: all words at once.` },
  { kind: "code", title: "Code it (optional)",
    html: `<p>Softmax is the one non-trivial function so far. Write it in NumPy, including the trick that stops huge numbers from overflowing.</p>`,
    render(a) { mount(a, "py-exercise", { id: "softmax" }); },
    recap: `You've written softmax. Next lesson you'll write the whole attention function.` },
] });

/* =============================== LESSON 4 =============================== */
const allW = (head = H1, opts) => TOY.attendAll(S, head, opts);
Lessons.add({ id: "l4", title: "Every word at once, and no peeking", notebook: "notebook 02", steps: [
  { kind: "learn", title: "Every word asks at the same time",
    html: `<p>So far only "it" asked a question. In a real model <b>every</b> word does the same thing simultaneously. Each row below is one word's attention over the sentence.</p>`,
    render(a) {
      a.append(heatmap(allW().map((r) => r.w), { rowLabels: S.map((w, i) => w + " " + i), colLabels: S, vmin: 0, vmax: 1, annotate: true, cell: 44, digits: 2, tip: (i, j, v) => `"${S[i]}" gives ${Math.round(v * 100)}% of its attention to "${S[j]}"` }),
        P(`Row 7 ("it") is the one you computed by hand. The other rows are flat (10% each) because in our hand-made head only pointer words ask a real question. Their query is [0, 0], so every score is 0. In a trained model every row does something useful; open the Lab's model inspector to see.`));
    },
    recap: `Attention runs for every word in parallel, giving a 10 × 10 grid of weights. Row = who's looking, column = who's looked at.` },
  { kind: "math", title: "All the scores in one multiplication",
    html: `<p>Stack all 10 queries into a table <b>Q</b> (10 × 2) and all keys into <b>K</b> (10 × 2). Then <b>Q × Kᵀ</b> (K flipped on its side) computes all 100 dot products at once. Hover any cell to see its sum.</p>`,
    render(a) {
      const R = allW(), Q = R.map((r) => r.q), K = R[0].K;
      a.append(el("div", { class: "w-row" }, smallMat(Q, "Q (10 × 2)", { rowLabels: S, colLabels: ["q₁", "q₂"] }), smallMat(K, "K (10 × 2)", { rowLabels: S, colLabels: ["k₁", "k₂"] }),
        el("div", { class: "w-col" }, el("span", { class: "w-label" }, "Q × Kᵀ = all scores (10 × 10)"), heatmap(R.map((r) => r.raw), { rowLabels: S, colLabels: S, annotate: true, cell: 36, digits: 1, tip: (i, j) => `${S[i]} · ${S[j]} = ${Q[i][0]}×${f1(K[j][0])} + ${Q[i][1]}×${f1(K[j][1])} = ${f2(R[i].raw[j])}` }))),
        P(`Matrix multiplication is just "many dot products in a grid". Graphics chips (GPUs) are built to do exactly this very fast, which is a big reason Transformers took off.`));
    },
    recap: `scores = Q × Kᵀ: (10 × 2) × (2 × 10) = 10 × 10. One multiplication gives every word's score for every word.` },
  { kind: "predict", title: "A cheating problem", gate: true,
    html: `<p>Models like GPT are trained to <b>guess the next word</b>. At "was" the right guess is "tired".</p>`,
    predict: { q: `During training, the whole sentence is fed in at once. Should "was" be allowed to look at "tired" while guessing the next word?`, options: ["Yes, more information is always better", "No: that's looking at the answer"], answer: 1,
      why: `If "was" could see "tired", it would just copy it. It would get perfect training scores and learn nothing useful, then fail when generating new text, where the future doesn't exist yet.` },
    render(a) { a.append(wordRow(S, { sel: 8, dim: (i) => i > 8, sub: (i) => (i > 8 ? "future!" : i === 8 ? "guessing →" : "past") })); },
    recap: `When predicting the next word, each word may only look at itself and the words before it.` },
  { kind: "try", title: "The causal mask",
    html: `<p>The fix is a <b>mask</b>: every score for a <i>future</i> word is replaced by <b>−∞</b> before the softmax. Toggle it.</p>`,
    render(a) {
      const out = el("div");
      const draw = (on) => {
        const R = S.map((_, i) => TOY.attend(S, i, H1, { mask: on ? S.map((_, j) => j <= i) : null }));
        out.replaceChildren(heatmap(R.map((r) => r.w), { rowLabels: S, colLabels: S, vmin: 0, vmax: 1, annotate: true, cell: 44, digits: 2 }),
          P(on ? `The top-right triangle is now 0: nobody sees the future. "it" (row 7) can't see "was" or "tired" any more, so its attention to "cat" rises to <b>${Math.round(R[7].w[1] * 100)}%</b>. Row 0 ("the") can only see itself: 100%.` : `No mask: every word sees every word. That's fine for reading a whole sentence (like the encoder of a translator), but not for guessing the next word.`));
      };
      a.append(seg([[false, "no mask"], [true, "causal mask"]], false, draw, "mask"), out); draw(false);
    },
    recap: `causal mask = set future scores to −∞, so their weight becomes exactly 0. The grid becomes a lower triangle.` },
  { kind: "math", title: "Why −∞ and not 0?",
    html: `<p>Look at row 7 ("it") with the mask on. Compare masking by setting the score to 0 with setting it to −∞.</p>`,
    render(a) {
      const r = TOY.attend(S, IT, H1), sc = r.scaled;
      const zero = sc.map((s, j) => (j > IT ? 0 : s)), ez = zero.map(Math.exp), tz = ez.reduce((x, y) => x + y, 0);
      const inf = TOY.attend(S, IT, H1, { mask: S.map((_, j) => j <= IT) });
      a.append(mathTable(["word", "score → 0", "e^0 …", "weight", "score → −∞", "e^−∞ …", "weight"], S.map((w, j) => [w, f2(zero[j]), f2(ez[j]), Math.round((ez[j] / tz) * 100) + "%", f2(inf.masked[j]), f2(inf.expsRaw[j]), Math.round(inf.w[j] * 100) + "%"]), { hi: (j) => j > IT }),
        P(`e<sup>0</sup> = 1, so a score of 0 still gets weight: "was" and "tired" would still be seen. e<sup>−∞</sup> = 0 exactly, so they vanish and the rest still adds up to 100%.`));
    },
    recap: `Masking must happen on the scores, with −∞, before the softmax.` },
  { kind: "try", title: "Why scores must not get too big",
    html: `<p>Real models use 64 or 128 numbers per query, not 2. Adding up more products makes dot products bigger. Typically they grow like √d<sub>k</sub>. Slide d<sub>k</sub> to see what that does to "it"'s attention if we <i>don't</i> divide.</p>`,
    render(a) {
      const r = att(), out = el("div");
      const draw = (dk) => {
        const grow = Math.sqrt(dk / 2), unscaled = r.scaled.map((s) => s * grow), w = TOY.softmax(unscaled);
        out.replaceChildren(el("div", { class: "w-row" }, el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, `without ÷√d_k (scores ×${f1(grow)})`), wordRow(S, { weights: w, max: 1 })),
          el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, "with ÷√d_k"), wordRow(S, { weights: r.w, max: 1 }))),
          P(dk >= 64 ? `With d<sub>k</sub> = ${dk}, unscaled attention is <b>${Math.round(Math.max(...w) * 100)}%</b> on one word. Softmax is saturated: tiny changes to the scores change nothing, so the slope that learning follows is ≈ 0, and learning stalls. Dividing by √d<sub>k</sub> keeps it healthy.` : `Small d_k: both versions look similar.`));
      };
      a.append(slider("d_k", { min: 2, max: 512, step: 2, value: 2 }, draw), out); draw(2);
    },
    recap: `Bigger vectors → bigger dot products → a softmax stuck at 100% / 0% → no learning signal. ÷√d<sub>k</sub> prevents it.` },
  { kind: "learn", title: "The price: every pair of words",
    html: `<p>Every word scores every word, so <b>N words need N × N scores</b>.</p>`,
    render(a) {
      const out = el("div");
      const draw = (e) => { const n = Math.round(10 ** e); out.replaceChildren(P(`${n.toLocaleString()} words → <b>${(n * n).toLocaleString()}</b> scores per head, per layer.`)); };
      a.append(slider("number of words (log scale)", { min: 1, max: 6, step: 0.5, value: 1, fmt: (v) => Math.round(10 ** v).toLocaleString() }, draw), out); draw(1);
      a.append(P(`Double the text and the work quadruples. That's why "long context" is hard, and why tricks like FlashAttention (the same maths, computed in small tiles) matter.`));
    },
    recap: `Attention costs grow with the square of the length: O(N²).` },
  { kind: "code", title: "Code it (optional)",
    html: `<p>Now write the full attention function, for all words at once, with an optional mask. This is equation (1) of the original paper.</p>`,
    render(a) { mount(a, "py-exercise", { id: "attention" }); },
    recap: `softmax(QKᵀ/√d<sub>k</sub>)V plus a mask: you can now write the core of every Transformer from memory.` },
] });

/* =============================== LESSON 5 =============================== */
const SHUF = ["the", "mat", "sat", "on", "the", "cat", "because", "it", "was", "tired"];
const withPE = (words) => words.map((w, i) => embOf(w).map((v, c) => v + TOY.pe(i)[c]));
Lessons.add({ id: "l5", title: "Word order: telling the model where words are", notebook: "notebook 03", steps: [
  { kind: "predict", title: "Shuffle the sentence", gate: true,
    html: `<p>Swap cat and mat: <i>"the <b>mat</b> sat on the <b>cat</b> because it was tired"</i>. A different situation entirely.</p>`,
    predict: { q: `Will "it"'s attention change in any way, apart from following the words to their new places?`, options: ["Yes: cat moved, so the percentages change", "No: the same percentages, just moved with the words"], answer: 1, why: `Attention only compares numbers: queries with keys. Nothing in those numbers says <i>where</i> a word is.` },
    render(a) {
      const r1 = TOY.attend(S, IT, H1), r2 = TOY.attend(SHUF, IT, H1);
      a.append(el("span", { class: "w-label" }, "original"), wordRow(S, { weights: r1.w, max: 0.6 }), el("span", { class: "w-label" }, "shuffled"), wordRow(SHUF, { weights: r2.w, max: 0.6 }),
        mathTable(["", "output of 'it'"], [["original", `[${r1.out.map(f2).join(", ")}]`], ["shuffled", `[${r2.out.map(f2).join(", ")}]`]]), P(`Identical to the last digit.`));
    },
    recap: `Attention is blind to word order. Shuffle the words and each word's output is exactly the same.` },
  { kind: "learn", title: "A bag of fridge magnets",
    html: `<p>Without extra help, a Transformer sees a sentence like words on fridge magnets thrown in a bag: it knows <i>which</i> words are there, not their order. "dog bites man" and "man bites dog" would look identical. Older models (RNNs) read word by word, so order was built in, but that also made them slow, because they couldn't process words in parallel.</p><p>So we must <b>write the position into each word's numbers</b>.</p>`,
    recap: `We need to add position information to every word.` },
  { kind: "predict", title: "Why not just add 0, 1, 2, …?", gate: true,
    predict: { q: `Could we simply add each word's position number (0, 1, 2 … 999) to its embedding?`, options: ["Yes, that works perfectly", "Bad idea: position 999 would drown out meanings that are about 0–1"], answer: 1,
      why: `Meaning numbers are around 0 to 1. Adding 999 would bury them, and the model has never seen positions bigger than its training texts. We need position codes that stay small, are different for every position, and make "3 words apart" look the same everywhere.` },
    recap: `Position codes must stay small and unique, and they should make relative distances easy to see.` },
  { kind: "try", title: "Position clocks",
    html: `<p>The original Transformer uses <b>clocks</b>. Each position gets clock hands that turn as you move along the sentence: a fast hand that turns 1 radian (about 57°) per word, and a slow hand that turns 100 times slower. The code is each hand's sideways reach (sin) and height (cos). It stays between −1 and 1 forever, and no two positions match.</p>`,
    render(a) {
      const out = el("div");
      const clock = (ang, lab) => { const ns = "http://www.w3.org/2000/svg", s = document.createElementNS(ns, "svg"); s.setAttribute("viewBox", "0 0 90 104"); s.setAttribute("width", "110");
        const mk = (t, at, tx) => { const n = document.createElementNS(ns, t); for (const k in at) n.setAttribute(k, at[k]); if (tx) n.textContent = tx; s.append(n); };
        mk("circle", { cx: 45, cy: 45, r: 38, fill: "none", stroke: Theme.v("--line"), "stroke-width": 3 }); mk("line", { x1: 45, y1: 45, x2: 45 + 36 * Math.sin(ang), y2: 45 - 36 * Math.cos(ang), stroke: Theme.v("--accent"), "stroke-width": 4, "stroke-linecap": "round" });
        mk("text", { x: 45, y: 100, "text-anchor": "middle", fill: Theme.v("--muted"), "font-size": 11 }, lab); return s; };
      const draw = (p) => {
        const v = TOY.pe(p);
        out.replaceChildren(wordRow(S, { sel: p, sub: (i) => "pos " + i }), el("div", { class: "w-row", style: { alignItems: "center" } }, clock(p, "fast hand"), clock(p / 100, "slow hand"),
          mathTable(["number", "formula", "value"], [["1: fast sin", `sin(${p})`, f2(v[0])], ["2: fast cos", `cos(${p})`, f2(v[1])], ["3: slow sin", `sin(${p} ÷ 100)`, f2(v[2])], ["4: slow cos", `cos(${p} ÷ 100)`, f2(v[3])]])),
          P(`Position ${p}'s code: <code>[${v.map(f2).join(", ")}]</code>. Real models use hundreds of hands at many speeds; the paper's formula picks the speeds 1, 1/10000<sup>2/d</sup>, 1/10000<sup>4/d</sup>, …`));
      };
      a.append(slider("position", { min: 0, max: 9, value: 1 }, draw), out); draw(1);
    },
    recap: `positional encoding = clock-hand readings (sin and cos) at several speeds. Small, unique, and smooth.` },
  { kind: "math", title: "Meaning + position",
    html: `<p>Each word's input is simply its meaning <b>plus</b> its position code, number by number. Click a word.</p>`,
    render(a) {
      const out = el("div");
      const draw = (k) => { const e = embOf(S[k]), p = TOY.pe(k), s = e.map((v, i) => v + p[i]);
        out.replaceChildren(wordRow(S, { sel: k, onClick: draw, sub: (i) => "pos " + i }), mathTable(["number", "meaning", "+ position", "= input"], TOY.FEAT_SHORT.map((f, i) => [f, f2(e[i]), f2(p[i]), f2(s[i])]))); };
      a.append(out); draw(1);
      a.append(P(`Now the two "the"s (positions 0 and 4) get <b>different</b> inputs, even though they're the same word.`));
    },
    recap: `input vector = embedding + positional encoding. Same word at different positions → different numbers.` },
  { kind: "try", title: "Shuffle again, now with positions",
    html: `<p>Same two sentences, but every word now carries its position.</p>`,
    render(a) {
      const r1 = TOY.attend(S, IT, H1, { embs: withPE(S) }), r2 = TOY.attend(SHUF, IT, H1, { embs: withPE(SHUF) });
      a.append(el("span", { class: "w-label" }, "original + positions"), wordRow(S, { weights: r1.w, max: 0.6 }), el("span", { class: "w-label" }, "shuffled + positions"), wordRow(SHUF, { weights: r2.w, max: 0.6 }),
        mathTable(["", "output of 'it'"], [["original", `[${r1.out.map(f2).join(", ")}]`], ["shuffled", `[${r2.out.map(f2).join(", ")}]`]]),
        P(`Now the outputs differ: order matters. Our hand-made head wasn't designed to use positions, so the change is small and arbitrary. A trained model learns to use them on purpose, for example a head that looks at "the previous word".`));
    },
    recap: `With positions added, the same words in a different order produce different results.` },
  { kind: "learn", title: "Other ways to encode position",
    render(a) {
      a.append(mathTable(["method", "used by", "idea in one line"], [["Sinusoidal (clocks)", "original Transformer (2017)", "the clock code above, added to the embeddings"], ["Learned positions", "BERT, GPT-2", "a learned vector per position, like a second embedding table"], ["RoPE (rotary)", "Llama, Mistral, most modern models", "rotate each query and key by its position, so scores depend on the distance between words"]]),
        P(`The same idea everywhere: give attention a way to know <i>where</i> words are, not just <i>what</i> they are.`));
    },
    recap: `Sinusoidal, learned and rotary encodings all inject word order. RoPE is today's default.` },
  { kind: "check", title: "Quick check",
    predict: { q: `Remove positional encoding from a trained translator. "24/11/2026" and "2026/11/24" contain exactly the same characters. What happens?`, options: ["It still translates both correctly", "It can't tell them apart, so at least one comes out wrong"], answer: 1,
      why: `Without positions the encoder sees the same bag of characters. In the Lab you can try this with a real model: without positions, accuracy falls from 100% to about 15%.` },
    recap: `Positions are not optional. Lesson 6: why one attention isn't enough.` },
] });

/* =============================== LESSON 6 =============================== */
const H2 = TOY.HEAD2;
Lessons.add({ id: "l6", title: "Many heads, many questions", notebook: "notebook 03", steps: [
  { kind: "learn", title: "One head = one question",
    html: `<p>To understand "it", a reader wants two things at once: <b>who</b> is it? (the cat) and <b>what about it</b>? (it was tired). One attention can only take one weighted mix, which answers one question. So Transformers run several attentions side by side, called <b>heads</b>, each with its own W<sub>Q</sub> and W<sub>K</sub>, so each asks its own question.</p>`,
    render(a) { a.append(el("div", { class: "cards3" }, el("div", { class: "card3", style: { borderColor: "var(--c-attn)" } }, el("b", {}, "Head 1"), el("span", {}, `"${H1.name}"`)), el("div", { class: "card3", style: { borderColor: "var(--c-ffn)" } }, el("b", {}, "Head 2"), el("span", {}, `"${H2.name}"`)))); },
    recap: `Multi-head attention = several attentions in parallel, each free to ask a different question.` },
  { kind: "try", title: "Two heads, side by side",
    html: `<p>Click any word to see where each head makes it look.</p>`,
    render(a) {
      const out = el("div");
      const draw = (k) => { const r1 = TOY.attend(S, k, H1), r2 = TOY.attend(S, k, H2);
        out.replaceChildren(wordRow(S, { sel: k, onClick: draw }), el("span", { class: "w-label" }, `head 1: ${H1.name}`), wordRow(S, { weights: r1.w, max: 0.6 }), el("span", { class: "w-label" }, `head 2: ${H2.name}`), wordRow(S, { weights: r2.w, max: 0.6 }),
          P(k === IT ? `"it": head 1 finds <b>cat</b>, head 2 finds <b>sat</b> and <b>tired</b>.` : k === 1 ? `"cat" asks head 2 "what am I doing?", and finds <b>sat</b> and <b>tired</b>.` : `This word asks no strong question in these two hand-made heads, so its attention is spread out.`)); };
      a.append(out); draw(IT);
    },
    recap: `Head 1 links "it" → cat. Head 2 links "it" → sat/tired. Same sentence, two different relationships.` },
  { kind: "math", title: "Head 2's numbers for 'it'",
    html: `<p>Head 2 has its own recipes: W<sub>Q</sub> turns living or pointer words into the question "wants an action/feeling", and W<sub>K</sub> makes action/feeling words answer "I'm an action/feeling".</p>`,
    render(a) {
      const r = TOY.attend(S, IT, H2);
      a.append(el("div", { class: "w-row" }, smallMat(H2.WQ, "W_Q (head 2)", { rowLabels: TOY.FEAT_SHORT, colLabels: H2.qLabels }), smallMat(H2.WK, "W_K (head 2)", { rowLabels: TOY.FEAT_SHORT, colLabels: H2.kLabels })),
        el("div", { class: "worked" }, `query of "it" = [${mulStr(embOf("it"), H2.WQ.map((row) => row[0]))}, …] = `, el("b", {}, `[${r.q.map(f2).join(", ")}]`)),
        mathTable(["word", "key", "score", "÷√2", "e^", "attention"], S.map((w, i) => [w, `[${r.K[i].map(f1).join(", ")}]`, f2(r.raw[i]), f2(r.scaled[i]), f2(r.expsRaw[i]), Math.round(r.w[i] * 100) + "%"]), { hi: (i) => i === 2 || i === 9 }));
    },
    recap: `Exactly the same recipe as Lesson 3, just with different W<sub>Q</sub> and W<sub>K</sub>. "it" → sat ${Math.round(TOY.attend(S, IT, H2).w[2] * 100)}%, tired ${Math.round(TOY.attend(S, IT, H2).w[9] * 100)}%.` },
  { kind: "watch", title: "Combine the heads",
    html: `<p>Each head returns a 4-number message. They are <b>joined end to end</b> (8 numbers), then a final weights table <b>W<sub>O</sub></b> mixes them back down to 4. Ours simply adds the two halves.</p>`,
    render(a) {
      const o1 = TOY.attend(S, IT, H1).out, o2 = TOY.attend(S, IT, H2).out, cat = [...o1, ...o2], mixed = o1.map((v, i) => v + o2[i]), x = embOf("it"), fin = x.map((v, i) => v + mixed[i]);
      const frames = [
        [vecBars(o1, { title: "head 1 output (who?)" }), "Head 1's answer: mostly 'living' (from cat)."],
        [vecBars(o2, { title: "head 2 output (doing what?)" }), "Head 2's answer: mostly 'action/feeling' (from sat, tired)."],
        [vecBars(cat, { labels: [...TOY.FEAT_SHORT.map((f) => "h1 " + f), ...TOY.FEAT_SHORT.map((f) => "h2 " + f)], colors: [...FEAT_COLORS, ...FEAT_COLORS] }), "Join them end to end: 8 numbers."],
        [vecBars(mixed, { title: "× W_O → back to 4 numbers" }), "W_O mixes the heads into one message."],
        [el("div", { class: "w-row" }, vecBars(x, { title: '"it" before' }), vecBars(fin, { title: '"it" after both heads', max: 1.8 })), `"it" is now living (from cat) AND tired/acting (from sat, tired): "a tired cat".`]];
      a.append(player(frames.length, (k) => ({ node: frames[k][0], caption: frames[k][1] }), { interval: 1500 }));
    },
    recap: `multi-head output = W<sub>O</sub> × [head 1 ; head 2 ; …]. Every head contributes its own piece of context.` },
  { kind: "learn", title: "Same cost as one big head",
    html: `<p>More heads doesn't mean more work. A model with 64 numbers per word and 4 heads gives each head a <b>slice of 16</b> of those numbers. Same total size, same number of weights, just split up. Below: 3 words × 8 numbers, split into heads (try 1, 2, 4, 8).</p>`,
    render(a) { mount(a, "split-heads"); },
    recap: `d<sub>model</sub> is split across heads (d<sub>k</sub> = d<sub>model</sub> / heads), so multi-head attention costs the same as one head.` },
  { kind: "try", title: "Real heads in a trained model",
    html: `<p>Our heads were hand-written. Here are the 4 heads of the <b>real trained</b> date translator from the Lab. Nobody designed these patterns; they emerged from training.</p>`,
    render(a) { mount(a, "real-heads"); },
    recap: `Trained heads specialise by themselves: some look at neighbours, some at separators, some link the digits of a number.` },
  { kind: "code", title: "Code it (optional)",
    html: `<p>Split into heads and merge back in NumPy. The classic bug produces the right shape but scrambled numbers, and the checker will tell you if you hit it.</p>`,
    render(a) { mount(a, "py-exercise", { id: "heads" }); },
    recap: `Splitting into heads is a reshape plus a transpose, done in the right order.` },
] });

/* =============================== LESSON 7 =============================== */
// a tiny hand-made feed-forward network (4 → 4 hidden → 4) whose hidden units have names
const FFN = { names: ["tired living thing", "object", "pointer still unresolved", "non-living"],
  W1: [[1, 0, -1, -1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]], b1: [-0.5, -0.5, -0.5, 0.3],
  W2: [[0.5, 0, 0.3, 0], [0, 0.3, 0, 0], [0, 0, 0, -0.3], [0, 0.2, 0, 0]], b2: [0, 0, 0, 0] };
function blockTrace() {
  const X = S.map(embOf), x = X[IT];   // positions left out so the numbers stay readable
  const o1 = TOY.attend(S, IT, H1, { embs: X }).out, o2 = TOY.attend(S, IT, H2, { embs: X }).out, att = o1.map((v, i) => v + o2[i]);
  const r1 = x.map((v, i) => v + att[i]);
  const ln = (v) => { const m = v.reduce((s, z) => s + z, 0) / v.length, va = v.reduce((s, z) => s + (z - m) ** 2, 0) / v.length; return { m, va, out: v.map((z) => (z - m) / Math.sqrt(va + 1e-5)) }; };
  const n1 = ln(r1), pre = FFN.W1[0].map((_, j) => n1.out.reduce((s, z, i) => s + z * FFN.W1[i][j], 0) + FFN.b1[j]), hid = pre.map((z) => Math.max(0, z));
  const ff = FFN.W2[0].map((_, j) => hid.reduce((s, h, i) => s + h * FFN.W2[i][j], 0) + FFN.b2[j]), r2 = n1.out.map((v, i) => v + ff[i]), n2 = ln(r2);
  return { x, att, r1, n1, pre, hid, ff, r2, n2 };
}
Lessons.add({ id: "l7", title: "The whole Transformer", notebook: "notebook 04", steps: [
  { kind: "learn", title: "A block: meet, then think",
    html: `<p>A Transformer is a stack of identical <b>blocks</b>. Each block does two things:</p><ol><li><b>Attention, the meeting.</b> Words exchange information (Lessons 3–6).</li><li><b>Feed-forward, thinking alone.</b> Each word processes what it just heard, on its own, with a small neural network.</li></ol><p>Around each of these sits an <b>"Add &amp; Norm"</b>: add the input back (a residual connection), then rescale the numbers (LayerNorm). Stack 6, 12 or 96 of these blocks and you have a Transformer.</p>`,
    render(a) { a.append(el("div", { class: "flow" }, ...["words + positions", "attention (meet)", "add & norm", "feed-forward (think)", "add & norm", "→ next block"].map((t, i) => el("span", { class: "shape " + ["emb", "attn", "norm", "ffn", "norm", ""][i] }, t)))); },
    recap: `block = attention → add & norm → feed-forward → add & norm. Repeat N times.` },
  { kind: "watch", title: "Follow 'it' through one block",
    html: `<p>Every number "it" holds, at every stage of one block. Press play, or drag the slider.</p>`,
    render(a) {
      const t = blockTrace();
      const fr = [
        [t.x, "1. Input: the meaning of 'it'", "Where every block starts. (In a real model, meaning + position; we leave positions out here so the numbers stay readable.)"],
        [t.att, "2. Attention output (2 heads, mixed)", "What 'it' gathered from the cat and from tired."],
        [t.r1, "3. Add: input + attention", "Keep the original, add the new information."],
        [t.n1.out, "4. Norm: rescale to average 0, spread 1", "Keeps numbers in a stable range."],
        [t.hid, "5. Feed-forward hidden layer (after ReLU)", "Four 'detectors' fire or stay silent. Only 'tired living thing' fires: attention already told 'it' about the cat and about being tired.", FFN.names],
        [t.ff, "6. Feed-forward output", "What the thinking step adds."],
        [t.r2, "7. Add: before + feed-forward", "Residual again."],
        [t.n2.out, "8. Norm: block output", "This vector goes into the next block."]];
      a.append(player(fr.length, (k) => ({ node: el("div", { class: "w-row" }, vecBars(fr[k][0], { title: fr[k][1], labels: fr[k][3] || TOY.FEAT_SHORT, max: 2 }), el("div", { class: "w-col" }, el("code", {}, `[${fr[k][0].map(f2).join(", ")}]`))), caption: fr[k][2] }), { interval: 1600 }));
    },
    recap: `One block transforms "it"'s 4 numbers eight times. The next steps open stages 3, 4 and 5–6.` },
  { kind: "math", title: "Residual: add, don't replace",
    html: `<p>Instead of replacing "it" with the attention output, we <b>add</b> them. Two reasons: the word never loses its own information, and during training the "blame" signal can flow straight back through the plus sign. That's what lets 96-block models train at all.</p>`,
    render(a) { const t = blockTrace(); a.append(mathTable(["feature", "input", "+ attention", "= result"], TOY.FEAT_SHORT.map((f, i) => [f, f2(t.x[i]), f2(t.att[i]), f2(t.r1[i])]))); },
    recap: `residual: x_new = x + sublayer(x).` },
  { kind: "math", title: "LayerNorm, every number",
    html: `<p>After many additions numbers can drift large or small. <b>LayerNorm</b> rescales each word's vector to average 0 and spread 1 (a real model then applies a learned scale and shift).</p>`,
    render(a) {
      const t = blockTrace(), v = t.r1, m = t.n1.m, sd = Math.sqrt(t.n1.va + 1e-5);
      a.append(el("div", { class: "worked" },
        el("div", {}, `average = (${v.map(f2).join(" + ")}) ÷ 4 = `, el("b", {}, f2(m))),
        el("div", {}, `variance = average of (value − ${f2(m)})² = (${v.map((z) => f2((z - m) ** 2)).join(" + ")}) ÷ 4 = `, el("b", {}, f2(t.n1.va))),
        el("div", {}, `spread (std) = √${f2(t.n1.va)} = `, el("b", {}, f2(sd)))),
        mathTable(["feature", "value", "− average", "÷ spread"], TOY.FEAT_SHORT.map((f, i) => [f, f2(v[i]), f2(v[i] - m), f2((v[i] - m) / sd)])));
    },
    recap: `LayerNorm: subtract the average, divide by the spread, for each word separately.` },
  { kind: "math", title: "Feed-forward: thinking alone",
    html: `<p>The feed-forward network is two weight tables with a <b>ReLU</b> between them (ReLU: negative numbers become 0). Our hand-made one has 4 hidden "detectors" with names. Real ones have thousands, unnamed, and hold most of the model's knowledge.</p>`,
    render(a) {
      const t = blockTrace();
      a.append(mathTable(["detector", "input × W₁ + bias", "ReLU", "meaning"], FFN.names.map((n, j) => [n, `${t.n1.out.map((z, i) => `${f2(z)}×${FFN.W1[i][j]}`).filter((_, i) => FFN.W1[i][j] !== 0).join(" + ")} ${FFN.b1[j] < 0 ? "−" : "+"} ${Math.abs(FFN.b1[j])} = ${f2(t.pre[j])}`, f2(t.hid[j]), t.hid[j] > 0 ? "fires" : "silent"])),
        P(`Then output = hidden × W₂ = <code>[${t.ff.map(f2).join(", ")}]</code>, which is added back (residual) and normalised again.`));
    },
    recap: `feed-forward = W₂ · ReLU(W₁ · x + b₁) + b₂, applied to every word on its own. About ⅔ of a real model's weights live here.` },
  { kind: "learn", title: "Stack blocks, then predict",
    html: `<p>The block's output goes into the next block, which can now ask better questions: after block 1, "it" already "knows" it's a tired cat. After the last block, a final layer compares each word's vector with every vocabulary word, giving one <b>score per word</b>. A softmax turns those scores into <b>probabilities for the next word</b>. That's how ChatGPT-style models pick each word they write.</p><p>The original Transformer had two stacks: an <b>encoder</b> that reads the input (no mask) and a <b>decoder</b> that writes the output (causal mask, plus cross-attention that looks at the encoder). The date translator in the Lab is exactly that.</p>`,
    render(a) { mount(a, "architecture"); },
    recap: `Transformer = embeddings + positions → N blocks → final layer → softmax over the vocabulary.` },
  { kind: "code", title: "Code it (optional)",
    html: `<p>Write LayerNorm and the feed-forward network in NumPy.</p>`,
    render(a) { mount(a, "py-exercise", { id: "layernorm" }); },
    recap: `With attention, LayerNorm and feed-forward, you've now written every piece of a Transformer block.` },
] });

/* =============================== LESSON 8 =============================== */
const IT_DATA = TOY.itDataset();
const itTrained = {};
function trainedIt(seed = 6, steps = 150) { const k = seed + ":" + steps; if (!itTrained[k]) { const m = new TOY.ItModel(16, seed); for (let s = 0; s < steps; s++) m.step(IT_DATA.train, 0.02); itTrained[k] = m; } return itTrained[k]; }
function itView(m, words) { const ex = { words, qi: words.indexOf("it") }, pr = m.predict(ex), f = pr.f; const top = f.p.map((p, i) => [TOY.IT_VOCAB[i], p]).sort((x, y) => y[1] - x[1]).slice(0, 4);
  return el("div", { class: "w-row" }, el("div", { class: "w-col w-grow" }, el("span", { class: "w-label" }, `where "it" looks`), wordRow(words, { weights: f.a, max: 1 })),
    el("div", { class: "w-col", style: { minWidth: "14rem" } }, el("span", { class: "w-label" }, `"it" refers to…`), bars(top.map(([w, p], i) => ({ label: w, value: p, cls: i === 0 ? "hi" : "" })), { max: 1, pct: true }))); }
Lessons.add({ id: "l8", title: "How a model learns its numbers", notebook: "notebook 05", steps: [
  { kind: "learn", title: "A model starts as random numbers",
    html: `<p>In the earlier lessons <i>we</i> chose W<sub>Q</sub>, W<sub>K</sub> and the embeddings. A real model starts with <b>random</b> numbers and has to find good ones itself. Let's watch it happen on a small model with 1 attention head, 16 numbers per word and 21 words, about 1,500 numbers in total. Its job: say who "it" is.</p><p>Before any training, it's clueless:</p>`,
    render(a) { a.append(itView(new TOY.ItModel(16, 6), S)); },
    recap: `An untrained model = random weights = random attention and random answers.` },
  { kind: "learn", title: "The job: who is 'it'?",
    html: `<p>We make ${IT_DATA.train.length + IT_DATA.test.length} sentences from 4 patterns, 6 animals and 6 objects. The answer is always the animal. Note the patterns where the animal comes <b>after</b> the object, so "it's always word 1" won't work. We hide ${IT_DATA.test.length} sentences (new animal + object pairs) to test it later.</p>`,
    render(a) { a.append(mathTable(["example training sentence", "answer"], [0, 7, 14, 21, 30, 45].map((i) => [IT_DATA.train[i].words.join(" "), IT_DATA.train[i].target]))); },
    recap: `Training data = examples with the right answers. Test data = examples the model never trains on.` },
  { kind: "math", title: "Loss: one number for 'how wrong'",
    html: `<p>For each example the model gives a probability to the right answer. The <b>loss</b> is −log(that probability): 0 when it's 100% sure and right, large when it gave the right answer little probability. Slide it.</p>`,
    render(a) {
      const out = el("div");
      const draw = (p) => out.replaceChildren(el("div", { class: "worked" }, `loss = −log(${f2(p)}) = `, el("b", { class: "big" }, f2(-Math.log(p)))));
      a.append(slider("probability given to the right answer", { min: 0.01, max: 1, step: 0.01, value: 0.05, fmt: (v) => Math.round(v * 100) + "%" }, draw), out,
        mathTable(["probability of the right answer", "loss"], [1, 0.9, 0.5, 0.25, 0.05, 0.01].map((p) => [Math.round(p * 100) + "%", f2(-Math.log(p))])),
        P(`Guessing at random among 21 words gives each word about 5%, so the loss starts near −log(1/21) = <b>${f2(Math.log(21))}</b>. Training means: make the average loss go down.`)); draw(0.05);
    },
    recap: `loss = −log(probability of the correct answer). Lower is better. Random guessing ≈ log(vocabulary size).` },
  { kind: "try", title: "Walking downhill",
    html: `<p>Imagine the loss as a landscape, and the model's numbers as your position on it. <b>Gradient descent</b>: feel which way is downhill (the slope, called the <i>gradient</i>), take a small step that way, repeat. The step size is the <b>learning rate</b>. Here's one number being trained.</p>`,
    render(a) { mount(a, "gd-ball"); },
    recap: `Training loop: compute the loss, compute the slope for every weight, step each weight a little downhill, repeat.` },
  { kind: "math", title: "Backpropagation: passing the blame back",
    html: `<p>With 1,500 numbers, how do we know each one's slope? <b>Backpropagation</b> applies the chain rule backwards from the loss. It starts with a beautifully simple first step: <b>blame on each output score = predicted probability − correct answer</b>.</p>`,
    render(a) {
      const m = new TOY.ItModel(16, 6), ex = IT_DATA.train[0], f = m.forward(ex), y = TOY.IT_VOCAB.indexOf(ex.target);
      const rows = f.p.map((p, i) => [TOY.IT_VOCAB[i], p, i === y ? 1 : 0]).sort((u, v) => v[1] - u[1]).slice(0, 6);
      if (!rows.find((r) => r[2])) rows.push([ex.target, f.p[y], 1]);
      a.append(P(`Sentence: <i>${ex.words.join(" ")}</i>, answer: <b>${ex.target}</b>, untrained model.`),
        mathTable(["word", "predicted", "correct", "blame = predicted − correct"], rows.map(([w, p, c]) => [w, Math.round(p * 100) + "%", String(c), f2(p - c)]), { hi: (i) => rows[i][2] === 1 }),
        P(`The right answer gets negative blame ("push its score up"); wrong answers get positive blame ("push down"). That blame flows backwards through the output layer, the weighted mix, the softmax, W<sub>Q</sub>, W<sub>K</sub> and the embeddings. Every number gets told which way to move. In our model the code that computes this is ~25 lines, written by hand.`));
    },
    recap: `backprop = the chain rule, run backwards. It starts from (prediction − truth) and gives every weight its slope.` },
  { kind: "try", title: "Train it! Watch attention appear",
    html: `<p>Press <b>Train</b>. Each step looks at all ${IT_DATA.train.length} training sentences, computes the loss and the slopes, and nudges every number. Watch where "it" looks in our sentence, and watch the <b>test accuracy</b>: sentences with animal + object pairs the model never trains on.</p>`,
    render(a) {
      let m, step = 0, timer = null, losses = [], seed = 6;
      const view = el("div"), chart = el("div"), stats = el("div", { class: "status-line" });
      const btn = button("Train", () => (timer ? stop() : start()), "primary");
      const reset = () => { stop(); m = new TOY.ItModel(16, seed); step = 0; losses = []; draw(); };
      const start = () => { btn.textContent = "Pause"; timer = setInterval(() => { for (let k = 0; k < 2; k++) { const r = m.step(IT_DATA.train, 0.02); losses.push(r.loss); step++; } draw(); if (step >= 200) stop(); }, 40); };
      const stop = () => { clearInterval(timer); timer = null; btn.textContent = step ? "Continue" : "Train"; };
      const draw = () => {
        const tr = m.evaluate(IT_DATA.train), te = m.evaluate(IT_DATA.test);
        stats.textContent = `step ${step} · loss ${f2(tr.loss)} · training accuracy ${Math.round(tr.acc * 100)}% · test accuracy (unseen sentences) ${Math.round(te.acc * 100)}%`;
        view.replaceChildren(itView(m, S));
        chart.replaceChildren(losses.length > 1 ? lineChart([{ name: "loss", values: losses }], { width: 420, height: 160, ymin: 0 }) : P(`Loss starts near ${f2(Math.log(21))}.`));
      };
      a.append(el("div", { class: "controls" }, btn, button("1 step", () => { stop(); const r = m.step(IT_DATA.train, 0.02); losses.push(r.loss); step++; draw(); }), button("Reset", reset),
        select("random start (seed)", [6, 4, 2, 5, 7, 3], seed, (v) => { seed = +v; reset(); })), stats, view, chart,
        P(`Things to notice: early on, attention is spread out and the answers are wrong. Then the loss drops sharply as "it" locks on to the animal. Try other seeds: training starts from random numbers, so every run is different. Some find odd solutions (seed 5, seed 7) or get stuck (seed 3).`));
      reset();
    },
    recap: `Nobody told the model "pointer words should look for animals". Gradient descent discovered that this lowers the loss.` },
  { kind: "predict", title: "A sentence it has never seen", gate: true,
    html: `<p>"the dog sat on the rug because it was tired": the dog + rug pair was held out, never seen in training.</p>`,
    predict: { q: `After training, where will "it" look in this unseen sentence?`, options: ["dog", "rug", "spread over everything"], answer: 0, why: `It learned a <b>rule</b> (look for the animal), not a memorised list:` },
    render(a) { a.append(itView(trainedIt(), ["the", "dog", "sat", "on", "the", "rug", "because", "it", "was", "tired"])); },
    recap: `Generalisation: a model that learned a rule works on new examples. That's the whole point of training.` },
  { kind: "try", title: "Test it yourself",
    html: `<p>Build any sentence from the pieces the model knows and see what it thinks.</p>`,
    render(a) {
      let t = 2, A = "horse", O = "box";
      const out = el("div");
      const draw = () => { const words = TOY.TEMPLATES[t].replace("A", A).replace("O", O).split(" "); out.replaceChildren(P(`<i>${words.join(" ")}</i>`), itView(trainedIt(), words)); };
      a.append(el("div", { class: "controls" }, select("pattern", TOY.TEMPLATES.map((s, i) => [i, s.replace("A", "[animal]").replace("O", "[object]")]), t, (v) => { t = +v; draw(); }), select("animal", TOY.ANIMALS, A, (v) => { A = v; draw(); }), select("object", TOY.OBJECTS, O, (v) => { O = v; draw(); })), out); draw();
    },
    recap: `The trained attention head finds the animal wherever it is in the sentence.` },
  { kind: "learn", title: "Models take shortcuts",
    html: `<p>Our trained model is right 100% of the time. But look at a sentence about a <b>cow</b>:</p>`,
    render(a) { a.append(itView(trainedIt(), ["the", "cow", "sat", "on", "the", "box", "because", "it", "was", "tired"]),
      P(`"it" doesn't look at "cow" at all, yet it answers "cow"! The model found a shortcut: <i>"if no animal's name tag lights up, the answer must be the cow"</i>. It's correct on every sentence we have, so the loss is happy, but it isn't the rule we had in mind. Every training run picks a different "default" animal.`),
      P(`Models learn whatever lowers the loss, not what we intended. That's why researchers inspect attention and test on new kinds of data, and why a good score alone doesn't prove a model "understands".`)); },
    recap: `Training optimises the loss, and sometimes finds strange shortcuts. Always inspect what the model actually does.` },
  { kind: "learn", title: "From here to ChatGPT",
    html: `<p>Big language models use exactly what you've learned, at enormous scale:</p><ul><li><b>Task:</b> guess the next word (token), on trillions of words of text.</li><li><b>Model:</b> embeddings + positions → ~100 blocks of multi-head attention (causal mask) and feed-forward → probabilities for the next token.</li><li><b>Training:</b> loss = −log(probability of the actual next word), backpropagation, gradient descent. The same loop you just watched.</li><li><b>Writing text:</b> pick a next word, append it, run again: one token at a time.</li></ul><p>The <b>Lab</b> has a real trained Transformer (the date translator) where you can watch every one of these pieces, including cross-attention and step-by-step generation.</p>`,
    render(a) { a.append(el("a", { class: "btn primary", href: "#lab-inspector" }, "Open the model inspector →")); },
    recap: `ChatGPT-style models = this course × scale. Same ideas, more numbers.` },
  { kind: "code", title: "Code it (optional)",
    html: `<p>Generation, one token at a time, driving the real trained date Transformer from your own Python loop.</p>`,
    render(a) { mount(a, "py-exercise", { id: "greedy" }); },
    recap: `You've written the generation loop every LLM uses.` },
] });

/* =============================== LESSON 9 =============================== */
QUIZZES.final = [
  ["What does an embedding turn a word into?", ["A list of numbers where similar words get similar numbers", "A single ID number", "A one-hot vector", "A sentence"], 0, "IDs and one-hot vectors carry no meaning; embeddings do."],
  ["In attention, what are the query, key and value?", ["My question, my name tag, my message", "Three copies of the same thing", "The input, output and loss", "Positions"], 0, "Score = query · key; output = weighted mix of values."],
  ["Why divide scores by √d_k?", ["Big vectors make big scores, which make softmax all-or-nothing and stop learning", "To make the sum 100%", "To save memory", "Because d_k is always 4"], 0, "You saw unscaled attention collapse to ~100% on one word as d_k grew."],
  ["Why −∞ in the causal mask?", ["e^−∞ = 0 exactly, so future words get zero weight and the rest still sum to 100%", "−∞ makes it faster", "It's the same as 0", "To make attention symmetric"], 0, "A score of 0 still gets weight because e^0 = 1."],
  ["Why does a Transformer need positional encoding?", ["Attention alone can't see word order", "To make words similar", "To mask the future", "To count parameters"], 0, "Shuffled sentence → identical attention, as you saw in Lesson 5."],
  ["Why use several heads?", ["Each can ask a different question (who? doing what?) at no extra cost", "To reduce the vocabulary", "Because one head can't be trained", "To add positions"], 0, "Head 1 found 'cat', head 2 found 'tired'."],
  ["What does the residual connection do?", ["Adds the input back so nothing is lost and training signals flow easily", "Removes the input", "Normalises numbers", "Masks the future"], 0, "x_new = x + sublayer(x)."],
  ["How does a model find its numbers?", ["Start random; repeatedly nudge every number downhill on the loss (gradient descent + backprop)", "Engineers write them", "They're copied from a dictionary", "They're fixed forever"], 0, "You watched 'it' learn to look at the animal."],
];
Lessons.add({ id: "l9", title: "Explain it like an expert", notebook: "notebook 06", steps: [
  { kind: "learn", title: "The whole story in 8 sentences",
    html: `<ol class="story"><li>Text is cut into <b>tokens</b>, and each becomes an ID. <a href="#l1">L1</a></li><li>An <b>embedding</b> table turns each ID into a vector whose direction means something. <a href="#l2">L2</a></li><li><b>Attention</b>: each word compares its query with every key (dot product ÷ √d<sub>k</sub>, softmax) and takes a weighted mix of values, so "it" becomes "cat"-like. <a href="#l3">L3</a></li><li>All words do it at once (QKᵀ); a <b>causal mask</b> hides the future. <a href="#l4">L4</a></li><li><b>Positional encodings</b> tell attention where words are. <a href="#l5">L5</a></li><li><b>Multiple heads</b> ask different questions in parallel. <a href="#l6">L6</a></li><li>A <b>block</b> = attention + feed-forward, each with add &amp; norm; stack many, then predict the next token. <a href="#l7">L7</a></li><li>All the numbers start random and are <b>learned</b> by gradient descent on the loss. <a href="#l8">L8</a></li></ol>`,
    recap: `If you can say these 8 sentences and explain the WHY behind each, you understand Transformers.` },
  { kind: "check", title: "Final quiz",
    render(a) { mount(a, "quiz", { id: "final" }); },
    recap: `Every answer is something you measured yourself in this course.` },
  { kind: "try", title: "Interview flashcards",
    html: `<p>Say each answer out loud (30–60 seconds, including the <b>why</b>), then reveal it.</p>`,
    render(a) { mount(a, "flashcards"); },
    recap: `Cite your own experiments: "with positions removed, accuracy fell from 100% to 15%" beats any textbook answer.` },
  { kind: "code", title: "Whiteboard: attention from memory",
    html: `<p>The classic interview task. Start the timer and write attention from an empty box. Target: under 3 minutes.</p>`,
    render(a) { mount(a, "py-exercise", { id: "memory" }); },
    recap: `🎉 You've finished the course. Keep exploring in the Lab.` },
] });
