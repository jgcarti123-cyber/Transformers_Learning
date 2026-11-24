/* ============================================================================
   toy.js: the hand-made world behind the beginner lessons.
   One sentence, tiny vectors with NAMED meanings, numbers you can check by hand,
   and a tiny attention model that learns "who is 'it'?" live in the browser.
   ========================================================================== */
const TOY = (() => {
  const SENT = ["the", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"];
  const FEATURES = ["living thing", "object/place", "action/feeling", "points back"];
  const FEAT_SHORT = ["living", "object", "action", "pointer"];
  // Hand-designed embeddings (d = 4). Each number answers one question about the word.
  const EMB = {
    the: [0.1, 0.1, 0.0, 0.0],
    cat: [1.0, 0.1, 0.0, 0.0],
    dog: [0.9, 0.2, 0.0, 0.0],
    sat: [0.1, 0.0, 1.0, 0.0],
    on: [0.0, 0.2, 0.1, 0.0],
    mat: [0.0, 1.0, 0.0, 0.0],
    rug: [0.0, 0.9, 0.1, 0.0],
    because: [0.0, 0.0, 0.2, 0.0],
    it: [0.3, 0.3, 0.0, 1.0],
    was: [0.0, 0.0, 0.3, 0.0],
    tired: [0.6, 0.0, 0.8, 0.0],
  };
  const VOCAB = Object.keys(EMB);
  const ID = Object.fromEntries(VOCAB.map((w, i) => [w, i]));

  // Head 1, "who is the living thing?":  only pointer words ask; keys advertise living / object.
  const HEAD1 = {
    name: "who is the living thing?",
    WQ: [[0, 0], [0, 0], [0, 0], [2, 0]],            // query = [looking for living, looking for object]
    WK: [[2, 0], [0, 2], [0, 0], [0, 0]],            // key   = [I am living, I am an object]
    qLabels: ["wants living", "wants object"], kLabels: ["is living", "is object"],
  };
  // Head 2, "what state am I in?":  nouns and pointers ask for action/feeling words.
  const HEAD2 = {
    name: "what is it doing or feeling?",
    WQ: [[1.5, 0], [0, 0], [0, 0], [1.5, 0]],        // query = [wants action/feeling, -]
    WK: [[0, 0], [0, 0], [2, 0], [0, 0]],            // key   = [is action/feeling, -]
    qLabels: ["wants action", "(unused)"], kLabels: ["is action", "(unused)"],
  };

  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const vecMat = (v, M) => M[0].map((_, j) => v.reduce((s, x, i) => s + x * M[i][j], 0));
  const r2 = (x) => Math.round(x * 100) / 100;
  const softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => (v === -Infinity ? 0 : Math.exp(v - m))); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };

  // Everything about ONE query attending over a sentence, with every intermediate number kept.
  function attend(words, qi, head = HEAD1, { scale = true, mask = null, embs = null } = {}) {
    const X = embs || words.map((w) => EMB[w]);
    const q = vecMat(X[qi], head.WQ);
    const K = X.map((x) => vecMat(x, head.WK));
    const dk = q.length;
    const raw = K.map((k) => dot(q, k));
    const scaled = raw.map((s) => (scale ? s / Math.sqrt(dk) : s));
    const masked = scaled.map((s, j) => (mask && !mask[j] ? -Infinity : s));
    const m = Math.max(...masked);
    const exps = masked.map((s) => (s === -Infinity ? 0 : Math.exp(s - m)));
    const expsRaw = masked.map((s) => (s === -Infinity ? 0 : Math.exp(s)));
    const sum = exps.reduce((a, b) => a + b, 0);
    const w = exps.map((e) => e / sum);
    const V = X;                                             // W_V = identity: a value is the word's meaning
    const out = V[0].map((_, c) => w.reduce((s, wj, j) => s + wj * V[j][c], 0));
    return { X, q, K, V, raw, scaled, masked, exps, expsRaw, sumRaw: expsRaw.reduce((a, b) => a + b, 0), w, out, dk };
  }
  function attendAll(words, head = HEAD1, opts = {}) { return words.map((_, i) => attend(words, i, head, opts)); }
  const cosine = (a, b) => dot(a, b) / (Math.sqrt(dot(a, a) * dot(b, b)) || 1);

  // small sinusoidal positions (d = 4)
  function pe(pos, d = 4) { const v = []; for (let i = 0; i < d; i += 2) { const a = pos / Math.pow(10000, i / d); v.push(Math.sin(a), Math.cos(a)); } return v; }

  // ---------------- a tiny model that LEARNS "who is 'it'?" ----------------
  const ANIMALS = ["cat", "dog", "cow", "bird", "fox", "horse"];
  const OBJECTS = ["mat", "rug", "box", "bed", "chair", "log"];
  const TEMPLATES = [
    "the A sat on the O because it was tired",
    "the A slept on the O because it was tired",
    "on the O sat the A because it was tired",
    "the O was under the A because it was tired",
  ];
  const TEST_PAIRS = new Set(["dog|rug", "cow|chair", "fox|bed", "bird|mat", "horse|box"]);
  function itDataset() {
    const train = [], test = [];
    for (const A of ANIMALS) for (const O of OBJECTS) TEMPLATES.forEach((t, ti) => {
      const words = t.replace("A", A).replace("O", O).split(" ");
      (TEST_PAIRS.has(A + "|" + O) ? test : train).push({ words, target: A, qi: words.indexOf("it"), template: ti });
    });
    return { train, test };
  }
  const IT_VOCAB = [...new Set(["the", "sat", "slept", "on", "under", "was", "because", "it", "tired", ...ANIMALS, ...OBJECTS])];
  const IT_ID = Object.fromEntries(IT_VOCAB.map((w, i) => [w, i]));

  class ItModel {
    constructor(d = 8, seed = 1) {
      let a = seed >>> 0; const r = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      const n = () => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      const V = IT_VOCAB.length;
      this.d = d; this.V = V;
      const mat = (R, C, s) => Array.from({ length: R }, () => Array.from({ length: C }, () => n() * s));
      this.P = { E: mat(V, d, 1), Wq: mat(d, d, 1 / Math.sqrt(d)), Wk: mat(d, d, 1 / Math.sqrt(d)), Wv: mat(d, d, 1 / Math.sqrt(d)), Wo: mat(d, V, 1 / Math.sqrt(d)) };
      this.M = {}; this.S = {}; this.t = 0;
      for (const k in this.P) { this.M[k] = this.P[k].map((row) => row.map(() => 0)); this.S[k] = this.P[k].map((row) => row.map(() => 0)); }
    }
    forward(ex) {
      const { E, Wq, Wk, Wv, Wo } = this.P, d = this.d;
      const ids = ex.words.map((w) => IT_ID[w]), X = ids.map((i) => E[i]);
      const q = vecMat(X[ex.qi], Wq), K = X.map((x) => vecMat(x, Wk)), Vv = X.map((x) => vecMat(x, Wv));
      const s = K.map((k) => dot(q, k) / Math.sqrt(d)), a = softmax(s);
      const o = Vv[0].map((_, c) => a.reduce((acc, aj, j) => acc + aj * Vv[j][c], 0));
      const logits = vecMat(o, Wo), p = softmax(logits);
      return { ids, X, q, K, Vv, s, a, o, logits, p };
    }
    predict(ex) { const f = this.forward(ex); let b = 0; f.p.forEach((x, i) => { if (x > f.p[b]) b = i; }); return { word: IT_VOCAB[b], p: f.p[b], f }; }
    // one Adam step on a batch; gradients derived by hand (chain rule), no autograd
    step(batch, lr = 0.03) {
      const d = this.d, V = this.V, sq = Math.sqrt(d);
      const G = {}; for (const k in this.P) G[k] = this.P[k].map((row) => row.map(() => 0));
      let loss = 0, correct = 0;
      for (const ex of batch) {
        const f = this.forward(ex), y = IT_ID[ex.target];
        loss -= Math.log(f.p[y] + 1e-12);
        let best = 0; f.p.forEach((x, i) => { if (x > f.p[best]) best = i; }); if (best === y) correct++;
        const dl = f.p.map((pi, i) => (pi - (i === y ? 1 : 0)) / batch.length);           // dLoss/dlogits = p - onehot
        for (let c = 0; c < d; c++) for (let v = 0; v < V; v++) G.Wo[c][v] += f.o[c] * dl[v];
        const dO = this.P.Wo.map((row) => dot(row, dl));                                // (d)
        const da = f.Vv.map((v) => dot(dO, v));
        const sa = f.a.reduce((acc, aj, j) => acc + aj * da[j], 0);
        const ds = f.a.map((aj, j) => aj * (da[j] - sa));                                // softmax backward
        const dq = new Array(d).fill(0);
        f.X.forEach((x, j) => {
          const dk = f.q.map((qc) => (ds[j] * qc) / sq), dv = dO.map((g) => f.a[j] * g);
          for (let c = 0; c < d; c++) dq[c] += (ds[j] * f.K[j][c]) / sq;
          const dx = new Array(d).fill(0);
          for (let i = 0; i < d; i++) for (let c = 0; c < d; c++) {
            G.Wk[i][c] += x[i] * dk[c]; G.Wv[i][c] += x[i] * dv[c];
            dx[i] += this.P.Wk[i][c] * dk[c] + this.P.Wv[i][c] * dv[c];
          }
          for (let i = 0; i < d; i++) G.E[f.ids[j]][i] += dx[i];
        });
        const xq = f.X[ex.qi];
        for (let i = 0; i < d; i++) for (let c = 0; c < d; c++) { G.Wq[i][c] += xq[i] * dq[c]; G.E[f.ids[ex.qi]][i] += this.P.Wq[i][c] * dq[c]; }
      }
      this.t++;
      const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
      for (const k in this.P) this.P[k].forEach((row, i) => row.forEach((_, j) => {
        const g = G[k][i][j];
        this.M[k][i][j] = b1 * this.M[k][i][j] + (1 - b1) * g;
        this.S[k][i][j] = b2 * this.S[k][i][j] + (1 - b2) * g * g;
        row[j] -= (lr * this.M[k][i][j] / c1) / (Math.sqrt(this.S[k][i][j] / c2) + 1e-8);
      }));
      return { loss: loss / batch.length, acc: correct / batch.length };
    }
    evaluate(set) { let c = 0, L = 0; for (const ex of set) { const f = this.forward(ex); const y = IT_ID[ex.target]; L -= Math.log(f.p[y] + 1e-12); let b = 0; f.p.forEach((x, i) => { if (x > f.p[b]) b = i; }); if (b === y) c++; } return { acc: c / set.length, loss: L / set.length }; }
  }

  return { SENT, FEATURES, FEAT_SHORT, EMB, VOCAB, ID, HEAD1, HEAD2, dot, vecMat, r2, softmax, attend, attendAll, cosine, pe,
    ANIMALS, OBJECTS, TEMPLATES, itDataset, IT_VOCAB, ItModel };
})();
if (typeof module !== "undefined") module.exports = TOY;
