/* ============================================================================
   engine.js: the maths, with no DOM.
   A real Transformer forward pass (weights trained in PyTorch, exported as fp16),
   word2vec (skip-gram) training, the synthetic corpus, positional encodings, PCA.
   ========================================================================== */
const TF = (() => {
  // ---------- random numbers (seeded, so demos are reproducible) ----------
  function rng(seed = 1) {
    let a = seed >>> 0;
    const next = () => {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    next.normal = () => {
      let u = 0, v = 0;
      while (u === 0) u = next();
      while (v === 0) v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    next.int = (n) => Math.floor(next() * n);
    next.pick = (arr) => arr[Math.floor(next() * arr.length)];
    return next;
  }

  // ---------- small linear algebra on plain arrays ----------
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const norm = (a) => Math.sqrt(dot(a, a));
  const cosine = (a, b) => dot(a, b) / (norm(a) * norm(b) + 1e-12);
  function softmax(z, temp = 1) {
    let m = -Infinity;
    for (const v of z) if (v > m) m = v;
    const e = z.map((v) => (v === -Infinity ? 0 : Math.exp((v - m) / temp)));
    const s = e.reduce((x, y) => x + y, 0);
    return e.map((v) => v / s);
  }
  function matmul(A, B) { // A: n×k (array of rows), B: k×m
    const n = A.length, k = B.length, m = B[0].length;
    const C = Array.from({ length: n }, () => new Array(m).fill(0));
    for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A[i][p]; for (let j = 0; j < m; j++) C[i][j] += a * B[p][j]; }
    return C;
  }
  const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));

  // ---------- positional encoding ----------
  function sinusoidalPE(maxLen, d) {
    const pe = new Float32Array(maxLen * d);
    for (let pos = 0; pos < maxLen; pos++)
      for (let i = 0; i < d; i += 2) {
        const angle = pos / Math.pow(10000, i / d);
        pe[pos * d + i] = Math.sin(angle);
        if (i + 1 < d) pe[pos * d + i + 1] = Math.cos(angle);
      }
    return pe;
  }

  // ---------- PCA to 2-D (power iteration on the covariance) ----------
  function pca2(rows) {
    const n = rows.length, d = rows[0].length;
    const mean = new Array(d).fill(0);
    rows.forEach((r) => r.forEach((v, i) => (mean[i] += v / n)));
    const X = rows.map((r) => r.map((v, i) => v - mean[i]));
    const C = Array.from({ length: d }, () => new Array(d).fill(0));
    X.forEach((r) => { for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += r[i] * r[j]; });
    const comps = [];
    for (let c = 0; c < 2; c++) {
      let v = Array.from({ length: d }, (_, i) => Math.sin(i + 1 + c * 7));
      for (let it = 0; it < 200; it++) {
        let w = C.map((row) => dot(row, v));
        comps.forEach((u) => { const p = dot(w, u); w = w.map((x, i) => x - p * u[i]); });
        const nv = norm(w) || 1;
        v = w.map((x) => x / nv);
      }
      comps.push(v);
    }
    // fix sign so the picture doesn't flip between runs
    comps.forEach((u, ci) => { const s = X.reduce((acc, r) => acc + Math.sign(dot(r, u)) * Math.abs(r[ci] || 0), 0); if (s < 0) u.forEach((_, i) => (u[i] = -u[i])); });
    return X.map((r) => [dot(r, comps[0]), dot(r, comps[1])]);
  }

  // ---------- fp16 weights ----------
  function halfToFloat(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function base64ToBytes(str) {
    if (typeof atob === "undefined") return new Uint8Array(Buffer.from(str, "base64"));
    const bin = atob(str), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function decodeFp16(bytes) {
    const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
    return out;
  }

  // ---------- the Transformer ----------
  class Model {
    constructor(cfg, tensors, flat, vocab) {
      this.cfg = cfg; this.vocab = vocab;
      this.d = cfg.d_model; this.h = cfg.n_heads; this.dk = cfg.d_model / cfg.n_heads; this.L = cfg.n_layers;
      this.V = cfg.vocab_size; this.dff = cfg.d_ff;
      this.P = {};
      for (const t of tensors) {
        const size = t.shape.reduce((a, b) => a * b, 1);
        this.P[t.name] = flat.subarray(t.offset, t.offset + size);
      }
      this.pe = sinusoidalPE(cfg.max_len, this.d);
      this.stoi = Object.fromEntries(vocab.map((s, i) => [s, i]));
    }
    // text <-> ids
    encodeText(text) {
      const ids = [], bad = [];
      for (const c of text) { if (c in this.stoi && this.stoi[c] > 2) ids.push(this.stoi[c]); else bad.push(c); }
      return { ids, bad: [...new Set(bad)] };
    }
    decodeIds(ids) { let s = ""; for (const i of ids) { if (i === 2) break; if (i > 2) s += this.vocab[i]; } return s; }
    tok(i) { const s = this.vocab[i]; return s === " " ? "␣" : s; }

    linear(x, T, inD, prefix) {
      const W = this.P[prefix + ".weight"], b = this.P[prefix + ".bias"], out = W.length / inD;
      const y = new Float32Array(T * out);
      for (let t = 0; t < T; t++)
        for (let o = 0; o < out; o++) {
          let s = b ? b[o] : 0;
          const wo = o * inD, xt = t * inD;
          for (let i = 0; i < inD; i++) s += x[xt + i] * W[wo + i];
          y[t * out + o] = s;
        }
      return y;
    }
    layerNorm(x, T, prefix) {
      const d = this.d, g = this.P[prefix + ".weight"], b = this.P[prefix + ".bias"], y = new Float32Array(x.length);
      for (let t = 0; t < T; t++) {
        let m = 0, v = 0;
        for (let i = 0; i < d; i++) m += x[t * d + i];
        m /= d;
        for (let i = 0; i < d; i++) v += (x[t * d + i] - m) ** 2;
        v /= d;
        const inv = 1 / Math.sqrt(v + 1e-5);
        for (let i = 0; i < d; i++) y[t * d + i] = (x[t * d + i] - m) * inv * g[i] + b[i];
      }
      return y;
    }
    add(a, b) { const y = new Float32Array(a.length); for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i]; return y; }
    embed(ids, table) {
      const d = this.d, E = this.P[table], sc = Math.sqrt(d), T = ids.length;
      const tok = new Float32Array(T * d), x = new Float32Array(T * d);
      for (let t = 0; t < T; t++)
        for (let i = 0; i < d; i++) {
          tok[t * d + i] = E[ids[t] * d + i] * sc;
          x[t * d + i] = tok[t * d + i] + (this.cfg.use_positions ? this.pe[t * d + i] : 0);
        }
      return { tok, x };
    }
    mha(xq, Tq, xkv, Tk, prefix, causal) {
      const d = this.d, h = this.h, dk = this.dk, sc = 1 / Math.sqrt(dk);
      const q = this.linear(xq, Tq, d, prefix + ".w_q"), k = this.linear(xkv, Tk, d, prefix + ".w_k"), v = this.linear(xkv, Tk, d, prefix + ".w_v");
      const concat = new Float32Array(Tq * d), attn = [], scores = [];
      for (let hh = 0; hh < h; hh++) {
        const W = new Float32Array(Tq * Tk), S = new Float32Array(Tq * Tk), off = hh * dk;
        for (let i = 0; i < Tq; i++) {
          let m = -Infinity;
          for (let j = 0; j < Tk; j++) {
            if (causal && j > i) { S[i * Tk + j] = -Infinity; continue; }
            let s = 0;
            for (let c = 0; c < dk; c++) s += q[i * d + off + c] * k[j * d + off + c];
            S[i * Tk + j] = s * sc;
            if (S[i * Tk + j] > m) m = S[i * Tk + j];
          }
          let z = 0;
          for (let j = 0; j < Tk; j++) { const e = S[i * Tk + j] === -Infinity ? 0 : Math.exp(S[i * Tk + j] - m); W[i * Tk + j] = e; z += e; }
          for (let j = 0; j < Tk; j++) W[i * Tk + j] /= z;
          for (let c = 0; c < dk; c++) { let s = 0; for (let j = 0; j < Tk; j++) s += W[i * Tk + j] * v[j * d + off + c]; concat[i * d + off + c] = s; }
        }
        attn.push(W); scores.push(S);
      }
      return { y: this.linear(concat, Tq, d, prefix + ".w_o"), q, k, v, attn, scores, concat };
    }
    ffn(x, T, prefix) {
      const hidden = this.linear(x, T, this.d, prefix + ".lin1");
      for (let i = 0; i < hidden.length; i++) if (hidden[i] < 0) hidden[i] = 0;
      return { y: this.linear(hidden, T, this.dff, prefix + ".lin2"), hidden };
    }
    encode(ids) {
      const T = ids.length, e = this.embed(ids, "src_embed.weight");
      let x = e.x;
      const layers = [];
      for (let l = 0; l < this.L; l++) {
        const p = `encoder_layers.${l}`;
        const a = this.mha(x, T, x, T, p + ".self_attn", false);
        const x1 = this.layerNorm(this.add(x, a.y), T, p + ".norm1");
        const f = this.ffn(x1, T, p + ".ffn");
        const x2 = this.layerNorm(this.add(x1, f.y), T, p + ".norm2");
        layers.push({ input: x, attn: a.attn, scores: a.scores, q: a.q, k: a.k, v: a.v, attnOut: a.y, afterNorm1: x1, ffnHidden: f.hidden, ffnOut: f.y, out: x2 });
        x = x2;
      }
      return { ids, T, tok: e.tok, emb: e.x, memory: x, layers };
    }
    decode(ids, memory, S) {
      const T = ids.length, e = this.embed(ids, "tgt_embed.weight");
      let y = e.x;
      const layers = [];
      for (let l = 0; l < this.L; l++) {
        const p = `decoder_layers.${l}`;
        const sa = this.mha(y, T, y, T, p + ".self_attn", true);
        const y1 = this.layerNorm(this.add(y, sa.y), T, p + ".norm1");
        const ca = this.mha(y1, T, memory, S, p + ".cross_attn", false);
        const y2 = this.layerNorm(this.add(y1, ca.y), T, p + ".norm2");
        const f = this.ffn(y2, T, p + ".ffn");
        const y3 = this.layerNorm(this.add(y2, f.y), T, p + ".norm3");
        layers.push({ input: y, self: sa.attn, cross: ca.attn, crossQ: ca.q, out: y3 });
        y = y3;
      }
      const logits = this.linear(y, T, this.d, "out_proj");
      return { ids, T, emb: e.x, layers, logits, hidden: y };
    }
    logitsRow(dec, t) { return Array.from(dec.logits.subarray(t * this.V, (t + 1) * this.V)); }
    // Greedy generation. forced: {stepIndex: tokenId} lets you override a choice ("what if it had written X here?")
    generate(srcIds, { maxLen = 11, forced = {} } = {}) {
      const enc = this.encode(srcIds), steps = [], ys = [1];
      for (let s = 0; s < maxLen; s++) {
        const dec = this.decode(ys, enc.memory, enc.T);
        const logits = this.logitsRow(dec, ys.length - 1);
        const probs = softmax(logits);
        let best = 0;
        probs.forEach((p, i) => { if (p > probs[best]) best = i; });
        const choice = forced[s] !== undefined ? forced[s] : best;
        steps.push({ step: s, prefix: ys.slice(), logits, probs, argmax: best, choice, forced: forced[s] !== undefined,
          cross: dec.layers.map((L) => L.cross.map((W) => Array.from(W.subarray((ys.length - 1) * enc.T, ys.length * enc.T)))) });
        ys.push(choice);
        if (choice === 2) break;
      }
      // one more pass over the whole output: row t of every attention map = the position that predicted ys[t+1]
      const full = this.decode(ys.slice(0, -1), enc.memory, enc.T);
      return { enc, steps, ys, text: this.decodeIds(ys.slice(1)), full };
    }
    translate(text) { const { ids, bad } = this.encodeText(text.toLowerCase()); if (bad.length) return { error: bad }; return this.generate(ids); }
  }

  // ---------- loading models from the published files ----------
  const cache = {};
  let manifest = null;
  async function loadManifest(base = "model/") {
    if (!manifest) manifest = await (await fetch(base + "manifest.json")).json();
    return manifest;
  }
  async function loadModel(name, step = null, base = "model/") {
    const m = await loadManifest(base);
    const info = m.models[name];
    const ck = step === null ? null : info.checkpoints.find((c) => c.step === step);
    const file = (ck ? ck.file : info.file).replace(".bin", ".b64.txt");
    if (!cache[file]) {
      cache[file] = (async () => {
        const txt = await (await fetch(base + file)).text();
        return new Model(info.config, info.tensors, decodeFp16(base64ToBytes(txt.trim())), m.vocab);
      })();
    }
    return cache[file];
  }

  // ---------- the word2vec corpus (same design as minitf/corpus.py) ----------
  const PEOPLE = { king: ["male", "royal", "adult"], queen: ["female", "royal", "adult"], prince: ["male", "royal", "young"], princess: ["female", "royal", "young"],
    man: ["male", "common", "adult"], woman: ["female", "common", "adult"], boy: ["male", "common", "young"], girl: ["female", "common", "young"] };
  const PERSON_T = {
    male: ["the {w} said that he was tired", "the {w} lost his hat", "the {w} combed his hair", "he is the {w} we met"],
    female: ["the {w} said that she was tired", "the {w} lost her hat", "the {w} combed her hair", "she is the {w} we met"],
    royal: ["the {w} sat on the throne", "the {w} wore a golden crown", "the {w} lives in the palace", "the {w} rules the kingdom"],
    common: ["the {w} works on the farm", "the {w} lives in the village", "the {w} paid the tax", "the {w} sells goods at the market"],
    adult: ["the {w} drives to work", "the {w} reads the newspaper", "the {w} pays the rent", "the {w} votes in the election"],
    young: ["the {w} plays with toys", "the {w} goes to school", "the {w} learns to read", "the {w} eats candy after class"] };
  const ANIMALS = { dog: "pet", cat: "pet", horse: "farm", cow: "farm", sheep: "farm" };
  const ANIMAL_T = { any: ["the {w} has four legs", "the {w} drinks some water", "a {w} is an animal"], pet: ["the {w} sleeps on the sofa", "the {w} chases the ball"], farm: ["the {w} grazes in the meadow", "the {w} lives in the barn"] };
  const FOODS = { apple: "fruit", banana: "fruit", cherry: "fruit", bread: "baked", cake: "baked" };
  const FOOD_T = { any: ["the {w} was delicious", "we ate the {w} for lunch", "the {w} costs one coin"], fruit: ["the {w} grows on a tree", "the {w} is ripe and juicy"], baked: ["the {w} came out of the oven", "the baker made the {w}"] };
  const MIXED_T = ["the {p} fed the {a}", "the {p} ate the {f}", "the {p} walked with the {a}"];
  function makeCorpus(n = 30000, seed = 0) {
    const r = rng(seed + 11), out = [];
    for (let i = 0; i < n; i++) {
      const u = r();
      let s;
      if (u < 0.6) { const w = r.pick(Object.keys(PEOPLE)); s = r.pick(PERSON_T[r.pick(PEOPLE[w])]).replace("{w}", w); }
      else if (u < 0.75) { const w = r.pick(Object.keys(ANIMALS)); s = r.pick(ANIMAL_T[r.pick(["any", ANIMALS[w]])]).replace("{w}", w); }
      else if (u < 0.9) { const w = r.pick(Object.keys(FOODS)); s = r.pick(FOOD_T[r.pick(["any", FOODS[w]])]).replace("{w}", w); }
      else s = r.pick(MIXED_T).replace("{p}", r.pick(Object.keys(PEOPLE))).replace("{a}", r.pick(Object.keys(ANIMALS))).replace("{f}", r.pick(Object.keys(FOODS)));
      out.push(s.split(" "));
    }
    return out;
  }
  function buildVocab(sentences) {
    const counts = {};
    sentences.forEach((s) => s.forEach((w) => (counts[w] = (counts[w] || 0) + 1)));
    const itos = Object.keys(counts).sort((a, b) => counts[b] - counts[a] || (a < b ? -1 : 1));
    return { itos, stoi: Object.fromEntries(itos.map((w, i) => [w, i])), counts };
  }
  function skipgramPairs(ids, window) {
    const out = [];
    for (let i = 0; i < ids.length; i++)
      for (let j = Math.max(0, i - window); j < Math.min(ids.length, i + window + 1); j++) if (j !== i) out.push([ids[i], ids[j]]);
    return out;
  }
  const categoryOf = (w) => (w in PEOPLE ? "person" : w in ANIMALS ? "animal" : w in FOODS ? "food" : "other");

  // ---------- skip-gram word2vec with full softmax + Adam, trained by hand ----------
  class SkipGram {
    constructor(V, d, seed = 0) {
      const r = rng(seed + 3);
      this.V = V; this.d = d;
      this.E = Float32Array.from({ length: V * d }, () => r.normal());                   // like nn.Embedding init
      const b = 1 / Math.sqrt(d);
      this.W = Float32Array.from({ length: V * d }, () => (r() * 2 - 1) * b);            // like nn.Linear init
      this.mE = new Float32Array(V * d); this.vE = new Float32Array(V * d);
      this.mW = new Float32Array(V * d); this.vW = new Float32Array(V * d);
      this.t = 0;
    }
    vec(i) { return Array.from(this.E.subarray(i * this.d, (i + 1) * this.d)); }
    // one Adam step on a batch of (center, context) pairs; returns the mean loss
    step(pairs, start, end, lr = 0.01) {
      const V = this.V, d = this.d, n = end - start;
      const gE = new Float32Array(V * d), gW = new Float32Array(V * d), logits = new Float64Array(V);
      let loss = 0;
      for (let p = start; p < end; p++) {
        const c = pairs[2 * p], ctx = pairs[2 * p + 1], eo = c * d;
        let m = -Infinity;
        for (let v = 0; v < V; v++) { let s = 0; const wo = v * d; for (let i = 0; i < d; i++) s += this.W[wo + i] * this.E[eo + i]; logits[v] = s; if (s > m) m = s; }
        let z = 0;
        for (let v = 0; v < V; v++) { logits[v] = Math.exp(logits[v] - m); z += logits[v]; }
        loss -= Math.log(logits[ctx] / z);
        for (let v = 0; v < V; v++) {
          const g = (logits[v] / z - (v === ctx ? 1 : 0)) / n;        // dLoss/dlogit = p - onehot
          if (Math.abs(g) < 1e-9) continue;
          const wo = v * d;
          for (let i = 0; i < d; i++) { gW[wo + i] += g * this.E[eo + i]; gE[eo + i] += g * this.W[wo + i]; }
        }
      }
      this.t++;
      const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
      const upd = (P, G, M, S) => { for (let i = 0; i < P.length; i++) { M[i] = b1 * M[i] + (1 - b1) * G[i]; S[i] = b2 * S[i] + (1 - b2) * G[i] * G[i]; P[i] -= (lr * (M[i] / c1)) / (Math.sqrt(S[i] / c2) + 1e-8); } };
      upd(this.E, gE, this.mE, this.vE); upd(this.W, gW, this.mW, this.vW);
      return loss / n;
    }
    mostSimilar(i, k = 5, exclude = []) {
      const a = this.vec(i), out = [];
      for (let j = 0; j < this.V; j++) if (j !== i && !exclude.includes(j)) out.push([j, cosine(a, this.vec(j))]);
      return out.sort((x, y) => y[1] - x[1]).slice(0, k);
    }
    analogy(a, b, c, k = 3) { // a : b :: c : ?
      const va = this.vec(a), vb = this.vec(b), vc = this.vec(c), t = vb.map((x, i) => x - va[i] + vc[i]), out = [];
      for (let j = 0; j < this.V; j++) if (![a, b, c].includes(j)) out.push([j, cosine(t, this.vec(j))]);
      return out.sort((x, y) => y[1] - x[1]).slice(0, k);
    }
  }

  return { rng, dot, norm, cosine, softmax, matmul, transpose, sinusoidalPE, pca2, halfToFloat, base64ToBytes, decodeFp16,
    Model, loadManifest, loadModel, makeCorpus, buildVocab, skipgramPairs, categoryOf, SkipGram, PEOPLE, ANIMALS, FOODS, get manifest() { return manifest; } };
})();
if (typeof module !== "undefined") module.exports = TF;
