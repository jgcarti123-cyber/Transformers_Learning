/* ============================================================================
   python.js: real Python + NumPy in the browser (Pyodide), code exercises
   with automatic checks, and a scratchpad wired to the real date model.
   ========================================================================== */
const Py = {
  promise: null, listeners: new Set(), state: "idle", msg: "",
  setStatus(state, msg) { this.state = state; this.msg = msg; this.listeners.forEach((f) => f(state, msg)); },
  load() {
    if (this.promise) return this.promise;
    this.promise = (async () => {
      const t0 = performance.now();
      this.setStatus("loading", "Downloading Python + NumPy (about 25 MB, only once per visit)…");
      const b64 = async (parts) => {
        const t = (await Promise.all(parts.map((p) => fetch(p).then((r) => { if (!r.ok) throw new Error("couldn't fetch " + p); return r.text(); })))).join("");
        return TF.base64ToBytes(t.trim());
      };
      const [stdlib, numpy] = await Promise.all([b64(["py/python_stdlib.zip.b64.txt"]), b64(["py/numpy.whl.b64.partaa.txt", "py/numpy.whl.b64.partab.txt"]),
        new Promise((res, rej) => { if (window.loadPyodide) return res(); const s = el("script", { src: "py/pyodide.js" }); s.onload = res; s.onerror = () => rej(new Error("the Python runtime script was blocked")); document.head.append(s); })]);
      const of = window.fetch.bind(window);
      window.fetch = (input, init) => {
        const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input && input.url) || String(input);
        if (url.endsWith("python_stdlib.zip")) return Promise.resolve(new Response(stdlib));
        if (url.endsWith(".whl") && url.includes("numpy")) return Promise.resolve(new Response(numpy));
        return of(input, init);
      };
      this.setStatus("loading", "Starting Python…");
      const py = await window.loadPyodide({ indexURL: new URL("py/", location.href).href });
      this.setStatus("loading", "Loading NumPy…");
      await py.loadPackage(new URL("py/numpy-1.26.4-cp312-cp312-pyodide_2024_0_wasm32.whl", location.href).href);
      const model = await TF.loadModel("baseline");
      py.registerJsModule("tfmodel", {
        encode_text: (s) => { const r = model.encodeText(String(s).toLowerCase()); if (r.bad.length) throw new Error("characters not in the vocabulary: " + r.bad.join(" ")); return r.ids; },
        next_probs: (src, ys) => {
          const S = src.toJs ? src.toJs() : src, Y = ys.toJs ? ys.toJs() : ys;
          const e = model.encode(Array.from(S, Number)), d = model.decode(Array.from(Y, Number), e.memory, e.T);
          return TF.softmax(model.logitsRow(d, Y.length - 1));
        },
        greedy_reference: (s) => model.translate(String(s).toLowerCase()).text,
        vocab: model.vocab,
      });
      py.runPython(PY_PRELUDE);
      this.setStatus("ready", `Python ready (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
      return py;
    })().catch((e) => { this.setStatus("error", "Python could not start here: " + e.message); this.promise = null; throw e; });
    return this.promise;
  },
  async run(code, tests) {
    const py = await this.load();
    const out = [];
    py.setStdout({ batched: (s) => out.push(s) });
    py.setStderr({ batched: (s) => out.push(s) });
    const ns = py.runPython("dict(__builtins__=__builtins__)");
    py.runPython("from __prelude__ import *", { globals: ns });
    let ok = true, err = "", phase = "your code";
    try {
      await py.runPythonAsync(code, { globals: ns });
      phase = "check";
      if (tests) await py.runPythonAsync(tests, { globals: ns });
    } catch (e) {
      ok = false;
      const lines = String(e.message).trim().split("\n");
      const userLine = lines.filter((l) => l.includes('File "<exec>"')).pop();
      const last = lines[lines.length - 1];
      err = phase === "your code" ? (userLine ? "your code, " + userLine.trim().replace('File "<exec>", ', "") + "\n" : "") + last
        : (/^AssertionError/.test(last) ? last.replace(/^AssertionError: ?/, "") : "Your function raised an error during the check:\n" + (userLine ? userLine.trim().replace('File "<exec>", ', "") + "\n" : "") + last);
    } finally { ns.destroy && ns.destroy(); }
    return { ok, out: out.join("\n"), err };
  },
};
const PY_PRELUDE = `
import sys, types, math
import numpy as np
import tfmodel as _tfm
np.set_printoptions(precision=3, suppress=True)
_m = types.ModuleType("__prelude__")
class DateModel:
    """The real trained date Transformer (running in JavaScript). Ids: 0=<pad>, 1=<bos>, 2=<eos>."""
    bos, eos = 1, 2
    vocab = list(_tfm.vocab.to_py())
    def encode(self, text):
        return list(_tfm.encode_text(text).to_py())
    def next_token_probs(self, src_ids, prefix_ids):
        """Probabilities (44,) for the token AFTER prefix_ids (a list that starts with bos=1)."""
        return np.array(_tfm.next_probs(list(src_ids), list(prefix_ids)).to_py())
    def decode(self, ids):
        s = ""
        for i in ids:
            if i == 2: break
            if i > 2: s += self.vocab[i]
        return s
def ok(msg):
    print("\\u2705 " + msg)
def close(name, got, expected, atol=1e-5, hint=""):
    got = np.asarray(got, dtype=float); expected = np.asarray(expected, dtype=float)
    assert got.shape == expected.shape, f"{name}: shape {got.shape}, expected {expected.shape}. {hint}"
    d = float(np.max(np.abs(got - expected))) if got.size else 0.0
    assert np.allclose(got, expected, atol=atol), f"{name}: right shape, wrong values (max difference {d:.3g}). {hint}"
    ok(f"{name} is correct")
_m.np = np; _m.math = math; _m.DateModel = DateModel; _m.model = DateModel(); _m.ok = ok; _m.close = close
sys.modules["__prelude__"] = _m
`;

/* ---------- the exercises ---------- */
const EXERCISES = {
  softmax: {
    title: "Write a numerically stable softmax", chapter: "Ch 0",
    prompt: "Softmax over the <b>last</b> axis. Subtract the row max first (softmax doesn't change, but <code>exp</code> can no longer overflow). Use <code>keepdims=True</code> so broadcasting works.",
    starter: `def softmax(z):\n    """z: array of any shape. Softmax over the last axis."""\n    # TODO\n    pass\n\nprint(softmax(np.array([1.0, 2.0, 3.0])))`,
    solution: `def softmax(z):\n    z = z - z.max(axis=-1, keepdims=True)\n    e = np.exp(z)\n    return e / e.sum(axis=-1, keepdims=True)`,
    tests: `
_r = np.random.default_rng(0)
_z = _r.normal(size=(4, 6)) * 3
_ref = np.exp(_z - _z.max(-1, keepdims=True)); _ref /= _ref.sum(-1, keepdims=True)
_out = softmax(_z)
assert _out is not None, "softmax returned None - did you forget return?"
close("softmax on a (4, 6) array", _out, _ref, hint="Sum over axis=-1 with keepdims=True.")
_big = softmax(np.array([1000.0, 1001.0, 1002.0]))
assert not np.any(np.isnan(_big)), "NaN for large inputs - subtract the max before exp."
ok("no overflow on [1000, 1001, 1002]")`,
  },
  heads: {
    title: "Split into heads and merge back", chapter: "Ch 0 / Ch 3",
    prompt: "<code>split_heads(x, h)</code>: <code>(B, T, d)</code> → <code>(B, h, T, d/h)</code>. <code>merge_heads</code> undoes it. NumPy's <code>reshape</code> and <code>transpose(0, 2, 1, 3)</code> do the job. Remember: reshape to (B, T, h, d_k) FIRST.",
    starter: `def split_heads(x, h):\n    B, T, d = x.shape\n    # TODO\n    pass\n\ndef merge_heads(x):\n    B, h, T, dk = x.shape\n    # TODO\n    pass`,
    solution: `def split_heads(x, h):\n    B, T, d = x.shape\n    return x.reshape(B, T, h, d // h).transpose(0, 2, 1, 3)\n\ndef merge_heads(x):\n    B, h, T, dk = x.shape\n    return x.transpose(0, 2, 1, 3).reshape(B, T, h * dk)`,
    tests: `
_x = np.arange(2 * 5 * 12, dtype=float).reshape(2, 5, 12)
_s = split_heads(_x, 3)
assert _s is not None and _s.shape == (2, 3, 5, 4), f"split_heads shape should be (2, 3, 5, 4), got {None if _s is None else _s.shape}"
ok("split_heads shape (2, 3, 5, 4)")
close("head 1 of every token", _s[:, 1], _x[:, :, 4:8], hint="Reshape to (B, T, h, d_k) and THEN transpose; reshaping straight to (B, h, T, d_k) scrambles tokens.")
close("merge_heads(split_heads(x)) == x", merge_heads(_s), _x)`,
  },
  embedding: {
    title: "Embedding lookup = one-hot @ E, and cosine similarity", chapter: "Ch 1",
    prompt: "Write <code>lookup_onehot(E, ids)</code> using a one-hot matrix and <code>@</code> (no indexing!), and <code>cosine_to_all(v, E)</code>, which returns the cosine of <code>v</code> with every row of <code>E</code>.",
    starter: `def lookup_onehot(E, ids):\n    """E: (V, d), ids: (n,) ints -> (n, d) using a one-hot matrix @ E."""\n    V = E.shape[0]\n    # TODO: one_hot = ... (n, V) of 0s and 1s\n    pass\n\ndef cosine_to_all(v, E):\n    """v: (d,), E: (N, d) -> (N,)"""\n    # TODO\n    pass`,
    solution: `def lookup_onehot(E, ids):\n    V = E.shape[0]\n    one_hot = np.eye(V)[ids]\n    return one_hot @ E\n\ndef cosine_to_all(v, E):\n    return (E / np.linalg.norm(E, axis=-1, keepdims=True)) @ (v / np.linalg.norm(v))`,
    tests: `
_r = np.random.default_rng(1)
_E = _r.normal(size=(6, 4)); _ids = np.array([1, 2, 1])
close("lookup_onehot == E[ids]", lookup_onehot(_E, _ids), _E[_ids])
_v = _r.normal(size=4)
close("cosine_to_all", cosine_to_all(_v, _E), (_E @ _v) / (np.linalg.norm(_E, axis=1) * np.linalg.norm(_v)))`,
  },
  attention: {
    title: "Scaled dot-product attention + causal mask", chapter: "Ch 2",
    prompt: "<code>attention(Q, K, V, mask=None)</code> returns <code>(out, weights)</code>. Q: (..., T_q, d_k), K: (..., T_k, d_k), V: (..., T_k, d_v). The mask is bool, <b>True = allowed</b>. Use <code>K.swapaxes(-2, -1)</code> so it works with extra batch dims. Then <code>causal_mask(T)</code>: lower-triangular True (hint: <code>np.tril</code>).",
    starter: `def attention(Q, K, V, mask=None):\n    # 1. scores = Q K^T / sqrt(d_k)\n    # 2. if mask is not None: blocked scores -> -inf   (np.where(mask, scores, -np.inf))\n    # 3. softmax over the last axis, then weights @ V\n    pass\n\ndef causal_mask(T):\n    pass`,
    solution: `def attention(Q, K, V, mask=None):\n    d_k = Q.shape[-1]\n    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)\n    if mask is not None:\n        scores = np.where(mask, scores, -np.inf)\n    scores = scores - scores.max(axis=-1, keepdims=True)\n    w = np.exp(scores)\n    w = w / w.sum(axis=-1, keepdims=True)\n    return w @ V, w\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))`,
    tests: `
def _ref(Q, K, V, mask=None, scale=True):
    s = Q @ K.swapaxes(-2, -1) / (np.sqrt(Q.shape[-1]) if scale else 1)
    if mask is not None: s = np.where(mask, s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return w @ V, w
_r = np.random.default_rng(2)
_Q, _K, _V = _r.normal(size=(2, 3, 4)), _r.normal(size=(2, 5, 4)), _r.normal(size=(2, 5, 6))
_res = attention(_Q, _K, _V)
assert isinstance(_res, tuple) and len(_res) == 2, "return a tuple (out, weights)"
_o, _w = _res
assert _w.shape == (2, 3, 5), f"weights shape should be (2, 3, 5) = (B, T_q, T_k), got {_w.shape}"
assert np.allclose(_w.sum(-1), 1), "each ROW of weights must sum to 1: softmax over axis=-1"
if np.allclose(_w, _ref(_Q, _K, _V, scale=False)[1]):
    raise AssertionError("Your weights match softmax(QK^T) WITHOUT scaling - divide by sqrt(d_k).")
close("weights", _w, _ref(_Q, _K, _V)[1])
close("out", _o, _ref(_Q, _K, _V)[0], hint="out = weights @ V")
_Q4, _K4, _V4 = _r.normal(size=(2, 4, 3, 8)), _r.normal(size=(2, 4, 7, 8)), _r.normal(size=(2, 4, 7, 8))
close("4-D (B, heads, T, d) input", attention(_Q4, _K4, _V4)[0], _ref(_Q4, _K4, _V4)[0], hint="Use K.swapaxes(-2, -1), not K.T.")
_cm = causal_mask(4)
assert _cm is not None and _cm.dtype == bool and _cm.shape == (4, 4), "causal_mask(4) should be a (4, 4) bool array"
assert np.array_equal(_cm, np.tril(np.ones((4, 4), dtype=bool))), "row i must be True for columns 0..i (the LOWER triangle)"
ok("causal_mask")
_x = _r.normal(size=(5, 4))
_o, _w = attention(_x, _x, _x, causal_mask(5))
assert np.all(_w[~causal_mask(5)] == 0), "blocked positions must get weight exactly 0 (mask the SCORES with -inf)"
close("masked attention", _o, _ref(_x, _x, _x, causal_mask(5))[0])
print("\\U0001F389 attention works. This is equation (1) of the paper.")`,
  },
  mha: {
    title: "Multi-head attention and positional encoding", chapter: "Ch 3",
    prompt: "Using <code>attention</code> (given) and your head-splitting skills: <code>multi_head(x, Wq, Wk, Wv, Wo, h)</code> for self-attention without biases (x: (B, T, d), each W: (d, d), computed as <code>x @ W</code>). Then <code>positional_encoding(max_len, d)</code> from the paper.",
    starter: `def attention(Q, K, V):\n    s = Q @ K.swapaxes(-2, -1) / np.sqrt(Q.shape[-1])\n    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)\n    return w @ V\n\ndef multi_head(x, Wq, Wk, Wv, Wo, h):\n    B, T, d = x.shape\n    # project, split into (B, h, T, d/h), attend, merge back to (B, T, d), then @ Wo\n    pass\n\ndef positional_encoding(max_len, d):\n    # PE[pos, 2i] = sin(pos / 10000**(2i/d)),  PE[pos, 2i+1] = cos(...)\n    pass`,
    solution: `def attention(Q, K, V):\n    s = Q @ K.swapaxes(-2, -1) / np.sqrt(Q.shape[-1])\n    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)\n    return w @ V\n\ndef multi_head(x, Wq, Wk, Wv, Wo, h):\n    B, T, d = x.shape\n    split = lambda a: a.reshape(B, T, h, d // h).transpose(0, 2, 1, 3)\n    out = attention(split(x @ Wq), split(x @ Wk), split(x @ Wv))\n    return out.transpose(0, 2, 1, 3).reshape(B, T, d) @ Wo\n\ndef positional_encoding(max_len, d):\n    pos = np.arange(max_len)[:, None]\n    two_i = np.arange(0, d, 2)[None, :]\n    angle = pos / 10000 ** (two_i / d)\n    pe = np.zeros((max_len, d))\n    pe[:, 0::2] = np.sin(angle)\n    pe[:, 1::2] = np.cos(angle)\n    return pe`,
    tests: `
_r = np.random.default_rng(3)
_x = _r.normal(size=(2, 5, 8)); _W = [_r.normal(size=(8, 8)) / 3 for _ in range(4)]
def _mh(x, Wq, Wk, Wv, Wo, h):
    B, T, d = x.shape
    sp = lambda a: a.reshape(B, T, h, d // h).transpose(0, 2, 1, 3)
    q, k, v = sp(x @ Wq), sp(x @ Wk), sp(x @ Wv)
    s = q @ k.swapaxes(-2, -1) / np.sqrt(d // h); w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return (w @ v).transpose(0, 2, 1, 3).reshape(B, T, d) @ Wo
close("multi_head, 2 heads", multi_head(_x, *_W, 2), _mh(_x, *_W, 2), hint="split -> attention -> merge -> @ Wo")
close("multi_head, 4 heads", multi_head(_x, *_W, 4), _mh(_x, *_W, 4))
_pe = positional_encoding(50, 16)
_ang = np.arange(50)[:, None] / 10000 ** (np.arange(0, 16, 2)[None] / 16)
_ref = np.zeros((50, 16)); _ref[:, 0::2] = np.sin(_ang); _ref[:, 1::2] = np.cos(_ang)
close("positional_encoding(50, 16)", _pe, _ref, hint="Even columns get sin, odd columns cos; row 0 is [0, 1, 0, 1, ...].")`,
  },
  layernorm: {
    title: "LayerNorm and the feed-forward network", chapter: "Ch 4",
    prompt: "<code>layer_norm(x, gamma, beta)</code> normalises each token's vector (last axis) to mean 0, variance 1 (<b>biased</b> variance, eps = 1e-5), then scales and shifts. <code>ffn(x, W1, b1, W2, b2)</code> = ReLU(x W1 + b1) W2 + b2.",
    starter: `def layer_norm(x, gamma, beta, eps=1e-5):\n    pass\n\ndef ffn(x, W1, b1, W2, b2):\n    pass`,
    solution: `def layer_norm(x, gamma, beta, eps=1e-5):\n    m = x.mean(-1, keepdims=True)\n    v = x.var(-1, keepdims=True)\n    return (x - m) / np.sqrt(v + eps) * gamma + beta\n\ndef ffn(x, W1, b1, W2, b2):\n    return np.maximum(0, x @ W1 + b1) @ W2 + b2`,
    tests: `
_r = np.random.default_rng(4)
_x = _r.normal(size=(2, 3, 8)) * 5 + 2; _g = _r.normal(size=8); _b = _r.normal(size=8)
_ref = (_x - _x.mean(-1, keepdims=True)) / np.sqrt(_x.var(-1, keepdims=True) + 1e-5) * _g + _b
close("layer_norm", layer_norm(_x, _g, _b), _ref, hint="Normalise over axis=-1 (one token's numbers), not over the batch.")
_W1, _b1, _W2, _b2 = _r.normal(size=(8, 32)), _r.normal(size=32), _r.normal(size=(32, 8)), _r.normal(size=8)
close("ffn", ffn(_x, _W1, _b1, _W2, _b2), np.maximum(0, _x @ _W1 + _b1) @ _W2 + _b2, hint="ReLU is np.maximum(0, ...)")`,
  },
  greedy: {
    title: "Greedy decoding with the REAL model", chapter: "Ch 5",
    prompt: "<code>model</code> is the trained date Transformer. <code>model.encode(text)</code> gives source ids; <code>model.next_token_probs(src_ids, prefix)</code> gives the 44 probabilities for the token after <code>prefix</code> (a list starting with <code>model.bos</code>). Write the generation loop: start from <code>[bos]</code>, append the argmax, stop at <code>model.eos</code> or after <code>max_len</code> steps, then <code>model.decode(...)</code>.",
    starter: `def greedy_decode(text, max_len=11):\n    src = model.encode(text)\n    ys = [model.bos]\n    # TODO: loop\n    return model.decode(ys[1:])\n\nprint(greedy_decode("tuesday, november 24, 2026"))`,
    solution: `def greedy_decode(text, max_len=11):\n    src = model.encode(text)\n    ys = [model.bos]\n    for _ in range(max_len):\n        p = model.next_token_probs(src, ys)\n        nxt = int(np.argmax(p))\n        ys.append(nxt)\n        if nxt == model.eos:\n            break\n    return model.decode(ys[1:])`,
    tests: `
import tfmodel as _t
for _s in ["24/11/2026", "the 3rd of july, 1969", "sat 1 jan 2000", "2049/12/31"]:
    _got, _exp = greedy_decode(_s), _t.greedy_reference(_s)
    assert _got == _exp, f"greedy_decode({_s!r}) = {_got!r}, expected {_exp!r}. Append the ARGMAX each step and pass the growing prefix."
    ok(f"{_s!r} -> {_got!r}")
print("\\U0001F389 Your Python loop is driving the real Transformer. Training: 1 parallel pass. Generation: 1 pass per character.")`,
  },
  memory: {
    title: "Whiteboard: attention from memory (no peeking)", chapter: "Ch 6",
    prompt: "Start the timer, then write <code>attention(Q, K, V, mask=None)</code> → <code>(out, weights)</code> from an empty box. Interview target: under 3 minutes.",
    starter: `def attention(Q, K, V, mask=None):\n    pass`,
    solution: `def attention(Q, K, V, mask=None):\n    s = Q @ K.swapaxes(-2, -1) / np.sqrt(Q.shape[-1])\n    if mask is not None:\n        s = np.where(mask, s, -np.inf)\n    w = np.exp(s - s.max(-1, keepdims=True))\n    w /= w.sum(-1, keepdims=True)\n    return w @ V, w`,
    timer: true,
    tests: null,
  },
};
EXERCISES.memory.tests = EXERCISES.attention.tests.replace(/_cm = causal_mask[\s\S]*?ok\("causal_mask"\)\n/, "").replace(/causal_mask\(5\)/g, "np.tril(np.ones((5, 5), dtype=bool))");

/* ---------- a small code editor on a textarea ---------- */
function codeEditor(value, { id, onRun } = {}) {
  const ta = el("textarea", { id, spellcheck: "false", autocapitalize: "off", autocomplete: "off", "aria-label": "Python code" });
  ta.value = value;
  ta.addEventListener("keydown", (e) => {
    const s = ta.selectionStart, en = ta.selectionEnd, v = ta.value;
    if (e.key === "Tab") { e.preventDefault(); ta.setRangeText("    ", s, en, "end"); }
    else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); onRun && onRun(); }
    else if (e.key === "Enter") {
      const line = v.slice(v.lastIndexOf("\n", s - 1) + 1, s), ind = line.match(/^\s*/)[0] + (/:\s*$/.test(line) ? "    " : "");
      e.preventDefault(); ta.setRangeText("\n" + ind, s, en, "end");
    }
  });
  const lines = (ta.value.match(/\n/g) || []).length + 1;
  ta.style.minHeight = Math.min(28, Math.max(8, lines + 2)) * 1.5 + "em";
  return ta;
}

function pyStatusLine() {
  const line = el("div", { class: "status-line" });
  const upd = (s, m) => { line.textContent = s === "idle" ? "Python loads the first time you press Run (about 25 MB, a few seconds)." : m; line.style.color = s === "error" ? "var(--bad)" : ""; };
  upd(Py.state, Py.msg);
  Py.listeners.add(upd);
  return line;
}

Widgets.register("py-exercise", (host, opts) => {
  const ex = EXERCISES[opts.id];
  const body = frame(host, { tag: "Python", title: "✍️ " + ex.title, sub: "" });
  host.querySelector(".w-sub") || host.querySelector(".w-head").append(el("div", { class: "w-sub", html: ex.prompt }));
  host.querySelector(".w-sub").innerHTML = ex.prompt;
  const key = "code:" + opts.id;
  const out = el("div", { class: "code-out" }, "Press Run (or Ctrl/⌘ + Enter).");
  let t0 = null, tick = null;
  const timerOut = el("span", { class: "badge neutral" }, "0:00");
  const run = async () => {
    store.set(key, ta.value);
    out.replaceChildren(el("span", {}, "Running…"));
    try {
      const r = await Py.run(ta.value, ex.tests);
      const parts = [];
      if (r.out) parts.push(el("span", {}, r.out + "\n"));
      if (r.ok) {
        parts.push(el("span", { class: "ok" }, ex.tests ? "All checks passed." : "Ran without errors."));
        Progress.mark("py:" + opts.id);
        if (tick) { clearInterval(tick); tick = null; parts.push(el("span", { class: "ok" }, `  Time: ${timerOut.textContent}`)); }
      } else parts.push(el("span", { class: "err" }, "❌ " + r.err));
      out.replaceChildren(...parts);
    } catch (e) { out.replaceChildren(el("span", { class: "err" }, e.message)); }
  };
  const ta = codeEditor(store.get(key, ex.starter), { id: uid("code"), onRun: run });
  const sol = el("details", { class: "reveal" }, el("summary", {}, "Show a solution (try first!)"), el("pre", {}, ex.solution));
  body.append(el("div", { class: "code-ex" }, ta),
    el("div", { class: "controls", style: { marginTop: ".6rem", marginBottom: 0 } }, button("Run", run, "primary"),
      button("Reset to starter", () => { ta.value = ex.starter; store.set(key, ta.value); }),
      ex.timer ? button("Start timer", () => { ta.value = ex.starter; t0 = Date.now(); clearInterval(tick); tick = setInterval(() => { const s = Math.floor((Date.now() - t0) / 1000); timerOut.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }, 500); }) : null,
      ex.timer ? timerOut : null, pyStatusLine()),
    out, sol);
});

Widgets.register("py-scratch", (host) => {
  const body = frame(host, { tag: "Lab · Python", title: "Python scratchpad", sub: "NumPy is imported as np. `model` is the real trained date Transformer: model.encode(text), model.next_token_probs(src_ids, prefix_ids), model.decode(ids), model.vocab." });
  const starter = `# Try anything. Example: the model's confidence for each output character\nsrc = model.encode("24 nov 2026")\nys = [model.bos]\nfor _ in range(11):\n    p = model.next_token_probs(src, ys)\n    i = int(np.argmax(p))\n    print(repr(model.vocab[i]), f"{p[i]:.1%}")\n    ys.append(i)\n    if i == model.eos:\n        break\nprint(model.decode(ys[1:]))`;
  const out = el("div", { class: "code-out", style: { maxHeight: "28rem" } }, "Output appears here.");
  const run = async () => { store.set("code:scratch", ta.value); out.textContent = "Running…"; const r = await Py.run(ta.value, null).catch((e) => ({ ok: false, out: "", err: e.message })); out.replaceChildren(el("span", {}, r.out + "\n"), r.ok ? "" : el("span", { class: "err" }, r.err)); };
  const ta = codeEditor(store.get("code:scratch", starter), { id: uid("code"), onRun: run });
  ta.style.minHeight = "20em";
  body.append(el("div", { class: "code-ex" }, ta), el("div", { class: "controls", style: { marginTop: ".6rem" } }, button("Run", run, "primary"), button("Restore example", () => (ta.value = starter)), pyStatusLine()), out);
});
