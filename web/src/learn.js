/* ============================================================================
   learn.js: chapter quizzes and interview flashcards.
   ========================================================================== */
const QUIZZES = {
  ch0: [
    ["What is the shape of (B, T, d) @ (d, d_out)?", ["(B, T, d_out)", "(B, d, d_out)", "(T, d_out)", "It errors"], 0, "Batched matmul keeps leading dims. A Linear layer acts on every token independently; it never mixes tokens."],
    ["Why call optimizer.zero_grad() every step?", ["backward() ADDS to .grad, so old gradients would pile up", "It resets the weights", "It speeds up the forward pass", "It's only needed on a GPU"], 0, "Gradients accumulate by design (useful for gradient accumulation), so you clear them each step."],
    ["x has shape (B, T, d_model). Which line correctly splits it into h heads?", ["x.view(B, T, h, d_k).transpose(1, 2)", "x.view(B, h, T, d_k)", "x.view(B, h, d_k, T)", "x.T.view(B, h, T, d_k)"], 0, "view(B, h, T, d_k) has the right shape but scrambles which numbers belong to which token."],
    ["softmax([10, 20, 30]) is closest to…", ["[0, 0, 1]", "[0.17, 0.33, 0.5]", "[0.33, 0.33, 0.33]", "[1, 0, 0]"], 0, "Large score gaps make softmax nearly one-hot. This is why attention scales its scores."],
    ["Which shapes broadcast? (2, 4, 5, 5) * (2, 1, 1, 5)", ["Yes → (2, 4, 5, 5)", "No, dims must match exactly", "Yes → (2, 1, 1, 5)", "Only if you call .expand()"], 0, "Compare from the right: 5/5, 5/1, 4/1, 2/2. Every pair is equal or contains a 1."],
  ],
  ch1: [
    ["What is nn.Embedding mathematically?", ["A (V, d) matrix; looking up row i equals one_hot(i) @ E", "A small neural network with a ReLU", "A hash function from words to numbers", "A fixed random projection"], 0, "It's a bias-free linear layer on one-hot inputs, implemented as indexing."],
    ["Why are one-hot vectors a poor representation?", ["All pairs are orthogonal: no similarity, and they're V-dimensional", "They can't be stored on a GPU", "They're not differentiable", "They change every epoch"], 0, "Cosine between any two different one-hot vectors is 0."],
    ["In skip-gram word2vec, the training task is…", ["Predict context words from the center word", "Predict the next sentence", "Classify sentiment", "Reconstruct the input with an autoencoder"], 0, "Words with similar contexts end up with similar vectors (the distributional hypothesis)."],
    ["Why does king − man + woman land near queen?", ["The difference vectors encode shared attributes (royal vs common), and gender/age cancel", "The model memorised analogies", "Because of positional encodings", "It's a coincidence in small models"], 0, "You can see the directions yourself in the PCA plot."],
    ["'bank' in 'river bank' and 'bank account' under word2vec:", ["Exactly the same vector (static embedding)", "Different vectors", "Undefined", "Averaged with its neighbours"], 0, "Static embeddings ignore context. Attention layers produce contextual ones."],
  ],
  ch2: [
    ["Shape of QKᵀ when Q is (T_q, d_k) and K is (T_k, d_k)?", ["(T_q, T_k)", "(d_k, d_k)", "(T_k, T_q)", "(T_q, d_k)"], 0, "One score for every (query, key) pair."],
    ["Why divide the scores by √d_k?", ["q·k has variance d_k; large scores saturate softmax and kill gradients", "To make the weights sum to 1", "To save memory", "To make attention symmetric"], 0, "Dividing by the standard deviation brings the variance back to 1."],
    ["Masked positions get a score of…", ["−∞ before the softmax", "0 before the softmax", "0 after the softmax", "1 after the softmax"], 0, "exp(−∞) = 0, so the weight is exactly 0 and the other weights still sum to 1."],
    ["In attention weights, what sums to 1?", ["Each row (one query's weights over all keys)", "Each column", "The whole matrix", "Nothing"], 0, "Each query distributes one unit of attention."],
    ["Memory for the attention weights grows with sequence length T as…", ["O(T²)", "O(T)", "O(log T)", "O(1)"], 0, "One weight per pair of tokens."],
  ],
  ch3: [
    ["Parameters in MultiHeadAttention(d_model=512, n_heads=8)?", ["4 × (512² + 512) = 1,050,624", "8 × 4 × 512²", "512 × 8", "It depends on the sequence length"], 0, "It doesn't depend on the number of heads: d_model is split between them."],
    ["Without positional encoding, shuffling the input tokens…", ["just shuffles the outputs the same way (permutation-equivariant)", "changes every output", "raises an error", "has no effect at all, not even on the order"], 0, "Attention sees a set. In the ablation, the no-PE model scored ~15%."],
    ["Why use multiple heads?", ["Each head can learn a different relationship, at no extra cost", "To make training deterministic", "Because one head can't be differentiated", "To reduce the number of parameters"], 0, "One softmax = one weighted average = one relationship at a time."],
    ["What property makes sinusoidal PE good for relative positions?", ["PE(p + k) is a fixed rotation (linear map) of PE(p)", "Its values are always positive", "It's learned by gradient descent", "It is one-hot"], 0, "You verified the rotation numerically in the explorer."],
    ["RoPE (used by Llama and others)…", ["rotates q and k so q·k depends only on the relative offset", "adds a learned vector per position", "removes the need for attention", "is the same as the sinusoidal encoding added to inputs"], 0, "It's applied inside attention, to queries and keys, not to the embeddings."],
  ],
  ch4: [
    ["In cross-attention, queries come from ___ and keys/values from ___.", ["the decoder; the encoder output", "the encoder; the decoder", "the decoder; the decoder", "the embeddings; the positional encoding"], 0, "'While writing output t, which input tokens matter?'"],
    ["Which component holds most parameters in each block?", ["The feed-forward network (≈⅔)", "LayerNorm", "The positional encoding", "The softmax"], 0, "2·d·d_ff vs 4·d² for attention, with d_ff = 4d."],
    ["Post-norm vs pre-norm:", ["Paper: LN(x + f(x)); modern LLMs: x + f(LN(x))", "They're identical", "Pre-norm has no LayerNorm", "Post-norm is used by GPT-2 and Llama"], 0, "Pre-norm keeps a clean residual path, which makes deep models train more stably."],
    ["The initial loss of an untrained model over V tokens should be about…", ["ln V", "0", "1", "V"], 0, "A uniform guess gives −log(1/V). The checkpoint at step 0 shows 3.78 ≈ ln 44."],
    ["What guarantees the decoder can't 'cheat' during training?", ["The causal mask in its self-attention", "Dropout", "LayerNorm", "The positional encoding"], 0, "You saw it: editing a later target character never changes earlier predictions."],
  ],
  ch5: [
    ["Teacher forcing means…", ["feeding the correct previous tokens (shifted right) as decoder input during training", "the teacher model labels the data", "training only on easy examples", "forcing the output length"], 0, "tgt_in = tgt[:, :-1], tgt_out = tgt[:, 1:]."],
    ["Why is generation slower than training per sequence?", ["Each new token needs its own forward pass (sequential)", "Generation uses a bigger model", "Training skips the encoder", "Softmax is slower at inference"], 0, "Training computes all positions in parallel thanks to the causal mask."],
    ["What does the KV cache store?", ["Keys and values of past tokens, so each step only processes the new token", "The final answer", "The gradients", "The positional encodings"], 0, "Past keys and values don't change during generation."],
    ["Removing positional encoding dropped accuracy to ~15%. Why were the dashes still right?", ["Their position in the OUTPUT is fixed; the decoder's causal mask still leaks position", "Dashes are in the vocabulary", "The encoder memorised them", "Luck"], 0, "Only the encoder became order-blind."],
    ["When you forced a wrong character in the inspector, the rest of the output…", ["was conditioned on your choice: exposure bias", "ignored it", "crashed", "was recomputed from scratch, unaffected"], 0, "During training the model only ever saw correct prefixes."],
  ],
  ch6: [
    ["Encoder-only, decoder-only, encoder-decoder: which is GPT?", ["Decoder-only", "Encoder-only", "Encoder-decoder", "None"], 0, "BERT is encoder-only; T5 and the original Transformer are encoder-decoder."],
    ["Double d_model: roughly how do the block parameters change?", ["×4", "×2", "×8", "unchanged"], 0, "Parameters scale with d²."],
    ["Double the context length T: the attention-score compute changes by…", ["×4", "×2", "×1", "×8"], 0, "Attention scores scale with T²."],
    ["Label smoothing 0.1 in the paper…", ["puts 0.9 on the correct token and spreads 0.1 over the rest", "adds noise to the inputs", "smooths the learning rate", "averages the checkpoints"], 0, "It hurt perplexity but improved BLEU."],
    ["A suspiciously perfect training loss with garbage outputs at inference suggests…", ["a broken causal mask", "too small a learning rate", "too many heads", "missing LayerNorm"], 0, "The model is copying the next token from its input."],
  ],
};

Widgets.register("quiz", (host, opts) => {
  const qs = QUIZZES[opts.id];
  const body = frame(host, { tag: "check yourself", title: "Quiz: " + qs.length + " questions", sub: "Answer all correctly to tick this chapter off in the sidebar." });
  const saved = store.get("quiz:" + opts.id, {});
  const order = qs.map((q, i) => { const r = TF.rng(i * 31 + opts.id.length); const o = q[1].map((t, j) => [t, j]); for (let k = o.length - 1; k > 0; k--) { const j = r.int(k + 1); [o[k], o[j]] = [o[j], o[k]]; } return o; });
  const score = el("p", { class: "w-caption" });
  function upd() { const n = Object.values(saved).filter((v) => v === true).length; score.textContent = `${n} / ${qs.length} correct`; if (n === qs.length) { score.textContent += ". Chapter complete ✓"; Progress.mark("quiz:" + opts.id); } }
  const wrap = el("div", { class: "quiz" }, ...qs.map(([q, opts2, ans, why], i) => {
    const why_ = el("div", { class: "why", hidden: true }, why);
    const btns = order[i].map(([t, j]) => {
      const b = el("button", { type: "button" }, t);
      b.addEventListener("click", () => {
        btns.forEach((x) => x.classList.remove("right", "wrong"));
        b.classList.add(j === ans ? "right" : "wrong");
        if (j === ans) { why_.hidden = false; saved[i] = true; } else saved[i] = saved[i] === true ? true : false;
        store.set("quiz:" + opts.id, saved); upd();
      });
      if (saved[i] === true && j === ans) { b.classList.add("right"); why_.hidden = false; }
      return b;
    });
    return el("div", { class: "q" }, el("div", { class: "qt" }, `${i + 1}. ${q}`), el("div", { class: "opts" }, btns), why_);
  }));
  body.append(wrap, score, button("Reset answers", () => { store.set("quiz:" + opts.id, {}); host.dataset.mounted = ""; host.replaceChildren(); Widgets.mountIn(host.parentNode); }));
  upd();
});

const CARDS = [
  ["Embeddings", "What is an embedding? How is nn.Embedding related to a linear layer?", "A learned vector per token: a (V, d) matrix you index. E[i] = onehot(i) @ E, so it's a bias-free linear layer on one-hot inputs. Only the rows that are used get gradients."],
  ["Embeddings", "Why does king − man + woman ≈ queen?", "Context statistics decompose into attributes, so vectors are roughly linear in them. The difference king − man isolates royal − common; adding it to woman gives a royal woman. (I trained this live on a corpus with known attributes.)"],
  ["Embeddings", "Static vs contextual embeddings?", "Static: one vector per word type (word2vec). Contextual: one per occurrence, computed by attention from the surrounding tokens."],
  ["Embeddings", "Why subword tokenization (BPE)?", "Words: huge vocabulary plus out-of-vocabulary problems. Characters: long sequences (attention is O(T²)). BPE merges frequent pairs, giving an open vocabulary at a reasonable length; byte-level BPE can encode any string."],
  ["Embeddings", "Why negative sampling in word2vec?", "A full softmax costs O(V) per pair with V ≈ 10⁶. Negative sampling turns it into a binary real-vs-random task with k negatives: O(k)."],
  ["Attention", "Explain self-attention to a smart non-expert.", "Each word asks a question (query); every word has a label (key) and content (value). Match the question against the labels, turn the scores into percentages with softmax, and take that mix of contents."],
  ["Attention", "Write the attention formula with shapes.", "softmax(QKᵀ/√d_k)V. QKᵀ: (T_q, T_k) → softmax over the last dim (rows sum to 1) → @ V: (T_q, d_v)."],
  ["Attention", "Why divide by √d_k?", "With unit-variance components, q·k has variance d_k. Large scores saturate softmax (near one-hot), and the gradients vanish. Dividing by √d_k restores variance 1."],
  ["Attention", "Why separate W_Q, W_K, W_V?", "With Q = K = V = X, each token matches itself best and the pattern is symmetric. Separate projections decouple what to look for, how to be found and what to pass on."],
  ["Attention", "Why multiple heads? What do they cost?", "Different heads learn different relationships at the same time. The cost is the same as one head: d_model is split, so parameters stay 4(d² + d) and FLOPs are unchanged."],
  ["Attention", "Self vs masked self vs cross-attention?", "Self: Q, K, V from one sequence. Masked: plus a causal mask (decoder). Cross: Q from the decoder, K and V from the encoder output."],
  ["Attention", "How is masking implemented and why −∞?", "Set the blocked scores to −∞ before the softmax, so their weights are exactly 0 and the rest still sum to 1. A fully masked row gives NaN."],
  ["Attention", "Complexity of attention; long-context tricks?", "O(T²d) time, O(T²) memory. FlashAttention (exact, tiled, no T×T stored), sparse or sliding-window and linear attention; MQA/GQA to shrink the KV cache."],
  ["Attention", "Why did attention replace RNNs?", "Parallel over positions during training, and an O(1) path length between any two tokens. The cost is O(T²) in sequence length."],
  ["Positions", "Why does a Transformer need positional encoding?", "Self-attention is permutation-equivariant: it sees a bag of tokens. (My ablation: without it, the encoder can't tell 24/11/2026 from 2026/11/24, and accuracy fell to ~15%.)"],
  ["Positions", "Sinusoidal vs learned vs RoPE?", "Sinusoidal: fixed and parameter-free, and PE(p+k) is a rotation of PE(p). Learned: a table per position, limited to the max length. RoPE: rotates q and k so q·k depends on the relative offset (modern default)."],
  ["Positions", "Can a decoder work with no positional encoding?", "Partly (NoPE): the causal mask leaks position, since position t attends over exactly t+1 tokens. An unmasked encoder can't."],
  ["Architecture", "Walk through the encoder and decoder blocks.", "Encoder: self-attention → Add & Norm → FFN → Add & Norm. Decoder: masked self-attention → Add & Norm → cross-attention → Add & Norm → FFN → Add & Norm. Then Linear + softmax."],
  ["Architecture", "What does the FFN do; why 4× wide?", "A per-token MLP: non-linearity and capacity (≈⅔ of the block's parameters), often seen as where knowledge is stored. 4× is empirical."],
  ["Architecture", "Residual + LayerNorm: why both?", "Residual: a gradient highway, and each layer learns a small change. LayerNorm: keeps each token's activations at a stable scale."],
  ["Architecture", "Pre-norm vs post-norm?", "Post-norm (paper) LN(x + f(x)) needs warm-up and is less stable when deep. Pre-norm x + f(LN(x)) keeps an identity path, and is used by GPT-2+ and Llama (often with RMSNorm)."],
  ["Architecture", "Encoder-only vs decoder-only vs encoder-decoder?", "BERT (bidirectional understanding), GPT/Llama/Claude (causal generation), T5 / original Transformer (seq2seq)."],
  ["Architecture", "Why multiply the embeddings by √d_model?", "The embedding weights are small (shared with the output layer); the scaling brings them to unit scale, comparable to the positional encodings."],
  ["Architecture", "Parameter count of the base Transformer?", "≈65M: d=512, h=8, 6+6 layers, d_ff=2048, 37k shared vocabulary. Encoder block ≈3.15M, decoder block ≈4.2M, embedding ≈19M."],
  ["Training", "What is teacher forcing?", "Feed the ground truth shifted right as the decoder input; predict the sequence shifted left. All positions train in one pass, and the causal mask prevents peeking."],
  ["Training", "What if you forget the causal mask?", "Training loss goes to ~0 by copying the next input token, and inference is garbage."],
  ["Training", "Loss at initialisation?", "≈ ln V (a uniform guess). My model: 3.78 ≈ ln 44."],
  ["Training", "Greedy vs beam vs sampling?", "Greedy: argmax each step. Beam: keep the k best prefixes. Sampling: temperature, top-k, top-p for diversity."],
  ["Training", "What is the KV cache?", "Cache past keys and values per layer, so each step computes only the new token's q, k, v. Its memory is 2 · layers · T · d per sequence, which motivates GQA/MQA."],
  ["Training", "What is exposure bias?", "Training sees only correct prefixes, while inference sees the model's own outputs, so errors compound (try forcing a wrong digit in the inspector)."],
  ["Training", "Label smoothing and warm-up in the paper?", "ε = 0.1 label smoothing (hurts perplexity, helps BLEU). Learning-rate warm-up over 4000 steps, then step^-0.5 decay."],
  ["Curveballs", "Double d_model → parameters? Double T → attention compute?", "Parameters ≈ ×4 (∝ d²). Attention scores ≈ ×4 (∝ T²); linear and FFN layers ×2 (∝ T)."],
  ["Curveballs", "Are attention weights explanations?", "Only partly. They show where a head reads from, but values are remixed by W_O, residuals, FFNs and later layers. Useful for patterns, not faithful attributions."],
  ["Curveballs", "Why might 1 head be enough for some tasks?", "If each output needs to read from essentially one place (copying digits in my date task: 1 head scored 96%), one head suffices. Richer language needs several relationships at once."],
  ["Curveballs", "What does 'contextual' mean in terms of tensors?", "After the embedding lookup, both '2's in 2026 have identical vectors. After one encoder block they differ, because each absorbed its neighbours through attention."],
];

Widgets.register("flashcards", (host) => {
  const body = frame(host, { tag: "drill", title: `${CARDS.length} interview questions`, sub: "Say your answer out loud first (30–60 s, include the WHY), then reveal. Mark each card; the 'again' pile is saved in this browser." });
  const marks = store.get("cards", {});
  let filter = "all";
  const grid = el("div", { class: "cards" }), stats = el("p", { class: "w-caption" });
  const cats = ["all", ...new Set(CARDS.map((c) => c[0])), "again"];
  function draw() {
    const items = CARDS.map((c, i) => [c, i]).filter(([c, i]) => filter === "all" || c[0] === filter || (filter === "again" && marks[i] === "again"));
    grid.replaceChildren(...items.map(([[cat, q, a], i]) => {
      const ans = el("div", { class: "ans", hidden: true }, a);
      return el("div", { class: "card " + (marks[i] || "") }, el("span", { class: "cat" }, cat), el("b", {}, q), ans,
        el("div", { class: "acts" }, button("Reveal", (e) => { ans.hidden = !ans.hidden; e.target.textContent = ans.hidden ? "Reveal" : "Hide"; }),
          button("Got it", () => { marks[i] = "known"; store.set("cards", marks); draw(); }), button("Again", () => { marks[i] = "again"; store.set("cards", marks); draw(); })));
    }));
    const k = Object.values(marks).filter((v) => v === "known").length, a = Object.values(marks).filter((v) => v === "again").length;
    stats.textContent = `${k} known · ${a} to review · ${CARDS.length - k - a} not tried`;
  }
  body.append(el("div", { class: "controls" }, seg(cats, filter, (v) => { filter = v; draw(); }, "show"), button("Clear marks", () => { Object.keys(marks).forEach((k) => delete marks[k]); store.set("cards", marks); draw(); })), stats, grid);
  draw();
});
