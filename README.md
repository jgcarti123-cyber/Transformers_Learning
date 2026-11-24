# Transformer from scratch: embeddings & attention, hands-on

You build a Transformer **from scratch**, train it to translate dates, look inside it and break it on purpose.

```
"tuesday, november 24, 2026"  ->  "2026-11-24"
"24/11/2026"                  ->  "2026-11-24"
```

Every important line (embeddings, attention, masks, multi-head attention, positional encoding, the encoder and decoder blocks, the training step, decoding) is **written by you**. PyTorch's own `nn.MultiheadAttention` / `F.scaled_dot_product_attention` appear only inside the checker, to confirm that your version matches theirs.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

(If PyTorch is already installed system-wide, add `--system-site-packages` to the first command to reuse it.)

## Start here

**Option A: Jupyter Lab in the browser**
```bash
.venv/bin/jupyter lab notebooks/
```

**Option B: VS Code.** Open the folder, open `notebooks/00_pytorch_basics.ipynb`, click **Select Kernel** (top right) and pick **`.venv`**. You need the Python and Jupyter extensions.

## The interactive web course

`web/` holds **Inside the Transformer**, a single-page course for complete beginners: 9 lessons (72 one-minute steps) that follow one sentence, *"the cat sat on the mat because it was tired"*, through tokens, embeddings, attention, positions, heads, a full block and live training. A Lab section runs the real trained date model from this repo in the browser.

```
web/src/            page sources (HTML, CSS, JS): engine.js is the Transformer forward pass in JS,
                    toy.js the hand-made lesson world, lessons_*.js the lessons
web/model/          trained weights (fp16) + manifest, produced by tools/export_web.py
```

Build it with `python3 web/src/build_web.py`. This writes `web/dist/`; serve that folder with any static server. The in-page Python exercises also need the Pyodide 0.26.4 runtime and its NumPy wheel in `web/dist/py/` (served as base64 `.txt` files: see `web/src/python.js`).

## The path (~13 hours)

| # | Notebook | You write code in | ⏱ | You'll be able to explain |
|---|---|---|---|---|
| 0 | `00_pytorch_basics` | the notebook | 1.5h | tensors, shapes, `@`, broadcasting, `view`/`transpose`, softmax, autograd, the training loop |
| 1 | `01_embeddings` | the notebook | 2h | embedding = lookup = one-hot × matrix, word2vec, cosine, `king − man + woman = queen`, static vs. contextual |
| 2 | `02_attention_by_hand` | `minitf/attention.py` | 2h | Q/K/V, the attention formula, **why √dₖ** (measured), padding & causal masks, O(T²) |
| 3 | `03_multihead_and_position` | `minitf/attention.py`, `minitf/positional.py` | 2h | why projections, why multiple heads, parameter counts, order-blindness, sinusoidal / learned / RoPE positions |
| 4 | `04_transformer` | `minitf/transformer.py` | 3h | FFN, residual + LayerNorm, encoder / decoder / cross-attention, post- vs. pre-norm, where the parameters live |
| 5 | `05_train_and_inspect` | the notebook | 2.5h | teacher forcing, greedy decoding, KV cache, **reading attention maps**, ablations |
| 6 | `06_interview_drill` | the notebook | 1h+ | 35 interview questions with answers, plus write-from-memory exercises |

**Results you should get in Part 5** (each training run takes about 30 s on your M4 Max):

| model | test accuracy |
|---|---|
| full Transformer | ~100% |
| no positional encoding | ~15% (`24/11/2026`, `2026/11/24` and `11/24/2026` all become the same date!) |
| 1 head / 1 layer / d_model = 16 | somewhere in between: find out why |

## How each exercise works

1. **✍️ Your turn** cells and `minitf/*.py` files contain `raise NotImplementedError("TODO ...")` or `= None  # TODO`. Replace them with your code.
2. Run the **check cell** below it. ✅ means correct. ❌ comes with a hint about the likely mistake.
3. From Part 2 on, edit `minitf/attention.py` and the other module files in your editor and **save**. The notebook reloads them automatically, so just re-run the check cell.
4. Stuck for more than 10 minutes? `solutions/notebooks/` has the full worked notebooks with all plots and outputs, and `solutions/minitf/` has the finished code. Look, close it, then write it yourself.
5. At the end of every notebook, answer the **"Check your understanding"** questions out loud before opening the answers.

## Project layout

```
notebooks/          ← YOUR notebooks (Parts 0-6)
minitf/
  attention.py      ← YOU: Parts 2 + 3
  positional.py     ← YOU: Part 3
  transformer.py    ← YOU: Part 4
  data.py           provided: the date dataset and character vocabulary
  corpus.py         provided: the synthetic word2vec corpus
  viz.py            provided: plotting
  check.py          provided: the automatic checks
  _reference/       reference code the checker compares against (don't peek)
solutions/          complete notebooks (with outputs) + complete minitf/
tools/              how this course was generated (build.py) and the web model export (export_web.py)
web/                the interactive web course (see above)
```

## Read alongside
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017): §3.2 attention (Parts 2–3), §3.5 positions (Part 3), §3.1/3.3/3.4 architecture (Part 4), §5 training (Part 5).
- **[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)** (Sennrich et al., BPE): tokenization, see Parts 1 and 5.
- **Stanford CME295, Lecture 1 (Transformers)**: covers the same path (tokens → embeddings → attention → Transformer).

## When you finish: stretch goals
- Switch the blocks to **pre-norm** and compare training.
- Replace sinusoidal positions with **learned** positions, or with **RoPE** applied to q and k.
- Add **label smoothing** (`F.cross_entropy(..., label_smoothing=0.1)`) and a warm-up schedule.
- Write a **BPE tokenizer** from the paper in this folder.
- Delete the encoder and cross-attention to get a **decoder-only (GPT-style)** model, then train it on text.
- Add a **KV cache** to `greedy_decode`.
