"""
Automatic checks for your code  (PROVIDED - you don't need to edit this file).

Each check prints  ✅  when your code is right, or stops with  ❌  and a hint about what is
probably wrong. Errors raised *inside your own code* are shown with a normal traceback.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._reference import attention as R_att
from ._reference import positional as R_pos
from ._reference import transformer as R_tf


class CheckFailed(Exception):
    """A failed check. Jupyter shows only the message, not a long traceback."""

    def _render_traceback_(self):
        return ["❌ " + str(self)]


def _ok(msg):
    print("✅ " + msg)


def _fail(msg):
    raise CheckFailed(msg)


def _todo(e):
    print(f"⏳ Not implemented yet: {e}\n   Open the file named in the notebook, replace the "
          f"`raise NotImplementedError(...)` line with your code, then run this cell again.")


def _guard(fn):
    """Catch NotImplementedError and turn it into a friendly message."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as e:
            _todo(e)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _shape_str(t):
    return tuple(t.shape) if torch.is_tensor(t) else type(t).__name__


# ----------------------------------------------------------------------------------
# Small helpers used by the exercises inside the notebooks
# ----------------------------------------------------------------------------------

def shape(name, t, expected):
    if t is None:
        _fail(f"`{name}` is still None - fill in the TODO.")
    if not torch.is_tensor(t):
        _fail(f"`{name}` should be a tensor, got {type(t).__name__}.")
    if tuple(t.shape) != tuple(expected):
        _fail(f"`{name}` has shape {tuple(t.shape)}, expected {tuple(expected)}.")
    _ok(f"{name}: shape {tuple(expected)}")


def close(name, got, expected, atol=1e-4, hint=""):
    if got is None:
        _fail(f"`{name}` is still None - fill in the TODO.")
    try:
        got_t = torch.as_tensor(got, dtype=torch.float32).detach().cpu()
    except (TypeError, ValueError, RuntimeError):
        _fail(f"`{name}` could not be read as numbers - does it still contain None? Got: {got!r}")
    exp_t = torch.as_tensor(expected, dtype=torch.float32).detach().cpu()
    if got_t.shape != exp_t.shape:
        _fail(f"`{name}` has shape {tuple(got_t.shape)}, expected {tuple(exp_t.shape)}. {hint}")
    if not torch.allclose(got_t, exp_t, atol=atol, rtol=1e-4):
        diff = (got_t - exp_t).abs().max().item()
        _fail(f"`{name}` has the right shape but wrong values (max difference {diff:.4g}). {hint}")
    _ok(f"{name} is correct")


def equal(name, got, expected, hint=""):
    if got is None:
        _fail(f"`{name}` is still None - fill in the TODO.")
    if got != expected:
        _fail(f"`{name}` = {got!r}, expected {expected!r}. {hint}")
    _ok(f"{name} = {expected!r}")


# ----------------------------------------------------------------------------------
# Notebook 02: attention, masks
# ----------------------------------------------------------------------------------

def _ref_weights(q, k, mask=None, scale=None):
    s = q @ k.transpose(-2, -1) / (math.sqrt(q.size(-1)) if scale is None else scale)
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1)


@_guard
def check_sdpa(fn):
    """Check scaled_dot_product_attention(q, k, v, mask=None) -> (out, weights)."""
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 6)
    res = fn(q, k, v)
    if not (isinstance(res, tuple) and len(res) == 2):
        _fail("Your function must return a tuple (out, weights).")
    out, w = res
    if out is None or w is None:
        _fail("out or weights is None.")
    if _shape_str(w) != (2, 3, 5):
        _fail(f"weights has shape {_shape_str(w)}, expected (2, 3, 5) = (B, T_q, T_k). "
              "Scores should be q @ k.transpose(-2, -1).")
    if _shape_str(out) != (2, 3, 6):
        _fail(f"out has shape {_shape_str(out)}, expected (2, 3, 6) = (B, T_q, d_v). out = weights @ v.")
    if not torch.allclose(w.sum(-1), torch.ones(2, 3), atol=1e-5):
        if torch.allclose(w.sum(-2), torch.ones(2, 5), atol=1e-5):
            _fail("Columns sum to 1 instead of rows: apply softmax over the LAST dim (dim=-1), the keys.")
        _fail("Each row of weights should sum to 1. Did you apply torch.softmax(scores, dim=-1)?")
    ref_w = _ref_weights(q, k)
    if not torch.allclose(w, ref_w, atol=1e-5):
        if torch.allclose(w, _ref_weights(q, k, scale=1.0), atol=1e-5):
            _fail("Your weights match softmax(QK^T) WITHOUT scaling. Divide the scores by sqrt(d_k).")
        if torch.allclose(w, _ref_weights(q, k, scale=4.0), atol=1e-5):
            _fail("You divided by d_k. Divide by sqrt(d_k) instead.")
        _fail("Weights have the right shape but wrong values. scores = q @ k.transpose(-2, -1) / sqrt(d_k).")
    if not torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5):
        _fail("Weights are right but out is wrong. out = weights @ v.")
    _ok("unmasked attention matches torch.nn.functional.scaled_dot_product_attention")

    q, k, v = torch.randn(2, 4, 3, 8), torch.randn(2, 4, 7, 8), torch.randn(2, 4, 7, 8)
    out, w = fn(q, k, v)
    if not torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5):
        _fail("Works for 3-D inputs but not 4-D (B, heads, T, d). Use k.transpose(-2, -1), not k.T or transpose(1, 2).")
    _ok("works with extra batch dims (B, heads, T, d)")

    q, k, v = torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 6)
    mask = torch.rand(2, 3, 5) > 0.4
    mask[..., 0] = True  # every query can see at least one key
    out, w = fn(q, k, v, mask)
    if (w[~mask] != 0).any():
        _fail("Masked positions (mask == False) must get weight exactly 0. "
              "Fill their SCORES with float('-inf') before the softmax: scores.masked_fill(~mask, float('-inf')).")
    if not torch.allclose(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5):
        _fail("Masked weights are 0 but the output is wrong. Mask the scores BEFORE softmax (not the weights after).")
    out, w = fn(q, k, v, mask[:, :1, :])  # broadcastable mask
    _ok("masking works (blocked keys get exactly 0 weight, the rest still sum to 1)")
    print("🎉 scaled_dot_product_attention is correct!")


@_guard
def check_padding_mask(fn):
    ids = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 0, 0, 0]])
    m = fn(ids, 0)
    if not torch.is_tensor(m):
        _fail("Return a tensor.")
    if m.dtype != torch.bool:
        _fail(f"Mask dtype should be torch.bool, got {m.dtype}. (ids != pad_id) is already bool.")
    if tuple(m.shape) != (2, 1, 1, 5):
        _fail(f"Mask shape should be (B, 1, 1, T) = (2, 1, 1, 5), got {tuple(m.shape)}. Try [:, None, None, :].")
    if not torch.equal(m, R_att.make_padding_mask(ids, 0)):
        _fail("Wrong values: True where ids != pad_id (real tokens), False for padding.")
    _ok("make_padding_mask is correct")


@_guard
def check_causal_mask(fn):
    m = fn(4)
    if not torch.is_tensor(m):
        _fail("Return a tensor.")
    if m.dtype != torch.bool:
        _fail(f"Mask dtype should be torch.bool, got {m.dtype}. Use dtype=torch.bool (or .bool()).")
    if tuple(m.shape) != (4, 4):
        _fail(f"Shape should be (T, T) = (4, 4), got {tuple(m.shape)}.")
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    if torch.equal(m, ~expected) or torch.equal(m, expected.T):
        _fail("Looks flipped: row i must be True for columns 0..i (the past) - that's the LOWER triangle (torch.tril).")
    if not torch.equal(m, expected):
        _fail("Wrong values: row i should be True for columns j <= i.")
    _ok("make_causal_mask is correct")


# ----------------------------------------------------------------------------------
# Notebook 03: multi-head attention, positional encoding
# ----------------------------------------------------------------------------------

@_guard
def check_mha(cls):
    torch.manual_seed(0)
    d, h = 16, 4
    mha = cls(d, h)
    for name in ["w_q", "w_k", "w_v", "w_o"]:
        layer = getattr(mha, name, None)
        if not isinstance(layer, nn.Linear):
            _fail(f"self.{name} should be an nn.Linear(d_model, d_model). Found: {type(layer).__name__}.")
        if layer.in_features != d or layer.out_features != d or layer.bias is None:
            _fail(f"self.{name} should be nn.Linear(d_model, d_model) with bias.")
    _ok("w_q, w_k, w_v, w_o exist with the right sizes")

    x = torch.randn(2, 5, d)
    s = mha.split_heads(x)
    if _shape_str(s) != (2, h, 5, d // h):
        _fail(f"split_heads: shape {_shape_str(s)}, expected (B, h, T, d_k) = (2, {h}, 5, {d // h}).")
    if not torch.equal(s[:, 1], x[..., d // h: 2 * d // h]):
        _fail("split_heads has the right SHAPE but scrambles the numbers. Head 1 of token t should be "
              "x[b, t, d_k:2*d_k]. Use x.view(B, T, h, d_k).transpose(1, 2), not x.view(B, h, T, d_k).")
    m = mha.merge_heads(s)
    if _shape_str(m) != (2, 5, d) or not torch.equal(m, x):
        _fail("merge_heads(split_heads(x)) should give back x exactly. Use x.transpose(1, 2).contiguous().view(B, T, d_model).")
    _ok("split_heads / merge_heads are correct")

    ref = nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.w_q.weight, mha.w_k.weight, mha.w_v.weight]))
        ref.in_proj_bias.copy_(torch.cat([mha.w_q.bias, mha.w_k.bias, mha.w_v.bias]))
        ref.out_proj.weight.copy_(mha.w_o.weight)
        ref.out_proj.bias.copy_(mha.w_o.bias)
    hint = (" Common causes: k and v must come from x_kv (q from x_q); apply w_o AFTER merging heads; "
            "call your scaled_dot_product_attention on the split heads.")

    out = mha(x, x)
    if _shape_str(out) != (2, 5, d):
        _fail(f"Output shape {_shape_str(out)}, expected (B, T_q, d_model) = (2, 5, {d}).")
    if not torch.allclose(out, ref(x, x, x, need_weights=False)[0], atol=1e-5):
        _fail("Self-attention output differs from torch.nn.MultiheadAttention with the same weights." + hint)
    _ok("self-attention matches torch.nn.MultiheadAttention")

    xq, xkv = torch.randn(2, 3, d), torch.randn(2, 6, d)
    out = mha(xq, xkv)
    if not torch.allclose(out, ref(xq, xkv, xkv, need_weights=False)[0], atol=1e-5):
        _fail("Cross-attention (x_q != x_kv) is wrong." + hint)
    if _shape_str(mha.last_attn) != (2, h, 3, 6):
        _fail(f"self.last_attn should hold the weights, shape (B, h, T_q, T_k) = (2, {h}, 3, 6); got {_shape_str(mha.last_attn)}.")
    _ok("cross-attention works (different lengths for queries and keys) and last_attn is stored")

    ids = torch.tensor([[3, 4, 5, 6, 0, 0], [3, 4, 0, 0, 0, 0]])
    pad = R_att.make_padding_mask(ids, 0)
    out = mha(xq, xkv, pad)
    expected = ref(xq, xkv, xkv, key_padding_mask=~pad[:, 0, 0, :], need_weights=False)[0]
    if not torch.allclose(out, expected, atol=1e-5):
        _fail("Padding mask is not applied correctly: pass `mask` straight into scaled_dot_product_attention.")
    causal = R_att.make_causal_mask(5)
    out = mha(x, x, causal)
    if not torch.allclose(out, ref(x, x, x, attn_mask=~causal, need_weights=False)[0], atol=1e-5):
        _fail("Causal mask is not applied correctly.")
    _ok("masks work")
    print("🎉 MultiHeadAttention is correct - it matches PyTorch's own implementation!")


@_guard
def check_positional(fn):
    pe = fn(50, 16)
    if not torch.is_tensor(pe) or tuple(pe.shape) != (50, 16):
        _fail(f"Expected a (max_len, d_model) = (50, 16) tensor, got {_shape_str(pe)}.")
    ref = R_pos.sinusoidal_positional_encoding(50, 16)
    if not torch.allclose(pe.float(), ref, atol=1e-5):
        swapped = torch.zeros_like(ref)
        swapped[:, 0::2], swapped[:, 1::2] = ref[:, 1::2], ref[:, 0::2]
        if torch.allclose(pe.float(), swapped, atol=1e-5):
            _fail("sin and cos are swapped: EVEN columns (0::2) get sin, ODD columns (1::2) get cos.")
        if torch.allclose(pe[0].float(), ref[0], atol=1e-5):
            _fail("Position 0 is right but later positions are wrong - check the frequencies: "
                  "angle = pos / 10000 ** (two_i / d_model), with two_i = 0, 2, 4, ...")
        _fail("Wrong values. Row 0 should be [0, 1, 0, 1, ...] (sin 0 = 0, cos 0 = 1).")
    _ok("sinusoidal_positional_encoding matches the paper's formula")


# ----------------------------------------------------------------------------------
# Notebook 04: the blocks and the full model
# ----------------------------------------------------------------------------------

def _copy_weights(student, ref, what):
    sd = student.state_dict()
    exp = set(ref.state_dict().keys())
    got = set(sd.keys())
    if exp != got:
        _fail(f"{what}: parameter names don't match the ones the instructions ask for.\n"
              f"   missing:    {sorted(exp - got)}\n   unexpected: {sorted(got - exp)}\n"
              "   Use exactly the attribute names given in the comments.")
    ref.load_state_dict(sd)


@_guard
def check_ffn(cls):
    torch.manual_seed(0)
    ffn, ref = cls(16, 32), R_tf.FeedForward(16, 32)
    _copy_weights(ffn, ref, "FeedForward")
    x = torch.randn(2, 5, 16)
    out = ffn(x)
    if _shape_str(out) != (2, 5, 16):
        _fail(f"Output shape should be (2, 5, 16), got {_shape_str(out)}.")
    if not torch.allclose(out, ref(x), atol=1e-5):
        _fail("Wrong values. FFN(x) = lin2(relu(lin1(x))).")
    _ok("FeedForward is correct")


@_guard
def check_encoder_block(cls):
    torch.manual_seed(0)
    blk, ref = cls(16, 4, 32), R_tf.EncoderBlock(16, 4, 32)
    _copy_weights(blk, ref, "EncoderBlock")
    x = torch.randn(2, 5, 16)
    ids = torch.tensor([[3, 4, 5, 6, 7], [3, 4, 5, 0, 0]])
    mask = R_att.make_padding_mask(ids, 0)
    out = blk(x, mask)
    if _shape_str(out) != (2, 5, 16):
        _fail(f"Output shape should be (B, S, d_model) = (2, 5, 16), got {_shape_str(out)}.")
    if not torch.allclose(out, ref(x, mask), atol=1e-5):
        _fail("Wrong values. Expected:  x = norm1(x + self_attn(x, x, src_mask));  x = norm2(x + ffn(x)).")
    _ok("EncoderBlock matches the reference")
    x2 = x.clone()
    x2[1, 3:] = torch.randn(2, 16) * 10
    if not torch.allclose(blk(x2, mask)[1, :3], out[1, :3], atol=1e-5):
        _fail("Padding test failed: changing PADDED positions changed the output at REAL positions. Pass src_mask to self_attn.")
    _ok("padding test: garbage in padded positions doesn't leak into real tokens")


@_guard
def check_decoder_block(cls):
    torch.manual_seed(0)
    blk, ref = cls(16, 4, 32), R_tf.DecoderBlock(16, 4, 32)
    _copy_weights(blk, ref, "DecoderBlock")
    y, mem = torch.randn(2, 6, 16), torch.randn(2, 5, 16)
    src_mask = R_att.make_padding_mask(torch.tensor([[3, 4, 5, 6, 7], [3, 4, 5, 0, 0]]), 0)
    tgt_mask = R_att.make_causal_mask(6) & R_att.make_padding_mask(torch.ones(2, 6, dtype=torch.long), 0)
    out = blk(y, mem, tgt_mask, src_mask)
    if _shape_str(out) != (2, 6, 16):
        _fail(f"Output shape should be (B, T, d_model) = (2, 6, 16), got {_shape_str(out)}.")
    if not torch.allclose(out, ref(y, mem, tgt_mask, src_mask), atol=1e-5):
        _fail("Wrong values. Expected:\n   y = norm1(y + self_attn(y, y, tgt_mask))\n"
              "   y = norm2(y + cross_attn(y, memory, src_mask))\n   y = norm3(y + ffn(y))")
    _ok("DecoderBlock matches the reference")
    y2 = y.clone()
    y2[:, 3:] = torch.randn(2, 3, 16) * 10
    if not torch.allclose(blk(y2, mem, tgt_mask, src_mask)[:, :3], out[:, :3], atol=1e-5):
        _fail("Causality test failed: changing FUTURE tokens changed the output at earlier positions.")
    _ok("causality test: position t never sees positions > t")
    mem2 = mem.clone()
    mem2[1, 3:] = torch.randn(2, 16) * 10
    if not torch.allclose(blk(y, mem2, tgt_mask, src_mask)[1], out[1], atol=1e-5):
        _fail("Padding test failed: cross-attention looks at padded encoder positions. Pass src_mask to cross_attn.")
    _ok("padding test: cross-attention ignores padded source positions")


@_guard
def check_transformer(cls):
    torch.manual_seed(0)
    cfg = dict(vocab_size=20, d_model=16, n_heads=4, n_layers=2, d_ff=32, max_len=32)
    model, ref = cls(**cfg), R_tf.Transformer(**cfg)
    _copy_weights(model, ref, "Transformer")
    src = torch.tensor([[5, 6, 7, 8, 9, 10, 11], [5, 6, 7, 0, 0, 0, 0]])
    tgt_in = torch.tensor([[1, 12, 13, 14, 15], [1, 12, 13, 0, 0]])

    res = model.encode(src)
    if not (isinstance(res, tuple) and len(res) == 2):
        _fail("encode must return a tuple (memory, src_mask).")
    memory, src_mask = res
    if _shape_str(memory) != (2, 7, 16):
        _fail(f"memory should be (B, S, d_model) = (2, 7, 16), got {_shape_str(memory)}.")
    ref_mem, _ = ref.encode(src)
    if not torch.allclose(memory, ref_mem, atol=1e-5):
        _fail("encode: wrong values. Embed with self.embed(src, self.src_embed), then run every encoder layer with src_mask.")
    _ok("encode matches the reference")

    logits = model(src, tgt_in)
    if _shape_str(logits) != (2, 5, 20):
        _fail(f"logits should be (B, T, vocab) = (2, 5, 20), got {_shape_str(logits)}.")
    if not torch.allclose(logits, ref(src, tgt_in), atol=1e-5):
        _fail("decode: wrong values. tgt_mask = make_causal_mask(T, device) & make_padding_mask(tgt_in, pad_id); "
              "embed with self.tgt_embed; run every decoder layer; finish with self.out_proj.")
    _ok("decode / forward match the reference")

    t2 = tgt_in.clone()
    t2[0, 3:] = torch.tensor([16, 17])
    if not torch.allclose(model(src, t2)[0, :3], logits[0, :3], atol=1e-5):
        _fail("Causality test failed for the full model.")
    src_more_pad = torch.cat([src, torch.zeros(2, 3, dtype=torch.long)], dim=1)
    if not torch.allclose(model(src_more_pad, tgt_in), logits, atol=1e-5):
        _fail("Adding extra padding to the source changed the output: the source padding mask isn't used everywhere.")
    _ok("end-to-end tests: causal, and extra padding changes nothing")
    print("🎉 Your Transformer is complete and correct!")
