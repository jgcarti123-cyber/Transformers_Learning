"""
Attention  -  YOU write this file.
    Notebook 02: scaled_dot_product_attention, make_padding_mask, make_causal_mask
    Notebook 03: MultiHeadAttention

Every place you need to write code says  raise NotImplementedError("TODO ...").
Delete that line and write your code there. Then re-run the check cell in the notebook
(the notebooks auto-reload this file, so there is no need to restart the kernel).

MASK CONVENTION used everywhere in this project:
    mask is a bool tensor.  True  = "this query MAY look at this key"
                            False = "blocked"
    This matches torch.nn.functional.scaled_dot_product_attention.
    (Careful: torch.nn.MultiheadAttention uses the OPPOSITE convention.)
"""
import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------------
# Notebook 02
# ----------------------------------------------------------------------------------

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V          (paper, equation 1)

    Args:
        q:    (..., T_q, d_k)  queries  - "what am I looking for?"
        k:    (..., T_k, d_k)  keys     - "what do I contain?"
        v:    (..., T_k, d_v)  values   - "what do I hand over if I'm picked?"
        mask: optional bool tensor that broadcasts to (..., T_q, T_k). True = allowed.
        ("..." means any number of leading batch dims, e.g. (B,) or (B, heads).)

    Returns:
        out:     (..., T_q, d_v)  for every query, a weighted average of the values
        weights: (..., T_q, T_k)  the attention weights; every row sums to 1
    """
    # Step 1 - scores: how well does every query match every key?
    #   (..., T_q, d_k) @ (..., d_k, T_k)  ->  (..., T_q, T_k), then divide by sqrt(d_k).
    #   Hints: k.transpose(-2, -1) swaps the last two dims.  d_k = q.size(-1).  math.sqrt.
    ### TODO: step 1 - compute the scaled scores
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    ### END

    # Step 2 - masking: blocked positions get -inf BEFORE the softmax,
    #   so that exp(-inf) = 0 and they receive exactly zero weight.
    #   Only do this if mask is not None.
    #   Hint: scores.masked_fill(~mask, float("-inf"))   (~ flips True/False)
    ### TODO: step 2 - apply the mask
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    ### END

    # Step 3 - softmax over the KEYS dimension (the last one), then use the
    #   weights to average the values:  (..., T_q, T_k) @ (..., T_k, d_v) -> (..., T_q, d_v)
    ### TODO: step 3 - softmax, then weighted sum of the values
    weights = torch.softmax(scores, dim=-1)
    out = weights @ v
    ### END
    return out, weights


def make_padding_mask(ids, pad_id):
    """
    ids: (B, T) token ids, right-padded with pad_id.
    Returns a bool mask of shape (B, 1, 1, T): True where the token is real, False where it is padding.

    Why (B, 1, 1, T)?  Attention scores have shape (B, heads, T_q, T_k). Whether a KEY is padding
    is the same for every head and every query, so we use size-1 dims and let broadcasting copy it.
    Hint: (ids != pad_id) gives (B, T). Then add two dims, e.g. with [:, None, None, :].
    """
    ### TODO: build the padding mask
    return (ids != pad_id)[:, None, None, :]
    ### END


def make_causal_mask(T, device=None):
    """
    Returns a (T, T) bool mask where row i is True for columns 0..i and False for columns > i:
    position i may look at itself and the past, never the future.
    Hint: torch.tril (lower triangle) of a (T, T) tensor of ones, as dtype=torch.bool.
    """
    ### TODO: build the causal mask
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    ### END


# ----------------------------------------------------------------------------------
# Notebook 03
# ----------------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention (paper, section 3.2.2).

        x_q  (B, T_q, d_model) --W_q--> split into heads -> (B, h, T_q, d_k) --.
        x_kv (B, T_k, d_model) --W_k--> split into heads -> (B, h, T_k, d_k) ---+--> attention per head
        x_kv (B, T_k, d_model) --W_v--> split into heads -> (B, h, T_k, d_k) --'      (B, h, T_q, d_k)
                                                                                         | merge heads
                                                                   (B, T_q, d_model) <-W_o--'

    Self-attention:  x_q and x_kv are the same tensor.
    Cross-attention: x_q comes from the decoder, x_kv from the encoder output.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Create four nn.Linear(d_model, d_model) layers (with bias), named EXACTLY:
        #   self.w_q, self.w_k, self.w_v, self.w_o
        # (the checker copies their weights into PyTorch's own implementation to compare)
        ### TODO: create the four projection layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        ### END
        self.last_attn = None  # we keep the latest attention weights here, for plotting

    def split_heads(self, x):
        """(B, T, d_model) -> (B, h, T, d_k).   Hint: .view(B, T, h, d_k) then .transpose(1, 2)"""
        ### TODO: split into heads
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        ### END

    def merge_heads(self, x):
        """(B, h, T, d_k) -> (B, T, d_model).   Hint: .transpose(1, 2), .contiguous(), .view(...)"""
        ### TODO: merge the heads back together
        B, h, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d_k)
        ### END

    def forward(self, x_q, x_kv, mask=None):
        """
        x_q:  (B, T_q, d_model)   x_kv: (B, T_k, d_model)
        mask: bool, broadcastable to (B, h, T_q, T_k), True = allowed
        Returns (B, T_q, d_model). Also stores the attention weights (B, h, T_q, T_k) in self.last_attn.
        """
        # 1. project: q from x_q; k and v from x_kv. Then split each into heads.
        # 2. scaled_dot_product_attention(q, k, v, mask)  (your function from notebook 02)
        # 3. self.last_attn = weights.detach()
        # 4. merge heads, then apply self.w_o
        ### TODO: forward pass of multi-head attention
        q = self.split_heads(self.w_q(x_q))
        k = self.split_heads(self.w_k(x_kv))
        v = self.split_heads(self.w_v(x_kv))
        out, weights = scaled_dot_product_attention(q, k, v, mask)
        self.last_attn = weights.detach()
        return self.w_o(self.merge_heads(out))
        ### END
