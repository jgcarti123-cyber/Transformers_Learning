"""
The Transformer  -  YOU write this file (notebook 04).

We follow the original paper ("post-norm"):   x = LayerNorm(x + Sublayer(x))
Use EXACTLY the attribute names given in the comments: the checker compares your blocks
with a reference implementation by copying weights across by name.
"""
import math

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, make_causal_mask, make_padding_mask
from .positional import sinusoidal_positional_encoding


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (paper, section 3.3):
        FFN(x) = ReLU(x W1 + b1) W2 + b2
    Applied to every token separately: (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model).
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        # Two layers named exactly self.lin1 = Linear(d_model -> d_ff), self.lin2 = Linear(d_ff -> d_model)
        ### TODO: create lin1 and lin2
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        ### END

    def forward(self, x):
        ### TODO: lin1 -> ReLU (torch.relu) -> lin2
        return self.lin2(torch.relu(self.lin1(x)))
        ### END


class EncoderBlock(nn.Module):
    """
        x -> [self-attention] -> add & norm -> [feed-forward] -> add & norm
    """

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # Name them exactly: self.self_attn (MultiHeadAttention), self.ffn (FeedForward),
        #                    self.norm1, self.norm2 (nn.LayerNorm(d_model))
        ### TODO: create the sub-layers
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        ### END

    def forward(self, x, src_mask):
        """x: (B, S, d_model).  src_mask: (B, 1, 1, S) padding mask.  Returns (B, S, d_model)."""
        # Sublayer 1: self-attention -- queries, keys and values all come from x.
        # Sublayer 2: feed-forward.
        # Each sublayer:  x = norm(x + sublayer(x))
        ### TODO: the two sublayers, each with a residual connection and LayerNorm
        x = self.norm1(x + self.self_attn(x, x, src_mask))
        x = self.norm2(x + self.ffn(x))
        return x
        ### END


class DecoderBlock(nn.Module):
    """
        y -> [masked self-attention] -> add & norm
          -> [cross-attention: queries from y, keys/values from the encoder output] -> add & norm
          -> [feed-forward] -> add & norm
    """

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # Name them exactly: self.self_attn, self.cross_attn (both MultiHeadAttention),
        #                    self.ffn (FeedForward), self.norm1, self.norm2, self.norm3 (nn.LayerNorm)
        ### TODO: create the sub-layers
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        ### END

    def forward(self, y, memory, tgt_mask, src_mask):
        """
        y:        (B, T, d_model)  decoder input (the target so far)
        memory:   (B, S, d_model)  encoder output
        tgt_mask: (B, 1, T, T)     causal AND padding mask for the decoder's self-attention
        src_mask: (B, 1, 1, S)     padding mask for the encoder output (used in cross-attention)
        Returns (B, T, d_model)
        """
        ### TODO: masked self-attention, cross-attention, feed-forward (each: residual + LayerNorm)
        y = self.norm1(y + self.self_attn(y, y, tgt_mask))
        y = self.norm2(y + self.cross_attn(y, memory, src_mask))
        y = self.norm3(y + self.ffn(y))
        return y
        ### END


class Transformer(nn.Module):
    """
    The full encoder-decoder Transformer (paper, figure 1).

        src ids (B, S) -> embed -> EncoderBlock x n_layers -> memory (B, S, d)
        tgt_in ids (B, T) -> embed -> DecoderBlock x n_layers (using memory) -> Linear -> logits (B, T, vocab)
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_len=64,
                 pad_id=0, use_positions=True):
        super().__init__()  # (__init__ is provided - read it so you know what exists)
        self.d_model = d_model
        self.pad_id = pad_id
        self.use_positions = use_positions
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        # Small initial embeddings (std = 1/sqrt(d_model)); embed() multiplies them by sqrt(d_model),
        # so token vectors start at roughly unit scale, like the positional encodings (values in [-1, 1]).
        nn.init.normal_(self.src_embed.weight, std=d_model ** -0.5)
        nn.init.normal_(self.tgt_embed.weight, std=d_model ** -0.5)
        # A buffer is saved with the model and moved by .to(device), but is NOT a trained parameter.
        self.register_buffer("pe", sinusoidal_positional_encoding(max_len, d_model))
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def embed(self, ids, table):
        """(B, T) ids -> (B, T, d_model):  table(ids) * sqrt(d_model)  + positional encoding."""
        x = table(ids) * math.sqrt(self.d_model)
        if self.use_positions:
            x = x + self.pe[: ids.size(1)]          # (T, d) broadcasts over the batch
        return x

    def encode(self, src):
        """
        src: (B, S) ids.  Returns (memory, src_mask):
            memory:   (B, S, d_model)  output of the last encoder block
            src_mask: (B, 1, 1, S)     padding mask (the decoder needs it for cross-attention)
        """
        # 1. src_mask = make_padding_mask(...)   2. x = self.embed(src, self.src_embed)
        # 3. run x through every block in self.encoder_layers
        ### TODO: the encoder
        src_mask = make_padding_mask(src, self.pad_id)
        x = self.embed(src, self.src_embed)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x, src_mask
        ### END

    def decode(self, tgt_in, memory, src_mask):
        """
        tgt_in: (B, T) ids (starting with <bos>).  Returns logits (B, T, vocab_size).
        """
        # 1. tgt_mask = causal mask (T, T)  AND  padding mask of tgt_in (B, 1, 1, T)
        #    (combine with &, broadcasting gives (B, 1, T, T)).  Pass device=tgt_in.device.
        # 2. y = self.embed(tgt_in, self.tgt_embed)
        # 3. run y through every block in self.decoder_layers
        # 4. return self.out_proj(y)
        ### TODO: the decoder
        T = tgt_in.size(1)
        tgt_mask = make_causal_mask(T, device=tgt_in.device) & make_padding_mask(tgt_in, self.pad_id)
        y = self.embed(tgt_in, self.tgt_embed)
        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask, src_mask)
        return self.out_proj(y)
        ### END

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)
