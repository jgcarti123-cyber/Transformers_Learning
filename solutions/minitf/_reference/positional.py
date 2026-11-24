"""
Positional encoding  -  YOU write this file (notebook 03).
"""
import torch


def sinusoidal_positional_encoding(max_len, d_model):
    """
    The paper's positional encoding (section 3.5):

        PE[pos, 2i]   = sin( pos / 10000^(2i / d_model) )
        PE[pos, 2i+1] = cos( pos / 10000^(2i / d_model) )

    pos = 0 .. max_len-1 is the position; i = 0 .. d_model/2 - 1 picks the pair of dimensions.
    Even dimensions get sin, odd dimensions get cos, and each pair (2i, 2i+1) shares one frequency.

    Returns a float tensor of shape (max_len, d_model).

    Hints (no loops needed):
        pos = torch.arange(max_len).unsqueeze(1).float()      # (max_len, 1)
        two_i = torch.arange(0, d_model, 2).float()           # (d_model/2,)   = 0, 2, 4, ...
        angle = pos / 10000 ** (two_i / d_model)              # broadcasting -> (max_len, d_model/2)
        pe[:, 0::2] = ...   (every even column)     pe[:, 1::2] = ...   (every odd column)
    """
    pos = torch.arange(max_len).unsqueeze(1).float()
    two_i = torch.arange(0, d_model, 2).float()
    angle = pos / 10000 ** (two_i / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe
