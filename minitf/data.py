"""
Date-translation dataset  (PROVIDED - you don't need to edit this file, but do read it).

The task: translate a date written the way people write it into ISO format.

    "tuesday, november 24, 2026"  ->  "2026-11-24"
    "24/11/2026"                  ->  "2026-11-24"

Why this task?
  * It is a real sequence-to-sequence "translation" task, like the one in the paper,
    so we need the full encoder-decoder Transformer.
  * It trains in a minute or two on a laptop.
  * Attention is easy to read: when the model writes "11" you can check
    whether it looks at "nov" / "november" / "11" in the input.

Tokens are single characters, so the vocabulary is tiny (~45 tokens).
"""
import datetime
import random

import torch

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# Each format turns a datetime.date into a human-style string.
FORMATS = {
    "long":           lambda d: f"{MONTHS[d.month-1]} {d.day}, {d.year}",              # november 24, 2026
    "short":          lambda d: f"{MONTHS[d.month-1][:3]} {d.day} {d.year}",           # nov 24 2026
    "day_first":      lambda d: f"{d.day} {MONTHS[d.month-1]} {d.year}",               # 24 november 2026
    "day_first_short": lambda d: f"{d.day} {MONTHS[d.month-1][:3]} {d.year}",          # 24 nov 2026
    "weekday_long":   lambda d: f"{WEEKDAYS[d.weekday()]}, {MONTHS[d.month-1]} {d.day}, {d.year}",  # tuesday, november 24, 2026
    "weekday_short":  lambda d: f"{WEEKDAYS[d.weekday()][:3]} {d.day} {MONTHS[d.month-1][:3]} {d.year}",  # tue 24 nov 2026
    "ordinal":        lambda d: f"the {_ordinal(d.day)} of {MONTHS[d.month-1]}, {d.year}",  # the 24th of november, 2026
    "slash":          lambda d: f"{d.day}/{d.month}/{d.year}",                          # 24/11/2026  (day first!)
    "dot":            lambda d: f"{d.day}.{d.month}.{d.year}",                          # 24.11.2026
    "year_first":     lambda d: f"{d.year}/{d.month}/{d.day}",                          # 2026/11/24
}


def to_iso(d: datetime.date) -> str:
    return d.strftime("%Y-%m-%d")


class CharVocab:
    """Maps characters <-> integer ids. Ids 0, 1, 2 are the special tokens <pad>, <bos>, <eos>."""

    CHARS = " ,./-0123456789abcdefghijklmnopqrstuvwxyz"

    def __init__(self):
        self.itos = [PAD, BOS, EOS] + list(self.CHARS)        # id -> string
        self.stoi = {s: i for i, s in enumerate(self.itos)}   # string -> id
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, add_bos=False, add_eos=False):
        unknown = sorted({c for c in text if c not in self.stoi})
        if unknown:
            raise ValueError(f"Characters {unknown} are not in the vocabulary. "
                             f"Allowed: {self.CHARS!r} (lowercase only). "
                             "This is a tokenization limit: the model can only see tokens it has an id for.")
        ids = [self.stoi[c] for c in text]
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids, stop_at_eos=True):
        out = []
        for i in (ids.tolist() if torch.is_tensor(ids) else ids):
            if i == self.eos_id and stop_at_eos:
                break
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            out.append(self.itos[i])
        return "".join(out)

    def tokens(self, ids):
        """Readable list of tokens, e.g. for plot labels."""
        ids = ids.tolist() if torch.is_tensor(ids) else ids
        return [self.itos[i] if self.itos[i] != " " else "␣" for i in ids]


def random_date(rng: random.Random, start_year=1900, end_year=2099) -> datetime.date:
    start = datetime.date(start_year, 1, 1).toordinal()
    end = datetime.date(end_year, 12, 31).toordinal()
    return datetime.date.fromordinal(rng.randint(start, end))


def random_pair(rng: random.Random, fmt: str = None):
    """Returns (source_text, target_text, format_name)."""
    d = random_date(rng)
    fmt = fmt or rng.choice(list(FORMATS))
    return FORMATS[fmt](d), to_iso(d), fmt


def make_batch(pairs, vocab: CharVocab, device="cpu"):
    """
    pairs: list of (source_text, target_text, ...)
    Returns:
        src: (B, S) long - source characters, right-padded with <pad>
        tgt: (B, T) long - <bos> target characters <eos>, right-padded with <pad>
    """
    src_ids = [vocab.encode(p[0]) for p in pairs]
    tgt_ids = [vocab.encode(p[1], add_bos=True, add_eos=True) for p in pairs]
    S = max(map(len, src_ids))
    T = max(map(len, tgt_ids))
    src = torch.full((len(pairs), S), vocab.pad_id, dtype=torch.long)
    tgt = torch.full((len(pairs), T), vocab.pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_ids, tgt_ids)):
        src[i, :len(s)] = torch.tensor(s)
        tgt[i, :len(t)] = torch.tensor(t)
    return src.to(device), tgt.to(device)


def sample_batch(batch_size, vocab: CharVocab, rng: random.Random, device="cpu"):
    """A fresh random batch every call (there are ~730,000 possible (date, format) pairs,
    far more than the model can memorise, so it has to learn the rules)."""
    return make_batch([random_pair(rng) for _ in range(batch_size)], vocab, device)


def make_test_set(n_per_format=100, seed=1234):
    """Fixed evaluation set: n examples of every format."""
    rng = random.Random(seed)
    return [random_pair(rng, fmt) for fmt in FORMATS for _ in range(n_per_format)]
