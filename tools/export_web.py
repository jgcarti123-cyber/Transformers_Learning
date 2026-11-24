"""
Trains the date Transformer (baseline with checkpoints + ablations) and exports everything the
web course needs into web/model/:  manifest.json  +  one float16 .bin weight file per model.

    .venv/bin/python tools/export_web.py
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solutions"))
from minitf.data import CharVocab, FORMATS, make_batch, sample_batch, make_test_set  # noqa: E402
from minitf.transformer import Transformer  # noqa: E402

OUT = ROOT / "web" / "model"
OUT.mkdir(parents=True, exist_ok=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
vocab = CharVocab()
test_set = make_test_set(n_per_format=100)
SAMPLES = ["tuesday, november 24, 2026", "24/11/2026", "2026/11/24", "the 3rd of july, 1969", "1 jan 2000"]
CHECKPOINTS = [0, 20, 50, 100, 200, 350, 600, 1000, 1500]
STEPS = 1500


@torch.no_grad()
def greedy(model, src, max_len=11):
    model.eval()
    mem, sm = model.encode(src)
    ys = torch.full((src.size(0), 1), vocab.bos_id, dtype=torch.long, device=src.device)
    for _ in range(max_len):
        ys = torch.cat([ys, model.decode(ys, mem, sm)[:, -1].argmax(-1, keepdim=True)], 1)
    model.train()
    return ys[:, 1:]


def evaluate(model):
    src, tgt = make_batch(test_set, vocab, device)
    pred = greedy(model, src)
    strs = [vocab.decode(p) for p in pred]
    per_fmt = {}
    for (s, t, f), p in zip(test_set, strs):
        per_fmt.setdefault(f, []).append(p == t)
    per_fmt = {f: round(sum(v) / len(v), 4) for f, v in per_fmt.items()}
    per_pos = (pred[:, :10] == tgt[:, 1:11]).float().mean(0).tolist()
    samples = {s: vocab.decode(greedy(model, torch.tensor([vocab.encode(s)], device=device))[0]) for s in SAMPLES}
    return {"accuracy": round(sum(per_fmt.values()) / len(per_fmt), 4), "per_format": per_fmt,
            "per_position": [round(x, 4) for x in per_pos], "samples": samples}


def export_weights(model, name):
    tensors, offset, chunks = [], 0, []
    for k, v in model.state_dict().items():
        if k == "pe":
            continue
        a = v.detach().float().cpu().numpy().astype(np.float16).ravel()
        tensors.append({"name": k, "shape": list(v.shape), "offset": offset})
        offset += a.size
        chunks.append(a)
    (OUT / f"{name}.bin").write_bytes(np.concatenate(chunks).astype("<f2").tobytes())
    return {"file": f"{name}.bin", "tensors": tensors, "n_params": int(offset)}


def run(name, cfg, checkpoints=()):
    torch.manual_seed(0)
    rng = random.Random(0)
    model = Transformer(len(vocab), **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses, ckpts, t0 = [], [], time.time()
    for step in range(STEPS + 1):
        if step in checkpoints:
            ev = evaluate(model)
            w = export_weights(model, f"{name}_step{step}")
            ckpts.append({"step": step, "loss": round(float(np.mean(losses[-20:])) if losses else float(np.log(len(vocab))), 4), **ev, **w})
            print(f"  {name} step {step}: acc {ev['accuracy']:.3f}  {ev['samples'][SAMPLES[0]]!r}", flush=True)
        if step == STEPS:
            break
        src, tgt = sample_batch(128, vocab, rng, device)
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), tgt[:, 1:].reshape(-1), ignore_index=vocab.pad_id)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    final = evaluate(model)
    weights = export_weights(model, name)
    print(f"{name}: acc {final['accuracy']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    # a reference vector so the browser engine can verify itself: logits for SAMPLES[0]
    model.eval().cpu()
    with torch.no_grad():
        src = torch.tensor([vocab.encode(SAMPLES[0])])
        tgt_in = torch.tensor([[vocab.bos_id] + vocab.encode("2026-11-24")])
        ref_logits = model(src, tgt_in)[0, :3].tolist()
    curve = [round(float(np.mean(losses[max(0, i - 9):i + 1])), 4) for i in range(0, len(losses), 5)]
    full_cfg = dict(vocab_size=len(vocab), d_model=64, n_heads=4, n_layers=2, d_ff=256, max_len=64, use_positions=True)
    full_cfg.update(cfg)
    return {"name": name, "config": full_cfg, "loss_curve_every5": curve, **final, **weights,
            "checkpoints": ckpts, "ref_logits_first3": [[round(x, 4) for x in row] for row in ref_logits]}


def main():
    models = {
        "baseline": run("baseline", {}, CHECKPOINTS),
        "no_positions": run("no_positions", {"use_positions": False}),
        "one_head": run("one_head", {"n_heads": 1}),
        "one_layer": run("one_layer", {"n_layers": 1}),
        "tiny": run("tiny", {"d_model": 16, "d_ff": 64}),
    }
    manifest = {"vocab": vocab.itos, "pad_id": 0, "bos_id": 1, "eos_id": 2, "formats": list(FORMATS),
                "samples": SAMPLES, "steps": STEPS, "batch_size": 128, "lr": 1e-3, "models": models}
    (OUT / "manifest.json").write_text(json.dumps(manifest))
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {OUT} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
