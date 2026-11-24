"""Plotting helpers  (PROVIDED - you don't need to edit this file)."""
import math

import matplotlib.pyplot as plt
import torch


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else x


def plot_matrix(m, title="", x_labels=None, y_labels=None, ax=None, cmap="viridis",
                vmin=None, vmax=None, annotate=False, xlabel=None, ylabel=None, colorbar=True):
    """Heatmap of a 2-D tensor."""
    m = _np(m)
    own = ax is None
    if own:
        h, w = m.shape
        fig, ax = plt.subplots(figsize=(min(1 + 0.45 * w, 14), min(1 + 0.4 * h, 10)))
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90 if max(len(str(s)) for s in x_labels) > 3 else 0)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    if annotate:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        color="white" if m[i, j] < (m.max() + m.min()) / 2 else "black", fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if own:
        plt.tight_layout()
        plt.show()


def plot_attention(weights, x_labels=None, y_labels=None, title="attention weights", annotate=None):
    """weights: (T_q, T_k). Rows = queries (who is looking), columns = keys (who is looked at)."""
    weights = _np(weights)
    if annotate is None:
        annotate = weights.shape[0] * weights.shape[1] <= 64
    plot_matrix(weights, title, x_labels, y_labels, vmin=0, vmax=1, annotate=annotate,
                xlabel="keys (looked at)", ylabel="queries (looking)")


def plot_attention_heads(weights, x_labels=None, y_labels=None, title=""):
    """weights: (h, T_q, T_k) - one panel per head, plus their average."""
    weights = _np(weights)
    h = weights.shape[0]
    panels = list(weights) + [weights.mean(0)]
    names = [f"head {i}" for i in range(h)] + ["average of heads"]
    cols = min(3, len(panels))
    rows = math.ceil(len(panels) / cols)
    T_q, T_k = weights.shape[1:]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * (1.2 + 0.3 * T_k), rows * (1.0 + 0.3 * T_q)),
                             squeeze=False)
    for ax, w, name in zip(axes.flat, panels, names):
        plot_matrix(w, name, x_labels, y_labels, ax=ax, vmin=0, vmax=1, colorbar=False)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def pca_2d(vectors):
    """Project (N, d) vectors to (N, 2) along the two directions of largest variance."""
    x = torch.as_tensor(_np(vectors))
    x = x - x.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return x @ vh[:2].T


def plot_embeddings_2d(vectors, labels, title="embeddings (PCA to 2-D)", colors=None, ax=None):
    """Scatter plot of (N, d) vectors after PCA, with a text label per point."""
    p = _np(pca_2d(vectors))
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(p[:, 0], p[:, 1], c=colors if colors is not None else "tab:blue", s=30)
    for (x, y), lab in zip(p, labels):
        ax.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=10)
    ax.set_title(title)
    ax.axhline(0, color="#ddd", lw=0.5, zorder=0)
    ax.axvline(0, color="#ddd", lw=0.5, zorder=0)
    if own:
        plt.tight_layout()
        plt.show()


def plot_curves(curves: dict, title="", xlabel="step", ylabel="loss", logy=False):
    """curves: {name: list_of_values}"""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, ys in curves.items():
        ax.plot(ys, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    if len(curves) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
