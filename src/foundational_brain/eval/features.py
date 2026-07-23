"""Extract frozen per-subject representations for linear probing.

The point of Phase 5 is to test one prediction the latent-width ablation made
sharp: the model's value is *temporal*, so the RNN hidden state should carry
subject phenotype while the encoder latent should probe no better than a linear
projection of the same data. To test that, we need three representations of
each subject, summarized the same way:

* ``pca``     — frames projected onto the top-k principal components (basis fit
  on training subjects only). The linear, model-free reference.
* ``encoder`` — the encoder latent ``z_t``. The learned *spatial* map. The
  ablation says this should behave like ``pca``.
* ``rnn``     — the RNN hidden state ``h_t``. The learned *temporal* map, the
  "foundation" component; the hypothesis is that this is where phenotype lives.

Each is a time series of vectors; all three are pooled to one vector per
subject by concatenating the temporal **mean and std**. Using the identical
pooling for all three keeps the comparison about the representation, not the
summary. (Per-region z-scoring makes each region's raw temporal mean ~0, so the
std term is what carries amplitude — this is why mean alone would be a poor
summary and both are kept.)
"""

from __future__ import annotations

import numpy as np
import torch


def pool_mean_std(x: np.ndarray) -> np.ndarray:
    """Summarize a ``(T, d)`` sequence as a ``(2d,)`` [mean, std] vector."""
    return np.concatenate([x.mean(axis=0), x.std(axis=0)]).astype(np.float32)


@torch.no_grad()
def model_features(
    model,
    series: list[np.ndarray],
    device,
    batch_frames: int = 4096,
) -> dict[str, np.ndarray]:
    """Encoder-latent and RNN-hidden features, pooled per subject.

    Each subject's full (normalized) series is run through the model as a single
    sequence — not windowed — so the RNN hidden state reflects the whole scan's
    history rather than an arbitrary 64-frame slice.
    """
    model.eval()
    enc_rows, rnn_rows = [], []
    for s in series:
        x = torch.from_numpy(np.ascontiguousarray(s)).unsqueeze(0).to(device)  # (1,T,R)
        z = model.encode(x)                       # (1, T, latent_dim)
        _, _, feats = model.latent_rnn(z)         # (1, T, rnn_hidden)
        enc_rows.append(pool_mean_std(z[0].cpu().numpy()))
        rnn_rows.append(pool_mean_std(feats[0].cpu().numpy()))
    return {
        "encoder": np.stack(enc_rows),
        "rnn": np.stack(rnn_rows),
    }


def pca_features(
    series: list[np.ndarray],
    train_series: list[np.ndarray],
    n_components: int = 128,
) -> np.ndarray:
    """Frames projected onto a training-fit PCA basis, pooled per subject.

    The basis is fit on ``train_series`` and applied to ``series`` so the
    reference is a genuine held-out linear projection, not one fit to the probe
    set.
    """
    from ..eval.baselines import fit_pca

    comps, mean = fit_pca(train_series, n_components=n_components)
    rows = []
    for s in series:
        scores = (np.asarray(s, np.float32) - mean) @ comps  # (T, k)
        rows.append(pool_mean_std(scores))
    return np.stack(rows)


def all_features(
    model,
    series: list[np.ndarray],
    train_series: list[np.ndarray],
    device,
    pca_components: int = 128,
) -> dict[str, np.ndarray]:
    """All three representations for the same subjects, keyed by name."""
    feats = model_features(model, series, device)
    feats["pca"] = pca_features(series, train_series, n_components=pca_components)
    return feats
