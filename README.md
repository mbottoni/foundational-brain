# foundational-brain

A **foundation model for fMRI** built from three composable parts:

```
        ┌───────────┐        ┌────────────────────────┐        ┌───────────┐
 fMRI   │           │ latent │   Foundation model     │ latent │           │  fMRI
 ─────► │  Encoder  │ ─────► │   over the latent      │ ─────► │  Decoder  │ ─────►
 signal │  (spatial)│  z_t   │   space (RNN dynamics) │  ẑ_t   │  (spatial)│  recon
        └───────────┘        └────────────────────────┘        └───────────┘
```

The goal is a **pure, self-supervised foundation model**: no labels, no downstream
task baked in. We learn the *dynamics of brain activity* by compressing each fMRI
frame into a latent vector, modeling the temporal evolution of those latents with a
recurrent network, and reconstructing the signal. Once trained, the encoder + latent
model produce general-purpose representations that transfer to downstream tasks
(decoding, subject/state classification, forecasting, etc.).

---

## Why this design

- **Encoder / Decoder** decouple *what a brain state looks like* (spatial structure)
  from *how it evolves* (temporal dynamics). The encoder learns a compact, denoised
  representation of a single fMRI frame; the decoder inverts it.
- **RNN in the latent space** is the "foundation" component. fMRI is a relatively
  short, slow (TR ≈ 0.7–2 s), noisy time series. An RNN (GRU/LSTM, later optionally a
  state-space model) is a strong, parameter-efficient prior for these dynamics and
  avoids the quadratic cost of attention over long scans.
- **Self-supervised objectives** (autoregressive next-frame prediction + masked
  latent modeling) require no labels, so we can pretrain on large heterogeneous fMRI
  corpora.

---

## Data representation

fMRI is 4D: `(X, Y, Z, T)`. We support two representations, starting with the first:

1. **Parcellated ROI time series (primary).** Project each volume onto a brain atlas
   (e.g. Schaefer-400, AAL, Gordon) → a matrix `T × N_regions`. Tractable, standard,
   comparable across subjects/scanners. This is what the initial model consumes.
2. **Voxel / patch-level (stretch goal).** 3D patches of the volume for a
   higher-resolution encoder once the pipeline is validated on parcels.

**Primary corpus: ABIDE-PCP** — 871 quality-checked resting-state subjects across
20 sites, distributed *already parcellated* (CC200 → `T × 200`), fully open, no
access application. Fetched by `src/foundational_brain/data/download.py`.
`development_fmri` (4D NIfTI) exercises the raw-volume → atlas path. Later scaling
candidates: OpenNeuro, HCP, ADHD-200.

See **[`docs/data_report.md`](docs/data_report.md)** for the profile of this corpus
and the measurements behind the hyperparameters in `configs/default.yaml`.

### What the data actually said

Five findings from profiling all 871 subjects that changed the plan:

1. **A brain frame is not low-rank.** 125 of 200 components are needed for 90% of
   frame variance (effective rank ≈ 86). The originally configured `latent_dim=256`
   *exceeded the input dimension* — an expansion, not a bottleneck. Now 128.
2. **Forecasting only works at 1 TR.** At a 2-TR horizon both persistence and AR(1)
   are already *worse* than predicting the mean. A multi-step forecasting objective
   would be fitting noise, so `forecast_horizon: 1`.
3. **The bar is AR(1) at 0.435 MSE** (56.6% of variance, z-scored). The latent RNN
   is only worth its parameters if it beats a per-region scalar.
4. **Sites do not share a TR.** Lag-1 autocorrelation ranges 0.534 to 0.874 across
   sites — a "1-step" window means a different amount of elapsed time depending on
   where the subject was scanned. Pooling naively averages over incompatible time
   axes.
5. **But the dynamics are not AR(1) either.** `acf(2)/acf(1)²` is 0.43–0.72 (negative
   at three sites) where a first-order Markov process would give 1.0 — so a model
   with memory has genuine headroom over the baseline.

---

## Model architecture

### 1. Encoder  `E: x_t → z_t`
- Input: one fMRI frame `x_t ∈ ℝ^{N_regions}` (or a 3D patch grid).
- MLP / 1D-conv encoder → latent `z_t ∈ ℝ^d` (e.g. `d = 128–256`).
- Optional variational bottleneck (VAE-style) for a smooth, regularized latent space.

### 2. Foundation model  `F: z_{1:t} → ẑ_{t+1}`
- Recurrent core (GRU or LSTM) rolling over the latent sequence.
- Learns temporal dynamics of brain states; the hidden state is the transferable
  "brain dynamics" representation.
- Trained to (a) predict the next latent autoregressively and (b) fill masked latents.

### 3. Decoder  `D: z_t → x̂_t`
- Mirrors the encoder; maps latent back to ROI space.
- Reconstruction loss ties the latent space to real signal.

### Training objectives (self-supervised, combined)
- **Reconstruction:** `‖D(E(x_t)) − x_t‖²` — keeps latents informative.
- **Latent forecasting:** `‖F(z_{1:t}) − z_{t+1}‖²` and/or reconstruct `x_{t+1}` from
  the predicted latent — the core dynamics objective.
- **Masked latent modeling:** randomly mask timepoints; reconstruct them from context.
- (Optional) **KL** term if the encoder is variational.

---

## Repository structure

```
foundational-brain/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml          # data / model / training hyperparameters
├── data/                     # (gitignored) raw + preprocessed fMRI
├── notebooks/                # exploration, visualization
├── src/
│   └── foundational_brain/
│       ├── data/             # loading, parcellation, normalization, datasets
│       ├── models/           # encoder, decoder, latent RNN, full model
│       ├── training/         # loops, losses, optimizers, checkpointing
│       └── eval/             # reconstruction/forecast metrics, probing tasks
├── scripts/                  # entrypoints: preprocess, pretrain, evaluate
└── tests/
```

---

## Roadmap

**Phase 0 — Scaffolding** ✅ (this commit)
- Repo, README plan, environment, project skeleton.

**Phase 1a — Data acquisition & exploration** ✅
- ABIDE-PCP fetchers + nilearn-based parcellation for raw NIfTI.
- Full corpus profile → `docs/data_report.md`, config set from measurements.
- Statistics tested against analytically-known synthetic answers.

**Phase 1b — Data pipeline** (next)
- Windowing into `seq_len=64` sequences; per-region z-scoring; flat-region drop.
- **Resolve the TR problem** (finding 4): source per-site TR and either resample to
  a common sampling interval, condition the model on TR, or restrict pretraining to
  one TR group. This is a prerequisite for pooling sites, not an optimization.
- `Dataset`/`DataLoader`, train/val/test split by subject, stratified by site.

**Phase 2 — Autoencoder (spatial only)**
- Implement Encoder + Decoder; train per-frame reconstruction.
- Validate the latent space (variance explained, reconstruction quality).

**Phase 3 — Latent dynamics model (the foundation)**
- Add the RNN over latents; autoregressive next-frame + masked-latent objectives.
- Train end-to-end (or freeze encoder first, then joint fine-tune).
- **Success criterion, set by the data:** beat 0.435 MSE at a 1-TR horizon,
  reported per site (the per-site bar ranges 0.240–0.713).

**Phase 4 — Scale pretraining**
- Multiple datasets/subjects; larger latent + RNN; logging (W&B/TensorBoard);
  checkpointing; mixed precision.

**Phase 5 — Evaluation & transfer**
- Intrinsic: reconstruction / forecasting error, latent trajectory analysis.
- Extrinsic (linear probing on frozen features): subject ID, task/state, age, etc.
- Compare against baselines (PCA, raw-signal models, published fMRI foundation models
  such as BrainLM).

**Phase 6 — Extensions**
- Variational / diffusion decoder, voxel-level encoder, state-space (Mamba/S4) latent
  core, cross-subject/cross-scanner generalization.

---

## Setup

Python **3.12** (3.14 has no torch wheels yet, and its bundled `ensurepip` is
broken on Homebrew):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Reproduce the data report (downloads ~870 files into the gitignored `data/`,
roughly 15 minutes on a first run, cached afterwards):

```bash
.venv/bin/python scripts/explore_data.py --n-subjects 0
.venv/bin/python -m pytest
```

## Status

🚧 Phases 0 and 1a complete: scaffolding, public data acquisition, and a full
profile of the ABIDE corpus with hyperparameters set from it. Next is Phase 1b —
the windowing/`DataLoader` pipeline, and resolving cross-site TR heterogeneity.

## License

MIT (see `LICENSE`).
