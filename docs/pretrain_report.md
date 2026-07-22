# Pretraining report

- 522 subjects in the TR-homogeneous pretraining group (of 871 total)
- 182 regions, seq_len 64, latent_dim 128
- split subjects: train=419, val=52, test=51
- windows: train=2199, val=141, test=138

## Baselines (fitted on train, scored on val)

| predictor | 1-TR forecast MSE |
|---|---|
| predict the mean | 1.0027 |
| persistence | 0.4367 |
| AR(1) | **0.3891** |

Reconstruction bar: PCA-128 at **0.0750**.

## Phase 2 — autoencoder

Validation reconstruction MSE **0.0859**, **does NOT beat** the bar (-14.5% vs 0.0750).

## Phase 3 — full model

Validation 1-TR forecast MSE **0.2679**, **beats** the bar (+31.1% vs 0.3891).
Test 1-TR forecast MSE **0.2723**.
Validation reconstruction MSE 0.1816.

## Cross-TR generalization (held-out sites)

349 subjects from sites whose TR was never seen in training.

- model 1-TR forecast MSE **0.3347**
- AR(1) on the same subjects: 0.4614
