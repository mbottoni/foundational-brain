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

Validation reconstruction MSE **0.0865**, **does NOT beat** the bar (-15.3% vs 0.0750).

## Phase 3 — full model

Validation 1-TR forecast MSE **0.2679**, **beats** the bar (+31.1% vs 0.3891).
Test 1-TR forecast MSE **0.2723**.
Validation reconstruction MSE 0.1816.

### Per-subject paired test vs AR(1)

The unit is the subject, not the window; the bootstrap resamples subjects. Win rate is shown alongside the mean so a result carried by a few outliers would be visible.

| split | n | model | AR(1) | improvement | win rate | 95% CI | p |
|---|---|---|---|---|---|---|---|
| val | 52 | 0.2679 | 0.3939 | +32.0% | 100.0% | [0.1152, 0.1364] | 3.50e-10 |
| test | 51 | 0.2725 | 0.4022 | +32.2% | 100.0% | [0.1182, 0.1412] | 5.15e-10 |

## Cross-TR generalization (held-out sites)

349 subjects from sites whose TR was never seen in training.

- model 1-TR forecast MSE **0.3347**
- AR(1) on the same subjects: 0.4614
