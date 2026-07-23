# Latent-width ablation: autoencoder vs PCA

Reconstruction MSE on the validation split, 182 regions. PCA is fitted at each width on the same training subjects the autoencoder trains on.

| latent_dim | autoencoder | PCA | AE − PCA | winner |
|---|---|---|---|---|
| 8 | 0.6052 | 0.6012 | +0.7% | PCA |
| 16 | 0.4794 | 0.4749 | +0.9% | PCA |
| 32 | 0.3622 | 0.3526 | +2.7% | PCA |
| 64 | 0.2294 | 0.2219 | +3.4% | PCA |
| 128 | 0.0825 | 0.0750 | +10.0% | PCA |

## Reading
- The autoencoder loses to PCA at **every** width: for pure reconstruction the nonlinear encoder is not earning its parameters. Its value, if any, has to come from the dynamics objective, not reconstruction.