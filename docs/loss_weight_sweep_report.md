# Loss-weight sweep: reconstruction weight vs forecasting quality

Forecast and masked weights fixed at 1.0. AR(1) bar: **0.3891**. Lower forecast MSE is better.

| w_reconstruction | val forecast MSE | val reconstruction MSE | vs AR(1) |
|---|---|---|---|
| 0.0 | 0.3303 | 1.5606 | beats |
| 0.1 | 0.3538 | 0.3505 | beats |
| 0.3 | 0.3464 | 0.2896 | beats |
| 1.0 | 0.3085 | 0.2074 | beats |

## Reading

- Best forecasting at **w_reconstruction = 1.0** (0.3085).
- The default 1:1:1 is at or near the best; reconstruction is not meaningfully hurting forecasting, so keeping it as a regulariser costs nothing.
- With reconstruction fully off (w=0), forecast MSE is 0.3303 and reconstruction MSE is 1.5606 — a check on whether the decoder is needed at all for the forecasting objective.