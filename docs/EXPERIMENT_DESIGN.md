# Applied Energy experiment design freeze

Note: scripts and result directories retain the working name
`applied_energy` from the extended journal version of this work.

## Forecasting protocol

- One forecast decision per local delivery day, issued at 12:00 on D-1.
- Target: the 24 hourly day-ahead prices of delivery day D.
- History: 336 hours ending immediately before the issue time.
- Markets: DE-LU, FR, NL and BE.
- Train / validation / test delivery periods: 2019-01-16--2022-12-31,
  2023-01-01--2023-06-30 and 2023-07-01--2024-12-31.
- DST delivery or D-1 context days with 23/25 hours are excluded and counted.
- Forecast-product load and wind/solar fields are used; actual-load files are
  excluded. Realized flow is shifted 24 h and financial closes are shifted one
  day. Normalization is estimated from the training period only.
- The D-1 published day-ahead price curve and target-day load/renewable
  forecast products are treated as known at issue time.

## CMA-Former structure

1. Market-wise patch embedding and shared intra-market temporal encoding.
2. Learnable market identity embeddings.
3. Target-day known-covariate encoding before cross-market exchange.
4. Aligned-patch cross-market attention with a learnable residual scale and
   fully data-driven pairwise gates.
5. A corrected direct-interconnection topology is evaluated only as a
   sensitivity analysis. It is not part of the main CMA-Former claim because
   the DE-LU--BE link became commercially available during the sample period.
6. A shared D-1 published-price residual anchor used by every neural model.
7. Quantile outputs at 0.1, 0.5 and 0.9.

## Comparisons

Core models use seeds 11, 22, 42, 77 and 123: CMA-Former, PatchTST, DLinear,
N-BEATSx, TFT, iTransformer, Temporal-GAT, independent-market PatchTST and a
directly concatenated multivariate Transformer. Deterministic controls are
LEAR and the previous-day published-price curve.

Structural controls use the same five seeds: no cross-market attention,
learnable topology-prior bias, fixed topology prior, shuffled topology prior
and target-day future information integrated only after cross-market attention.

## Development-only diagnostics

Earlier results under hourly overlapping origins, mixed load products, the
unfrozen attention block, or the pre-freeze future-covariate path are preserved
for traceability but are excluded from formal tables. Pilot results were used
to diagnose protocol and architecture failures. After this freeze, the model
structure and evaluation protocol must not be changed in response to formal
test rankings.

## Scope notes

The topology prior is a market-connectivity control, not a power-flow
constraint. Cross-market attention diagnostics are model-internal
information-use summaries, not causal spillover estimates.
