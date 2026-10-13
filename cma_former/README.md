# CMA-Former

Code release for the paper:

**CMA-Former: Context-Conditioned Cross-Market Attention for Day-Ahead Electricity
Price Forecasting**

CMA-Former forecasts the 24 hourly day-ahead electricity prices of four coupled
European bidding zones (DE-LU, FR, NL, BE) once per delivery day. It combines:

1. **Market-wise temporal encoding** — historical patches are encoded
   independently per market with shared temporal weights and learnable market
   identity embeddings.
2. **Context-conditioned cross-market attention** — projected delivery-day
   covariates (published D-1 price curve, target-day load and renewable
   forecasts) condition gated attention across markets at temporally aligned
   patch positions.
3. **Persistence-residual quantile head** — predictions are residuals around
   the published previous-day price curve, corrected by a shared hourly
   context adapter, with quantile outputs at 0.1/0.5/0.9.

## Setup

```bash
conda create -n cma_former python=3.10
conda activate cma_former
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

Raw inputs are obtained from public sources and placed under `data/raw/`
(see `src/entsoe_downloader.py` and the download notes in the paper):

- ENTSO-E Transparency Platform: day-ahead prices, forecast load,
  wind/solar generation forecasts, cross-border physical flows
  (DE-LU, FR, NL, BE; 2019-01-01 to 2024-12-31, hourly)
- EEX / public sources: daily TTF gas and EUA carbon prices

Build the processed dataset:

```bash
python experiments/prepare_applied_energy_data.py
python experiments/summarize_applied_energy_dataset.py   # integrity summary
```

## Running experiments

```bash
python experiments/run_applied_energy_matrix.py --config configs/applied_energy.yaml
python experiments/run_cpu_dayahead_baselines.py          # LEAR + naive controls
python experiments/evaluate_applied_energy.py             # aggregate metrics
python experiments/verify_applied_energy_results.py       # verification
```

A single model/seed run for debugging:

```bash
python experiments/run_applied_energy.py --model cma_former --seed 42
```

Unit tests:

```bash
pytest tests/ -q
```

## Protocol notes

The forecasting protocol is frozen in
`docs/EXPERIMENT_DESIGN_FREEZE.md`: one forecast per delivery day issued at
12:00 local time on D-1, 23/25-hour DST days excluded, training-period-only
normalization, and causal lag rules for flow and financial series. All
reported comparisons follow this protocol without modification.

## License

MIT License. See [LICENSE](LICENSE).
