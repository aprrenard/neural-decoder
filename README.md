# Neural Lick Decoder

Predicting continuous licking behaviour from population neural activity, with a focus on two axes of decoder robustness: cross-animal generalisation and within-session representational plasticity.

## Scientific questions

**Q1 — Cross-animal generalisation:** How much of the neural lick code is conserved across individuals? Models are trained on a subset of animals and evaluated on a held-out naive animal (leave-one-animal-out cross-validation). Decoder performance quantifies how stereotyped vs. idiosyncratic the lick representation is across individuals.

**Q2 — Within-session plasticity:** Does a decoder trained on early-session activity generalise to late-session activity? Each session is split into early / mid / late epochs; the model is trained on the early epoch and evaluated on mid and late. Decoder degradation or improvement is a readout of how much the neural code reorganised during fast within-session learning.

## Data

- **Neural recordings:** extracellular electrophysiology, one session per animal; spike trains binned at 10 ms
- **Behavioural signal:** continuous jaw/lick trace from DeepLabCut pose estimation
- **Trial structure:** auditory and whisker stimuli presented randomly; mice lick in response
- **Input format:** trial-aligned segments, `-2 s` to `+3 s` around stimulus onset
- **Format:** NWB (read via [CICADA NWB](https://gitlab.com/cossartlab/cicada_nwb))

Raw and processed data are not tracked in git.

## Models

| Model | Role | Status |
|---|---|---|
| Ridge regression | Baseline and interpretability anchor (lagged population vector → lick value) | Implemented |
| LSTM | Sequence decoder exploiting temporal structure | Planned |
| Transformer | NDT-style encoder: masked spike-count pretraining (Poisson loss), then fine-tuning for lick decoding; shared encoder with per-animal read-in/read-out layers | Planned |

Every model must outperform ridge regression to justify its complexity. All runs are tracked in MLflow.

The transformer follows the Neural Data Transformer design (Ye & Pandarinath, 2021). Design choices, literature context, and the evaluation protocol are documented in [`docs/transformer-design-notes.md`](docs/transformer-design-notes.md).

## Evaluation

- **Q1:** leave-one-animal-out cross-validation. Self-supervised pretraining only ever sees the training animals of a fold; hyperparameters are fixed before the final fold sweep; results are reported per held-out animal.
- **Q2:** early → mid / late epoch transfer within each session, with no leakage across epochs.
- **Metrics:** R², Pearson r, MSE on the held-out lick trace.

## Project structure

```
neural-decoder/
├── docs/
│   └── transformer-design-notes.md
├── src/
│   ├── data/
│   │   ├── ingestion.py       # NWB → trial-aligned .npz (neural PSTH + DLC)
│   │   └── preprocessing.py   # resampling, design matrices, LOAO / epoch splits
│   ├── models/
│   │   ├── baseline.py        # ridge regression
│   │   ├── lstm.py            # planned
│   │   └── transformer.py     # planned
│   ├── training/
│   │   ├── train.py           # training + MLflow logging
│   │   └── evaluate.py        # within-session (Q2) + LOAO (Q1) evaluation
│   ├── pipeline/
│   │   └── flow.py            # planned — Prefect flow: ingest → preprocess → train → evaluate
│   └── api/
│       └── main.py            # planned — FastAPI: POST /predict
├── tests/
├── notebooks/
└── data/                      # gitignored; lives locally or in GCP storage
```

## Stack

Python 3.12 · PyTorch · NumPy / SciPy / pandas / scikit-learn · MLflow · Prefect · FastAPI · Evidently AI · Docker · GitHub Actions

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Ingest NWB sessions into trial-aligned arrays (data/nwb → data/processed)
uv run python -m src.data.ingestion

# Fit and evaluate the ridge baseline (logged to MLflow)
uv run python -m src.training.evaluate

# Browse runs
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
