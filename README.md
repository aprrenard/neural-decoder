# Neural Lick Decoder

Predicting continuous licking behaviour from population neural activity, with a focus on two axes of decoder robustness: cross-animal generalisation and within-session representational plasticity.

## Scientific questions

**Q1 — Cross-animal generalisation:** How much of the neural lick code is conserved across individuals? Models are trained on a subset of animals and evaluated on a held-out naive animal (leave-one-animal-out cross-validation). Decoder performance quantifies how stereotyped vs. idiosyncratic the lick representation is across individuals.

**Q2 — Within-session plasticity:** Does a decoder trained on early-session activity generalise to late-session activity? Each session is split into early / mid / late epochs; the model is trained on the early epoch and evaluated on mid and late. Decoder degradation or improvement is a readout of how much the neural code reorganised during fast within-session learning.

## Data

- **Neural recordings:** calcium imaging or electrophysiology (one session per animal)
- **Behavioural signal:** continuous jaw/lick trace from DeepLabCut pose estimation
- **Trial structure:** auditory and whisker stimuli presented randomly; mice lick in response
- **Input format:** trial-aligned segments, `-0.5 s` to `+2 s` around stimulus onset
- **Format:** NWB (read via [CICADA NWB](https://gitlab.com/cossartlab/cicada_nwb))

Raw and processed data are not tracked in git.

## Models

| Model | Role |
|---|---|
| Ridge regression | Baseline and interpretability anchor |
| LSTM | Sequence decoder exploiting temporal structure |
| Transformer | Attention-based sequence decoder |

Every model must outperform ridge regression to justify its complexity. All runs are tracked in MLflow.

## Project structure

```
neural-decoder/
├── src/
│   ├── data/
│   │   ├── ingestion.py       # NWB → numpy arrays
│   │   └── preprocessing.py   # filtering, alignment, epoch splitting
│   ├── models/
│   │   ├── baseline.py        # ridge regression
│   │   ├── lstm.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── train.py           # training loop + MLflow logging
│   │   └── evaluate.py        # LOAO eval (Q1) + epoch eval (Q2)
│   ├── pipeline/
│   │   └── flow.py            # Prefect flow: ingest → preprocess → train → evaluate
│   └── api/
│       └── main.py            # FastAPI: POST /predict
├── tests/
├── notebooks/
│   └── exploration.ipynb
└── data/                      # gitignored; lives locally or in GCP storage
```

## Stack

Python 3.12 · PyTorch · NumPy / SciPy / pandas · MLflow · Prefect · FastAPI · Evidently AI · Docker · GitHub Actions

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Run the full pipeline
uv run prefect run src/pipeline/flow.py
```

