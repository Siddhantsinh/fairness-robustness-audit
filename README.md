# Fairness and Robustness Audit of an ML Model

An educational audit of a binary classification model: overall performance,
group-disaggregated fairness metrics, robustness under noise/missingness/
threshold/shift, and one mitigation experiment. This is a methodology and
transparency exercise, not a claim that any single metric proves a system
"fair."

## Research question (fill this in before writing any model code)

> Does a binary classification model perform differently across
> **sex and race groups**, and how do robustness conditions and a selected mitigation
> method affect both predictive performance and group-level disparities?

See `configs/config.yaml` for the locked parameters (dataset, target,
groups, metrics, seed, thresholds).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py  # acquires data/raw.csv
```

## Project layout

```text
fairness-robustness-audit/
├── README.md
├── requirements.txt
├── data/README.md              # dataset source, license, download steps
├── configs/config.yaml         # locked experiment configuration
├── notebooks/01_data_exploration.ipynb
├── src/audit_project/
│   ├── data.py                 # loading + validation
│   ├── features.py             # preprocessing pipeline
│   ├── train.py                # model fitting
│   ├── evaluate.py             # overall metrics
│   ├── fairness.py             # group metrics
│   └── robustness.py           # noise / missingness / threshold / shift
├── tests/
├── reports/figures/
└── scripts/                    # CLI entry points
```

## Running the pipeline

```bash
python scripts/run_pipeline.py --config configs/config.yaml --data data/raw.csv
```

## Status

- [x] Dataset selected, license recorded
- [x] Data card written (`data/README.md`)
- [x] Evaluation protocol locked
- [x] Baseline pipeline runs end to end
- [x] Fairness audit
- [x] Robustness audit
- [ ] Mitigation experiment
- [x] Tests passing
- [ ] Model card + technical report
